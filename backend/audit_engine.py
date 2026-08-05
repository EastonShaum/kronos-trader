"""
Audit Engine — periodic AI-driven review of trading performance.
Generates structured recommendations stored for manual approval.

LLM calls use Claude Code headless mode (`claude -p`) so no Console
API key is required — relies on `claude auth login` subscription.
"""
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from database import AuditRecommendation, Prediction, Trade, Portfolio, MarketType, TradeStatus

logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Safety bounds for proposed parameter values
# -------------------------------------------------------
PARAM_BOUNDS = {
    "stop_loss_pct":          (-10.0, -1.0),
    "take_profit_pct":        (2.0,  20.0),
    "min_confidence":         (50.0, 95.0),
    "min_predicted_move_pct": (0.5,  10.0),
    "position_size_pct":      (5.0,  30.0),
}


def _safe_float(val, default=0.0):
    """Return float, replacing None/NaN/Inf with default."""
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# -------------------------------------------------------
# Claude Code headless caller
# -------------------------------------------------------

def _find_claude_binary() -> str:
    """
    Return path to the `claude` binary.
    Checks CLAUDE_BINARY env var first, then PATH, then common npm global locations.
    Raises FileNotFoundError if not found.
    """
    env_override = os.getenv("CLAUDE_BINARY", "").strip()
    if env_override and os.path.isfile(env_override) and os.access(env_override, os.X_OK):
        return env_override

    found = shutil.which("claude")
    if found:
        return found
    candidates = [
        os.path.expanduser("~/.npm-global/bin/claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/usr/local/bin/claude",
        "/usr/bin/claude",
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "claude binary not found. Run: npm install -g @anthropic-ai/claude-code && claude auth login"
    )


def call_claude_headless(prompt: str, timeout_sec: int = 120) -> dict:
    """
    Invoke Claude Code in headless mode and return the parsed inner JSON.
    Does NOT use ANTHROPIC_API_KEY — relies on `claude auth login`.

    The `claude -p` envelope response looks like:
      {"type": "result", "result": "<text>", "is_error": false, ...}
    We extract .result and JSON-parse it.
    """
    claude_bin = _find_claude_binary()
    tmp_path = None

    try:
        prompt_bytes = prompt.encode("utf-8")
        use_file = len(prompt_bytes) > 8_000_000

        if use_file:
            fd, tmp_path = tempfile.mkstemp(suffix=".txt")
            with os.fdopen(fd, "w") as f:
                f.write(prompt)
            cmd_prompt = (
                f"Read the audit data and instructions from {tmp_path} "
                f"and follow them exactly, responding with only the "
                f"JSON object it specifies."
            )
            allowed_tools = "Read"
        else:
            cmd_prompt = prompt
            allowed_tools = ""

        cmd = [
            claude_bin, "-p", cmd_prompt,
            "--output-format", "json",
            "--allowedTools", allowed_tools,
            "--max-turns", "1",
        ]

        logger.info(f"[audit] Calling claude headless (timeout={timeout_sec}s, file={use_file})")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        if result.returncode != 0:
            stderr_snippet = result.stderr.strip()[:500]
            raise RuntimeError(f"claude -p exited {result.returncode}: {stderr_snippet}")

        if not result.stdout.strip():
            raise RuntimeError("claude -p returned empty stdout")

        envelope = json.loads(result.stdout)

        if envelope.get("is_error"):
            raise RuntimeError(f"claude -p returned is_error=true: {envelope.get('result', '')[:300]}")

        raw_text = envelope.get("result", "").strip()
        if not raw_text:
            raise RuntimeError(f"claude -p envelope has empty result field: {envelope}")

        # Strip markdown fences defensively
        raw_text = (
            raw_text
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        return json.loads(raw_text)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# -------------------------------------------------------
# Data collection
# -------------------------------------------------------

def collect_audit_data(db: Session, lookback_days: int) -> dict:
    """
    Pull and aggregate trading performance over lookback_days.
    Returns a JSON-serializable dict — all floats are safe (no NaN/Inf).
    """
    import pathlib as _pl
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    # --- Trades in window ---
    trades = db.query(Trade).filter(Trade.opened_at >= cutoff).all()
    closed_trades = [
        t for t in trades
        if t.status == TradeStatus.CLOSED and t.realized_pnl is not None
    ]

    # Per-symbol stats
    symbol_stats: dict = {}
    for t in closed_trades:
        s = symbol_stats.setdefault(t.symbol, {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0})
        s["count"] += 1
        s["pnl"] += _safe_float(t.realized_pnl)
        if t.realized_pnl >= 0:
            s["wins"] += 1
        else:
            s["losses"] += 1

    symbol_summary = {}
    for sym, st in symbol_stats.items():
        total = st["count"]
        symbol_summary[sym] = {
            "trade_count":  total,
            "win_rate_pct": round(st["wins"] / total * 100, 1) if total else 0,
            "avg_pnl":      round(st["pnl"] / total, 4) if total else 0,
            "total_pnl":    round(st["pnl"], 4),
        }

    # --- Prediction accuracy in window ---
    completed_preds = db.query(Prediction).filter(
        Prediction.created_at >= cutoff,
        Prediction.was_correct.isnot(None),
    ).all()

    pred_by_symbol: dict = {}
    for p in completed_preds:
        bucket = pred_by_symbol.setdefault(p.symbol, {"correct": 0, "total": 0, "conf_buckets": {}})
        bucket["total"] += 1
        if p.was_correct:
            bucket["correct"] += 1
        conf_pct = round(_safe_float(p.confidence) * 100)
        bkt_key = f"{(conf_pct // 10) * 10}-{(conf_pct // 10) * 10 + 9}%"
        cb = bucket["conf_buckets"].setdefault(bkt_key, {"total": 0, "correct": 0})
        cb["total"] += 1
        if p.was_correct:
            cb["correct"] += 1

    pred_accuracy = {}
    for sym, b in pred_by_symbol.items():
        pred_accuracy[sym] = {
            "accuracy_pct":      round(b["correct"] / b["total"] * 100, 1) if b["total"] else 0,
            "total_predictions": b["total"],
            "calibration": {
                k: {
                    "hit_rate_pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
                    "sample_count": v["total"],
                }
                for k, v in b["conf_buckets"].items()
            },
        }

    # Dual-model accuracy comparison
    small_preds = [p for p in completed_preds if p.model_name == "small"]
    mini_preds  = [p for p in completed_preds if p.model_name == "mini"]

    def _acc(preds):
        if not preds:
            return None
        correct = sum(1 for p in preds if p.was_correct)
        return {"accuracy_pct": round(correct / len(preds) * 100, 1), "count": len(preds)}

    model_comparison = {"small": _acc(small_preds), "mini": _acc(mini_preds)}

    # --- Portfolio-level stats ---
    portfolios = db.query(Portfolio).all()
    portfolio_summary = []
    for port in portfolios:
        market_trades = [t for t in closed_trades if t.portfolio_id == port.id]
        total_pnl = sum(_safe_float(t.realized_pnl) for t in market_trades)
        portfolio_summary.append({
            "market":           port.market.value,
            "starting_balance": _safe_float(port.starting_balance),
            "current_equity":   _safe_float(port.cash_balance + port.total_pnl),
            "total_pnl":        round(total_pnl, 2),
            "trade_count":      len(market_trades),
        })

    # --- Current settings ---
    settings_file = _pl.Path(__file__).parent / "settings.json"
    current_settings: dict = {}
    if settings_file.exists():
        try:
            current_settings = json.loads(settings_file.read_text())
        except Exception:
            pass

    return {
        "lookback_days":       lookback_days,
        "generated_at":        datetime.utcnow().isoformat(),
        "symbol_performance":  symbol_summary,
        "prediction_accuracy": pred_accuracy,
        "model_comparison":    model_comparison,
        "portfolio_summary":   portfolio_summary,
        "current_settings":    current_settings,
        "total_closed_trades": len(closed_trades),
        "total_open_trades":   len([t for t in trades if t.status.value == "open"]),
    }


# -------------------------------------------------------
# Prompt builder — returns a single combined string for claude -p
# -------------------------------------------------------

def build_audit_prompt(data: dict, audit_type: str) -> str:
    categories = ["risk_param", "symbol_list"]
    if audit_type == "monthly":
        categories.append("model_strategy")

    schema_example = {
        "recommendations": [
            {
                "category":       "risk_param | symbol_list | model_strategy",
                "param":          "stop_loss_pct",
                "current_value":  "-3.0",
                "proposed_value": "-2.5",
                "confidence":     "low | medium | high",
                "reasoning":      "2-3 sentences citing specific numbers from the data.",
            }
        ],
        "summary": "One-paragraph executive summary of findings.",
    }

    instructions = (
        "You are a quantitative analyst reviewing a paper trading system's performance data. "
        "Your role is to recommend specific, bounded parameter changes ONLY when the data clearly supports them. "
        "Do not recommend a change without at least 10 trades of supporting evidence for that symbol/parameter. "
        f"Categories you may recommend: {categories}. "
        "For risk_param: param must be one of stop_loss_pct, take_profit_pct, min_confidence, "
        "min_predicted_move_pct, position_size_pct. "
        "For symbol_list: param is 'add_symbol' or 'remove_symbol', proposed_value is the symbol name. "
        "For model_strategy: param is 'active_model' or 'dual_model_mode' or 'require_agreement'. "
        "All numeric values as plain decimals (e.g. '-3.5' for stop_loss_pct meaning -3.5%). "
        "Respond ONLY with valid JSON matching the schema — no prose, no markdown fences."
    )

    return (
        f"{instructions}\n\n"
        f"Audit type: {audit_type}\n\n"
        f"Performance data:\n{json.dumps(data, indent=2)}\n\n"
        f"Respond with JSON matching this schema exactly:\n{json.dumps(schema_example, indent=2)}"
    )


# -------------------------------------------------------
# Bounds validation
# -------------------------------------------------------

def _validate_bounds(rec: dict) -> dict:
    """
    Check proposed_value against PARAM_BOUNDS.
    Sets confidence='out_of_bounds' and clamps the value if outside range.
    """
    if rec.get("category") != "risk_param":
        return rec

    param = rec.get("param", "")
    bounds = PARAM_BOUNDS.get(param)
    if bounds is None:
        return rec

    try:
        val = float(rec.get("proposed_value", ""))
    except (TypeError, ValueError):
        return rec

    lo, hi = bounds
    if val < lo or val > hi:
        clamped = max(lo, min(hi, val))
        rec["confidence"]     = "out_of_bounds"
        rec["proposed_value"] = str(clamped)
        rec["reasoning"] = (
            f"[CLAMPED from {val} to {clamped} — outside allowed bounds "
            f"[{lo}, {hi}]] " + rec.get("reasoning", "")
        )
    return rec


# -------------------------------------------------------
# Main audit runner
# -------------------------------------------------------

def run_audit(db: Session, audit_type: str = "weekly") -> list[dict]:
    """
    Collect data, call Claude via headless CLI, validate bounds, store rows.
    Returns list of stored recommendation dicts for logging/notification.
    """
    logger.info(f"[audit] Starting {audit_type} audit...")

    lookback_days = int(os.getenv(
        "AUDIT_LOOKBACK_DAYS_MONTHLY" if audit_type == "monthly" else "AUDIT_LOOKBACK_DAYS_WEEKLY",
        30 if audit_type == "monthly" else 7,
    ))
    timeout_sec = int(os.getenv("AUDIT_CLAUDE_TIMEOUT_SEC", 120))

    data = collect_audit_data(db, lookback_days)

    if data["total_closed_trades"] == 0:
        logger.info("[audit] No closed trades in lookback window — skipping audit.")
        return []

    prompt = build_audit_prompt(data, audit_type)

    try:
        parsed = call_claude_headless(prompt, timeout_sec=timeout_sec)
    except FileNotFoundError as e:
        logger.error(f"[audit] Claude binary not found: {e}")
        return []
    except subprocess.TimeoutExpired:
        logger.error(f"[audit] claude -p timed out after {timeout_sec}s")
        return []
    except (RuntimeError, json.JSONDecodeError) as e:
        logger.error(f"[audit] Claude call failed: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"[audit] Unexpected error calling Claude: {e}", exc_info=True)
        return []

    recs_raw = parsed.get("recommendations", [])
    if not isinstance(recs_raw, list):
        logger.warning(f"[audit] Unexpected recommendations structure: {type(recs_raw)}")
        return []

    stored = []
    for rec in recs_raw:
        if not isinstance(rec, dict):
            continue
        rec = _validate_bounds(rec)
        db.add(AuditRecommendation(
            audit_type     = audit_type,
            category       = rec.get("category", ""),
            param          = rec.get("param", ""),
            current_value  = str(rec.get("current_value", "")),
            proposed_value = str(rec.get("proposed_value", "")),
            confidence     = rec.get("confidence", "low"),
            reasoning      = rec.get("reasoning", ""),
            status         = "pending",
        ))
        stored.append(rec)

    db.commit()
    logger.info(f"[audit] {audit_type} audit complete — {len(stored)} recommendations stored.")
    return stored