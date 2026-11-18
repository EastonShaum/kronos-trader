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
import time
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

import notifications
from database import (
    AuditRecommendation, Prediction, Trade, Portfolio, MarketType,
    TradeStatus, TradeDirection,
)
# MIN_TRADES_FOR_CI is re-exported so existing callers and tests keep working.
from stats_bootstrap import MIN_TRADES_FOR_CI, block_bootstrap_ci

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
            stdout_snippet = result.stdout.strip()[:500]
            raise RuntimeError(
                f"claude -p exited {result.returncode}: stderr={stderr_snippet!r} stdout={stdout_snippet!r}"
            )

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
    # A round trip is stored as two rows: a BUY (entry) and a SELL (exit).
    # Only the SELL carries the result. BUY rows default realized_pnl to 0.0,
    # so counting them doubles trade counts AND scores every entry as a win
    # under `realized_pnl >= 0`. Filter to exits.
    trades = db.query(Trade).filter(Trade.opened_at >= cutoff).all()
    closed_trades = [
        t for t in trades
        if t.status == TradeStatus.CLOSED
        and t.direction == TradeDirection.SELL
        and t.realized_pnl is not None
    ]

    def _net(t) -> float:
        """Net of fees and slippage where recorded; realized is gross."""
        return _safe_float(t.net_pnl if t.net_pnl is not None else t.realized_pnl)

    # Per-symbol stats
    symbol_stats: dict = {}
    for t in closed_trades:
        s = symbol_stats.setdefault(t.symbol, {"wins": 0, "losses": 0, "pnl": 0.0,
                                               "count": 0, "returns": []})
        s["count"] += 1
        pnl = _net(t)
        s["pnl"] += pnl
        s["returns"].append(pnl)
        if pnl >= 0:
            s["wins"] += 1
        else:
            s["losses"] += 1

    symbol_summary = {}
    for sym, st in symbol_stats.items():
        total = st["count"]
        entry = {
            "trade_count":  total,
            "win_rate_pct": round(st["wins"] / total * 100, 1) if total else 0,
            "avg_pnl":      round(st["pnl"] / total, 4) if total else 0,
            "total_pnl":    round(st["pnl"], 4),
            "pnl_basis":    "net_pnl (after fees and slippage)",
        }
        # An interval, so a mean built on a handful of trades cannot be read as
        # a finding. Symbols below the bar get no CI and are marked explicitly.
        if total >= MIN_TRADES_FOR_CI:
            ci = block_bootstrap_ci(st["returns"], reps=2000, seed=0)
            entry["mean_pnl_95ci"] = [round(ci["lower"], 4), round(ci["upper"], 4)]
            entry["ci_excludes_zero"] = ci["upper"] < 0 or ci["lower"] > 0
            entry["insufficient_sample"] = False
        else:
            entry["mean_pnl_95ci"] = None
            entry["ci_excludes_zero"] = False
            entry["insufficient_sample"] = True
            entry["sample_note"] = (
                f"only {total} closed round trips (< {MIN_TRADES_FOR_CI}); "
                f"not enough to distinguish this symbol from noise"
            )
        symbol_summary[sym] = entry

    # --- Prediction accuracy in window ---
    completed_preds = db.query(Prediction).filter(
        Prediction.created_at >= cutoff,
        Prediction.was_correct.isnot(None),
    ).all()

    # Entry requires confidence >= min_confidence, so predictions below it can
    # never open a trade. Predicted-move buckets are restricted to the
    # tradeable population; otherwise they describe signals the engine ignores.
    min_conf_setting = 0.75
    try:
        _sf = _pl.Path(__file__).parent / "settings.json"
        if _sf.exists():
            min_conf_setting = _safe_float(
                json.loads(_sf.read_text()).get("risk", {}).get("min_confidence", 0.75), 0.75)
    except Exception:
        pass

    pred_by_symbol: dict = {}
    for p in completed_preds:
        bucket = pred_by_symbol.setdefault(
            p.symbol, {"correct": 0, "total": 0, "conf_buckets": {}, "move_buckets": {}})
        bucket["total"] += 1
        if p.was_correct:
            bucket["correct"] += 1

        # Bucket keys are prefixed with the quantity they bucket. An earlier
        # audit read a bare "0-9%" confidence key as a predicted-move band and
        # recommended changing min_predicted_move_pct on the strength of it.
        conf_pct = round(_safe_float(p.confidence) * 100)
        bkt_key = f"confidence_{(conf_pct // 10) * 10}-{(conf_pct // 10) * 10 + 9}%"
        cb = bucket["conf_buckets"].setdefault(bkt_key, {"total": 0, "correct": 0})
        cb["total"] += 1
        if p.was_correct:
            cb["correct"] += 1

        # Predicted-move buckets: the evidence min_predicted_move_pct actually
        # needs, and which the audit previously had no data for at all.
        cur = _safe_float(p.current_price)
        if cur > 0 and _safe_float(p.confidence) >= min_conf_setting:
            move_pct = abs(_safe_float(p.predicted_close) - cur) / cur * 100
            lo = min(int(move_pct // 2) * 2, 10)
            mv_key = f"predicted_move_{lo}-{lo + 2}%" if lo < 10 else "predicted_move_>=10%"
            mb = bucket["move_buckets"].setdefault(mv_key, {"total": 0, "correct": 0})
            mb["total"] += 1
            if p.was_correct:
                mb["correct"] += 1

    def _rates(buckets: dict) -> dict:
        return {
            k: {
                "hit_rate_pct": round(v["correct"] / v["total"] * 100, 1) if v["total"] else 0,
                "sample_count": v["total"],
            }
            for k, v in sorted(buckets.items())
        }

    pred_accuracy = {}
    for sym, b in pred_by_symbol.items():
        pred_accuracy[sym] = {
            "accuracy_pct":      round(b["correct"] / b["total"] * 100, 1) if b["total"] else 0,
            "total_predictions": b["total"],
            "calibration_by_confidence": _rates(b["conf_buckets"]),
            "calibration_by_predicted_move": _rates(b["move_buckets"]),
            "predicted_move_buckets_note": (
                f"restricted to predictions with confidence >= {min_conf_setting}, "
                f"i.e. those that could actually open a trade"
            ),
        }

    # Dual-model accuracy comparison
    small_preds = [p for p in completed_preds if p.model_name == "small"]
    mini_preds  = [p for p in completed_preds if p.model_name == "mini"]

    def _acc(preds):
        if not preds:
            return None
        correct = sum(1 for p in preds if p.was_correct)
        stamps = [p.created_at for p in preds if p.created_at]
        return {
            "accuracy_pct": round(correct / len(preds) * 100, 1),
            "count": len(preds),
            "window_start": min(stamps).isoformat() if stamps else None,
            "window_end": max(stamps).isoformat() if stamps else None,
        }

    model_comparison = {"small": _acc(small_preds), "mini": _acc(mini_preds)}

    # Only one model is active at a time outside dual-model mode, so within a
    # lookback window the two models' predictions can come from disjoint
    # periods. Comparing those is a comparison of market conditions wearing a
    # model label. Detect it and say so, rather than leaving it to be inferred.
    overlap_hours = 0.0
    confounded = False
    a, b = model_comparison["small"], model_comparison["mini"]
    if a and b and a["window_start"] and b["window_start"]:
        lo = max(datetime.fromisoformat(a["window_start"]),
                 datetime.fromisoformat(b["window_start"]))
        hi = min(datetime.fromisoformat(a["window_end"]),
                 datetime.fromisoformat(b["window_end"]))
        overlap_hours = max(0.0, (hi - lo).total_seconds() / 3600)
        confounded = overlap_hours <= 0

    model_comparison["comparison_validity"] = {
        "overlap_hours": round(overlap_hours, 1),
        "confounded_by_period": confounded,
        "note": (
            "The two models predicted over non-overlapping periods. Their "
            "accuracies differ by market conditions as much as by model, so "
            "this comparison CANNOT support switching active_model. Say so "
            "instead of recommending a switch."
            if confounded else
            "Prediction windows overlap, so the accuracy comparison is "
            "between models rather than between time periods."
        ),
    }

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
        f"Categories you may recommend: {categories}. "
        "For risk_param: param must be one of stop_loss_pct, take_profit_pct, min_confidence, "
        "min_predicted_move_pct, position_size_pct. "
        "For symbol_list: param is 'add_symbol' or 'remove_symbol', proposed_value is the symbol name. "
        "For model_strategy: param is 'active_model' or 'dual_model_mode' or 'require_agreement'. "
        "All numeric values as plain decimals (e.g. '-3.5' for stop_loss_pct meaning -3.5%). "
        "\n\nEVIDENCE RULES — each one exists because a previous audit got it wrong:\n"
        f"1. Do not recommend removing a symbol whose symbol_performance entry has "
        f"insufficient_sample=true. Fewer than {MIN_TRADES_FOR_CI} round trips cannot "
        f"distinguish a symbol from noise. Prefer symbols where ci_excludes_zero=true.\n"
        "2. Bucket keys name the quantity they bucket. 'confidence_0-9%' buckets "
        "CONFIDENCE, not predicted move, and predictions below min_confidence never "
        "open a trade, so their hit rate says nothing about entry quality. To reason "
        "about min_predicted_move_pct use calibration_by_predicted_move, which is "
        "already restricted to tradeable predictions.\n"
        "3. If model_comparison.comparison_validity.confounded_by_period is true, you "
        "MUST NOT recommend changing active_model on accuracy grounds. Report the "
        "confound instead.\n"
        "4. Before recommending a threshold change, state what it does to trade "
        "volume. A change that removes most of the tradeable population is a major "
        "change, not a tweak, and needs the surviving band to be demonstrably better "
        "in calibration_by_predicted_move — not merely the excluded band to be bad.\n"
        "5. Every number in your reasoning must be copied from the data payload. Do "
        "not compute, estimate or infer figures that are not present.\n"
        "6. Recommending no change is a valid and often correct outcome. An empty "
        "recommendations list is preferred over a weakly-supported change.\n"
        "\nRespond ONLY with valid JSON matching the schema — no prose, no markdown fences."
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


def _flag(rec: dict, label: str, message: str) -> dict:
    rec["confidence"] = label
    rec["reasoning"] = f"[{label.upper()}: {message}] " + rec.get("reasoning", "")
    return rec


def _validate_evidence(rec: dict, data: dict) -> dict:
    """
    Reject recommendations the payload does not support.

    The prompt states these rules, but a prompt is a request. This is the
    enforcement, and it runs on data rather than on the model's prose.
    """
    param = rec.get("param", "")

    # 1. Symbol removal on a sample too small to mean anything.
    if param == "remove_symbol":
        symbol = str(rec.get("proposed_value", "")).strip()
        perf = data.get("symbol_performance", {}).get(symbol)
        if perf is None:
            return _flag(rec, "no_evidence",
                         f"{symbol} has no closed round trips in the lookback window")
        if perf.get("insufficient_sample"):
            return _flag(rec, "insufficient_evidence",
                         f"{symbol} has {perf.get('trade_count')} round trips, "
                         f"below the {MIN_TRADES_FOR_CI} needed for a usable interval")
        if not perf.get("ci_excludes_zero"):
            return _flag(rec, "ci_includes_zero",
                         f"{symbol} mean P&L 95% CI {perf.get('mean_pnl_95ci')} "
                         f"includes zero, so it is not distinguishable from break-even")

    # 2. Model switch justified by accuracy measured over disjoint periods.
    if param == "active_model":
        validity = data.get("model_comparison", {}).get("comparison_validity", {})
        if validity.get("confounded_by_period"):
            return _flag(rec, "confounded",
                         "the two models' prediction windows do not overlap "
                         f"(overlap_hours={validity.get('overlap_hours')}), so the "
                         "accuracy gap is confounded with market period")

    # 3. Entry-threshold change that removes most of what the system can trade.
    if param == "min_predicted_move_pct":
        try:
            proposed = float(rec.get("proposed_value"))
            current = float(data.get("current_settings", {})
                            .get("risk", {}).get("min_predicted_move_pct", 0.02)) * 100
        except (TypeError, ValueError):
            return rec
        if proposed > current:
            kept = total = 0
            for sym in data.get("prediction_accuracy", {}).values():
                for key, bucket in sym.get("calibration_by_predicted_move", {}).items():
                    n = bucket.get("sample_count", 0)
                    total += n
                    lo = 10.0 if ">=10" in key else float(key.split("_")[-1].split("-")[0])
                    if lo >= proposed:
                        kept += n
            if total:
                share = kept / total * 100
                rec["reasoning"] = (
                    f"[VOLUME IMPACT: raising the gate from {current:.1f}% to "
                    f"{proposed:.1f}% keeps {kept}/{total} tradeable predictions "
                    f"({share:.1f}%)] " + rec.get("reasoning", "")
                )
                if share < 50:
                    return _flag(rec, "volume_collapse",
                                 f"only {share:.1f}% of tradeable predictions survive "
                                 f"this threshold")
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

    max_attempts = 2
    parsed = None
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            parsed = call_claude_headless(prompt, timeout_sec=timeout_sec)
            break
        except FileNotFoundError as e:
            last_error = str(e)
            logger.error(f"[audit] Claude binary not found: {e}")
            break  # retrying won't help
        except subprocess.TimeoutExpired:
            last_error = f"claude -p timed out after {timeout_sec}s"
            logger.error(f"[audit] {last_error} (attempt {attempt}/{max_attempts})")
        except (RuntimeError, json.JSONDecodeError) as e:
            last_error = str(e)
            logger.error(f"[audit] Claude call failed (attempt {attempt}/{max_attempts}): {e}", exc_info=True)
        except Exception as e:
            last_error = str(e)
            logger.error(f"[audit] Unexpected error calling Claude (attempt {attempt}/{max_attempts}): {e}", exc_info=True)

        if attempt < max_attempts:
            time.sleep(10)

    if parsed is None:
        notifications.send_audit_failure(audit_type, last_error or "unknown error")
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
        rec = _validate_evidence(rec, data)
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