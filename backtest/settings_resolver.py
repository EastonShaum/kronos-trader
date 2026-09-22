"""
Settings resolution for replay.

Replay must use the parameters that were actually in effect, not today's
settings.json. What is actually recoverable is limited, and the limits are
enforced here rather than papered over:

  RECOVERABLE (recorded ground truth)
    active_model     - predictions.model_name, per prediction row
    take_profit_pct  - audit_recommendations (one change: 6.0 -> 3.0)
    symbol enable/disable - audit_recommendations (remove_symbol)

  NOT RECOVERABLE (no record anywhere)
    position_size_pct, min_confidence, stop_loss_pct, min_predicted_move_pct
    and the entire costs block.

settings.json is gitignored, so it has no git history, and there is no
settings snapshot table. The unrecoverable parameters must therefore be
supplied explicitly by the operator. There is deliberately no fallback to the
current settings.json: silently assuming today's values held in July is how a
backtest starts lying.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Parameters with no historical record. The operator must pass these.
REQUIRED_OPERATOR_PARAMS = (
    "position_size_pct",
    "min_confidence",
    "stop_loss_pct",
    "min_predicted_move_pct",
)

# The model-switch date asserted in the patch brief. Retained as a hard guard
# even though the recorded history does not show a switch on this date - see
# derive_model_boundaries(). Guarding both the asserted and the derived dates
# is strictly safer than trusting either alone.
DECLARED_MODEL_BOUNDARY = date(2026, 7, 23)


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class ChangeEvent:
    ts: datetime
    param: str
    old_value: str
    new_value: str
    source: str

    @property
    def day(self) -> date:
        return self.ts.date()


@dataclass
class ResolvedSettings:
    start: date
    end: date
    values: dict[str, Any]
    provenance: dict[str, str]
    model_timeline: list[tuple[date, date, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def log(self) -> None:
        """Reconciliation results are meaningless without the assumptions."""
        logger.info("Resolved settings for %s .. %s", self.start, self.end)
        for key in sorted(self.values):
            logger.info("  %-24s = %-10s [%s]", key, self.values[key],
                        self.provenance.get(key, "unknown"))
        for lo, hi, model in self.model_timeline:
            logger.info("  active model %s .. %s -> %s", lo, hi, model)
        for warn in self.warnings:
            logger.warning("  %s", warn)


def load_change_events(conn: sqlite3.Connection) -> list[ChangeEvent]:
    """Applied parameter changes from audit_recommendations."""
    rows = conn.execute(
        "SELECT param, current_value, proposed_value, applied_at "
        "FROM audit_recommendations "
        "WHERE status = 'applied' AND applied_at IS NOT NULL "
        "ORDER BY applied_at"
    ).fetchall()

    events: list[ChangeEvent] = []
    for row in rows:
        ts = _parse_ts(row["applied_at"])
        if ts is None:
            continue
        events.append(ChangeEvent(
            ts=ts,
            param=row["param"],
            old_value=str(row["current_value"]),
            new_value=str(row["proposed_value"]),
            source="audit_recommendations",
        ))
    return events


def derive_model_boundaries(conn: sqlite3.Connection) -> list[date]:
    """
    Dates where the model regime actually changed, from predictions.model_name.

    Two kinds of change matter, and both are derived rather than assumed:
      - the set of models producing predictions changed (dual-mode on/off)
      - which single model was live changed
    """
    rows = conn.execute(
        "SELECT DATE(created_at) AS d, model_name, COUNT(*) AS n "
        "FROM predictions GROUP BY d, model_name ORDER BY d"
    ).fetchall()

    per_day: dict[date, set[str]] = {}
    for row in rows:
        day = _parse_ts(row["d"])
        if day is None:
            continue
        per_day.setdefault(day.date(), set()).add(row["model_name"])

    boundaries: list[date] = []
    prev_models: set[str] | None = None
    for day in sorted(per_day):
        models = per_day[day]
        if prev_models is not None and models != prev_models:
            boundaries.append(day)
        prev_models = models
    return boundaries


def model_timeline(conn: sqlite3.Connection,
                   start: date, end: date) -> list[tuple[date, date, str]]:
    """Contiguous runs of the prediction-model regime across the range."""
    rows = conn.execute(
        "SELECT DATE(created_at) AS d, model_name, COUNT(*) AS n "
        "FROM predictions WHERE DATE(created_at) BETWEEN ? AND ? "
        "GROUP BY d, model_name ORDER BY d",
        (start.isoformat(), end.isoformat()),
    ).fetchall()

    per_day: dict[date, set[str]] = {}
    for row in rows:
        day = _parse_ts(row["d"])
        if day is None:
            continue
        per_day.setdefault(day.date(), set()).add(row["model_name"])

    timeline: list[tuple[date, date, str]] = []
    run_start: date | None = None
    run_label: str | None = None
    prev_day: date | None = None

    for day in sorted(per_day):
        label = "+".join(sorted(per_day[day]))
        if label != run_label:
            if run_label is not None and run_start is not None and prev_day is not None:
                timeline.append((run_start, prev_day, run_label))
            run_start, run_label = day, label
        prev_day = day

    if run_label is not None and run_start is not None and prev_day is not None:
        timeline.append((run_start, prev_day, run_label))
    return timeline


class ModelBoundaryError(RuntimeError):
    """Raised when a replay range crosses a model change without an override."""


class SettingsUnavailableError(RuntimeError):
    """Raised when required settings have no record and none were supplied."""


def check_model_boundary(conn: sqlite3.Connection, start: date, end: date,
                         allow_model_boundary: bool) -> list[date]:
    """
    Refuse ranges that span a model change unless explicitly allowed.

    Crossing one silently produces a reconciliation mismatch that looks like a
    code bug but is not.
    """
    derived = derive_model_boundaries(conn)
    candidates = sorted({*derived, DECLARED_MODEL_BOUNDARY})
    crossed = [b for b in candidates if start < b <= end]

    if crossed and not allow_model_boundary:
        detail = ", ".join(b.isoformat() for b in crossed)
        raise ModelBoundaryError(
            f"Replay range {start}..{end} spans model change(s): {detail}. "
            f"Pass --allow-model-boundary to proceed. Note the brief asserts a "
            f"boundary on {DECLARED_MODEL_BOUNDARY}; the recorded prediction "
            f"history instead shows changes on "
            f"{', '.join(b.isoformat() for b in derived) or '(none)'}."
        )
    return crossed


def resolve(conn: sqlite3.Connection,
            start: date,
            end: date,
            operator_settings: dict[str, Any] | None = None,
            allow_model_boundary: bool = False) -> ResolvedSettings:
    """
    Resolve the settings in effect over [start, end].

    Precedence:
      1. Recorded history (audit_recommendations / predictions.model_name).
      2. Explicit operator-supplied values, asserted stable over the range.
    There is no step 3. Unsupplied, unrecorded parameters are a hard error.
    """
    crossed = check_model_boundary(conn, start, end, allow_model_boundary)

    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    warnings: list[str] = []

    if crossed:
        warnings.append(
            "Range crosses model change(s) "
            f"{', '.join(c.isoformat() for c in crossed)} - override in effect."
        )

    events = load_change_events(conn)

    # --- Recorded: take_profit_pct -----------------------------------------
    tp_events = [e for e in events if e.param == "take_profit_pct"]
    spanning_tp = [e for e in tp_events if start < e.day <= end]
    if spanning_tp:
        detail = ", ".join(f"{e.day} {e.old_value}->{e.new_value}" for e in spanning_tp)
        raise SettingsUnavailableError(
            f"Range {start}..{end} spans a recorded take_profit_pct change "
            f"({detail}). Split the replay at that date - a single value cannot "
            f"represent the range."
        )

    prior_tp = [e for e in tp_events if e.day <= start]
    if prior_tp:
        latest = max(prior_tp, key=lambda e: e.ts)
        values["take_profit_pct"] = float(latest.new_value) / 100.0
        provenance["take_profit_pct"] = f"audit_recommendations {latest.day}"
    elif tp_events:
        earliest = min(tp_events, key=lambda e: e.ts)
        values["take_profit_pct"] = float(earliest.old_value) / 100.0
        provenance["take_profit_pct"] = (
            f"audit_recommendations pre-change value (before {earliest.day})"
        )

    # --- Recorded: model regime --------------------------------------------
    timeline = model_timeline(conn, start, end)

    # --- Unrecorded: operator must supply ----------------------------------
    supplied = dict(operator_settings or {})
    missing = [p for p in REQUIRED_OPERATOR_PARAMS if p not in supplied]
    if missing:
        raise SettingsUnavailableError(
            "No historical record exists for: " + ", ".join(missing) + ". "
            "settings.json is gitignored (no git history) and there is no "
            "settings snapshot table. Supply these explicitly via --settings "
            "together with a date range over which you know them to be stable. "
            "This harness will not fall back to the current settings.json."
        )

    for param in REQUIRED_OPERATOR_PARAMS:
        values[param] = supplied[param]
        provenance[param] = "operator-supplied (NO historical record)"

    # take_profit_pct may also be overridden explicitly.
    if "take_profit_pct" in supplied:
        if "take_profit_pct" in values and values["take_profit_pct"] != supplied["take_profit_pct"]:
            warnings.append(
                f"Operator take_profit_pct={supplied['take_profit_pct']} overrides "
                f"recorded value {values['take_profit_pct']}."
            )
        values["take_profit_pct"] = supplied["take_profit_pct"]
        provenance["take_profit_pct"] = "operator-supplied (overrides record)"

    if "take_profit_pct" not in values:
        raise SettingsUnavailableError(
            "take_profit_pct could not be resolved from audit_recommendations "
            "for this range; supply it explicitly via --settings."
        )

    warnings.append(
        "Cost model (fees/slippage bps) has no historical record; replay reuses "
        "each trade's stored cost columns where available."
    )

    resolved = ResolvedSettings(
        start=start, end=end, values=values, provenance=provenance,
        model_timeline=timeline, warnings=warnings,
    )
    resolved.log()
    return resolved
