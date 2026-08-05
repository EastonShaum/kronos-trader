"""
Idempotent: submit one pending symbol-pause recommendation per symbol
(BTC/USDT, ETH/USDT, SIDU) to the audit approval queue.

Source: symbol-fit diagnostic session (Aug 2026) — distinct from scheduled
weekly/monthly audit jobs.

Action on approval: adds symbol to settings["disabled_symbols"] via the
normal remove_symbol approval path, blocking new entry signals without
closing any existing position.

Safe to rerun — skips symbols that already have a pending recommendation
from this diagnostic session.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from database import SessionLocal, AuditRecommendation, init_db

EVIDENCE = {
    "BTC/USDT": {
        "n":        37,
        "win_pct":  29.7,
        "avg_pnl":  -1.55,
        "ci_lo":    -2.33,
        "ci_hi":    -0.84,
    },
    "ETH/USDT": {
        "n":        42,
        "win_pct":  33.3,
        "avg_pnl":  -1.45,
        "ci_lo":    -2.46,
        "ci_hi":    -0.39,
    },
    "SIDU": {
        "n":        29,
        "win_pct":  31.0,
        "avg_pnl":  -1.54,
        "ci_lo":    -2.71,
        "ci_hi":    -0.32,
    },
}

MODEL_ERA_CAVEAT = (
    "CAVEAT — model era: this evidence is drawn almost entirely from the "
    "Kronos-small model era (Apr 17 – Jul 21 2026), which accounts for 282 "
    "of the 318 total trades. The system switched to Kronos-mini on Jul 23 "
    "with no direct side-by-side validation. If mini behaves meaningfully "
    "differently on these symbols, this recommendation may need revisiting "
    "once more mini-era data has accumulated (currently 36 mini trades total, "
    "spread across 12 symbols)."
)


def _build_reasoning(sym: str, ev: dict) -> str:
    return (
        f"Symbol-fit diagnostic session (Aug 2026) — one-off deep analysis, "
        f"NOT a routine weekly/monthly audit.\n\n"
        f"Statistical evidence ({sym}):\n"
        f"  · n = {ev['n']} closed trades\n"
        f"  · Win rate: {ev['win_pct']}%\n"
        f"  · Avg net P&L per trade: ${ev['avg_pnl']:.2f}\n"
        f"  · Bootstrap 95% CI on mean net P&L: "
        f"[${ev['ci_lo']:.2f}, ${ev['ci_hi']:.2f}]\n"
        f"  → The entire 95% CI lies below zero — statistically significant "
        f"negative expected value under current rules.\n\n"
        f"Proposed action: pause new-entry signal generation for {sym}.\n"
        f"  · Do NOT force-close any currently open position — let existing "
        f"positions resolve via their normal stop-loss / take-profit / "
        f"bearish-signal exit rules.\n"
        f"  · Implementation: adds {sym} to settings[\"disabled_symbols\"], "
        f"which filters it out of the scheduler's symbol list before predictions "
        f"are run.\n\n"
        f"{MODEL_ERA_CAVEAT}"
    )


def submit(dry_run: bool = False) -> list[dict]:
    init_db()  # idempotent — applies source column migration if needed
    db = SessionLocal()
    results = []
    try:
        for sym, ev in EVIDENCE.items():
            # Idempotency check: skip if a pending diagnostic rec already exists
            existing = (
                db.query(AuditRecommendation)
                .filter(
                    AuditRecommendation.category       == "symbol_list",
                    AuditRecommendation.param          == "remove_symbol",
                    AuditRecommendation.proposed_value == sym,
                    AuditRecommendation.source         == "diagnostic_session",
                    AuditRecommendation.status         == "pending",
                )
                .first()
            )
            if existing:
                results.append({
                    "symbol":  sym,
                    "action":  "skipped — pending diagnostic recommendation already exists",
                    "rec_id":  existing.id,
                })
                continue

            rec = AuditRecommendation(
                audit_type     = "diagnostic",
                source         = "diagnostic_session",
                category       = "symbol_list",
                param          = "remove_symbol",
                current_value  = "active",
                proposed_value = sym,
                confidence     = "high",
                reasoning      = _build_reasoning(sym, ev),
                status         = "pending",
            )

            if not dry_run:
                db.add(rec)
                db.flush()  # get the id before commit
                results.append({
                    "symbol":  sym,
                    "action":  "inserted",
                    "rec_id":  rec.id,
                })
            else:
                results.append({
                    "symbol": sym,
                    "action": "dry-run — would insert",
                })

        if not dry_run:
            db.commit()
    finally:
        db.close()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n=== SYMBOL-PAUSE RECOMMENDATION SUBMISSION ===\n")
    results = submit(dry_run=args.dry_run)
    for r in results:
        print(f"  {r['symbol']:<15} {r['action']}"
              + (f"  (id={r['rec_id']})" if "rec_id" in r else ""))
    if args.dry_run:
        print("\n  [DRY RUN — no records written]")
    else:
        inserted = sum(1 for r in results if r["action"] == "inserted")
        skipped  = sum(1 for r in results if r["action"].startswith("skipped"))
        print(f"\n  Inserted: {inserted}  Skipped (already exists): {skipped}")
