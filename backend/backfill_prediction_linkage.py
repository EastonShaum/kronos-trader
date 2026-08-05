"""
Schema migration + backfill: entry_prediction_id on Trade rows.

Adds a nullable Integer column `entry_prediction_id` to the trades table.
For every SELL (closed) trade, finds the most-recent BUY trade on the same
symbol/portfolio opened at or before the SELL, then copies that BUY trade's
prediction_id into entry_prediction_id.

Run from backend/:
    ~/Trading/kronos-trader/venv/bin/python backfill_prediction_linkage.py [--dry-run]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from database import SessionLocal, Trade, TradeStatus, TradeDirection, init_db


def backfill(db, dry_run: bool = False) -> dict:
    # Ensure schema is up-to-date (adds entry_prediction_id if not present)
    init_db()
    added = True  # migration handled by init_db

    sell_trades = (
        db.query(Trade)
        .filter(Trade.status == TradeStatus.CLOSED, Trade.direction == TradeDirection.SELL)
        .all()
    )

    updated = 0
    skipped_no_buy = 0
    skipped_no_pred = 0
    already_set = 0

    for sell in sell_trades:
        if sell.entry_prediction_id is not None:
            already_set += 1
            continue

        buy = (
            db.query(Trade)
            .filter(
                Trade.portfolio_id == sell.portfolio_id,
                Trade.symbol       == sell.symbol,
                Trade.direction    == TradeDirection.BUY,
                Trade.opened_at    <= sell.opened_at,
            )
            .order_by(Trade.opened_at.desc())
            .first()
        )

        if buy is None:
            skipped_no_buy += 1
            continue
        if buy.prediction_id is None:
            skipped_no_pred += 1
            continue

        if not dry_run:
            sell.entry_prediction_id = buy.prediction_id
            db.add(sell)
        updated += 1

    if not dry_run:
        db.commit()

    return {
        "column_added": "via_init_db",
        "total_sell_trades": len(sell_trades),
        "updated": updated,
        "already_set": already_set,
        "skipped_no_buy": skipped_no_buy,
        "skipped_no_pred": skipped_no_pred,
        "dry_run": dry_run,
    }


def verify(db) -> dict:
    """Post-backfill verification: check confidence values of linked entry predictions."""
    from database import Prediction

    sell_trades = (
        db.query(Trade)
        .filter(Trade.status == TradeStatus.CLOSED, Trade.direction == TradeDirection.SELL)
        .all()
    )

    below_75 = []
    no_entry_pred = 0

    for sell in sell_trades:
        if not sell.entry_prediction_id:
            no_entry_pred += 1
            continue
        pred = db.query(Prediction).get(sell.entry_prediction_id)
        if pred and pred.confidence is not None and pred.confidence < 0.75:
            below_75.append({
                "trade_id": sell.id,
                "symbol": sell.symbol,
                "opened_at": str(sell.opened_at)[:19],
                "entry_pred_id": sell.entry_prediction_id,
                "confidence_pct": round(pred.confidence * 100, 2),
            })

    return {
        "total_sell_trades": len(sell_trades),
        "no_entry_pred_id": no_entry_pred,
        "below_75_count": len(below_75),
        "below_75_samples": below_75[:10],
        "status": (
            "PASS — all entry predictions meet ≥75% threshold"
            if len(below_75) == 0
            else f"NOTE: {len(below_75)} trades have entry confidence < 75% (likely from earlier lower threshold)"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        print("\n=== ENTRY PREDICTION ID BACKFILL ===\n")
        result = backfill(db, dry_run=args.dry_run)
        print(f"  Schema migration: {result['column_added']}")
        print(f"  Total SELL trades:   {result['total_sell_trades']}")
        print(f"  Updated:             {result['updated']}")
        print(f"  Already set:         {result['already_set']}")
        print(f"  Skipped (no BUY):    {result['skipped_no_buy']}")
        print(f"  Skipped (no pred):   {result['skipped_no_pred']}")
        if args.dry_run:
            print("\n  [DRY RUN — no changes written]")

        if not args.dry_run:
            print("\n── Verification ──")
            v = verify(db)
            print(f"  {v['status']}")
            print(f"  Trades with no entry_prediction_id: {v['no_entry_pred_id']}")
            if v['below_75_samples']:
                print("  Sample trades below 75%:")
                for row in v['below_75_samples']:
                    print(f"    trade {row['trade_id']} {row['symbol']} {row['opened_at']} "
                          f"conf={row['confidence_pct']}%")
    finally:
        db.close()
