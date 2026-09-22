"""
Snapshot tooling.

The live DB is written by the prediction cycle (hourly) and the price monitor
(every 3 min), so a filesystem copy can land mid-transaction. The SQLite backup
API takes a consistent copy of a live database instead.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_DB = REPO_ROOT / "backend" / "kronos_trader.db"
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"


def create_snapshot(live_db: Path | None = None, dest_dir: Path | None = None) -> Path:
    """Copy the live DB to backtest/snapshots/ via the SQLite backup API."""
    live_db = Path(live_db) if live_db else LIVE_DB
    dest_dir = Path(dest_dir) if dest_dir else SNAPSHOT_DIR

    if not live_db.exists():
        raise FileNotFoundError(f"Live DB not found: {live_db}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = dest_dir / f"snapshot_{stamp}.db"

    src = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    logger.info("Snapshot written: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def open_readonly(snapshot_path: Path | str) -> sqlite3.Connection:
    """Open a snapshot read-only. Any write raises sqlite3.OperationalError."""
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    conn = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(create_snapshot())
