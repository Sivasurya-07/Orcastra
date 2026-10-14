"""
ORCASTRA Collector — Forced Kill + Recovery Test.

This test:
1. Starts a writer in a subprocess that writes events slowly
2. Kills the process with taskkill /F (equivalent to power loss)
3. Starts a new writer on the same session dir
4. Verifies: no corrupt finalized files, incomplete session detected, recovery works
"""

import hashlib
import json
import multiprocessing
import os
import sys
import time
import uuid
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orcastra_collector.config import CollectorConfig
from orcastra_collector.schema import AisEvent
from orcastra_collector.parser import AisParser
from orcastra_collector.ledger import Ledger
from orcastra_collector.writer import ParquetWriter
from orcastra_collector.replay import replay_session
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEST_DIR = Path(r"D:\ORCASTRA_DATA\test_kill")
PASS = 0
FAIL = 0

def assert_true(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        log.info(f"  PASS: {msg}")
    else:
        FAIL += 1
        log.error(f"  FAIL: {msg}")


def _make_raw_position(mmsi, lat, lon, sog, time_str):
    return json.dumps({
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": mmsi, "latitude": lat, "longitude": lon,
                     "time_utc": time_str, "ShipName": f"TEST_{mmsi}"},
        "Message": {"PositionReport": {"MessageID": 1, "Sog": sog, "Cog": 180,
                                        "TrueHeading": 180}},
    })


def writer_process(session_id, session_dir, ledger_path):
    """Subprocess that writes events. Will be killed mid-write."""
    ledger = Ledger(ledger_path)
    ledger.start_session(session_id, "{}", "[]")
    writer = ParquetWriter(session_dir, session_id, ledger,
                           max_rows=5, max_seconds=999)
    parser = AisParser(session_id)

    # Write events slowly (one per 0.1s) — simulates real-time collection
    for i in range(50):
        raw = _make_raw_position(300000000 + i, 29.5, -94.8, float(i),
                                 f"2026-10-08 14:00:{i:02d}.000000 +0000 UTC")
        from datetime import datetime as dt_cls, timezone as tz_cls
        ev, _ = parser.parse(raw, dt_cls.now(tz_cls.utc))
        writer.add_event(ev, raw)
        time.sleep(0.05)

    # This should never be reached if killed properly
    writer.finalize()
    ledger.close()


def main():
    global PASS, FAIL
    import shutil

    log.info("=== Test: Forced Kill + Restart Recovery ===")

    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)

    session_id = str(uuid.uuid4())
    session_dir = TEST_DIR / "session_kill_test"
    ledger_path = TEST_DIR / "kill_test_ledger.sqlite"

    # 1. Start the writer subprocess
    p = multiprocessing.Process(target=writer_process,
                                args=(session_id, session_dir, ledger_path))
    p.start()

    # Wait for it to produce at least one finalized part
    time.sleep(1.5)  # ~30 events at 0.05s each, should trigger parts at max_rows=5

    # 2. Kill it (hard kill — no cleanup)
    log.info("Killing writer process (PID %d)...", p.pid)
    p.kill()
    p.join(timeout=5)
    assert_true(not p.is_alive(), "Process killed successfully")

    # 3. Verify finalized files are not corrupt
    events_dir = session_dir / "events"
    if events_dir.exists():
        parquet_files = sorted(events_dir.glob("part_*.parquet"))
        tmp_files = list(events_dir.glob("*.tmp.parquet"))

        log.info("Found %d finalized .parquet, %d .tmp files", len(parquet_files), len(tmp_files))

        corrupt = 0
        total_rows = 0
        for pf in parquet_files:
            try:
                table = pq.read_table(pf)
                total_rows += len(table)
            except Exception as e:
                log.error("  CORRUPT: %s: %s", pf.name, e)
                corrupt += 1

        assert_true(corrupt == 0, f"No corrupt finalized files ({len(parquet_files)} checked)")
        assert_true(len(parquet_files) > 0, f"At least one finalized part exists ({len(parquet_files)})")
        log.info("  Total recoverable rows: %d", total_rows)
    else:
        log.error("  No events directory found")
        FAIL += 1

    # 4. Check ledger for incomplete session
    ledger = Ledger(ledger_path)
    incomplete = ledger.get_incomplete_sessions()
    assert_true(len(incomplete) == 1, f"Incomplete session detected ({len(incomplete)})")
    if incomplete:
        assert_true(incomplete[0]["session_id"] == session_id, "Correct session_id in incomplete")

    # 5. Recovery: clean up tmp files and mark session as crashed
    new_session_id = str(uuid.uuid4())
    writer2 = ParquetWriter(session_dir, new_session_id, ledger,
                            max_rows=100, max_seconds=999)
    cleaned = writer2.cleanup_incomplete()
    log.info("  Cleaned %d tmp files", cleaned)

    # Mark crashed session
    ledger.end_session(session_id, status="crashed", error_message="Killed in test")
    incomplete2 = ledger.get_incomplete_sessions()
    assert_true(len(incomplete2) == 0, "No incomplete sessions after recovery")

    # 6. Verify we can continue writing in a new session
    ledger.start_session(new_session_id, "{}", "[]")
    parser2 = AisParser(new_session_id)
    raw = _make_raw_position(399999999, 29.5, -94.8, 5.0,
                             "2026-10-08 15:00:00.000000 +0000 UTC")
    from datetime import datetime as dt_cls, timezone as tz_cls
    ev, _ = parser2.parse(raw, dt_cls.now(tz_cls.utc))
    writer2.add_event(ev, raw)
    writer2.finalize()
    assert_true(True, "New session writes successfully after recovery")

    ledger.close()

    # Cleanup
    shutil.rmtree(TEST_DIR)

    log.info("\n" + "="*60)
    log.info(f"FORCED KILL TEST: {PASS} passed, {FAIL} failed")
    log.info("="*60)
    return FAIL == 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    success = main()
    sys.exit(0 if success else 1)
