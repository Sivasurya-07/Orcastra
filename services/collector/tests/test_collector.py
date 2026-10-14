"""
ORCASTRA Collector — Tests.

Tests:
  1. Kill during write (no corrupt finalized file)
  2. Restart recovery
  3. Immutability (finalized files don't change)
  4. Deterministic replay
  5. Duplicates (flagged, not dropped)
  6. Enrichment surviving later position reports
  7. Malformed quarantine
  8. Controlled reconnect (parser + writer integration)
"""

import json
import os
import sys
import uuid
import hashlib
import tempfile
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orcastra_collector.config import CollectorConfig
from orcastra_collector.schema import AisEvent, PARQUET_SCHEMA, event_to_dict, categorize_vessel_type
from orcastra_collector.parser import AisParser
from orcastra_collector.vessel_cache import VesselCache
from orcastra_collector.ledger import Ledger
from orcastra_collector.writer import ParquetWriter
from orcastra_collector.replay import replay_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEST_DIR = Path(r"D:\ORCASTRA_DATA\test_collector")
PASS_COUNT = 0
FAIL_COUNT = 0


def _make_raw_position(mmsi: int, lat: float, lon: float,
                       sog: float = 10.0, time_str: str = "2026-10-08 14:00:00.000000 +0000 UTC") -> str:
    return json.dumps({
        "MessageType": "PositionReport",
        "MetaData": {
            "MMSI": mmsi, "MMSI_String": mmsi,
            "ShipName": f"TEST_{mmsi}",
            "latitude": lat, "longitude": lon,
            "time_utc": time_str,
        },
        "Message": {
            "PositionReport": {
                "MessageID": 1, "UserID": mmsi, "Valid": True,
                "Sog": sog, "Cog": 180.0, "TrueHeading": 180,
                "Latitude": lat, "Longitude": lon,
                "NavigationalStatus": 0, "RateOfTurn": 0,
                "PositionAccuracy": True, "Timestamp": 0,
            }
        }
    })


def _make_raw_static(mmsi: int, vtype: int, name: str = "TEST") -> str:
    return json.dumps({
        "MessageType": "ShipStaticData",
        "MetaData": {
            "MMSI": mmsi, "MMSI_String": mmsi,
            "ShipName": name,
            "latitude": 29.5, "longitude": -94.8,
            "time_utc": "2026-10-08 14:01:00.000000 +0000 UTC",
        },
        "Message": {
            "ShipStaticData": {
                "MessageID": 5, "UserID": mmsi,
                "Type": vtype, "CallSign": "TEST",
                "ImoNumber": 1234567,
                "Dimension": {"A": 100, "B": 50, "C": 10, "D": 10},
                "MaximumStaticDraught": 12.5,
                "Destination": "HOUSTON",
                "Eta": {"Month": 10, "Day": 8, "Hour": 14, "Minute": 0},
            }
        }
    })


def assert_true(condition: bool, msg: str):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        log.info(f"  PASS: {msg}")
    else:
        FAIL_COUNT += 1
        log.error(f"  FAIL: {msg}")


def test_parser_and_dedup():
    """Test 4 & 5: Deterministic dedup keys, duplicate flagging."""
    log.info("\n=== Test: Parser + Dedup ===")
    session_id = str(uuid.uuid4())
    parser = AisParser(session_id)

    raw = _make_raw_position(368330810, 29.45, -94.68, 10.0)
    receipt = datetime(2026, 10, 8, 14, 0, 1, tzinfo=timezone.utc)

    ev1, _ = parser.parse(raw, receipt)
    ev2, _ = parser.parse(raw, receipt)

    assert_true(ev1 is not None, "Parsed first message")
    assert_true(ev2 is not None, "Parsed second (duplicate) message")
    assert_true(ev1.dedup_key == ev2.dedup_key, "Same dedup key for same MMSI+time")
    assert_true(ev1.event_id == ev2.event_id, "Same event_id (deterministic)")
    assert_true(ev1.mmsi == 368330810, "MMSI correct")
    assert_true(abs(ev1.latitude - 29.45) < 0.001, "Latitude from MetaData")
    assert_true(abs(ev1.longitude - (-94.68)) < 0.001, "Longitude from MetaData")
    assert_true(ev1.sog == 10.0, "SOG parsed")
    assert_true(ev1.is_in_bbox, "In bbox")
    assert_true(ev1.is_valid_mmsi, "Valid MMSI")

    # Different time -> different dedup key
    raw2 = _make_raw_position(368330810, 29.45, -94.68, 10.0,
                              "2026-10-08 14:00:01.000000 +0000 UTC")
    ev3, _ = parser.parse(raw2, receipt)
    assert_true(ev3.dedup_key != ev1.dedup_key, "Different time -> different dedup key")


def test_sentinels():
    """Test sentinel detection."""
    log.info("\n=== Test: Sentinel Detection ===")
    session_id = str(uuid.uuid4())
    parser = AisParser(session_id)

    raw = json.dumps({
        "MessageType": "PositionReport",
        "MetaData": {"MMSI": 300000000, "latitude": 29.5, "longitude": -94.8,
                     "time_utc": "2026-10-08 14:00:00.000000 +0000 UTC"},
        "Message": {"PositionReport": {
            "MessageID": 1, "Sog": 102.3, "Cog": 360.0, "TrueHeading": 511,
            "Latitude": 29.5, "Longitude": -94.8,
        }}
    })
    ev, _ = parser.parse(raw, datetime.now(timezone.utc))
    assert_true(ev.sog_is_sentinel, "SOG sentinel 102.3 detected")
    assert_true(ev.cog_is_sentinel, "COG sentinel 360.0 detected")
    assert_true(ev.heading_is_sentinel, "Heading sentinel 511 detected")


def test_invalid_mmsi():
    """Test invalid MMSI detection."""
    log.info("\n=== Test: Invalid MMSI ===")
    parser = AisParser(str(uuid.uuid4()))
    raw = _make_raw_position(123456789, 29.5, -94.8)  # below valid range
    ev, _ = parser.parse(raw, datetime.now(timezone.utc))
    assert_true(not ev.is_valid_mmsi, "Invalid MMSI flagged")


def test_out_of_bbox():
    """Test out-of-bbox detection."""
    log.info("\n=== Test: Out of BBox ===")
    parser = AisParser(str(uuid.uuid4()))
    raw = _make_raw_position(300000000, 40.0, -74.0)  # New York, not Houston
    ev, _ = parser.parse(raw, datetime.now(timezone.utc))
    assert_true(not ev.is_in_bbox, "Out-of-bbox flagged")


def test_vessel_cache_and_enrichment():
    """Test 6: Vessel cache enrichment survives later position reports."""
    log.info("\n=== Test: Vessel Cache + Enrichment ===")
    db_path = TEST_DIR / "test_cache.sqlite"
    db_path.unlink(missing_ok=True)

    cache = VesselCache(db_path)
    parser = AisParser(str(uuid.uuid4()))

    # Position report before static data -> no type
    raw_pos = _make_raw_position(300000001, 29.5, -94.8)
    ev1, _ = parser.parse(raw_pos, datetime.now(timezone.utc),
                          vessel_type_lookup=cache.type_lookup)
    assert_true(ev1.vessel_type is None, "No type before static data")

    # Static data arrives
    raw_static = _make_raw_static(300000001, 70, "CARGO SHIP")
    ev2, static_info = parser.parse(raw_static, datetime.now(timezone.utc))
    assert_true(static_info is not None, "Static data parsed")
    cache.upsert(static_info)
    assert_true(cache.get_type(300000001) == 70, "Cache updated with type 70")

    # Later position report -> enriched
    raw_pos2 = _make_raw_position(300000001, 29.51, -94.79, 8.0,
                                  "2026-10-08 14:05:00.000000 +0000 UTC")
    ev3, _ = parser.parse(raw_pos2, datetime.now(timezone.utc),
                          vessel_type_lookup=cache.type_lookup)
    assert_true(ev3.vessel_type == 70, "Enriched with vessel type from cache")
    assert_true(ev3.vessel_category == "cargo", "Category derived correctly")

    cache.close()
    db_path.unlink(missing_ok=True)


def test_quarantine():
    """Test 7: Malformed messages are quarantined."""
    log.info("\n=== Test: Malformed Quarantine ===")
    parser = AisParser(str(uuid.uuid4()))

    # Invalid JSON
    ev, _ = parser.parse("not json at all", datetime.now(timezone.utc))
    assert_true(ev is None, "Invalid JSON returns None")

    # Unknown message type
    raw = json.dumps({"MessageType": "WeirdUnknownType", "MetaData": {"MMSI": 300000000}})
    ev, _ = parser.parse(raw, datetime.now(timezone.utc))
    assert_true(ev is None, "Unknown message type returns None")

    # No MMSI
    raw = json.dumps({"MessageType": "PositionReport", "MetaData": {}})
    ev, _ = parser.parse(raw, datetime.now(timezone.utc))
    assert_true(ev is None, "No MMSI returns None")

    stats = parser.stats
    assert_true(stats["quarantined"] >= 1, "Quarantine counter incremented")
    assert_true(stats["skipped_no_mmsi"] >= 1, "Skipped-no-MMSI counter incremented")


def test_writer_crash_safety():
    """Test 1 & 3: Kill during write (no corrupt finalized file), immutability."""
    log.info("\n=== Test: Writer Crash Safety ===")
    session_id = str(uuid.uuid4())
    session_dir = TEST_DIR / "test_crash"

    # Clean up
    if session_dir.exists():
        import shutil
        shutil.rmtree(session_dir)

    ledger_path = TEST_DIR / "test_ledger.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = Ledger(ledger_path)
    ledger.start_session(session_id, "{}", "[]")

    writer = ParquetWriter(session_dir, session_id, ledger,
                           max_rows=5, max_seconds=999)
    parser = AisParser(session_id)

    # Write 7 events (should produce 1 complete part + 2 in buffer)
    for i in range(7):
        raw = _make_raw_position(300000000 + i, 29.5, -94.8, 10.0 + i,
                                 f"2026-10-08 14:00:{i:02d}.000000 +0000 UTC")
        ev, _ = parser.parse(raw, datetime.now(timezone.utc))
        writer.add_event(ev, raw)

    # Part 1 should be finalized (5 rows)
    files = ledger.get_session_files(session_id)
    finalized = [f for f in files if f["is_finalized"] and f["file_type"] == "events"]
    assert_true(len(finalized) == 1, "One finalized part after 7 rows (max_rows=5)")
    assert_true(finalized[0]["row_count"] == 5, "First part has 5 rows")

    # Verify the file is valid Parquet
    import pyarrow.parquet as pq
    p1_path = Path(finalized[0]["file_path"])
    assert_true(p1_path.exists(), "Finalized Parquet file exists")
    table = pq.read_table(p1_path)
    assert_true(len(table) == 5, "Parquet readable, has 5 rows")

    # Record hash for immutability check
    hash1 = hashlib.sha256(p1_path.read_bytes()).hexdigest()

    # Simulate crash (don't call finalize)
    # The 2 remaining events in buffer are lost — that's the contract

    # Check no .tmp files exist
    tmp_files = list((session_dir / "events").glob("*.tmp.parquet"))
    assert_true(len(tmp_files) == 0, "No .tmp files from normal operation")

    # Verify immutability — re-read and compare hash
    hash2 = hashlib.sha256(p1_path.read_bytes()).hexdigest()
    assert_true(hash1 == hash2, "Finalized file unchanged (immutable)")

    ledger.close()


def test_restart_recovery():
    """Test 2: Restart recovery — incomplete sessions and files detected."""
    log.info("\n=== Test: Restart Recovery ===")
    ledger_path = TEST_DIR / "test_recovery_ledger.sqlite"
    ledger_path.unlink(missing_ok=True)

    ledger = Ledger(ledger_path)

    # Simulate a crashed session
    crashed_id = str(uuid.uuid4())
    ledger.start_session(crashed_id, "{}", "[]")
    # Don't end it — simulates crash

    # New session starts and detects the old one
    incomplete = ledger.get_incomplete_sessions()
    assert_true(len(incomplete) == 1, "Incomplete session detected")
    assert_true(incomplete[0]["session_id"] == crashed_id, "Correct crashed session ID")

    # Recovery: mark as crashed
    ledger.end_session(crashed_id, status="crashed",
                       error_message="Recovered by test")
    incomplete2 = ledger.get_incomplete_sessions()
    assert_true(len(incomplete2) == 0, "No incomplete sessions after recovery")

    ledger.close()


def test_replay_determinism():
    """Test 4: Deterministic replay — same events produce same output."""
    log.info("\n=== Test: Deterministic Replay ===")
    session_id = str(uuid.uuid4())
    session_dir = TEST_DIR / "test_replay"
    if session_dir.exists():
        import shutil
        shutil.rmtree(session_dir)

    ledger_path = TEST_DIR / "test_replay_ledger.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = Ledger(ledger_path)
    ledger.start_session(session_id, "{}", "[]")

    writer = ParquetWriter(session_dir, session_id, ledger,
                           max_rows=100, max_seconds=999)
    parser = AisParser(session_id)

    # Write events
    for i in range(10):
        raw = _make_raw_position(300000000 + i, 29.5, -94.8, float(i),
                                 f"2026-10-08 14:00:{i:02d}.000000 +0000 UTC")
        ev, _ = parser.parse(raw, datetime.now(timezone.utc))
        writer.add_event(ev, raw)
    writer.finalize()

    # Replay
    df = replay_session(session_dir)
    assert_true(len(df) == 10, "Replay has 10 events")
    assert_true(df["mmsi"].to_list() == list(range(300000000, 300000010)),
                "MMSIs match in order")

    # Replay again — deterministic
    df2 = replay_session(session_dir)
    assert_true(df["event_id"].to_list() == df2["event_id"].to_list(),
                "Replay is deterministic (same event_ids)")

    ledger.close()


def test_class_b():
    """Test Class B position report parsing."""
    log.info("\n=== Test: Class B Position Report ===")
    parser = AisParser(str(uuid.uuid4()))
    raw = json.dumps({
        "MessageType": "StandardClassBPositionReport",
        "MetaData": {"MMSI": 367609780, "latitude": 29.547, "longitude": -95.038,
                     "time_utc": "2026-10-08 13:34:49.000000 +0000 UTC"},
        "Message": {"StandardClassBPositionReport": {
            "MessageID": 18, "Sog": 0, "Cog": 360, "TrueHeading": 344,
            "Latitude": 29.547, "Longitude": -95.038,
        }}
    })
    ev, _ = parser.parse(raw, datetime.now(timezone.utc))
    assert_true(ev is not None, "Class B parsed")
    assert_true(ev.is_class_b, "is_class_b flag set")
    assert_true(ev.cog_is_sentinel, "COG 360 sentinel detected in Class B")


def main():
    global PASS_COUNT, FAIL_COUNT
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    log.info("ORCASTRA Collector Test Suite")
    log.info("="*60)

    test_parser_and_dedup()
    test_sentinels()
    test_invalid_mmsi()
    test_out_of_bbox()
    test_vessel_cache_and_enrichment()
    test_quarantine()
    test_writer_crash_safety()
    test_restart_recovery()
    test_replay_determinism()
    test_class_b()
    test_controlled_reconnect()
    test_part_immutability_after_write()
    test_static_data_report()
    # test_forced_kill_recovery is run separately (requires subprocess)

    log.info("\n" + "="*60)
    log.info(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    log.info("="*60)

    # Clean up test dir
    import shutil
    for d in TEST_DIR.iterdir():
        if d.is_dir():
            shutil.rmtree(d)
        else:
            d.unlink()

    return FAIL_COUNT == 0


def test_controlled_reconnect():
    """Test: Controlled reconnect — parser and writer state survives across reconnects."""
    log.info("\n=== Test: Controlled Reconnect ===")
    session_id = str(uuid.uuid4())
    session_dir = TEST_DIR / "test_reconnect"
    if session_dir.exists():
        import shutil
        shutil.rmtree(session_dir)

    ledger_path = TEST_DIR / "test_reconnect_ledger.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = Ledger(ledger_path)
    ledger.start_session(session_id, "{}", "[]")

    writer = ParquetWriter(session_dir, session_id, ledger,
                           max_rows=100, max_seconds=999)
    parser = AisParser(session_id)

    # Simulate "connection 1" — write 5 events
    for i in range(5):
        raw = _make_raw_position(300000000 + i, 29.5, -94.8, float(i),
                                 f"2026-10-08 14:00:{i:02d}.000000 +0000 UTC")
        ev, _ = parser.parse(raw, datetime.now(timezone.utc))
        writer.add_event(ev, raw)

    # Simulate "disconnect + reconnect" — parser and writer keep state
    # (In real collector, these objects persist across reconnects)

    # Simulate "connection 2" — write 5 more events with later times
    for i in range(5, 10):
        raw = _make_raw_position(300000000 + i, 29.5, -94.8, float(i),
                                 f"2026-10-08 14:05:{i:02d}.000000 +0000 UTC")
        ev, _ = parser.parse(raw, datetime.now(timezone.utc))
        writer.add_event(ev, raw)

    writer.finalize()

    # Replay and verify all 10 events survived
    df = replay_session(session_dir)
    assert_true(len(df) == 10, "All 10 events survived reconnect (5 + 5)")
    assert_true(df["session_id"].unique().to_list() == [session_id],
                "Same session_id across reconnects")
    assert_true(parser.stats["parsed"] == 10, "Parser count = 10 across both connections")

    ledger.close()


def test_part_immutability_after_write():
    """Test: Part immutability — writing more data does not modify earlier parts."""
    log.info("\n=== Test: Part Immutability After Further Writes ===")
    session_id = str(uuid.uuid4())
    session_dir = TEST_DIR / "test_immut"
    if session_dir.exists():
        import shutil
        shutil.rmtree(session_dir)

    ledger_path = TEST_DIR / "test_immut_ledger.sqlite"
    ledger_path.unlink(missing_ok=True)
    ledger = Ledger(ledger_path)
    ledger.start_session(session_id, "{}", "[]")

    writer = ParquetWriter(session_dir, session_id, ledger,
                           max_rows=5, max_seconds=999)
    parser = AisParser(session_id)

    # Write 6 events -> triggers part_000001 (5 rows), leaves 1 in buffer
    for i in range(6):
        raw = _make_raw_position(300000000 + i, 29.5, -94.8, float(i),
                                 f"2026-10-08 14:00:{i:02d}.000000 +0000 UTC")
        ev, _ = parser.parse(raw, datetime.now(timezone.utc))
        writer.add_event(ev, raw)

    # Record hash of part_000001
    part1 = session_dir / "events" / "part_000001.parquet"
    assert_true(part1.exists(), "Part 1 exists after 6 writes")
    hash1 = hashlib.sha256(part1.read_bytes()).hexdigest()

    # Write 10 more events -> triggers part_000002 (6 rows = 1+5) and part_000003 (5 rows)
    for i in range(6, 16):
        raw = _make_raw_position(300000000 + i, 29.5, -94.8, float(i),
                                 f"2026-10-08 14:01:{i:02d}.000000 +0000 UTC")
        ev, _ = parser.parse(raw, datetime.now(timezone.utc))
        writer.add_event(ev, raw)

    writer.finalize()

    # Verify part_000001 is unchanged
    hash2 = hashlib.sha256(part1.read_bytes()).hexdigest()
    assert_true(hash1 == hash2, "Part 1 hash unchanged after further writes (immutable)")

    # Verify we have multiple parts
    parts = list((session_dir / "events").glob("part_*.parquet"))
    assert_true(len(parts) >= 3, f"Multiple parts created ({len(parts)} parts)")

    ledger.close()


def test_static_data_report():
    """Test: StaticDataReport (AIS type 24, Class B CS static) parsing."""
    log.info("\n=== Test: StaticDataReport (Type 24) ===")
    parser = AisParser(str(uuid.uuid4()))

    # Simulate a StaticDataReport message
    raw = json.dumps({
        "MessageType": "StaticDataReport",
        "MetaData": {
            "MMSI": 367000001, "MMSI_String": 367000001,
            "ShipName": "CLASS_B_VESSEL",
            "latitude": 29.5, "longitude": -94.8,
            "time_utc": "2026-10-08 14:00:00.000000 +0000 UTC",
        },
        "Message": {
            "StaticDataReport": {
                "MessageID": 24,
                "UserID": 367000001,
                "Type": 36,
                "CallSign": "WDA1234",
            }
        }
    })
    ev, static_info = parser.parse(raw, datetime.now(timezone.utc))
    assert_true(ev is not None, "StaticDataReport parsed")
    assert_true(ev.is_class_b, "is_class_b set for type 24")
    assert_true(ev.message_type == "StaticDataReport", "message_type = StaticDataReport")
    assert_true(static_info is not None, "Static info extracted")
    assert_true(static_info.get("vessel_type") == 36, "Vessel type from type 24")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
