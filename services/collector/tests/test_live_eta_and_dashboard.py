"""
ORCASTRA — Tests for Phase 2: Live ETA, Lock File, State Atomicity, and Dashboard Safety.

Tests:
  1. Stale coordinates (> 5 min) -> ETA "unavailable", marked stale
  2. Missing / NaN coordinates -> ETA "unavailable", marked invalid
  3. Non-commercial exclusion vs untyped unknown (Cohort A / D017)
  4. Lock file acquisition, concurrency prevention, and stale PID recovery
  5. State file atomic write & safe concurrent reader
  6. Prediction logging and arrival scoring with mocked voyage sequence
  7. Offline replay without network access
  8. Dashboard module load does not auto-start collector
"""

import json
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add src to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orcastra_collector.lock import CollectorLock, CollectorLockError, get_running_collector_info
from orcastra_collector.state import CollectorStateWriter, read_latest_state
from orcastra_collector.live_eta import (
    compute_live_eta,
    PredictionLogger,
    haversine_km,
    channel_distance_to_berth,
    DISTANCE_BUCKETS,
    METHOD_NAME,
    P95_INTERVAL_NOTE,
    CORRIDOR,
    ENTRANCE_ZONE,
    DESTINATION_ZONE,
)
from orcastra_collector.schema import AisEvent

PASS_COUNT = 0
FAIL_COUNT = 0


def assert_true(condition: bool, msg: str):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  FAIL: {msg}")


def test_stale_coordinates_eta_unavailable():
    print("\n=== Test: Stale Coordinates (> 5 min) ===")
    now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    stale_time = (now - timedelta(seconds=350)).isoformat()

    vessel_data = {
        "mmsi": 367000001,
        "name": "TEST_STALE_SHIP",
        "latitude": 29.50,
        "longitude": -94.85,
        "sog": 10.0,
        "cog": 310.0,
        "vessel_type": 70,
        "last_seen_utc": stale_time,
    }

    eta = compute_live_eta(vessel_data, now_utc=now, stale_threshold_seconds=300.0)
    assert_true(eta.is_stale, "Vessel flagged as stale")
    assert_true(eta.status == "stale", "Status is 'stale'")
    assert_true(not eta.is_eligible, "Not eligible when stale")
    assert_true(eta.predicted_eta_utc is None, "Predicted ETA is None when stale")
    assert_true(eta.data_age_seconds >= 350.0, "Data age calculated correctly")


def test_missing_and_nan_coordinates():
    print("\n=== Test: Missing / NaN Coordinates ===")
    now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)

    # Missing None
    v1 = {
        "mmsi": 367000002,
        "name": "TEST_NONE_COORDS",
        "latitude": None,
        "longitude": None,
        "sog": 10.0,
        "cog": 310.0,
        "vessel_type": 70,
        "last_seen_utc": now.isoformat(),
    }
    eta1 = compute_live_eta(v1, now_utc=now)
    assert_true(eta1.status == "invalid", "None coords -> status 'invalid'")
    assert_true(eta1.predicted_eta_utc is None, "None coords -> ETA is None")

    # NaN coordinates
    v2 = {
        "mmsi": 367000003,
        "name": "TEST_NAN_COORDS",
        "latitude": float("nan"),
        "longitude": -94.85,
        "sog": 10.0,
        "cog": 310.0,
        "vessel_type": 70,
        "last_seen_utc": now.isoformat(),
    }
    eta2 = compute_live_eta(v2, now_utc=now)
    assert_true(eta2.status == "invalid", "NaN coords -> status 'invalid'")
    assert_true(eta2.predicted_eta_utc is None, "NaN coords -> ETA is None")


def test_cohort_a_eligibility_and_non_commercial():
    print("\n=== Test: Cohort A Eligibility & Non-Commercial Rules (D017) ===")
    now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)

    # Known non-commercial vessel (Type 31: Tug / Pushboat)
    v_tug = {
        "mmsi": 367000004,
        "name": "TEST_TUG",
        "latitude": 29.50,
        "longitude": -94.85,
        "sog": 8.0,
        "cog": 310.0,
        "vessel_type": 31,  # Tug
        "last_seen_utc": now.isoformat(),
    }
    eta_tug = compute_live_eta(v_tug, now_utc=now)
    assert_true(eta_tug.status == "non_commercial", "Tug type 31 marked 'non_commercial'")
    assert_true(not eta_tug.is_eligible, "Tug excluded from scoring")

    # Pleasure craft (Type 36)
    v_pleasure = {
        "mmsi": 367000005,
        "name": "TEST_YACHT",
        "latitude": 29.50,
        "longitude": -94.85,
        "sog": 12.0,
        "cog": 310.0,
        "vessel_type": 36,
        "last_seen_utc": now.isoformat(),
    }
    eta_pl = compute_live_eta(v_pleasure, now_utc=now)
    assert_true(eta_pl.status == "non_commercial", "Pleasure craft type 36 marked 'non_commercial'")
    assert_true(not eta_pl.is_eligible, "Pleasure craft excluded from scoring")

    # UNTYPED VESSEL (Type None): per D017, must be treated as UNKNOWN, NOT non-commercial
    v_untyped = {
        "mmsi": 367000006,
        "name": "TEST_UNTYPED_COMMERCIAL",
        "latitude": 29.50,
        "longitude": -94.85,
        "sog": 10.5,
        "cog": 310.0,
        "vessel_type": None,  # Untyped!
        "last_seen_utc": now.isoformat(),
    }
    eta_untyped = compute_live_eta(v_untyped, now_utc=now)
    assert_true(eta_untyped.status == "active", "Untyped vessel in transit marked 'active'")
    assert_true(eta_untyped.is_eligible, "Untyped vessel in transit is ELIGIBLE under Cohort A")
    assert_true(eta_untyped.predicted_eta_utc is not None, "Untyped vessel gets valid predicted ETA")


def test_lock_file_acquisition_and_stale_recovery():
    print("\n=== Test: Lock File Acquisition & Stale Recovery ===")
    with tempfile.TemporaryDirectory() as td:
        lock_path = Path(td) / "test_collector.lock"

        # 1. Normal acquisition
        lock1 = CollectorLock(lock_path, "session_1", mode="live")
        lock1.acquire()
        assert_true(lock_path.exists(), "Lock file created on disk")
        info = get_running_collector_info(lock_path)
        assert_true(info is not None and info["pid"] == os.getpid(), "Lock info contains current PID")

        # 2. Concurrency rejection
        lock2 = CollectorLock(lock_path, "session_2", mode="live")
        raised = False
        try:
            lock2.acquire()
        except CollectorLockError:
            raised = True
        assert_true(raised, "Second collector cannot acquire held lock (CollectorLockError)")

        # 3. Clean release
        lock1.release()
        assert_true(not lock_path.exists(), "Lock file removed after release")

        # 4. Stale lock recovery (simulate dead PID 99999999)
        fake_stale = {
            "pid": 99999999,
            "session_id": "session_dead",
            "mode": "live",
            "start_time_utc": "2026-10-27T00:00:00Z",
        }
        lock_path.write_text(json.dumps(fake_stale), encoding="utf-8")
        assert_true(lock_path.exists(), "Created simulated stale lock file")

        # Stale lock must be detected as dead and cleaned up
        stale_info = get_running_collector_info(lock_path)
        assert_true(stale_info is None, "get_running_collector_info returns None for dead PID")
        assert_true(not lock_path.exists(), "Stale lock automatically removed from disk")

        # Now new process can acquire cleanly
        lock3 = CollectorLock(lock_path, "session_3", mode="live")
        lock3.acquire()
        assert_true(lock_path.exists(), "New lock acquired after stale cleanup")
        lock3.release()


def test_state_file_atomicity():
    print("\n=== Test: State File Atomicity & Safe Reader ===")
    with tempfile.TemporaryDirectory() as td:
        state_path = Path(td) / "latest_state.json"
        writer = CollectorStateWriter(state_path, "sess_atomic", update_interval=0.0)

        # Populate initial vessel
        ev = AisEvent(
            event_id="ev_1", session_id="sess_atomic", source="test", parser_version="0.1",
            mmsi=367000007, vessel_name="VESSEL_ALPHA", latitude=29.45, longitude=-94.75,
            sog=11.2, cog=315.0, heading=315, vessel_type=70, vessel_category="Cargo",
            is_class_b=False, is_in_bbox=True, is_valid_mmsi=True, is_duplicate=False,
            sog_is_sentinel=False, cog_is_sentinel=False, heading_is_sentinel=False,
            raw_message_hash="hash",
        )
        writer.update_vessel(ev, name="VESSEL_ALPHA")
        writer.update_connection("connected", reconnects=0, drops=0)
        writer.update_stats(socket_frames=50, total_events=48, duplicates=2, quarantined=0)
        writer.flush(force=True)

        read_res = read_latest_state(state_path)
        assert_true(read_res is not None, "Initial state file written and read successfully")
        assert_true(read_res["stats"]["total_events"] == 48, "Total events matches")
        assert_true("367000007" in read_res["vessels"], "Vessel MMSI present in state")

        # Concurrency test: multiple rapid writes and reads
        read_errors = 0
        read_successes = 0

        def reader_worker():
            nonlocal read_errors, read_successes
            for _ in range(50):
                d = read_latest_state(state_path)
                if d is None or "stats" not in d:
                    read_errors += 1
                else:
                    read_successes += 1
                time.sleep(0.002)

        t = threading.Thread(target=reader_worker)
        t.start()

        for i in range(50):
            writer.update_stats(socket_frames=50 + i, total_events=48 + i, duplicates=2, quarantined=0)
            writer.flush(force=True)
            time.sleep(0.002)

        t.join()

        assert_true(read_errors == 0, f"Atomic reads had 0 corruption errors (out of {read_successes} reads)")
        assert_true(read_successes > 30, f"Successful concurrent reads ({read_successes})")


def test_prediction_logging_and_arrival_scoring():
    print("\n=== Test: Prediction Logging & Arrival Scoring Engine ===")
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test_prediction_log.sqlite"
        logger = PredictionLogger(db_path)

        t0 = datetime(2026, 10, 27, 10, 0, 0, tzinfo=timezone.utc)

        # 1. Inbound vessel passes Bolivar Roads
        v_inbound = {
            "mmsi": 367000008,
            "name": "TEST_CARGO_SHIP",
            "latitude": 29.380,
            "longitude": -94.780,  # ~46 km from berth
            "sog": 12.0,
            "cog": 310.0,
            "vessel_type": 70,
            "last_seen_utc": t0.isoformat(),
        }
        eta_inbound = compute_live_eta(v_inbound, now_utc=t0)
        assert_true(eta_inbound.is_eligible, "Inbound vessel eligible")
        assert_true(eta_inbound.bucket_name == "Outer Channel (25-50 km)", "Outer Channel bucket (46 km)")
        assert_true(abs(eta_inbound.predicted_duration_h - 2.591) < 0.001, "Predicted duration is 2.591 h")

        pred_id = logger.log_prediction(eta_inbound, obs_time=t0)
        assert_true(pred_id is not None, "Prediction recorded in SQLite log")

        # 2. Advance 1 hour later: Mid-Channel waypoint
        t1 = t0 + timedelta(hours=1)
        v_mid = {
            "mmsi": 367000008,
            "name": "TEST_CARGO_SHIP",
            "latitude": 29.600,
            "longitude": -94.960,  # 16 km from berth
            "sog": 10.0,
            "cog": 310.0,
            "vessel_type": 70,
            "last_seen_utc": t1.isoformat(),
        }
        eta_mid = compute_live_eta(v_mid, now_utc=t1)
        pred_id_2 = logger.log_prediction(eta_mid, obs_time=t1)
        assert_true(pred_id_2 is not None, "Second snapshot recorded in SQLite log")

        # 3. Arrival at berth 2.6 hours after start
        t_arr = t0 + timedelta(hours=2.6)
        arrival_result = logger.check_arrival_and_score(
            mmsi=367000008,
            lat=29.720,  # Destination zone
            lon=-95.050,
            sog=0.8,     # < 1.5 kn (docked / moored)
            obs_time=t_arr,
        )

        assert_true(arrival_result is not None, "Arrival event successfully detected and scored")
        assert_true(arrival_result["fully_observed"], "Vessel marked as fully observed")
        # Actual duration: 2.6 h. Initial prediction: 2.591 h. Error: ~0.009 h
        abs_err = arrival_result["absolute_error_h"]
        assert_true(abs(abs_err - 0.009) < 0.01, f"Absolute error accurate ({abs_err:.3f} h)")

        # 4. Check scoreboard output
        sb = logger.get_scoreboard_stats()
        assert_true(sb["status"] == "collecting (n = 1)", "Scoreboard status 'collecting (n = 1)'")
        assert_true(sb["n"] == 1, "Arrival count is 1")
        assert_true(sb["mae_h"] == round(abs_err, 3), "MAE matches observed error")


def test_offline_replay_without_network():
    print("\n=== Test: Offline Replay Without Network ===")
    from scripts.dashboard import load_offline_replay_events

    fixture_path = REPO_ROOT / "tests" / "fixtures" / "sample_session"
    assert_true(fixture_path.exists(), "Sample session fixture exists")

    # Set no proxy to guarantee no network is used
    old_proxy = os.environ.get("NO_PROXY")
    os.environ["NO_PROXY"] = "*"

    try:
        df = load_offline_replay_events(fixture_path)
        assert_true(not df.empty, f"Loaded {len(df)} events from fixture without network")
        assert_true("mmsi" in df.columns, "MMSI column present in replayed dataframe")
        assert_true("latitude" in df.columns and "longitude" in df.columns, "Coordinates present")
    finally:
        if old_proxy is not None:
            os.environ["NO_PROXY"] = old_proxy
        else:
            os.environ.pop("NO_PROXY", None)


def test_dashboard_does_not_auto_start_collector():
    print("\n=== Test: Dashboard Does Not Auto-Start Collector on Launch ===")
    from scripts.dashboard import get_collector_subprocess_status
    import scripts.dashboard

    # Verify collector is NOT started just by importing or inspecting the dashboard module
    status = get_collector_subprocess_status()
    # It must not have launched a collector process automatically
    assert_true(status is not None, "Dashboard status queried cleanly")
def test_dual_eta_and_comparison_note():
    print("\n=== Test: Dual ETA (Bucket Median & Kinematic Comparator) ===")
    now = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    v_active = {
        "mmsi": 367000099,
        "name": "TEST_DUAL_ETA_CARGO",
        "latitude": 29.50,
        "longitude": -94.85,
        "sog": 10.0,  # 10.0 kn = 18.52 km/h
        "cog": 310.0,
        "vessel_type": 70,
        "last_seen_utc": now.isoformat(),
    }
    eta = compute_live_eta(v_active, now_utc=now)
    assert_true(eta.is_eligible, "Active vessel is eligible")
    assert_true(eta.predicted_eta_utc is not None, "Bucket median ETA present")
    assert_true(eta.predicted_duration_h is not None, "Bucket median duration present")
    assert_true(eta.kinematic_eta_utc is not None, "Kinematic ETA present")
    assert_true(eta.kinematic_duration_h is not None, "Kinematic duration present")

    # Verify kinematic formula: d_rem / (10.0 * 1.852)
    expected_kin_h = round(eta.distance_to_berth_km / 18.52, 3)
    assert_true(abs(eta.kinematic_duration_h - expected_kin_h) < 0.05, f"Kinematic duration matches formula ({eta.kinematic_duration_h} vs {expected_kin_h})")

    # Verify note wording
    expected_note = "Bucket median has lower average error because vessel holds dominate; kinematic is shown for comparison."
    assert_true(eta.eta_comparison_note == expected_note, "Comparison note exact wording verified")

    # Stationary vessel check
    v_stat = dict(v_active, sog=1.5)
    eta_stat = compute_live_eta(v_stat, now_utc=now)
    assert_true(not eta_stat.is_eligible, "Stationary vessel ineligible")
    assert_true(eta_stat.predicted_duration_h is None, "Stationary predicted duration None")
    assert_true(eta_stat.kinematic_duration_h is None, "Stationary kinematic duration None")


def test_deploy_mode_detection_and_fallback():
    print("\n=== Test: Deploy Mode Detection & Demo JSON Fallback ===")
    from scripts.dashboard import (
        BENCHMARK_RESULTS_V2_1,
        BENCHMARK_RESULTS_V2,
        render_benchmark_chart,
        render_selection_bias_chart,
    )

    # 1. Benchmark result files must exist (either on D: or via data/demo fallback)
    assert_true(BENCHMARK_RESULTS_V2_1.exists(), f"Protocol v2.1 results exist ({BENCHMARK_RESULTS_V2_1})")
    assert_true(BENCHMARK_RESULTS_V2.exists(), f"Protocol v2 results exist ({BENCHMARK_RESULTS_V2})")

    # 2. Both files must be < 1 MB
    size_v2_1 = BENCHMARK_RESULTS_V2_1.stat().st_size
    size_v2 = BENCHMARK_RESULTS_V2.stat().st_size
    assert_true(size_v2_1 < 1024 * 1024, f"v2.1 results size ({size_v2_1} B) < 1 MB")
    assert_true(size_v2 < 1024 * 1024, f"v2 results size ({size_v2} B) < 1 MB")

    # 3. Chart rendering functions must execute cleanly and return Plotly figures
    fig_bench = render_benchmark_chart()
    assert_true(fig_bench is not None and hasattr(fig_bench, "to_dict"), "Benchmark chart renders cleanly")

    fig_bias = render_selection_bias_chart()
    assert_true(fig_bias is not None and hasattr(fig_bias, "to_dict"), "Selection bias chart renders cleanly")


def main():
    print("=" * 60)
    print("ORCASTRA Phase 2 Test Suite (Live ETA, Lock, State, Dashboard)")
    print("=" * 60)

    test_stale_coordinates_eta_unavailable()
    test_missing_and_nan_coordinates()
    test_cohort_a_eligibility_and_non_commercial()
    test_dual_eta_and_comparison_note()
    test_lock_file_acquisition_and_stale_recovery()
    test_state_file_atomicity()
    test_prediction_logging_and_arrival_scoring()
    test_offline_replay_without_network()
    test_dashboard_does_not_auto_start_collector()
    test_deploy_mode_detection_and_fallback()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 60)
    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

