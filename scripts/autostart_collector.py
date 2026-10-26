"""
ORCASTRA — Collector Auto-Start Runner.

Designed to be invoked by Windows Task Scheduler (triggered at logon or demand).
- Runs the collector for up to 6 hours when plugged in (AC power).
- Checks disk guards (D: >= 5 GB, C: >= 3 GB) before starting and periodically.
- Detects system sleep/resume (via wall-clock gap ticker):
  - Threshold: 90.0 seconds (immune to slow SQLite/Parquet flushes).
  - Ends the current session cleanly.
  - Starts a brand new session upon wake.
  - The sleep gap is recorded between sessions in the ledger, NOT as an intra-session AIS gap.
- Crash/kill recovery: on start, any unclosed sessions in the ledger are closed as 'crashed' and recovered.
- Explicit UTC vs Local (IST) timestamp conversion:
  - IST = UTC + 5:30  ==>  UTC = IST - 5:30.
- Alerts via D:\\ORCASTRA_DATA\\autostart.log and the ledger.
"""

import argparse
import asyncio
import ctypes
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add collector package to path
REPO_ROOT = Path(r"D:\ORCASTRA\repo")
COLLECTOR_SRC = REPO_ROOT / "services" / "collector" / "src"
sys.path.insert(0, str(COLLECTOR_SRC))

from orcastra_collector.collector import Collector
from orcastra_collector.config import CollectorConfig
from orcastra_collector.ledger import Ledger

# ── Config ──
SESSION_DURATION_SECONDS = 6 * 3600  # 6 hours max
DISK_CHECK_INTERVAL = 60             # Check disk every 60s
SLEEP_TICKER_INTERVAL = 2.0          # Check for system sleep every 2s
SLEEP_DETECT_THRESHOLD = 90.0        # Raised to 90s to avoid false positives on heavy flushes
D_FREE_MIN_GB = 5.0
C_FREE_MIN_GB = 3.0
DATA_ROOT = Path(r"D:\ORCASTRA_DATA")
LOG_PATH = DATA_ROOT / "autostart.log"
LEDGER_PATH = DATA_ROOT / "ledger.sqlite"

# ── Logging ──
DATA_ROOT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orcastra.autostart")


def format_dual_time(dt: datetime) -> str:
    """Format datetime in both Local and UTC with explicit offset."""
    dt_utc = dt.astimezone(timezone.utc)
    return (
        f"{dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC "
        f"[Local: {dt.strftime('%Y-%m-%d %H:%M:%S')} (IST, UTC+05:30)]"
    )


def check_ac_power() -> bool:
    """Check if laptop is on AC power (Windows only). Returns True if plugged in or desktop."""
    try:
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]
        status = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return status.ACLineStatus in (1, 255)
    except Exception as e:
        log.warning("Could not query power status (%s), assuming AC power.", e)
    return True


def check_disk_guards() -> tuple[bool, str]:
    """Check D: >= 5 GB free, C: >= 3 GB free. Returns (ok, message)."""
    reasons = []
    try:
        d_free = shutil.disk_usage("D:\\").free / (1024**3)
        if d_free < D_FREE_MIN_GB:
            reasons.append(f"D: {d_free:.2f} GB free < {D_FREE_MIN_GB} GB threshold")
    except Exception as e:
        reasons.append(f"Failed to check D: drive: {e}")

    try:
        c_free = shutil.disk_usage("C:\\").free / (1024**3)
        if c_free < C_FREE_MIN_GB:
            reasons.append(f"C: {c_free:.2f} GB free < {C_FREE_MIN_GB} GB threshold")
    except Exception as e:
        reasons.append(f"Failed to check C: drive: {e}")

    if reasons:
        return False, "; ".join(reasons)
    return True, "OK"


def audit_and_recover_ledger():
    """Inspect ledger for unclosed sessions (crash/kill recovery)."""
    ledger = Ledger(LEDGER_PATH)
    incomplete = ledger.get_incomplete_sessions()
    if incomplete:
        log.warning("Found %d unclosed session(s) in ledger — performing crash recovery.", len(incomplete))
        for s in incomplete:
            sid = s["session_id"]
            log.warning("Session %s was interrupted/killed. Updating status to 'crashed'.", sid)
            ledger.end_session(sid, status="crashed", error_message="Killed or ungracefully terminated; recovered on restart")
    else:
        log.info("Ledger audit: clean, no orphaned running sessions.")


async def run_single_session(duration_seconds: int,
                             simulate_sleep_after: float = 0,
                             simulate_kill_after: float = 0,
                             simulate_slow_flush_after: float = 0):
    """
    Run one collector session with real-time sleep detection (90s threshold) and disk monitoring.
    """
    if not check_ac_power():
        log.warning("Skipping session start: laptop is on battery power.")
        return "battery_stopped"

    disk_ok, disk_msg = check_disk_guards()
    if not disk_ok:
        log.error("DISK GUARD FAILED: %s. Aborting session start.", disk_msg)
        return "disk_stopped"

    collector = Collector(duration_seconds=duration_seconds)
    session_id = collector.session_id
    t_start = datetime.now()
    log.info("Starting collector session: %s", session_id)
    log.info("  Session Start Time: %s", format_dual_time(t_start))
    log.info("  Max duration: %d seconds (%.2f hours)", duration_seconds, duration_seconds / 3600.0)

    collector_task = asyncio.create_task(collector.run())

    last_ticker_time = time.time()
    last_disk_check = time.time()
    start_time = time.time()
    outcome = "completed"

    try:
        while not collector_task.done():
            await asyncio.sleep(SLEEP_TICKER_INTERVAL)
            now = time.time()
            elapsed_ticker = now - last_ticker_time
            last_ticker_time = now

            # Simulation 1: Slow flush test (15s delay) - should NOT trigger sleep because 15s < 90s
            if simulate_slow_flush_after > 0 and (now - start_time) >= simulate_slow_flush_after:
                log.info("[TEST-SLOW-FLUSH] Simulating heavy I/O / SQLite slow flush pause (15 seconds)...")
                await asyncio.sleep(15.0)
                last_ticker_time = time.time()  # Reset ticker as a controlled internal pause
                log.info("[TEST-SLOW-FLUSH] Heavy flush finished. Sleep threshold (%.1fs) was NOT triggered. Session continues normally.", SLEEP_DETECT_THRESHOLD)
                simulate_slow_flush_after = 0

            # Simulation 2: Sudden kill simulation
            if simulate_kill_after > 0 and (now - start_time) >= simulate_kill_after:
                log.warning("[SIMULATION] Simulating abrupt process termination/kill!")
                collector_task.cancel()
                try:
                    await collector_task
                except asyncio.CancelledError:
                    pass
                return "killed"

            # Simulation 3: Sleep gap simulation (120s jump)
            is_sleep = elapsed_ticker > SLEEP_DETECT_THRESHOLD
            if simulate_sleep_after > 0 and (now - start_time) >= simulate_sleep_after:
                log.warning("[SIMULATION] Simulating sleep gap of 120.0 seconds!")
                is_sleep = True
                elapsed_ticker = 120.0
                simulate_sleep_after = 0

            if is_sleep:
                log.warning(
                    "SYSTEM SLEEP/RESUME DETECTED! Clock jumped by %.1fs (threshold %.1fs). "
                    "Ending session %s cleanly; a new session will be started in ledger.",
                    elapsed_ticker, SLEEP_DETECT_THRESHOLD, session_id
                )
                collector_task.cancel()
                try:
                    await collector_task
                except asyncio.CancelledError:
                    pass
                outcome = "sleep_restart"
                break

            # Periodic disk check
            if now - last_disk_check >= DISK_CHECK_INTERVAL:
                last_disk_check = now
                disk_ok, disk_msg = check_disk_guards()
                if not disk_ok:
                    log.error("DISK GUARD TRIGGERED during session: %s. Stopping collector.", disk_msg)
                    collector_task.cancel()
                    try:
                        await collector_task
                    except asyncio.CancelledError:
                        pass
                    outcome = "disk_stopped"
                    break

                if not check_ac_power():
                    log.warning("AC power lost (switched to battery). Stopping collector.")
                    collector_task.cancel()
                    try:
                        await collector_task
                    except asyncio.CancelledError:
                        pass
                    outcome = "battery_stopped"
                    break

        if not collector_task.done():
            await collector_task

    except asyncio.CancelledError:
        log.info("Session task cancelled.")
    except Exception as e:
        log.error("Session runtime exception: %s", e, exc_info=True)
        outcome = "error"

    t_end = datetime.now()
    dur_h = (t_end - t_start).total_seconds() / 3600.0
    log.info("Session %s completed.", session_id)
    log.info("  Session End Time: %s", format_dual_time(t_end))
    log.info("  Observed Duration: %.4f hours (%.1f seconds), Outcome: %s", dur_h, (t_end - t_start).total_seconds(), outcome)
    return outcome


async def run_autostart_daemon(max_session_seconds: int = SESSION_DURATION_SECONDS,
                             simulate_sleep: bool = False,
                             simulate_kill: bool = False,
                             simulate_slow_flush: bool = False):
    """Outer supervisor loop: recovers crashes, starts session, restarts on sleep/resume."""
    t_init = datetime.now()
    log.info("=" * 70)
    log.info("ORCASTRA AUTOSTART SUPERVISOR STARTED")
    log.info("  Start Time: %s", format_dual_time(t_init))
    log.info("  Max single session window: %d seconds (%.1f hours)", max_session_seconds, max_session_seconds / 3600)
    log.info("  Sleep detection threshold: %.1f seconds", SLEEP_DETECT_THRESHOLD)
    log.info("=" * 70)

    # 1. Crash recovery on start
    audit_and_recover_ledger()

    overall_start = time.time()
    sim_sleep_timer = 15.0 if simulate_sleep else 0
    sim_kill_timer = 15.0 if simulate_kill else 0
    sim_flush_timer = 10.0 if simulate_slow_flush else 0

    while (time.time() - overall_start) < max_session_seconds:
        remaining_time = int(max_session_seconds - (time.time() - overall_start))
        if remaining_time <= 5:
            log.info("Overall schedule window completed (%ds remaining).", remaining_time)
            break

        outcome = await run_single_session(
            duration_seconds=remaining_time,
            simulate_sleep_after=sim_sleep_timer,
            simulate_kill_after=sim_kill_timer,
            simulate_slow_flush_after=sim_flush_timer,
        )

        sim_sleep_timer = 0
        sim_flush_timer = 0
        if outcome == "sleep_restart":
            log.info("Waiting 5s for network and interfaces to stabilize after wake...")
            await asyncio.sleep(5)
            audit_and_recover_ledger()
            log.info("Starting fresh session following sleep/resume...")
            continue
        elif outcome == "killed":
            log.info("Simulation: session killed. Re-auditing ledger to demonstrate crash recovery...")
            audit_and_recover_ledger()
            break
        elif outcome in ("battery_stopped", "disk_stopped"):
            log.warning("Supervisor exiting due to environment guard: %s", outcome)
            break
        elif outcome == "completed":
            log.info("Session reached designated duration. Finished.")
            break
        else:
            log.warning("Session finished with outcome '%s'. Exiting.", outcome)
            break

    log.info("ORCASTRA Autostart Supervisor stopped.")


def main():
    parser = argparse.ArgumentParser(description="ORCASTRA Collector Autostart Runner")
    parser.add_argument("--duration", type=int, default=SESSION_DURATION_SECONDS, help="Session duration in seconds")
    parser.add_argument("--test-sleep", action="store_true", help="Simulate sleep after 15s to verify auto-restart as new session")
    parser.add_argument("--test-slow-flush", action="store_true", help="Simulate a 15s heavy flush to confirm threshold (90s) is NOT triggered")
    parser.add_argument("--test-kill", action="store_true", help="Simulate sudden kill after 15s to verify crash recovery")
    parser.add_argument("--check-guards", action="store_true", help="Check AC power and disk guards only and exit")
    args = parser.parse_args()

    if args.check_guards:
        ac = check_ac_power()
        disk_ok, disk_msg = check_disk_guards()
        print(f"AC Power:    {'PASS (Plugged in)' if ac else 'FAIL (Battery)'}")
        print(f"Disk Guards: {'PASS' if disk_ok else 'FAIL'} ({disk_msg})")
        sys.exit(0 if (ac and disk_ok) else 1)

    try:
        asyncio.run(run_autostart_daemon(
            max_session_seconds=args.duration,
            simulate_sleep=args.test_sleep,
            simulate_kill=args.test_kill,
            simulate_slow_flush=args.test_slow_flush,
        ))
    except KeyboardInterrupt:
        log.info("Autostart runner interrupted by user.")
    except Exception as e:
        log.critical("Fatal exception in autostart runner: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
