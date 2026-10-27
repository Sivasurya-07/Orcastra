"""
ORCASTRA Collector — Process Lock Manager.

Enforces single-instance execution across Task Scheduler, CLI, and Dashboard.
Stale lock files left by terminated processes are automatically detected and cleared.
"""

import atexit
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil

log = logging.getLogger(__name__)


class CollectorLockError(Exception):
    """Raised when another collector instance is already running."""
    pass


def is_pid_alive(pid: int) -> bool:
    """Check if process with given PID exists and is not a zombie."""
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    except Exception:
        return False


def get_running_collector_info(lock_path: Path) -> Optional[dict]:
    """
    Check if a collector is running via the lock file.
    If the lock file exists but the PID is dead, the stale lock is removed.
    Returns lock info dict if running, else None.
    """
    if not lock_path.exists():
        return None

    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = data.get("pid")
        if pid and is_pid_alive(pid):
            return data
        else:
            log.warning("Stale collector lock detected (PID %s dead). Removing %s", pid, lock_path)
            try:
                lock_path.unlink(missing_ok=True)
            except Exception as e:
                log.warning("Could not unlink stale lock: %s", e)
            return None
    except Exception as e:
        log.warning("Could not read collector lock file %s: %s", lock_path, e)
        return None


class CollectorLock:
    """
    Exclusive lock ensuring only one collector runs at a time.
    """

    def __init__(self, lock_path: Path, session_id: str, mode: str = "live"):
        self.lock_path = Path(lock_path)
        self.session_id = session_id
        self.mode = mode
        self._acquired = False

    def acquire(self) -> "CollectorLock":
        existing = get_running_collector_info(self.lock_path)
        if existing is not None:
            pid = existing.get("pid")
            start = existing.get("start_time_utc", "unknown")
            mode = existing.get("mode", "unknown")
            raise CollectorLockError(
                f"Another collector is already running (PID: {pid}, mode: {mode}, started: {start}). "
                f"Only one collector instance may run at a time."
            )

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.lock_path.with_name(f"{self.lock_path.name}.{os.getpid()}.tmp")

        lock_data = {
            "pid": os.getpid(),
            "session_id": self.session_id,
            "mode": self.mode,
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
        }

        try:
            tmp_path.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")
            os.replace(tmp_path, self.lock_path)
            self._acquired = True
            atexit.register(self.release)
            log.info("Acquired collector lock: %s (PID %d)", self.lock_path, os.getpid())
            return self
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise CollectorLockError(f"Failed to acquire collector lock: {e}") from e

    def release(self) -> None:
        if not self._acquired:
            return

        if self.lock_path.exists():
            try:
                data = json.loads(self.lock_path.read_text(encoding="utf-8"))
                # Only release if we own it
                if data.get("pid") == os.getpid():
                    self.lock_path.unlink(missing_ok=True)
                    log.info("Released collector lock: %s (PID %d)", self.lock_path, os.getpid())
            except Exception as e:
                log.warning("Error releasing lock %s: %s", self.lock_path, e)
        self._acquired = False

    def __enter__(self) -> "CollectorLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
