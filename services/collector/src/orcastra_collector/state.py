"""
ORCASTRA Collector — Atomic Latest State Manager.

Maintains in-memory real-time state (latest vessel positions, message counters,
connection status) and flushes atomically to `latest_state.json` every 2-5s.
Guarantees readers never see partially-written or corrupt files.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .schema import AisEvent

log = logging.getLogger(__name__)


class CollectorStateWriter:
    """Atomic state writer for the live collector."""

    def __init__(
        self,
        state_path: Path,
        session_id: str,
        pid: Optional[int] = None,
        update_interval: float = 2.0,
    ):
        self.state_path = Path(state_path)
        self.session_id = session_id
        self.pid = pid or os.getpid()
        self.update_interval = update_interval

        self._last_flush_time = 0.0
        self.connection_status = "initializing"
        self.last_message_utc: Optional[str] = None
        self.reconnects = 0
        self.drops = 0
        self.socket_frames = 0
        self.total_events = 0
        self.total_duplicates = 0
        self.total_quarantined = 0
        self.last_error: Optional[str] = None

        # MMSI -> dict of latest vessel observation
        self._vessels: dict[int, dict[str, Any]] = {}

    def update_connection(
        self,
        status: str,
        reconnects: Optional[int] = None,
        drops: Optional[int] = None,
        last_message_time: Optional[datetime] = None,
        error: Optional[str] = None,
    ) -> None:
        self.connection_status = status
        if reconnects is not None:
            self.reconnects = reconnects
        if drops is not None:
            self.drops = drops
        if last_message_time is not None:
            self.last_message_utc = last_message_time.astimezone(timezone.utc).isoformat()
        if error is not None:
            self.last_error = error

    def update_stats(
        self,
        socket_frames: int,
        total_events: int,
        duplicates: int,
        quarantined: int,
    ) -> None:
        self.socket_frames = socket_frames
        self.total_events = total_events
        self.total_duplicates = duplicates
        self.total_quarantined = quarantined

    def update_vessel(self, event: AisEvent, name: Optional[str] = None) -> None:
        """Update or insert latest observation for a vessel."""
        # Only record if coordinates are present
        if event.latitude is None or event.longitude is None:
            return

        now_utc = datetime.now(timezone.utc).isoformat()
        obs_utc = (
            event.observation_time.astimezone(timezone.utc).isoformat()
            if event.observation_time
            else now_utc
        )

        existing = self._vessels.get(event.mmsi, {})
        vessel_name = name or existing.get("name") or f"MMSI_{event.mmsi}"

        self._vessels[event.mmsi] = {
            "mmsi": event.mmsi,
            "name": vessel_name,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "sog": event.sog,
            "cog": event.cog,
            "heading": event.heading,
            "vessel_type": event.vessel_type,
            "vessel_category": event.vessel_category,
            "is_class_b": event.is_class_b,
            "is_in_bbox": event.is_in_bbox,
            "last_seen_utc": now_utc,
            "observation_time_utc": obs_utc,
        }

    def flush(self, force: bool = False) -> None:
        """Atomically write latest state to JSON file."""
        now = time.time()
        if not force and (now - self._last_flush_time) < self.update_interval:
            return

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_name(f"{self.state_path.name}.{self.pid}.tmp")

        payload = {
            "version": "1.0",
            "session_id": self.session_id,
            "pid": self.pid,
            "status": self.connection_status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "connection": {
                "status": self.connection_status,
                "last_message_received_utc": self.last_message_utc,
                "reconnects": self.reconnects,
                "drops": self.drops,
                "last_error": self.last_error,
            },
            "stats": {
                "socket_frames": self.socket_frames,
                "total_events": self.total_events,
                "total_duplicates": self.total_duplicates,
                "total_quarantined": self.total_quarantined,
                "unique_vessels": len(self._vessels),
            },
            "vessels": self._vessels,
        }

        try:
            tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            replaced = False
            for _ in range(5):
                try:
                    os.replace(tmp_path, self.state_path)
                    replaced = True
                    break
                except (PermissionError, OSError):
                    time.sleep(0.01)

            if replaced:
                self._last_flush_time = now
        except Exception as e:
            log.warning("Failed to atomically flush latest state to %s: %s", self.state_path, e)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass


def read_latest_state(state_path: Path, max_retries: int = 3) -> Optional[dict]:
    """
    Safely read latest state file with retry to prevent race conditions during file replacement.
    Returns dict if readable, else None.
    """
    path = Path(state_path)
    if not path.exists():
        return None

    for attempt in range(max_retries):
        try:
            content = path.read_text(encoding="utf-8")
            return json.loads(content)
        except (json.JSONDecodeError, PermissionError, FileNotFoundError):
            if attempt < max_retries - 1:
                time.sleep(0.05)
            else:
                return None
        except Exception as e:
            log.warning("Unexpected error reading latest state from %s: %s", state_path, e)
            return None
    return None
