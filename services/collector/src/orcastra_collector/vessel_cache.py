"""
ORCASTRA Collector — SQLite Vessel Type Cache.

Persists ShipStaticData so that position reports can be enriched
with vessel type even if the static message arrived in a previous session.

unknown type is NOT non-commercial — it means we haven't received
a ShipStaticData message for that MMSI yet.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vessel_cache (
    mmsi            INTEGER PRIMARY KEY,
    imo             TEXT,
    call_sign       TEXT,
    vessel_name     TEXT,
    vessel_type     INTEGER,
    vessel_category TEXT,
    length          REAL,
    width           REAL,
    draught         REAL,
    destination     TEXT,
    eta_month       INTEGER,
    eta_day         INTEGER,
    eta_hour        INTEGER,
    eta_minute      INTEGER,
    first_seen      TEXT NOT NULL,
    last_updated    TEXT NOT NULL,
    update_count    INTEGER DEFAULT 1
);
"""


class VesselCache:
    """Thread-safe SQLite vessel type cache."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

        # In-memory lookup for fast access
        self._type_lookup: dict[int, int] = {}
        self._load_lookup()

    def _load_lookup(self):
        """Load mmsi->vessel_type mapping into memory."""
        rows = self._conn.execute(
            "SELECT mmsi, vessel_type FROM vessel_cache WHERE vessel_type IS NOT NULL"
        ).fetchall()
        self._type_lookup = {r[0]: r[1] for r in rows}
        log.info("Vessel cache loaded: %d vessels with type info", len(self._type_lookup))

    @property
    def type_lookup(self) -> dict[int, int]:
        """Returns the in-memory mmsi -> vessel_type dict."""
        return self._type_lookup

    def upsert(self, info: dict) -> None:
        """Insert or update vessel info from ShipStaticData."""
        mmsi = info.get("mmsi")
        if mmsi is None:
            return

        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute("""
            INSERT INTO vessel_cache
                (mmsi, imo, call_sign, vessel_name, vessel_type, vessel_category,
                 length, width, draught, destination,
                 eta_month, eta_day, eta_hour, eta_minute,
                 first_seen, last_updated, update_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(mmsi) DO UPDATE SET
                imo = COALESCE(excluded.imo, imo),
                call_sign = COALESCE(excluded.call_sign, call_sign),
                vessel_name = COALESCE(excluded.vessel_name, vessel_name),
                vessel_type = COALESCE(excluded.vessel_type, vessel_type),
                vessel_category = COALESCE(excluded.vessel_category, vessel_category),
                length = COALESCE(excluded.length, length),
                width = COALESCE(excluded.width, width),
                draught = COALESCE(excluded.draught, draught),
                destination = excluded.destination,
                eta_month = excluded.eta_month,
                eta_day = excluded.eta_day,
                eta_hour = excluded.eta_hour,
                eta_minute = excluded.eta_minute,
                last_updated = excluded.last_updated,
                update_count = update_count + 1
        """, (
            mmsi, info.get("imo"), info.get("call_sign"),
            info.get("vessel_name", "").strip() if info.get("vessel_name") else None,
            info.get("vessel_type"), info.get("vessel_category"),
            info.get("length"), info.get("width"), info.get("draught"),
            info.get("destination", "").strip() if info.get("destination") else None,
            info.get("eta_month"), info.get("eta_day"),
            info.get("eta_hour"), info.get("eta_minute"),
            now, now,
        ))
        self._conn.commit()

        # Update in-memory lookup
        vtype = info.get("vessel_type")
        if vtype is not None:
            self._type_lookup[mmsi] = vtype

    def get_type(self, mmsi: int) -> Optional[int]:
        """Get vessel type for an MMSI, or None if unknown."""
        return self._type_lookup.get(mmsi)

    def count(self) -> int:
        """Total vessels in cache."""
        return len(self._type_lookup)

    def close(self):
        self._conn.close()
