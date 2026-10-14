"""
ORCASTRA Collector — SQLite Session/Heartbeat/File Ledger.

Tracks:
  - Sessions: start/end, config, status
  - Heartbeats: periodic liveness + counters
  - Files: every Parquet part with row count, hash, finalized flag
  - Quality counters: sentinels, invalid MMSI, out-of-bbox, quarantine
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    start_time      TEXT NOT NULL,
    end_time        TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    config_json     TEXT,
    bbox_json       TEXT,
    total_events    INTEGER DEFAULT 0,
    total_duplicates INTEGER DEFAULT 0,
    total_quarantined INTEGER DEFAULT 0,
    total_files     INTEGER DEFAULT 0,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS heartbeats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    events_total    INTEGER DEFAULT 0,
    events_since_last INTEGER DEFAULT 0,
    unique_vessels  INTEGER DEFAULT 0,
    reconnects      INTEGER DEFAULT 0,
    duplicates      INTEGER DEFAULT 0,
    quarantined     INTEGER DEFAULT 0,
    memory_mb       REAL,
    cpu_pct         REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS files (
    file_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    file_path       TEXT NOT NULL UNIQUE,
    file_type       TEXT NOT NULL DEFAULT 'events',
    row_count       INTEGER DEFAULT 0,
    size_bytes      INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL,
    finalized_at    TEXT,
    is_finalized    INTEGER DEFAULT 0,
    sha256_prefix   TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS quality_counters (
    session_id          TEXT PRIMARY KEY,
    sog_sentinel        INTEGER DEFAULT 0,
    cog_sentinel        INTEGER DEFAULT 0,
    heading_sentinel    INTEGER DEFAULT 0,
    invalid_mmsi        INTEGER DEFAULT 0,
    out_of_bbox         INTEGER DEFAULT 0,
    class_a_msgs        INTEGER DEFAULT 0,
    class_b_msgs        INTEGER DEFAULT 0,
    static_msgs         INTEGER DEFAULT 0,
    quarantined         INTEGER DEFAULT 0,
    duplicates          INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_session ON heartbeats(session_id);
CREATE INDEX IF NOT EXISTS idx_files_session ON files(session_id);
"""


class Ledger:
    """SQLite ledger for sessions, heartbeats, files, and quality."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        log.info("Ledger initialized: %s", db_path)

    # ── Sessions ──

    def start_session(self, session_id: str, config_json: str,
                      bbox_json: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO sessions (session_id, start_time, config_json, bbox_json) "
            "VALUES (?, ?, ?, ?)",
            (session_id, now, config_json, bbox_json)
        )
        self._conn.execute(
            "INSERT INTO quality_counters (session_id) VALUES (?)",
            (session_id,)
        )
        self._conn.commit()

    def end_session(self, session_id: str, status: str = "completed",
                    total_events: int = 0, total_duplicates: int = 0,
                    total_quarantined: int = 0, total_files: int = 0,
                    error_message: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE sessions SET end_time=?, status=?, total_events=?, "
            "total_duplicates=?, total_quarantined=?, total_files=?, error_message=? "
            "WHERE session_id=?",
            (now, status, total_events, total_duplicates,
             total_quarantined, total_files, error_message, session_id)
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return None
        cols = [d[0] for d in self._conn.execute(
            "SELECT * FROM sessions LIMIT 0").description]
        return dict(zip(cols, row))

    def get_incomplete_sessions(self) -> list[dict]:
        """Find sessions that were not properly closed (crash recovery)."""
        rows = self._conn.execute(
            "SELECT session_id, start_time FROM sessions WHERE status='running'"
        ).fetchall()
        return [{"session_id": r[0], "start_time": r[1]} for r in rows]

    # ── Heartbeats ──

    def record_heartbeat(self, session_id: str, events_total: int,
                         events_since_last: int, unique_vessels: int,
                         reconnects: int, duplicates: int, quarantined: int,
                         memory_mb: float = 0, cpu_pct: float = 0) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO heartbeats "
            "(session_id, timestamp, events_total, events_since_last, "
            "unique_vessels, reconnects, duplicates, quarantined, memory_mb, cpu_pct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, now, events_total, events_since_last,
             unique_vessels, reconnects, duplicates, quarantined,
             memory_mb, cpu_pct)
        )
        self._conn.commit()

    # ── Files ──

    def register_file(self, session_id: str, file_path: str,
                      file_type: str = "events") -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR IGNORE INTO files "
            "(session_id, file_path, file_type, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, file_path, file_type, now)
        )
        self._conn.commit()

    def finalize_file(self, file_path: str, row_count: int,
                      size_bytes: int, sha256_prefix: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE files SET is_finalized=1, finalized_at=?, "
            "row_count=?, size_bytes=?, sha256_prefix=? "
            "WHERE file_path=?",
            (now, row_count, size_bytes, sha256_prefix, file_path)
        )
        self._conn.commit()

    def get_unfinalized_files(self, session_id: str) -> list[str]:
        """Find files that were registered but not finalized (crash recovery)."""
        rows = self._conn.execute(
            "SELECT file_path FROM files WHERE session_id=? AND is_finalized=0",
            (session_id,)
        ).fetchall()
        return [r[0] for r in rows]

    def get_session_files(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT file_path, file_type, row_count, size_bytes, "
            "is_finalized, sha256_prefix FROM files WHERE session_id=?",
            (session_id,)
        ).fetchall()
        return [{"file_path": r[0], "file_type": r[1], "row_count": r[2],
                 "size_bytes": r[3], "is_finalized": bool(r[4]),
                 "sha256_prefix": r[5]} for r in rows]

    # ── Quality counters ──

    def increment_quality(self, session_id: str, **kwargs) -> None:
        """Increment quality counters. kwargs: sog_sentinel, cog_sentinel, etc."""
        sets = []
        vals = []
        for key, val in kwargs.items():
            if val and val > 0:
                sets.append(f"{key} = {key} + ?")
                vals.append(val)
        if not sets:
            return
        vals.append(session_id)
        self._conn.execute(
            f"UPDATE quality_counters SET {', '.join(sets)} WHERE session_id=?",
            vals
        )
        # Commit in batches (caller should commit periodically)

    def commit(self):
        self._conn.commit()

    def get_quality(self, session_id: str) -> dict:
        row = self._conn.execute(
            "SELECT * FROM quality_counters WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return {}
        cols = [d[0] for d in self._conn.execute(
            "SELECT * FROM quality_counters LIMIT 0").description]
        return dict(zip(cols, row))

    def close(self):
        self._conn.close()
