"""
ORCASTRA Collector — Crash-safe Immutable Parquet Writer.

Design:
  1. Events accumulate in an in-memory buffer.
  2. When the buffer hits part_max_rows or part_max_seconds, it is flushed.
  3. Flush writes to a .tmp file first, then atomically renames to .parquet.
  4. The ledger registers the file BEFORE write, finalizes AFTER rename.
  5. If the process crashes mid-write, the .tmp file is not finalized in the ledger
     and will be cleaned up on recovery. The previous finalized .parquet files
     are intact and immutable.
  6. Once a .parquet part is finalized, it is NEVER modified.

Raw payloads are written to a separate .jsonl file (one JSON per line),
flushed together with events for traceability.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import AisEvent, PARQUET_SCHEMA, event_to_dict
from .ledger import Ledger

log = logging.getLogger(__name__)


class ParquetWriter:
    """Crash-safe, append-only Parquet part writer."""

    def __init__(self, session_dir: Path, session_id: str,
                 ledger: Ledger,
                 max_rows: int = 10_000,
                 max_seconds: int = 300,
                 compression: str = "zstd"):
        self.session_dir = session_dir
        self.session_id = session_id
        self.ledger = ledger
        self.max_rows = max_rows
        self.max_seconds = max_seconds
        self.compression = compression

        # Directories
        self.events_dir = session_dir / "events"
        self.raw_dir = session_dir / "raw"
        self.quarantine_dir = session_dir / "quarantine"
        for d in (self.events_dir, self.raw_dir, self.quarantine_dir):
            d.mkdir(parents=True, exist_ok=True)

        # State
        self._buffer: list[dict] = []
        self._raw_buffer: list[str] = []
        self._part_number = 0
        self._part_start_time = time.time()
        self._total_rows_written = 0
        self._total_files = 0

        # Find highest existing part number (for restart recovery)
        existing = list(self.events_dir.glob("part_*.parquet"))
        if existing:
            nums = [int(f.stem.split("_")[1]) for f in existing]
            self._part_number = max(nums)
            log.info("Resuming from part %06d", self._part_number)

    def add_event(self, event: AisEvent, raw_json: str) -> None:
        """Add an event to the buffer. Auto-flushes when limits reached."""
        d = event_to_dict(event)
        d["ingestion_time"] = datetime.now(timezone.utc)
        self._buffer.append(d)
        self._raw_buffer.append(raw_json if isinstance(raw_json, str)
                                else json.dumps(raw_json, separators=(",", ":")))

        if (len(self._buffer) >= self.max_rows or
                time.time() - self._part_start_time >= self.max_seconds):
            self.flush()

    def add_quarantine(self, raw_json: str, reason: str) -> None:
        """Write a malformed message to the quarantine file."""
        qf = self.quarantine_dir / "quarantine.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "reason": reason,
            "raw": raw_json[:2000] if isinstance(raw_json, str) else str(raw_json)[:2000],
        }
        with open(qf, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def flush(self) -> Optional[str]:
        """
        Flush the buffer to a new Parquet part.
        Returns the file path if written, None if buffer was empty.
        
        Crash-safety protocol:
          1. Register file path in ledger (unfinalized)
          2. Write to .tmp file
          3. Rename .tmp -> .parquet (atomic on same filesystem)
          4. Finalize in ledger with row count and hash
        """
        if not self._buffer:
            return None

        self._part_number += 1
        part_name = f"part_{self._part_number:06d}"
        parquet_path = self.events_dir / f"{part_name}.parquet"
        tmp_path = self.events_dir / f"{part_name}.tmp.parquet"
        raw_path = self.raw_dir / f"{part_name}.jsonl"

        # 1. Register in ledger (unfinalized)
        self.ledger.register_file(self.session_id, str(parquet_path), "events")

        try:
            # 2. Build PyArrow table and write to tmp
            table = pa.table(
                {col: [row[col] for row in self._buffer]
                 for col in PARQUET_SCHEMA.names},
                schema=PARQUET_SCHEMA
            )
            pq.write_table(table, tmp_path, compression=self.compression)

            # Write raw payloads
            with open(raw_path, "w", encoding="utf-8") as f:
                for line in self._raw_buffer:
                    f.write(line + "\n")

            # 3. Atomic rename
            tmp_path.rename(parquet_path)

            # 4. Finalize in ledger
            size = parquet_path.stat().st_size
            h = hashlib.sha256(parquet_path.read_bytes()).hexdigest()[:16]
            row_count = len(self._buffer)
            self.ledger.finalize_file(str(parquet_path), row_count, size, h)
            self.ledger.register_file(self.session_id, str(raw_path), "raw")
            self.ledger.finalize_file(str(raw_path), row_count,
                                     raw_path.stat().st_size, "")

            self._total_rows_written += row_count
            self._total_files += 1

            log.info("Part %s: %d rows, %d bytes, hash=%s",
                     part_name, row_count, size, h)

        except Exception as e:
            log.error("Failed to flush part %s: %s", part_name, e)
            # Clean up tmp file if it exists
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        finally:
            # Clear buffers regardless
            self._buffer.clear()
            self._raw_buffer.clear()
            self._part_start_time = time.time()

        return str(parquet_path)

    def finalize(self) -> dict:
        """Flush remaining buffer and return summary."""
        self.flush()
        return {
            "total_rows": self._total_rows_written,
            "total_files": self._total_files,
            "session_dir": str(self.session_dir),
        }

    def cleanup_incomplete(self) -> int:
        """Remove any .tmp.parquet files from a previous crash."""
        cleaned = 0
        for tmp in self.events_dir.glob("*.tmp.parquet"):
            log.warning("Removing incomplete tmp file: %s", tmp)
            tmp.unlink()
            cleaned += 1
        return cleaned
