"""
ORCASTRA Collector — Main collection loop.

Connects to AISStream, parses messages, writes Parquet parts,
maintains vessel cache and ledger, handles reconnect with backoff.
"""

import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil

from .config import CollectorConfig, DEFAULT_CONFIG
from .parser import AisParser
from .schema import AisEvent
from .writer import ParquetWriter
from .vessel_cache import VesselCache
from .ledger import Ledger
from .lock import CollectorLock, CollectorLockError
from .state import CollectorStateWriter
from .live_eta import PredictionLogger, compute_live_eta

log = logging.getLogger(__name__)


class Collector:
    """AISStream live collector with reconnect, dedup, and crash safety."""

    def __init__(self, config: CollectorConfig = DEFAULT_CONFIG,
                 api_key: Optional[str] = None,
                 duration_seconds: Optional[int] = None):
        self.config = config
        self.api_key = api_key or self._load_api_key()
        self.duration = duration_seconds  # None = run until interrupted

        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now(timezone.utc)
        session_name = self.session_start.strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = config.recordings_dir / session_name

        # Components
        self.ledger = Ledger(config.ledger_path)
        self.vessel_cache = VesselCache(config.vessel_cache_path)
        self.parser = AisParser(self.session_id, config)
        self.writer = ParquetWriter(
            self.session_dir, self.session_id, self.ledger,
            max_rows=config.parquet_part_max_rows,
            max_seconds=config.parquet_part_max_seconds,
            compression=config.parquet_compression,
        )

        # Process lock and atomic live state
        self.lock = CollectorLock(config.lock_file_path, self.session_id, mode="live")
        self.state_writer = CollectorStateWriter(
            config.latest_state_path,
            self.session_id,
            update_interval=config.state_update_interval,
        )
        self.prediction_logger = PredictionLogger(config.prediction_log_path)

        # Dedup set (mmsi_timestamp -> True)
        self._seen_keys: set[str] = set()

        # Stats
        self._reconnects = 0
        self._drops = 0
        self._socket_frames = 0     # raw WebSocket frames received (independent of parser)
        self._total_events = 0
        self._total_duplicates = 0
        self._total_quarantined = 0
        self._unique_mmsis: set[int] = set()
        self._last_heartbeat = time.time()
        self._last_heartbeat_count = 0

        # Quality batch counters (flushed to ledger periodically)
        self._quality_batch = {
            "sog_sentinel": 0, "cog_sentinel": 0, "heading_sentinel": 0,
            "invalid_mmsi": 0, "out_of_bbox": 0,
            "class_a_msgs": 0, "class_b_msgs": 0, "static_msgs": 0,
            "quarantined": 0, "duplicates": 0,
        }

    def _load_api_key(self) -> str:
        env_path = Path(r"D:\ORCASTRA\repo\.env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("AISSTREAM_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if k and k != "your_api_key_here":
                        return k
        raise ValueError("No valid AISSTREAM_API_KEY found in .env")

    async def run(self) -> dict:
        """Main collection loop. Returns session summary."""
        import websockets

        log.info("="*60)
        log.info("ORCASTRA Collector starting")
        log.info("  Session: %s", self.session_id)
        log.info("  Dir: %s", self.session_dir)
        log.info("  BBoxes: %s", self.config.bounding_boxes)
        log.info("  Duration: %s", f"{self.duration}s" if self.duration else "until interrupted")
        log.info("  Vessel cache: %d vessels loaded", self.vessel_cache.count())
        log.info("="*60)

        # Clean up any incomplete files from previous crash
        cleaned = self.writer.cleanup_incomplete()
        if cleaned:
            log.warning("Cleaned %d incomplete tmp files from previous crash", cleaned)

        # Check for incomplete sessions
        incomplete = self.ledger.get_incomplete_sessions()
        for s in incomplete:
            if s["session_id"] != self.session_id:
                log.warning("Previous session %s was not properly closed", s["session_id"])
                self.ledger.end_session(s["session_id"], status="crashed",
                                       error_message="Recovered by new session")

        # Register session
        self.ledger.start_session(
            self.session_id,
            config_json=json.dumps({
                "parser_version": self.config.parser_version,
                "bounding_boxes": self.config.bounding_boxes,
                "message_types": list(self.config.message_types),
            }),
            bbox_json=json.dumps(self.config.bounding_boxes),
        )

        # Acquire lock and initialize state file
        self.lock.acquire()
        self.state_writer.update_connection("initializing")
        self.state_writer.flush(force=True)

        subscription = {
            "APIKey": self.api_key,
            "BoundingBoxes": self.config.bounding_boxes,
            "FilterMessageTypes": list(self.config.message_types),
        }

        start_time = time.time()

        try:
            while True:
                # Check duration limit
                if self.duration and (time.time() - start_time) >= self.duration:
                    log.info("Duration limit reached (%ds)", self.duration)
                    break

                try:
                    async with websockets.connect(
                        self.config.aisstream_url,
                        compression="deflate",
                        open_timeout=self.config.ws_open_timeout,
                        close_timeout=self.config.ws_close_timeout,
                    ) as ws:
                        await ws.send(json.dumps(subscription))
                        if self._reconnects > 0:
                            log.info("Reconnected (#%d)", self._reconnects)
                        else:
                            log.info("Connected to AISStream")

                        self.state_writer.update_connection("connected", reconnects=self._reconnects, drops=self._drops)
                        self.state_writer.flush(force=True)

                        while True:
                            if self.duration and (time.time() - start_time) >= self.duration:
                                break

                            try:
                                raw = await asyncio.wait_for(
                                    ws.recv(),
                                    timeout=self.config.ws_recv_timeout
                                )
                                if isinstance(raw, bytes):
                                    raw = raw.decode("utf-8")

                                self._socket_frames += 1
                                receipt_time = datetime.now(timezone.utc)
                                self._process_message(raw, receipt_time)

                                # Periodic heartbeat
                                if time.time() - self._last_heartbeat >= self.config.heartbeat_interval:
                                    self._emit_heartbeat()

                            except asyncio.TimeoutError:
                                self._drops += 1
                                log.warning("No message for %.0fs (drop #%d)",
                                            self.config.ws_recv_timeout, self._drops)
                                self.state_writer.update_connection("connected", drops=self._drops)
                                self.state_writer.flush(force=False)

                except Exception as e:
                    self._reconnects += 1
                    delay = min(
                        self.config.reconnect_base_delay * (2 ** min(self._reconnects, 6)),
                        self.config.reconnect_max_delay
                    )
                    log.warning("Connection error (#%d): %s. Retry in %.0fs",
                                self._reconnects, str(e)[:100], delay)
                    self.state_writer.update_connection(
                        "reconnecting",
                        reconnects=self._reconnects,
                        drops=self._drops,
                        error=str(e)[:100],
                    )
                    self.state_writer.flush(force=True)

                    if self._reconnects > self.config.reconnect_max_retries:
                        log.error("Max reconnect retries exceeded")
                        break

                    await asyncio.sleep(delay)

        except KeyboardInterrupt:
            log.info("Interrupted by user")
        except Exception as e:
            log.error("Fatal error: %s", e, exc_info=True)
        finally:
            summary = self._finalize()

        return summary

    def _process_message(self, raw: str, receipt_time: datetime) -> None:
        """Parse, dedup, enrich, validate, write one message."""
        event, static_data = self.parser.parse(
            raw, receipt_time,
            vessel_type_lookup=self.vessel_cache.type_lookup
        )

        if event is None:
            # Check if it should be quarantined
            try:
                msg = json.loads(raw)
                msg_type = msg.get("MessageType", "")
                if msg_type not in ("SubscriptionConfirmation", ""):
                    self.writer.add_quarantine(raw, f"Unknown type: {msg_type}")
                    self._total_quarantined += 1
                    self._quality_batch["quarantined"] += 1
            except Exception:
                self.writer.add_quarantine(raw, "Unparseable JSON")
                self._total_quarantined += 1
                self._quality_batch["quarantined"] += 1
            return

        # Update vessel cache with static data
        if static_data:
            self.vessel_cache.upsert(static_data)
            self._quality_batch["static_msgs"] += 1

        # Dedup check
        if event.dedup_key in self._seen_keys:
            event.is_duplicate = True
            self._total_duplicates += 1
            self._quality_batch["duplicates"] += 1
        else:
            self._seen_keys.add(event.dedup_key)

        # Quality counters
        if event.sog_is_sentinel: self._quality_batch["sog_sentinel"] += 1
        if event.cog_is_sentinel: self._quality_batch["cog_sentinel"] += 1
        if event.heading_is_sentinel: self._quality_batch["heading_sentinel"] += 1
        if not event.is_valid_mmsi: self._quality_batch["invalid_mmsi"] += 1
        if not event.is_in_bbox: self._quality_batch["out_of_bbox"] += 1
        if event.is_class_b:
            self._quality_batch["class_b_msgs"] += 1
        elif event.message_type != "ShipStaticData":
            self._quality_batch["class_a_msgs"] += 1

        # Track vessel
        self._unique_mmsis.add(event.mmsi)

        # Write (including duplicates — flagged, not dropped)
        self.writer.add_event(event, raw)
        self._total_events += 1

        # Update latest state
        self.state_writer.update_connection("connected", last_message_time=receipt_time)
        self.state_writer.update_vessel(event, name=event.vessel_name)
        self.state_writer.update_stats(
            socket_frames=self._socket_frames,
            total_events=self._total_events,
            duplicates=self._total_duplicates,
            quarantined=self._total_quarantined,
        )
        self.state_writer.flush(force=False)

        # Log prediction and check arrival scoring
        if event.latitude is not None and event.longitude is not None and not event.is_duplicate:
            v_dict = {
                "mmsi": event.mmsi,
                "name": event.vessel_name,
                "latitude": event.latitude,
                "longitude": event.longitude,
                "sog": event.sog,
                "cog": event.cog,
                "vessel_type": event.vessel_type,
                "last_seen_utc": receipt_time.isoformat(),
            }
            live_eta = compute_live_eta(v_dict, now_utc=receipt_time)
            if live_eta.is_eligible:
                self.prediction_logger.log_prediction(
                    live_eta, obs_time=event.observation_time or receipt_time
                )
            if event.sog is not None and not event.sog_is_sentinel:
                self.prediction_logger.check_arrival_and_score(
                    event.mmsi, event.latitude, event.longitude, event.sog,
                    obs_time=event.observation_time or receipt_time
                )

    def _emit_heartbeat(self) -> None:
        """Record a heartbeat in the ledger."""
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        cpu = process.cpu_percent(interval=0.1)

        events_since = self._total_events - self._last_heartbeat_count
        self.ledger.record_heartbeat(
            self.session_id,
            events_total=self._total_events,
            events_since_last=events_since,
            unique_vessels=len(self._unique_mmsis),
            reconnects=self._reconnects,
            duplicates=self._total_duplicates,
            quarantined=self._total_quarantined,
            memory_mb=round(mem_mb, 1),
            cpu_pct=round(cpu, 1),
        )

        # Flush quality counters
        self.ledger.increment_quality(self.session_id, **self._quality_batch)
        self.ledger.commit()
        self._quality_batch = {k: 0 for k in self._quality_batch}

        self._last_heartbeat = time.time()
        self._last_heartbeat_count = self._total_events

        log.info("Heartbeat: %d events, %d vessels, %.1f MB RSS, %.1f%% CPU",
                 self._total_events, len(self._unique_mmsis), mem_mb, cpu)

    def _finalize(self) -> dict:
        """Flush buffers, close ledger, return summary."""
        writer_summary = self.writer.finalize()

        # Final quality flush
        self.ledger.increment_quality(self.session_id, **self._quality_batch)

        # End session
        self.ledger.end_session(
            self.session_id,
            status="completed",
            total_events=self._total_events,
            total_duplicates=self._total_duplicates,
            total_quarantined=self._total_quarantined,
            total_files=writer_summary["total_files"],
        )

        quality = self.ledger.get_quality(self.session_id)
        files = self.ledger.get_session_files(self.session_id)

        summary = {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "start_time": self.session_start.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "socket_frames": self._socket_frames,
            "total_events": self._total_events,
            "total_duplicates": self._total_duplicates,
            "total_quarantined": self._total_quarantined,
            "unique_vessels": len(self._unique_mmsis),
            "reconnects": self._reconnects,
            "drops": self._drops,
            "parser_stats": self.parser.stats,
            "quality": quality,
            "files": files,
            "writer": writer_summary,
            "vessel_cache_size": self.vessel_cache.count(),
        }

        # Save summary
        summary_path = self.session_dir / "session_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        self.vessel_cache.close()
        self.ledger.close()

        # Finalize latest state and release process lock
        self.state_writer.update_connection("stopped")
        self.state_writer.update_stats(
            socket_frames=self._socket_frames,
            total_events=self._total_events,
            duplicates=self._total_duplicates,
            quarantined=self._total_quarantined,
        )
        self.state_writer.flush(force=True)
        self.lock.release()

        log.info("="*60)
        log.info("Session complete: %s", self.session_id)
        log.info("  Events: %d, Duplicates: %d, Quarantined: %d",
                 self._total_events, self._total_duplicates, self._total_quarantined)
        log.info("  Vessels: %d, Reconnects: %d, Files: %d",
                 len(self._unique_mmsis), self._reconnects, writer_summary["total_files"])
        log.info("  Dir: %s", self.session_dir)
        log.info("="*60)

        return summary


def main():
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    ap = argparse.ArgumentParser(description="ORCASTRA AIS Collector")
    ap.add_argument("--duration", type=int, default=None,
                    help="Recording duration in seconds (default: until Ctrl+C)")
    ap.add_argument("--max-rows", type=int, default=10_000,
                    help="Max rows per Parquet part (default: 10000)")
    ap.add_argument("--max-seconds", type=int, default=300,
                    help="Max seconds per Parquet part (default: 300)")
    args = ap.parse_args()

    config = CollectorConfig(
        parquet_part_max_rows=args.max_rows,
        parquet_part_max_seconds=args.max_seconds,
    )
    collector = Collector(config=config, duration_seconds=args.duration)
    summary = asyncio.run(collector.run())
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
