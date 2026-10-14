"""
ORCASTRA Collector — Canonical AIS Event Schema.

This module defines the canonical event that the collector produces.
It is SEPARATE from the raw AISStream payload, which is preserved verbatim.

Timestamp semantics (CORRECTED 2026-10-09):
  aisstream_received_time — when AISStream received/processed the AIS message (MetaData.time_utc).
                            This is NOT the vessel's AIS transmission time.
                            Evidence-backed assumption: official docs do not define time_utc.
  ais_payload_timestamp   — AIS payload's own transponder timestamp (0-59 seconds)
                            from Message.*.Timestamp. This is only the seconds portion,
                            not an absolute UTC datetime.
  receipt_time            — when the collector process received the WebSocket message frame.
  session_start           — when the recording session began (set once at session start).
  ingestion_time          — when the event was written to Parquet (set by the writer).

Dedup key (REDESIGNED 2026-10-09):
  (mmsi, aisstream_received_time truncated to second, message_type[, ais_payload_timestamp])
  Rationale: AISStream reception time is server-side reception, not vessel transmission.
  Different message types (e.g. PositionReport + ShipStaticData) received in the
  same second for the same MMSI are distinct messages, not duplicates.
  True duplicates are identical MMSI + reception second + message type.
  The first received is kept; later ones are flagged is_duplicate=True.

Traceability:
  Every event carries session_id, source, parser_version, and raw_message_hash
  so it can be traced back to the exact session, pipeline version, and raw payload.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pyarrow as pa

# ── Canonical event dataclass ──

@dataclass
class AisEvent:
    """One canonical AIS event. Produced by the parser, consumed by the writer."""

    # Identity
    event_id: str                    # deterministic UUID from dedup_key
    session_id: str                  # UUID of the recording session
    source: str                      # "aisstream_live" or "noaa_replay"
    parser_version: str              # e.g. "0.1.0"

    # Vessel
    mmsi: int
    imo: Optional[str] = None
    vessel_name: Optional[str] = None
    vessel_type: Optional[int] = None     # AIS type code (from cache or static msg)
    vessel_category: Optional[str] = None # derived: cargo, tanker, tug_towing, etc.
    call_sign: Optional[str] = None

    # Timestamps (all UTC, all distinct)
    aisstream_received_time: Optional[datetime] = None  # AISStream reception time (NOT vessel tx time)
    ais_payload_timestamp: Optional[int] = None        # 0-59 seconds from AIS payload
    receipt_time: Optional[datetime] = None            # when collector received the WebSocket frame
    ingestion_time: Optional[datetime] = None          # when written to Parquet (set by writer)

    # Position
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    sog: Optional[float] = None       # knots
    cog: Optional[float] = None       # degrees
    heading: Optional[int] = None     # degrees
    rot: Optional[float] = None       # rate of turn
    nav_status: Optional[int] = None
    position_accuracy: Optional[bool] = None

    # Message metadata
    message_type: str = ""           # PositionReport, StandardClassBPositionReport, etc.
    ais_message_id: Optional[int] = None  # AIS message type number (1,2,3,5,18,19,24)
    is_class_b: bool = False
    raw_message_hash: str = ""       # SHA-256 of the raw JSON payload

    # Quality flags
    is_valid_mmsi: bool = True
    sog_is_sentinel: bool = False    # True if sog == 102.3
    cog_is_sentinel: bool = False    # True if cog == 360.0
    heading_is_sentinel: bool = False # True if heading == 511
    is_in_bbox: bool = True          # True if lat/lon within configured bbox

    # Dedup
    dedup_key: str = ""              # "mmsi_YYYYMMDDTHHMMSS_msgtype[_payloadts]"
    is_duplicate: bool = False

    @property
    def observation_time(self) -> Optional[datetime]:
        """Backward-compatible alias for aisstream_received_time."""
        return self.aisstream_received_time

    @observation_time.setter
    def observation_time(self, value: Optional[datetime]):
        self.aisstream_received_time = value

    @staticmethod
    def make_dedup_key(mmsi: int, received_time: Optional[datetime],
                       message_type: str = "", payload_ts: Optional[int] = None) -> str:
        """Deterministic dedup key: mmsi + reception second + message type + payload ts."""
        if received_time is None:
            ts = "unknown"
        else:
            ts = received_time.strftime("%Y%m%dT%H%M%S")
        parts = [str(mmsi), ts]
        if message_type:
            parts.append(message_type)
        if payload_ts is not None:
            parts.append(str(payload_ts))
        return "_".join(parts)

    @staticmethod
    def make_event_id(dedup_key: str) -> str:
        """Deterministic UUID5 from dedup key (reproducible across replays)."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"orcastra.ais.{dedup_key}"))

    @staticmethod
    def hash_raw(raw_json: str) -> str:
        """SHA-256 of the raw message for traceability."""
        return hashlib.sha256(raw_json.encode("utf-8")).hexdigest()[:16]


# ── Vessel type categorization ──

def categorize_vessel_type(vt: Optional[int]) -> Optional[str]:
    """Map AIS vessel type code to category. Returns None if unknown."""
    if vt is None:
        return None
    if 70 <= vt <= 79: return "cargo"
    if 80 <= vt <= 89: return "tanker"
    if 60 <= vt <= 69: return "passenger"
    if vt in (31, 32, 52): return "tug_towing"
    if vt == 30: return "fishing"
    if vt == 50: return "pilot"
    if vt in (36, 37): return "pleasure_sailing"
    if vt == 51: return "sar"
    return "other"


# ── PyArrow schema for Parquet output ──

PARQUET_SCHEMA = pa.schema([
    pa.field("event_id", pa.string(), nullable=False),
    pa.field("session_id", pa.string(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("parser_version", pa.string(), nullable=False),

    pa.field("mmsi", pa.int64(), nullable=False),
    pa.field("imo", pa.string(), nullable=True),
    pa.field("vessel_name", pa.string(), nullable=True),
    pa.field("vessel_type", pa.int32(), nullable=True),
    pa.field("vessel_category", pa.string(), nullable=True),
    pa.field("call_sign", pa.string(), nullable=True),

    pa.field("aisstream_received_time", pa.timestamp("us", tz="UTC"), nullable=True),
    pa.field("ais_payload_timestamp", pa.int8(), nullable=True),
    pa.field("receipt_time", pa.timestamp("us", tz="UTC"), nullable=True),
    pa.field("ingestion_time", pa.timestamp("us", tz="UTC"), nullable=True),

    pa.field("latitude", pa.float64(), nullable=True),
    pa.field("longitude", pa.float64(), nullable=True),
    pa.field("sog", pa.float32(), nullable=True),
    pa.field("cog", pa.float32(), nullable=True),
    pa.field("heading", pa.int16(), nullable=True),
    pa.field("rot", pa.float32(), nullable=True),
    pa.field("nav_status", pa.int8(), nullable=True),
    pa.field("position_accuracy", pa.bool_(), nullable=True),

    pa.field("message_type", pa.string(), nullable=False),
    pa.field("ais_message_id", pa.int8(), nullable=True),
    pa.field("is_class_b", pa.bool_(), nullable=False),
    pa.field("raw_message_hash", pa.string(), nullable=False),

    pa.field("is_valid_mmsi", pa.bool_(), nullable=False),
    pa.field("sog_is_sentinel", pa.bool_(), nullable=False),
    pa.field("cog_is_sentinel", pa.bool_(), nullable=False),
    pa.field("heading_is_sentinel", pa.bool_(), nullable=False),
    pa.field("is_in_bbox", pa.bool_(), nullable=False),

    pa.field("dedup_key", pa.string(), nullable=False),
    pa.field("is_duplicate", pa.bool_(), nullable=False),
])


def event_to_dict(ev: AisEvent) -> dict:
    """Convert an AisEvent to a dict matching PARQUET_SCHEMA columns."""
    return {
        "event_id": ev.event_id,
        "session_id": ev.session_id,
        "source": ev.source,
        "parser_version": ev.parser_version,
        "mmsi": ev.mmsi,
        "imo": ev.imo,
        "vessel_name": ev.vessel_name.strip() if ev.vessel_name else None,
        "vessel_type": ev.vessel_type,
        "vessel_category": ev.vessel_category,
        "call_sign": ev.call_sign.strip() if ev.call_sign else None,
        "aisstream_received_time": ev.aisstream_received_time,
        "ais_payload_timestamp": ev.ais_payload_timestamp,
        "receipt_time": ev.receipt_time,
        "ingestion_time": ev.ingestion_time,
        "latitude": ev.latitude,
        "longitude": ev.longitude,
        "sog": ev.sog,
        "cog": ev.cog,
        "heading": ev.heading,
        "rot": ev.rot,
        "nav_status": ev.nav_status,
        "position_accuracy": ev.position_accuracy,
        "message_type": ev.message_type,
        "ais_message_id": ev.ais_message_id,
        "is_class_b": ev.is_class_b,
        "raw_message_hash": ev.raw_message_hash,
        "is_valid_mmsi": ev.is_valid_mmsi,
        "sog_is_sentinel": ev.sog_is_sentinel,
        "cog_is_sentinel": ev.cog_is_sentinel,
        "heading_is_sentinel": ev.heading_is_sentinel,
        "is_in_bbox": ev.is_in_bbox,
        "dedup_key": ev.dedup_key,
        "is_duplicate": ev.is_duplicate,
    }
