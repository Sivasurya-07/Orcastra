"""
ORCASTRA Collector — AISStream Message Parser.

Converts raw AISStream JSON messages into canonical AisEvent objects.
The raw payload is preserved separately; this module only extracts
the canonical fields.

Field mapping (from 15-min live test raw samples):
  MetaData.MMSI          -> mmsi
  MetaData.latitude      -> latitude   (LOWERCASE in AISStream)
  MetaData.longitude     -> longitude  (LOWERCASE in AISStream)
  MetaData.ShipName      -> vessel_name
  MetaData.time_utc      -> observation_time
  Message.PositionReport.Sog  -> sog
  Message.PositionReport.Cog  -> cog
  Message.PositionReport.TrueHeading -> heading
  etc.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from .schema import AisEvent, categorize_vessel_type
from .config import CollectorConfig, DEFAULT_CONFIG

log = logging.getLogger(__name__)

# AISStream time format: "2026-10-08 13:34:49.836459668 +0000 UTC"
# Sometimes also: "2026-10-08T13:34:49Z"
TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S.%f",   # nanosecond gets truncated by Python to microsecond
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
]


def _parse_ais_time(time_str: Optional[str]) -> Optional[datetime]:
    """Parse AISStream time_utc string to datetime (UTC)."""
    if not time_str:
        return None
    # Strip timezone suffix like " +0000 UTC"
    clean = time_str.strip()
    for suffix in [" +0000 UTC", " UTC", "+00:00", "Z"]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
    # Truncate nanoseconds to microseconds (Python max precision)
    # e.g. "2026-10-08 13:34:49.836459668" -> "2026-10-08 13:34:49.836459"
    parts = clean.split(".")
    if len(parts) == 2 and len(parts[1]) > 6:
        clean = parts[0] + "." + parts[1][:6]

    for fmt in TIME_FORMATS:
        try:
            dt = datetime.strptime(clean, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    log.debug("Could not parse time: %s", time_str)
    return None


def _check_bbox(lat: Optional[float], lon: Optional[float],
                bboxes: list) -> bool:
    """Check if lat/lon falls within any configured bounding box."""
    if lat is None or lon is None:
        return False
    for bbox in bboxes:
        if (bbox[0][0] <= lat <= bbox[1][0] and
                bbox[0][1] <= lon <= bbox[1][1]):
            return True
    return False


class AisParser:
    """Parses raw AISStream JSON into canonical AisEvent objects."""

    def __init__(self, session_id: str, config: CollectorConfig = DEFAULT_CONFIG):
        self.session_id = session_id
        self.config = config
        self._stats = {
            "parsed": 0,
            "position": 0,
            "static": 0,
            "skipped_no_mmsi": 0,
            "skipped_subscription": 0,
            "quarantined": 0,
        }

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def parse(self, raw_json: str, receipt_time: datetime,
              vessel_type_lookup: dict[int, int] | None = None
              ) -> tuple[Optional[AisEvent], Optional[dict]]:
        """
        Parse a raw AISStream JSON message.

        Returns:
            (event, static_data) where:
            - event: AisEvent if successfully parsed, None if skipped/quarantined
            - static_data: dict with vessel info if ShipStaticData, else None
        
        Quarantined messages return (None, None) and increment quarantine counter.
        """
        try:
            msg = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except (json.JSONDecodeError, TypeError) as e:
            self._stats["quarantined"] += 1
            log.debug("JSON decode error: %s", e)
            return None, None

        msg_type = msg.get("MessageType", "")

        # Skip subscription confirmations
        if msg_type == "SubscriptionConfirmation":
            self._stats["skipped_subscription"] += 1
            return None, None

        # Extract metadata
        meta = msg.get("MetaData", {})
        mmsi = meta.get("MMSI")
        if mmsi is None or not isinstance(mmsi, (int, float)):
            self._stats["skipped_no_mmsi"] += 1
            return None, None
        mmsi = int(mmsi)

        # Lat/lon from MetaData (lowercase keys!)
        lat = meta.get("latitude")
        lon = meta.get("longitude")
        vessel_name = meta.get("ShipName")
        time_str = meta.get("time_utc")
        obs_time = _parse_ais_time(time_str)

        raw_hash = AisEvent.hash_raw(raw_json if isinstance(raw_json, str)
                                     else json.dumps(msg, separators=(",", ":")))

        msg_body = msg.get("Message", {})
        payload_ts = None
        for key in [msg_type, "PositionReport", "StandardClassBPositionReport",
                    "ExtendedClassBPositionReport", "ShipStaticData", "StaticDataReport"]:
            if key in msg_body:
                payload_ts = msg_body[key].get("Timestamp")
                break

        # Build dedup key and event_id (mmsi + reception second + msg_type [+ payload_ts])
        dedup_key = AisEvent.make_dedup_key(mmsi, obs_time, msg_type, payload_ts)
        event_id = AisEvent.make_event_id(dedup_key)

        # Quality checks
        is_valid_mmsi = (self.config.mmsi_valid_min <= mmsi <= self.config.mmsi_valid_max)
        is_in_bbox = _check_bbox(lat, lon, self.config.bounding_boxes)

        # Look up vessel type from cache
        vtype = None
        if vessel_type_lookup and mmsi in vessel_type_lookup:
            vtype = vessel_type_lookup[mmsi]

        # Common event fields
        event = AisEvent(
            event_id=event_id,
            session_id=self.session_id,
            source=self.config.source_live,
            parser_version=self.config.parser_version,
            mmsi=mmsi,
            vessel_name=vessel_name,
            vessel_type=vtype,
            vessel_category=categorize_vessel_type(vtype),
            aisstream_received_time=obs_time,
            ais_payload_timestamp=payload_ts,
            receipt_time=receipt_time,
            latitude=lat,
            longitude=lon,
            message_type=msg_type,
            raw_message_hash=raw_hash,
            is_valid_mmsi=is_valid_mmsi,
            is_in_bbox=is_in_bbox,
            dedup_key=dedup_key,
        )

        static_data = None

        if msg_type == "PositionReport":
            self._parse_position(event, msg_body.get("PositionReport", {}))
            event.is_class_b = False
            self._stats["position"] += 1

        elif msg_type == "StandardClassBPositionReport":
            self._parse_position(event, msg_body.get("StandardClassBPositionReport", {}))
            event.is_class_b = True
            self._stats["position"] += 1

        elif msg_type == "ExtendedClassBPositionReport":
            self._parse_position(event, msg_body.get("ExtendedClassBPositionReport", {}))
            event.is_class_b = True
            self._stats["position"] += 1

        elif msg_type == "ShipStaticData":
            ssd = msg_body.get("ShipStaticData", {})
            static_data = self._parse_static(event, ssd)
            self._stats["static"] += 1

        elif msg_type == "StaticDataReport":
            # AIS type 24 (Class B CS static data)
            sdr = msg_body.get("StaticDataReport", {})
            static_data = self._parse_static(event, sdr)
            event.is_class_b = True
            self._stats["static"] += 1

        else:
            # Unknown message type — quarantine
            self._stats["quarantined"] += 1
            return None, None

        self._stats["parsed"] += 1
        return event, static_data

    def _parse_position(self, event: AisEvent, pos: dict):
        """Extract position report fields."""
        event.ais_message_id = pos.get("MessageID")
        event.sog = pos.get("Sog")
        event.cog = pos.get("Cog")
        event.heading = pos.get("TrueHeading")
        event.rot = pos.get("RateOfTurn")
        event.nav_status = pos.get("NavigationalStatus")
        event.position_accuracy = pos.get("PositionAccuracy")

        # Sentinel checks
        if event.sog is not None:
            event.sog_is_sentinel = (abs(event.sog - self.config.sog_sentinel) < 0.01)
        if event.cog is not None:
            event.cog_is_sentinel = (abs(event.cog - self.config.cog_sentinel) < 0.01)
        if event.heading is not None:
            event.heading_is_sentinel = (event.heading == self.config.heading_sentinel)

    def _parse_static(self, event: AisEvent, ssd: dict) -> dict:
        """Extract ShipStaticData fields, return vessel info dict for cache."""
        event.ais_message_id = ssd.get("MessageID")
        event.imo = str(ssd.get("ImoNumber")) if ssd.get("ImoNumber") else None
        event.call_sign = ssd.get("CallSign")

        vtype = ssd.get("Type")
        if vtype is not None:
            event.vessel_type = vtype
            event.vessel_category = categorize_vessel_type(vtype)

        dim = ssd.get("Dimension", {})
        eta = ssd.get("Eta", {})

        return {
            "mmsi": event.mmsi,
            "imo": event.imo,
            "call_sign": event.call_sign,
            "vessel_name": event.vessel_name,
            "vessel_type": vtype,
            "vessel_category": categorize_vessel_type(vtype),
            "length": (dim.get("A", 0) or 0) + (dim.get("B", 0) or 0) if dim else None,
            "width": (dim.get("C", 0) or 0) + (dim.get("D", 0) or 0) if dim else None,
            "draught": ssd.get("MaximumStaticDraught"),
            "destination": ssd.get("Destination"),
            "eta_month": eta.get("Month"),
            "eta_day": eta.get("Day"),
            "eta_hour": eta.get("Hour"),
            "eta_minute": eta.get("Minute"),
        }
