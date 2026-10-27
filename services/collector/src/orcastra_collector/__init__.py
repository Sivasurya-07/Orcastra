"""ORCASTRA Collector — AIS data collection service."""

from .collector import Collector, main
from .config import CollectorConfig, DEFAULT_CONFIG
from .lock import CollectorLock, CollectorLockError, get_running_collector_info
from .state import CollectorStateWriter, read_latest_state
from .live_eta import (
    compute_live_eta,
    VesselLiveEta,
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

__all__ = [
    "Collector",
    "CollectorConfig",
    "DEFAULT_CONFIG",
    "CollectorLock",
    "CollectorLockError",
    "get_running_collector_info",
    "CollectorStateWriter",
    "read_latest_state",
    "compute_live_eta",
    "VesselLiveEta",
    "PredictionLogger",
    "haversine_km",
    "channel_distance_to_berth",
    "DISTANCE_BUCKETS",
    "METHOD_NAME",
    "P95_INTERVAL_NOTE",
    "CORRIDOR",
    "ENTRANCE_ZONE",
    "DESTINATION_ZONE",
    "main",
]
