"""
ORCASTRA Collector — Configuration.

All paths, thresholds, and tuning constants in one place.
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

# ── Data root (outside repo, on D: on Windows, or environment variable) ──
_data_root_env = os.environ.get("ORCASTRA_DATA_ROOT")
if _data_root_env:
    DATA_ROOT = Path(_data_root_env)
elif sys.platform.startswith("win"):
    DATA_ROOT = Path(r"D:\ORCASTRA_DATA")
else:
    DATA_ROOT = Path("/data/orcastra_data")

# ── Houston Ship Channel bounding box ──
# Matches NOAA archive sample exactly.
HOUSTON_BBOX = [[29.0, -95.3], [29.9, -94.4]]


@dataclass(frozen=True)
class CollectorConfig:
    """Immutable collector configuration."""

    # AISStream
    aisstream_url: str = "wss://stream.aisstream.io/v0/stream"
    bounding_boxes: list = field(default_factory=lambda: [HOUSTON_BBOX])
    message_types: tuple = (
        "PositionReport",
        "StandardClassBPositionReport",
        "ExtendedClassBPositionReport",
        "ShipStaticData",
        "StaticDataReport",  # AIS type 24 (Class B CS static)
    )

    # Reconnect / backoff
    reconnect_max_retries: int = 50
    reconnect_base_delay: float = 2.0       # seconds
    reconnect_max_delay: float = 120.0      # seconds
    ws_recv_timeout: float = 30.0           # seconds before treating as drop
    ws_open_timeout: float = 15.0
    ws_close_timeout: float = 10.0

    # Writer
    parquet_part_max_rows: int = 10_000     # flush to new part file
    parquet_part_max_seconds: int = 300     # 5 min max before rotation
    parquet_compression: str = "zstd"

    # Ledger
    heartbeat_interval: int = 60            # seconds

    # Data quality sentinel values
    sog_sentinel: float = 102.3
    cog_sentinel: float = 360.0
    heading_sentinel: int = 511
    mmsi_valid_min: int = 200_000_000
    mmsi_valid_max: int = 799_999_999

    # Paths
    data_root: Path = DATA_ROOT
    recordings_dir: Path = DATA_ROOT / "recordings"
    vessel_cache_path: Path = DATA_ROOT / "vessel_cache.sqlite"
    ledger_path: Path = DATA_ROOT / "ledger.sqlite"
    quarantine_dir: Path = DATA_ROOT / "quarantine"
    latest_state_path: Path = DATA_ROOT / "latest_state.json"
    lock_file_path: Path = DATA_ROOT / "collector.lock"
    prediction_log_path: Path = DATA_ROOT / "prediction_log.sqlite"

    # Live state & staleness
    state_update_interval: float = 2.0      # seconds between atomic latest_state flushes
    stale_vessel_threshold_seconds: float = 300.0  # 5 min staleness threshold

    # Parser
    parser_version: str = "0.1.0"
    source_live: str = "aisstream_live"
    source_replay: str = "noaa_replay"


# Singleton default config
DEFAULT_CONFIG = CollectorConfig()
