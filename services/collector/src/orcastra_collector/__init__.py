"""ORCASTRA Collector — AIS data collection service."""

from .collector import Collector, main
from .config import CollectorConfig, DEFAULT_CONFIG

__all__ = ["Collector", "CollectorConfig", "DEFAULT_CONFIG", "main"]
