"""
ORCASTRA Collector — Basic Parquet Replay.

Reads finalized Parquet parts from a session directory and replays
them as a stream of AisEvent dicts. Used for testing determinism
and for offline analysis.
"""

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)


def replay_session(session_dir: Path,
                   include_duplicates: bool = False) -> pl.DataFrame:
    """
    Replay a session from its Parquet parts.
    
    Returns a Polars DataFrame with all events, sorted by observation_time.
    Duplicates are excluded by default (is_duplicate == False).
    """
    events_dir = session_dir / "events"
    if not events_dir.exists():
        raise FileNotFoundError(f"No events directory: {events_dir}")

    parts = sorted(events_dir.glob("part_*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No Parquet parts in {events_dir}")

    log.info("Replaying %d parts from %s", len(parts), session_dir)

    dfs = []
    for part in parts:
        df = pl.read_parquet(part)
        log.info("  %s: %d rows", part.name, len(df))
        dfs.append(df)

    combined = pl.concat(dfs, how="diagonal")

    if not include_duplicates:
        before = len(combined)
        combined = combined.filter(pl.col("is_duplicate") == False)
        after = len(combined)
        if before != after:
            log.info("  Filtered %d duplicates", before - after)

    time_col = "aisstream_received_time" if "aisstream_received_time" in combined.columns else "observation_time"
    combined = combined.sort(time_col)
    log.info("Total: %d events from %d parts", len(combined), len(parts))
    return combined


def replay_to_events(session_dir: Path,
                     include_duplicates: bool = False) -> list[dict]:
    """Replay as a list of dicts (for testing)."""
    df = replay_session(session_dir, include_duplicates)
    return df.to_dicts()
