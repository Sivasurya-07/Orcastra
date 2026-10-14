"""
ORCASTRA M0 — Fixes 1,3,4 and dedup recomputation.

Fix 1: Recompute dedup from raw JSONL with redesigned key.
Fix 3: NOAA per-10-min windows at matching UTC hours.
Fix 4: Arrival detector on 75-min live recording.
"""
import json, sys, statistics
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

import polars as pl
import duckdb

DATA_ROOT = Path(r"D:\ORCASTRA_DATA")
SESSION_DIR = DATA_ROOT / "recordings" / "session_20261008_160107"
RAW_DIR = SESSION_DIR / "raw"
NOAA_DIR = DATA_ROOT / "historical"

# ======================================================================
# FIX 1: Recompute dedup from raw JSONL
# ======================================================================
print("=" * 70)
print("FIX 1: Dedup Recomputation from Raw JSONL")
print("=" * 70)

# Parse all raw JSONL, extract the AIS payload Timestamp field (0-59 seconds)
# and the MetaData.time_utc (AISStream reception time)
def parse_ais_time(time_str):
    if not time_str:
        return None
    clean = time_str.strip()
    for suffix in [" +0000 UTC", " UTC", "+00:00", "Z"]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
    parts = clean.split(".")
    if len(parts) == 2 and len(parts[1]) > 6:
        clean = parts[0] + "." + parts[1][:6]
    for fmt in ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

raw_msgs = []
for jsonl in sorted(RAW_DIR.glob("part_*.jsonl")):
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            try:
                msg = json.loads(line)
                meta = msg.get("MetaData", {})
                mmsi = meta.get("MMSI")
                if mmsi is None:
                    continue
                mmsi = int(mmsi)
                
                # AISStream reception time
                recv_time = parse_ais_time(meta.get("time_utc"))
                
                # AIS payload timestamp (0-59 seconds)
                msg_type = msg.get("MessageType", "")
                msg_body = msg.get("Message", {})
                ais_ts = None
                for key in [msg_type, "PositionReport", "StandardClassBPositionReport",
                            "ExtendedClassBPositionReport", "ShipStaticData", "StaticDataReport"]:
                    if key in msg_body:
                        ais_ts = msg_body[key].get("Timestamp")
                        break
                
                raw_msgs.append({
                    "mmsi": mmsi,
                    "aisstream_received_time": recv_time,
                    "ais_payload_timestamp_s": ais_ts,  # 0-59 or None
                    "message_type": msg_type,
                    "raw_hash": msg.get("_hash", ""),
                    "lat": meta.get("latitude"),
                    "lon": meta.get("longitude"),
                    "sog": None,
                    "ship_name": meta.get("ShipName"),
                })
                
                # Extract SOG for arrival detection
                for key in ["PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport"]:
                    if key in msg_body:
                        raw_msgs[-1]["sog"] = msg_body[key].get("Sog")
                        break
            except Exception as e:
                pass

print(f"Total raw messages parsed: {len(raw_msgs)}")

# Analyze AIS payload timestamp field
has_ts = sum(1 for m in raw_msgs if m["ais_payload_timestamp_s"] is not None)
print(f"Messages with AIS payload Timestamp field: {has_ts} ({has_ts/len(raw_msgs)*100:.1f}%)")
if has_ts > 0:
    ts_vals = [m["ais_payload_timestamp_s"] for m in raw_msgs if m["ais_payload_timestamp_s"] is not None]
    print(f"  Timestamp range: {min(ts_vals)} to {max(ts_vals)} (expected 0-59)")
    print(f"  Note: This is only the SECONDS portion, not an absolute time")

# OLD dedup key: mmsi + aisstream_received_time truncated to second
# Problem: rationale was "AIS transponders transmit at most once per second"
# but the timestamp is AISStream reception, not AIS transmission
# Two AIS transmissions received in the same second would be falsely deduped

# NEW dedup key design:
# mmsi + aisstream_received_time truncated to second + raw_message_hash
# Actually, a simpler approach: mmsi + aisstream_received_time + message_type
# since the same MMSI can send both position and static in the same second
# But the real question: are there actual duplicates in AISStream's output?

# Let's check with the OLD key first
old_keys = Counter()
for m in raw_msgs:
    if m["aisstream_received_time"]:
        ts = m["aisstream_received_time"].strftime("%Y%m%dT%H%M%S")
        old_key = f"{m['mmsi']}_{ts}"
    else:
        old_key = f"{m['mmsi']}_unknown"
    old_keys[old_key] += 1

old_dups = sum(v - 1 for v in old_keys.values() if v > 1)
print(f"\nOLD dedup key (mmsi + reception_time_to_second):")
print(f"  Unique keys: {len(old_keys)}")
print(f"  Duplicates: {old_dups}")

# NEW dedup key: mmsi + aisstream_received_time + message_type
# Rationale: AISStream may deliver position and static messages for the
# same MMSI in the same second. These are NOT duplicates.
# True duplicates are: same MMSI, same reception second, same message type.
new_keys = Counter()
for m in raw_msgs:
    if m["aisstream_received_time"]:
        ts = m["aisstream_received_time"].strftime("%Y%m%dT%H%M%S")
        new_key = f"{m['mmsi']}_{ts}_{m['message_type']}"
    else:
        new_key = f"{m['mmsi']}_unknown_{m['message_type']}"
    new_keys[new_key] += 1

new_dups = sum(v - 1 for v in new_keys.values() if v > 1)
print(f"\nNEW dedup key (mmsi + reception_time_to_second + message_type):")
print(f"  Unique keys: {len(new_keys)}")
print(f"  Duplicates: {new_dups}")

# Show what the duplicates look like
dup_examples = [(k, v) for k, v in new_keys.items() if v > 1]
if dup_examples:
    print(f"\n  Duplicate key examples:")
    for k, v in dup_examples[:5]:
        print(f"    {k}: {v} occurrences")

# Report the change
print(f"\nDEDUP REDESIGN SUMMARY:")
print(f"  OLD key: mmsi + aisstream_received_time (to second)")
print(f"  OLD rationale: 'AIS transponders transmit at most once per second' -- INVALID")
print(f"    (aisstream_received_time is AISStream reception, not vessel transmission)")
print(f"  NEW key: mmsi + aisstream_received_time (to second) + message_type")
print(f"  NEW rationale: Two messages from the same MMSI with the same message type,")
print(f"    received by AISStream in the same second, are relay duplicates.")
print(f"    Different message types (e.g. PositionReport + ShipStaticData) in the same")
print(f"    second are NOT duplicates. The reception-time interpretation is an")
print(f"    evidence-backed assumption (official docs do not define time_utc).")
print(f"  Old duplicates: {old_dups}")
print(f"  New duplicates: {new_dups}")
print(f"  Difference: {old_dups - new_dups} false positives removed")

# ======================================================================
# FIX 3: NOAA per-10-min windows at matching UTC hours
# ======================================================================
print("\n" + "=" * 70)
print("FIX 3: NOAA per-10-min Windows at Matching UTC Hours")
print("=" * 70)

noaa_parquet = NOAA_DIR / "noaa" / "houston" / "houston_sample.parquet"
if noaa_parquet.exists():
    ndf = pl.read_parquet(noaa_parquet)
    print(f"NOAA sample loaded: {len(ndf)} rows")
    
    # Get the timestamp column name
    ts_col = None
    for c in ndf.columns:
        if "time" in c.lower() or "timestamp" in c.lower() or "basedatetime" in c.lower():
            ts_col = c
            break
    if ts_col is None:
        # Try geometry-based
        print(f"  Columns: {ndf.columns[:10]}")
        ts_col = ndf.columns[0] if "BaseDateTime" not in ndf.columns else "BaseDateTime"
    
    print(f"  Timestamp column: {ts_col}")
    print(f"  Time range: {ndf[ts_col].min()} to {ndf[ts_col].max()}")
    
    # Filter to UTC hours 13-14 and 16-17 (matching the live tests)
    # NOAA is Mon-Wed Jan 15-17 2024; live was Thu Oct 8 2026
    # Same weekday type: both weekdays
    
    ndf_ts = ndf.with_columns(pl.col(ts_col).cast(pl.Datetime).alias("ts_parsed"))
    ndf_ts = ndf_ts.with_columns(pl.col("ts_parsed").dt.hour().alias("utc_hour"))
    
    for hour_range_name, hours in [("UTC 13-14", [13, 14]), ("UTC 16-17", [16, 17])]:
        subset = ndf_ts.filter(pl.col("utc_hour").is_in(hours))
        if len(subset) == 0:
            print(f"\n  {hour_range_name}: No data")
            continue
        
        # Per-10-min windows
        subset_10m = subset.with_columns(
            pl.col("ts_parsed").dt.truncate("10m").alias("window_10m")
        )
        per_10m = subset_10m.group_by("window_10m").agg(
            pl.len().alias("events"),
            pl.col("mmsi").n_unique().alias("unique_vessels"),
        ).sort("window_10m")
        
        events_list = per_10m["events"].to_list()
        vessels_list = per_10m["unique_vessels"].to_list()
        
        print(f"\n  {hour_range_name} (NOAA archive, {len(per_10m)} windows):")
        print(f"    Events/10min:  mean={statistics.mean(events_list):.0f}, median={statistics.median(events_list):.0f}")
        print(f"    Vessels/10min: mean={statistics.mean(vessels_list):.0f}, median={statistics.median(vessels_list):.0f}")
        
        # Show individual windows (sample)
        print(f"    Sample windows:")
        for row in per_10m.head(6).iter_rows(named=True):
            w = str(row['window_10m'])[:19]
            print(f"      {w} UTC: {row['events']:5d} events, {row['unique_vessels']:3d} vessels")
    
    # Live comparison
    print(f"\n  LIVE per-10-min windows (from 75-min session, UTC 16-17):")
    print(f"    Events/10min: mean=504, median=540")
    print(f"    Vessels/10min: mean=222, median=223")
    print(f"    (From prior analysis)")
    
    print(f"\n  PIPELINE COMPARISON (not proof of missing traffic):")
    print(f"    Source difference: NOAA NAIS coastal stations vs AISStream relay")
    print(f"    Temporal difference: Jan 2024 vs Oct 2026")
    print(f"    Same weekday type: both weekdays")
    print(f"    Same bbox: [29.0, 29.9] x [-95.3, -94.4]")
else:
    print(f"NOAA sample not found at {noaa_parquet}")
    # Try to find it
    for p in NOAA_DIR.rglob("*.parquet"):
        print(f"  Found: {p}")

# ======================================================================
# FIX 4: Arrival detector on 75-min live recording
# ======================================================================
print("\n" + "=" * 70)
print("FIX 4: Arrival Detection on 75-min Live Recording")
print("=" * 70)

# Approach definition from the archive analysis:
# Zone 1: Houston Ship Channel entrance (29.30-29.45 N, 94.85-94.65 W)
# Zone 2: Galveston Bay inner (29.55-29.75 N, 95.10-94.85 W)
# Approach: vessel enters zone AND has >= 1 report with SOG < 3.0 kn
# One approach per unique MMSI per zone

ZONES = {
    "HSC_entrance": {"lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65},
    "Galveston_inner": {"lat_min": 29.55, "lat_max": 29.75, "lon_min": -95.10, "lon_max": -94.85},
}

# Filter position messages with valid lat/lon/SOG
pos_msgs = [m for m in raw_msgs if m["message_type"] in 
            ("PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport")
            and m["lat"] is not None and m["lon"] is not None]

print(f"Position messages with lat/lon: {len(pos_msgs)}")

# Check which messages have SOG
has_sog = sum(1 for m in pos_msgs if m["sog"] is not None)
print(f"Position messages with SOG: {has_sog} ({has_sog/len(pos_msgs)*100:.1f}%)")

# Per-vessel observation count
vessel_obs = Counter()
for m in pos_msgs:
    vessel_obs[m["mmsi"]] += 1

print(f"Unique vessels with position data: {len(vessel_obs)}")
obs_counts = sorted(vessel_obs.values(), reverse=True)
print(f"Per-vessel observation count:")
print(f"  Mean: {statistics.mean(obs_counts):.1f}")
print(f"  Median: {statistics.median(obs_counts):.0f}")
print(f"  Min: {min(obs_counts)}, Max: {max(obs_counts)}")
print(f"  Vessels with >= 5 observations: {sum(1 for c in obs_counts if c >= 5)}")
print(f"  Vessels with >= 10 observations: {sum(1 for c in obs_counts if c >= 10)}")
print(f"  Vessels with >= 20 observations: {sum(1 for c in obs_counts if c >= 20)}")

# Detect approaches
for zone_name, zone in ZONES.items():
    in_zone = [m for m in pos_msgs if 
               zone["lat_min"] <= m["lat"] <= zone["lat_max"] and
               zone["lon_min"] <= m["lon"] <= zone["lon_max"]]
    
    in_zone_mmsis = set(m["mmsi"] for m in in_zone)
    slow_in_zone = [m for m in in_zone if m["sog"] is not None and m["sog"] < 3.0
                    and not (abs(m["sog"] - 102.3) < 0.01)]  # exclude SOG sentinel
    slow_mmsis = set(m["mmsi"] for m in slow_in_zone)
    
    print(f"\n{zone_name}:")
    print(f"  Messages in zone: {len(in_zone)}")
    print(f"  Unique vessels in zone: {len(in_zone_mmsis)}")
    print(f"  Messages with SOG < 3.0 in zone: {len(slow_in_zone)}")
    print(f"  Approaches (unique MMSI with SOG < 3.0 in zone): {len(slow_mmsis)}")

# Overall
all_approach_mmsis = set()
for zone_name, zone in ZONES.items():
    in_zone = [m for m in pos_msgs if 
               zone["lat_min"] <= m["lat"] <= zone["lat_max"] and
               zone["lon_min"] <= m["lon"] <= zone["lon_max"]]
    slow = [m for m in in_zone if m["sog"] is not None and m["sog"] < 3.0
            and not (abs(m["sog"] - 102.3) < 0.01)]
    all_approach_mmsis.update(m["mmsi"] for m in slow)

print(f"\nTotal approaches (both zones): {len(all_approach_mmsis)}")

# Effect of ~3-minute reporting interval
print(f"\nEffect of ~3-min reporting interval:")
print(f"  Median inter-message interval: 178.9s (~3 min)")
print(f"  A vessel transiting through a geofence zone at 10 kn (~5.1 m/s)")
print(f"  HSC entrance zone width: ~0.15 deg lat = ~16.7 km")
print(f"  Transit time at 10 kn: ~54 min = ~18 observations at 3-min intervals")
print(f"  HSC entrance zone length: ~0.20 deg lon = ~18.5 km")
print(f"  A vessel inside the zone for 54 min would be observed ~18 times.")
print(f"  A fast vessel (20 kn) through narrow axis: ~28 min = ~9 observations.")
print(f"  EFFECT: 3-min intervals are ADEQUATE for detecting approaches in these zones.")
print(f"  SHORT TRANSITS through a small zone could be missed, but these zones are large enough.")

# Commercial arrivals (using vessel type from cache)
import sqlite3
cache_conn = sqlite3.connect(str(DATA_ROOT / "vessel_cache.sqlite"))
cache_types = dict(cache_conn.execute(
    "SELECT mmsi, vessel_type FROM vessel_cache WHERE vessel_type IS NOT NULL"
).fetchall())
cache_conn.close()

commercial_types = set()
for mmsi_v in all_approach_mmsis:
    vt = cache_types.get(mmsi_v)
    if vt is not None and (70 <= vt <= 79 or 80 <= vt <= 89 or 60 <= vt <= 69):
        commercial_types.add(mmsi_v)

unknown_type = sum(1 for m in all_approach_mmsis if cache_types.get(m) is None)
print(f"\nCommercial arrivals (from typed vessels only):")
print(f"  Total approaches: {len(all_approach_mmsis)}")
print(f"  With known type: {len(all_approach_mmsis) - unknown_type}")
print(f"  With unknown type: {unknown_type}")
print(f"  Commercial (cargo+tanker+passenger, known type): {len(commercial_types)}")
print(f"  This is MEASUREMENT, not extrapolation.")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
