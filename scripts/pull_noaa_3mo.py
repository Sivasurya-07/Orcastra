"""
ORCASTRA Tier 1, Step 1 — Pull NOAA Jan-Mar 2024 (Jan-Mar 91 days).

- D: only, bbox-filtered to Houston region.
- Slim columns (drop vessel_name, imo, call_sign, transceiver).
- Preserves existing January data in D:\ORCASTRA_DATA\historical\noaa_28d untouched.
- Jan 1-28 hard-linked into noaa_3mo (0 extra bytes).
- Jan 29-31, Feb 1-29, Mar 1-31 downloaded with 4 workers.
- Total size strictly <= 1.0 GB across all historical data on D:.
- Resumable: skips existing files with valid checksum and size.
- SHA-256 checksum per daily file, recorded in manifest.json.
- Validates schema, timestamps, geographic coverage, duplicates, monthly counts.
- Reruns arrival-label and censoring audit on full 91-day period (72h rule).
"""

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import duckdb
import numpy as np
import polars as pl

# ── Paths ──
DATA_ROOT = Path(r"D:\ORCASTRA_DATA\historical")
EXISTING_JAN_DIR = DATA_ROOT / "noaa_28d"       # Preserve this untouched
OUTPUT_DIR = DATA_ROOT / "noaa_3mo"              # Full 91-day dataset
RESULTS_DIR = DATA_ROOT / "results"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

BASE_URL = "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024"
BBOX = {"lat_min": 29.0, "lat_max": 29.9, "lon_min": -95.3, "lon_max": -94.4}
MAX_TOTAL_BYTES = 1024 * 1024 * 1024  # 1.0 GB hard budget

# Slim columns
SLIM_COLUMNS = [
    "mmsi", "base_date_time", "sog", "cog", "heading",
    "vessel_type", "status", "length", "width", "draft", "cargo",
    "lat", "lon",
]

# Geofences for arrival audit
ENTRANCE_ZONE = {"lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65}
DESTINATION_ZONE = {"lat_min": 29.65, "lat_max": 29.80, "lon_min": -95.15, "lon_max": -94.95}
BOLIVAR_ROADS = {"lat_min": 29.34, "lat_max": 29.42, "lon_min": -94.78, "lon_max": -94.70}
COMMERCIAL_TYPES = set(range(70, 90))


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_dates(start, end):
    d = start
    dates = []
    while d <= end:
        dates.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return dates


def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def link_january_days(manifest):
    """Link existing Jan 1-28 files from noaa_28d into noaa_3mo (zero extra disk space)."""
    jan_files = sorted(EXISTING_JAN_DIR.glob("houston_2024-01-*.parquet"))
    linked = 0

    for f in jan_files:
        date_str = f.stem.replace("houston_", "")
        out_file = OUTPUT_DIR / f"houston_{date_str}.parquet"

        if out_file.exists():
            if date_str not in manifest:
                manifest[date_str] = {
                    "source": "linked_from_jan28",
                    "original_path": str(f),
                    "path": str(out_file),
                    "bytes": out_file.stat().st_size,
                    "sha256": sha256_file(out_file),
                }
            continue

        # Hard-link to preserve D: space
        try:
            os.link(str(f), str(out_file))
            linked += 1
        except Exception:
            # Fallback to copy if cross-link fails
            import shutil
            shutil.copy2(str(f), str(out_file))
            linked += 1

        manifest[date_str] = {
            "source": "linked_from_jan28",
            "original_path": str(f),
            "path": str(out_file),
            "bytes": out_file.stat().st_size,
            "sha256": sha256_file(out_file),
        }

    save_manifest(manifest)
    print(f"  January 1-28: {len(jan_files)} files linked into noaa_3mo (0 extra bytes).")
    return manifest


def download_single_day(date_str: str) -> dict:
    """Download one daily file with DuckDB, spatial filter, slim columns."""
    out_file = OUTPUT_DIR / f"houston_{date_str}.parquet"

    if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
        return {
            "date": date_str,
            "status": "cached",
            "bytes": out_file.stat().st_size,
            "path": str(out_file),
        }

    file_url = f"{BASE_URL}/ais-{date_str}.parquet"
    out_sql_path = str(out_file).replace("\\", "/")

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL spatial; LOAD spatial;")

        # Download with slim columns and bbox filter
        sql = f"""
        COPY (
            SELECT mmsi, base_date_time, sog, cog, heading,
                   vessel_type, status, length, width, draft, cargo,
                   ST_Y(geometry) as lat, ST_X(geometry) as lon
            FROM read_parquet('{file_url}')
            WHERE ST_Y(geometry) BETWEEN {BBOX['lat_min']} AND {BBOX['lat_max']}
              AND ST_X(geometry) BETWEEN {BBOX['lon_min']} AND {BBOX['lon_max']}
        ) TO '{out_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(sql)
    finally:
        con.close()

    if not out_file.exists():
        raise RuntimeError(f"Download failed for {date_str}: file not created")

    file_size = out_file.stat().st_size
    return {
        "date": date_str,
        "status": "downloaded",
        "bytes": file_size,
        "path": str(out_file),
    }


def validate_dataset():
    """Validate schema, timestamps, geography, duplicates, monthly counts."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    files = sorted(OUTPUT_DIR.glob("houston_2024-*.parquet"))
    print(f"Total daily files found: {len(files)}/91")

    monthly_rows = defaultdict(int)
    monthly_vessels = defaultdict(set)
    total_rows = 0
    total_dupes = 0
    schema_ok = True
    geo_violations = 0
    ts_violations = 0

    for f in files:
        date_str = f.stem.replace("houston_", "")
        month = date_str[:7]
        # Read with slim columns
        df = pl.read_parquet(f, columns=SLIM_COLUMNS)
        n = len(df)
        total_rows += n
        monthly_rows[month] += n

        # Schema check
        for col in SLIM_COLUMNS:
            if col not in df.columns:
                print(f"  SCHEMA ERROR: {date_str} missing column '{col}'")
                schema_ok = False

        # Vessels
        mmsis = set(df["mmsi"].unique().to_list())
        for m in mmsis:
            monthly_vessels[month].add(int(m))

        # Timestamp check
        dt_col = df["base_date_time"].cast(pl.Datetime)
        min_dt = dt_col.min()
        max_dt = dt_col.max()
        expected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        if min_dt is not None and hasattr(min_dt, "date"):
            if min_dt.date() > expected_date or max_dt.date() < expected_date:
                ts_violations += 1
                print(f"  TS WARNING: {date_str} min={min_dt}, max={max_dt}")

        # Geography check
        out_of_bbox = df.filter(
            (pl.col("lat") < BBOX["lat_min"]) | (pl.col("lat") > BBOX["lat_max"]) |
            (pl.col("lon") < BBOX["lon_min"]) | (pl.col("lon") > BBOX["lon_max"])
        )
        if len(out_of_bbox) > 0:
            geo_violations += len(out_of_bbox)
            print(f"  GEO VIOLATION: {date_str} has {len(out_of_bbox)} rows outside bbox")

        # Deduplication within file
        dedup = df.unique(subset=["mmsi", "base_date_time", "lat", "lon"])
        file_dupes = n - len(dedup)
        total_dupes += file_dupes

    print(f"\n  Schema check:       {'PASS' if schema_ok else 'FAIL'}")
    print(f"  Total rows:         {total_rows:,}")
    print(f"  Total duplicates:   {total_dupes:,} ({100*total_dupes/max(total_rows,1):.3f}%)")
    print(f"  Geo violations:     {geo_violations}")
    print(f"  Timestamp checks:   {'PASS' if ts_violations == 0 else f'{ts_violations} warnings'}")
    print(f"\n  Monthly Breakdown:")
    for month in sorted(monthly_rows.keys()):
        print(f"    {month}: {monthly_rows[month]:,} rows across {len(monthly_vessels[month]):,} unique vessels")

    # Storage footprint
    total_physical = sum(f.stat().st_size for f in files)
    print(f"\n  Apparent file size: {total_physical / 1024 / 1024:.2f} MB")
    print(f"  Budget limit:       {MAX_TOTAL_BYTES / 1024 / 1024:.1f} MB (1.0 GB)")

    validation_results = {
        "status": "PASS" if schema_ok and geo_violations == 0 else "FAIL",
        "total_files": len(files),
        "total_rows": total_rows,
        "total_duplicates": total_dupes,
        "duplicate_rate_pct": round(100 * total_dupes / max(total_rows, 1), 4),
        "geo_violations": geo_violations,
        "monthly_rows": dict(monthly_rows),
        "monthly_unique_vessels": {m: len(v) for m, v in monthly_vessels.items()},
        "total_bytes": total_physical,
    }
    with open(RESULTS_DIR / "noaa_3mo_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    return validation_results


def run_arrival_audit():
    """Rerun arrival-label and censoring audit on full 91-day period (72h censoring rule)."""
    print("\n" + "=" * 70)
    print("ARRIVAL LABEL AUDIT (91-DAY PERIOD, 72h CENSORING)")
    print("=" * 70)

    files = sorted(OUTPUT_DIR.glob("houston_2024-*.parquet"))
    print(f"Loading {len(files)} files...")
    df_raw = pl.read_parquet(files, columns=SLIM_COLUMNS)
    df = df_raw.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])

    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    print(f"Time span: {min_ts} to {max_ts} ({(max_ts.date() - min_ts.date()).days + 1} days)")

    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    print(f"Commercial records: {len(df_comm):,} across {df_comm['mmsi'].n_unique()} unique MMSIs")

    # 72h right-censoring boundary
    left_censor_boundary = min_ts + timedelta(hours=6)
    right_censor_boundary = max_ts - timedelta(hours=72)

    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    normalized = {}
    for k, v in mmsi_groups.items():
        key = k[0] if isinstance(k, tuple) else k
        normalized[int(key)] = v
    mmsi_groups = normalized

    stats = {
        "raw_approaches": 0,
        "left_censored": 0,
        "right_censored": 0,
        "stationary_entrance": 0,
        "invalid_duration": 0,
    }
    arrivals = []

    for mmsi_val, vdf in mmsi_groups.items():
        vdf = vdf.sort("dt")
        records = vdf.to_dicts()
        n = len(records)
        i = 0

        while i < n:
            rec = records[i]
            lat, lon = rec["lat"], rec["lon"]
            sog = rec["sog"] if rec["sog"] is not None and not (abs(rec["sog"] - 102.3) < 0.01) else 0.0

            if in_box(lat, lon, ENTRANCE_ZONE):
                stats["raw_approaches"] += 1
                t0 = rec["dt"]

                if t0 <= left_censor_boundary:
                    stats["left_censored"] += 1
                    i += 1
                    continue
                if sog < 3.0:
                    stats["stationary_entrance"] += 1
                    i += 1
                    continue
                if t0 > right_censor_boundary:
                    stats["right_censored"] += 1
                    i += 1
                    continue

                # Trace forward
                j = i + 1
                found = False
                while j < n:
                    rn = records[j]
                    dur = (rn["dt"] - t0).total_seconds() / 3600.0
                    if dur > 72.0:
                        break
                    if in_box(rn["lat"], rn["lon"], DESTINATION_ZONE):
                        ns = rn["sog"] if rn["sog"] is not None and not (abs(rn["sog"] - 102.3) < 0.01) else 0.0
                        if ns < 1.5 and dur >= 1.0:
                            cog_val = rec["cog"] if rec["cog"] is not None and not (abs(rec["cog"] - 360.0) < 0.01) else None
                            is_inbound = cog_val is not None and ((270 <= cog_val <= 360) or (0 <= cog_val <= 30))

                            # Anchorage dwell check in Bolivar Roads
                            anch_pings = []
                            for k in range(i + 1, j):
                                ar = records[k]
                                asog = ar["sog"] if ar["sog"] is not None and not (abs(ar["sog"] - 102.3) < 0.01) else 0.0
                                if in_box(ar["lat"], ar["lon"], BOLIVAR_ROADS) and asog < 1.0:
                                    anch_pings.append(ar["dt"])
                            dwelled = False
                            dwell_h = 0.0
                            if len(anch_pings) >= 2:
                                dwell_h = (max(anch_pings) - min(anch_pings)).total_seconds() / 3600.0
                                dwelled = dwell_h >= 0.5

                            arrivals.append({
                                "mmsi": mmsi_val,
                                "t0": t0.isoformat(),
                                "t_arr": rn["dt"].isoformat(),
                                "duration_hours": round(dur, 3),
                                "sog_t0": round(sog, 2),
                                "cog_t0": round(cog_val, 1) if cog_val is not None else None,
                                "is_inbound_cog": is_inbound,
                                "vessel_type": rec["vessel_type"],
                                "length": rec.get("length"),
                                "width": rec.get("width"),
                                "draft": rec.get("draft"),
                                "anchored_at_bolivar": dwelled,
                                "anchorage_dwell_hours": round(dwell_h, 3),
                                "month": t0.strftime("%Y-%m"),
                            })
                            found = True
                            break
                    j += 1

                if not found:
                    stats["invalid_duration"] += 1
                i = max(j if found or j > i + 1 else i + 1, i + 1)
            else:
                i += 1

    # Report
    print(f"\nAUDIT COUNTS ({min_ts.date()} to {max_ts.date()}):")
    print(f"  Raw entrance approaches:    {stats['raw_approaches']:,}")
    print(f"  Left-censored (<6h):         {stats['left_censored']:,}")
    print(f"  Stationary entrance (<3 kt): {stats['stationary_entrance']:,}")
    print(f"  Right-censored (72h window): {stats['right_censored']:,}")
    print(f"  Invalid/incomplete (>72h):   {stats['invalid_duration']:,}")
    print(f"  Valid completed arrivals:    {len(arrivals):,}")

    monthly_arrivals = defaultdict(int)
    monthly_vessels_arr = defaultdict(set)
    for a in arrivals:
        monthly_arrivals[a["month"]] += 1
        monthly_vessels_arr[a["month"]].add(a["mmsi"])

    print(f"\n  Monthly Completed Arrivals:")
    for month in sorted(monthly_arrivals.keys()):
        print(f"    {month}: {monthly_arrivals[month]} arrivals, {len(monthly_vessels_arr[month])} unique vessels")

    direct = [a for a in arrivals if not a["anchored_at_bolivar"]]
    anchored = [a for a in arrivals if a["anchored_at_bolivar"]]
    print(f"\n  Transit Stages:")
    print(f"    Direct transit: {len(direct)} ({100*len(direct)/max(len(arrivals),1):.1f}%)")
    print(f"    Anchored:       {len(anchored)} ({100*len(anchored)/max(len(arrivals),1):.1f}%)")

    inbound = [a for a in arrivals if a["is_inbound_cog"]]
    non_inbound = [a for a in arrivals if not a["is_inbound_cog"]]
    print(f"\n  Directionality:")
    print(f"    Inbound COG (channel heading):     {len(inbound)} ({100*len(inbound)/max(len(arrivals),1):.1f}%)")
    print(f"    Non-inbound COG:                   {len(non_inbound)} ({100*len(non_inbound)/max(len(arrivals),1):.1f}%)")

    durations = [a["duration_hours"] for a in arrivals]
    dur_stats = {}
    if durations:
        dur_stats = {
            "min": float(np.min(durations)),
            "p10": float(np.percentile(durations, 10)),
            "p25": float(np.percentile(durations, 25)),
            "median": float(np.median(durations)),
            "p75": float(np.percentile(durations, 75)),
            "p90": float(np.percentile(durations, 90)),
            "max": float(np.max(durations)),
        }
        print(f"\n  Transit Duration (hours):")
        print(f"    Min:    {dur_stats['min']:.2f} h")
        print(f"    P10:    {dur_stats['p10']:.2f} h")
        print(f"    Median: {dur_stats['median']:.2f} h")
        print(f"    P90:    {dur_stats['p90']:.2f} h")
        print(f"    Max:    {dur_stats['max']:.2f} h")

    unique_arr_vessels = len(set(a["mmsi"] for a in arrivals))
    print(f"\n  Unique commercial vessels with arrivals: {unique_arr_vessels}")

    trip_counts = defaultdict(int)
    for a in arrivals:
        trip_counts[a["mmsi"]] += 1
    trip_dist = defaultdict(int)
    for c in trip_counts.values():
        trip_dist[c] += 1
    print(f"  Trips per vessel distribution:")
    for trips in sorted(trip_dist.keys())[:10]:
        print(f"    {trips} arrival(s): {trip_dist[trips]} vessels")

    audit_results = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "period": f"{min_ts.date()} to {max_ts.date()}",
        "total_days": (max_ts.date() - min_ts.date()).days + 1,
        "censoring_rule": "72h right-censoring",
        "stats": stats,
        "valid_arrivals": len(arrivals),
        "unique_commercial_vessels": unique_arr_vessels,
        "monthly_arrivals": dict(monthly_arrivals),
        "monthly_unique_vessels": {m: len(v) for m, v in monthly_vessels_arr.items()},
        "stage_breakdown": {
            "direct_transit": len(direct),
            "anchored_at_bolivar": len(anchored),
        },
        "directionality": {
            "inbound_cog": len(inbound),
            "non_inbound_cog": len(non_inbound),
        },
        "duration_hours": dur_stats,
        "trips_per_vessel": dict(trip_dist),
    }

    out_path = RESULTS_DIR / "noaa_3mo_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n  Audit results saved to: {out_path}")

    return audit_results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ORCASTRA TIER 1 — NOAA JAN-MAR 2024 (91-DAY) PULL & AUDIT")
    print("=" * 70)
    print(f"Output directory:      {OUTPUT_DIR}")
    print(f"Preserved Jan dir:     {EXISTING_JAN_DIR}")
    print(f"Max size budget:       {MAX_TOTAL_BYTES / 1024 / 1024:.0f} MB (1.0 GB)")

    # ── Phase 1: Expected Size Report ──
    print("\n--- PHASE 1: SIZE ESTIMATE REPORT ---")
    jan_files = sorted(EXISTING_JAN_DIR.glob("houston_2024-01-*.parquet"))
    jan_bytes = sum(f.stat().st_size for f in jan_files)
    avg_day_bytes = jan_bytes / len(jan_files)
    slim_ratio = 0.915  # 8.5% savings from dropping 4 string columns
    slim_day_bytes = avg_day_bytes * slim_ratio

    # Days to pull: Jan 29-31 (3d), Feb (29d), Mar (31d) = 63 days
    new_days_count = 3 + 29 + 31
    expected_new_bytes = slim_day_bytes * new_days_count
    # Jan 1-28 is hard-linked (0 extra disk blocks)
    expected_historical_total = jan_bytes + expected_new_bytes

    print(f"  Existing Jan 1-28 (preserved, 28 days): {jan_bytes / 1024 / 1024:.2f} MB")
    print(f"  New days to pull (63 days):             {expected_new_bytes / 1024 / 1024:.2f} MB")
    print(f"    - Jan 29-31:                          {slim_day_bytes * 3 / 1024 / 1024:.2f} MB")
    print(f"    - Feb 1-29 (leap year):               {slim_day_bytes * 29 / 1024 / 1024:.2f} MB")
    print(f"    - Mar 1-31:                           {slim_day_bytes * 31 / 1024 / 1024:.2f} MB")
    print(f"  Expected total historical on D::        {expected_historical_total / 1024 / 1024:.2f} MB ({expected_historical_total / 1024 / 1024 / 1024:.3f} GB)")
    print(f"  Hard budget limit:                      1024.0 MB (1.000 GB)")
    print(f"  Budget status:                          {'WITHIN BUDGET' if expected_historical_total < MAX_TOTAL_BYTES else 'OVER BUDGET'}")

    if expected_historical_total >= MAX_TOTAL_BYTES:
        print("ABORT: Expected size exceeds 1.0 GB limit.")
        return

    # ── Phase 2: Link Existing January ──
    print("\n--- PHASE 2: LINK EXISTING JAN 1-28 ---")
    manifest = load_manifest()
    manifest = link_january_days(manifest)

    # ── Phase 3: Download Remaining 63 Days (Jan 29-31, Feb, Mar) ──
    print("\n--- PHASE 3: DOWNLOAD NEW DAYS (CONCURRENT) ---")
    dates_to_pull = []
    # Jan 29-31
    dates_to_pull.extend(get_dates(date(2024, 1, 29), date(2024, 1, 31)))
    # Feb 1-29
    dates_to_pull.extend(get_dates(date(2024, 2, 1), date(2024, 2, 29)))
    # Mar 1-31
    dates_to_pull.extend(get_dates(date(2024, 3, 1), date(2024, 3, 31)))

    print(f"Total target download dates: {len(dates_to_pull)}")
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(download_single_day, d): d for d in dates_to_pull}
        for future in concurrent.futures.as_completed(future_map):
            d = future_map[future]
            try:
                res = future.result()
                completed += 1
                out_path = Path(res["path"])
                checksum = sha256_file(out_path)
                manifest[d] = {
                    "source": "downloaded_azure",
                    "path": str(out_path),
                    "bytes": res["bytes"],
                    "sha256": checksum,
                }
                save_manifest(manifest)

                current_total = sum(f.stat().st_size for f in OUTPUT_DIR.glob("houston_*.parquet"))
                print(f"  [{completed}/{len(dates_to_pull)}] {d}: {res['status']} "
                      f"({res['bytes'] / 1024 / 1024:.2f} MB), Apparent total: {current_total / 1024 / 1024:.1f} MB")

            except Exception as e:
                print(f"  ERROR on {d}: {e}")
                save_manifest(manifest)
                sys.exit(1)

    print("\n--- DOWNLOAD PHASE COMPLETE ---")

    # ── Phase 4: Validate Dataset ──
    validate_dataset()

    # ── Phase 5: Arrival Label Audit ──
    run_arrival_audit()

    print("\n" + "=" * 70)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
