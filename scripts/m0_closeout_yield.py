"""
M0 Close-out: Realistic short-horizon yield + NOAA type cache seeding test.
"""
import json
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl

# ── Paths ──
NOAA_28D_DIR = Path(r"D:\ORCASTRA_DATA\historical\noaa_28d")
LIVE_DIR = Path(r"D:\ORCASTRA_DATA\recordings\session_20261008_160107")
RESULTS_DIR = Path(r"D:\ORCASTRA_DATA\historical\results")

COMMERCIAL_TYPES = set(range(70, 90))

# ── Geofences ──
ENTRANCE_ZONE = {"lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65}
DESTINATION_ZONE = {"lat_min": 29.65, "lat_max": 29.80, "lon_min": -95.15, "lon_max": -94.95}
BOLIVAR_ROADS = {"lat_min": 29.34, "lat_max": 29.42, "lon_min": -94.78, "lon_max": -94.70}


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def main():
    print("=" * 70)
    print("M0 CLOSE-OUT: NOAA TYPE CACHE + REALISTIC YIELD SIMULATION")
    print("=" * 70)

    # ────────────────────────────────────────────────────────────────
    # Part 1: Build NOAA MMSI → vessel_type lookup
    # ────────────────────────────────────────────────────────────────
    print("\n--- Part 1: NOAA MMSI → vessel_type table ---")
    noaa_files = sorted(NOAA_28D_DIR.glob("houston_*.parquet"))
    print(f"Loading {len(noaa_files)} NOAA files...")

    # Only read mmsi and vessel_type columns for speed
    df_noaa = pl.read_parquet(noaa_files, columns=["mmsi", "vessel_type"])
    noaa_type_map = {}
    for row in df_noaa.select(["mmsi", "vessel_type"]).unique().iter_rows():
        mmsi_val, vt = row
        if vt is not None and vt > 0:
            noaa_type_map[int(mmsi_val)] = int(vt)

    print(f"NOAA MMSI->type table: {len(noaa_type_map)} unique MMSIs with a known type")
    noaa_commercial = {m for m, t in noaa_type_map.items() if t in COMMERCIAL_TYPES}
    print(f"  Of those, commercial (70-89): {len(noaa_commercial)}")

    # ────────────────────────────────────────────────────────────────
    # Part 2: Load live recording MMSIs
    # ────────────────────────────────────────────────────────────────
    print("\n--- Part 2: Live Recording MMSI Coverage ---")
    live_parquet_files = sorted((LIVE_DIR / "events").glob("*.parquet"))
    
    if live_parquet_files:
        df_live = pl.read_parquet(live_parquet_files)
        live_mmsis = set(df_live["mmsi"].unique().to_list())
        live_mmsis_int = {int(m) for m in live_mmsis}
        print(f"Live recording: {len(live_mmsis_int)} unique MMSIs")
        
        # Check how many live MMSIs are in NOAA type table
        found_in_noaa = live_mmsis_int & set(noaa_type_map.keys())
        found_commercial = live_mmsis_int & noaa_commercial
        not_found = live_mmsis_int - set(noaa_type_map.keys())
        
        print(f"  Found in NOAA type table:     {len(found_in_noaa)} ({100*len(found_in_noaa)/len(live_mmsis_int):.1f}%)")
        print(f"  Found as commercial:          {len(found_commercial)} ({100*len(found_commercial)/len(live_mmsis_int):.1f}%)")
        print(f"  NOT found (no NOAA history):  {len(not_found)} ({100*len(not_found)/len(live_mmsis_int):.1f}%)")
        
        type_cache_fraction = len(found_in_noaa) / len(live_mmsis_int)
        commercial_fraction = len(found_commercial) / len(live_mmsis_int)
    else:
        print("No live Parquet files found.")
        type_cache_fraction = 0.5
        commercial_fraction = 0.15
        live_mmsis_int = set()
        found_in_noaa = set()
        found_commercial = set()

    # ────────────────────────────────────────────────────────────────
    # Part 3: Realistic Short-Horizon Yield Simulation
    # ────────────────────────────────────────────────────────────────
    print("\n--- Part 3: Realistic Short-Horizon Yield Simulation ---")
    print("Assumptions:")
    print("  - Sessions: 4 of 7 days (not every day)")
    print(f"  - Live detection fraction: ~0.34 of NOAA vessel density")
    print(f"  - Type cache hit rate (from NOAA seeding): {type_cache_fraction*100:.1f}%")
    print(f"  - Commercial fraction among live MMSIs (NOAA-seeded): {commercial_fraction*100:.1f}%")

    # Load arrivals
    print("\nRecomputing arrivals from NOAA 28-day data (72h right-censoring)...")
    df_full = pl.read_parquet(noaa_files)
    df = df_full.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])
    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    
    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    
    right_censor_boundary = max_ts - timedelta(hours=72)
    left_censor_boundary = min_ts + timedelta(hours=6)
    
    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    normalized = {}
    for k, v in mmsi_groups.items():
        key = k[0] if isinstance(k, tuple) else k
        normalized[int(key)] = v
    mmsi_groups = normalized

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
                t0 = rec["dt"]
                if t0 <= left_censor_boundary or sog < 3.0 or t0 > right_censor_boundary:
                    i += 1
                    continue
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
                            arrivals.append({"mmsi": mmsi_val, "t0": t0, "t_arr": rn["dt"],
                                             "duration_hours": dur})
                            found = True
                            break
                    j += 1
                i = max(j, i + 1)
            else:
                i += 1

    print(f"Arrivals (72h censor): {len(arrivals)}")

    # Short-horizon simulation at NOAA level
    first_day = min_ts.date()
    last_day = max_ts.date()

    configs = []
    for horizon_h in [1.0, 2.0]:
        for sess_h in [5, 7]:
            for start_h in [14, 18]:
                total_days = 0
                scorable = 0
                current_day = first_day
                while current_day <= last_day:
                    ss = datetime(current_day.year, current_day.month, current_day.day,
                                  start_h, 0, 0)
                    se = ss + timedelta(hours=sess_h)
                    if se > max_ts:
                        break
                    total_days += 1
                    for a in arrivals:
                        snap = a["t_arr"] - timedelta(hours=horizon_h)
                        if ss <= snap and a["t_arr"] <= se:
                            scorable += 1
                    current_day += timedelta(days=1)
                
                noaa_rate = scorable / total_days if total_days > 0 else 0
                configs.append({
                    "horizon_h": horizon_h, "sess_h": sess_h, "start_h": start_h,
                    "noaa_total": scorable, "noaa_days": total_days,
                    "noaa_rate": noaa_rate,
                })

    print("\n" + "=" * 70)
    print("REALISTIC SHORT-HORIZON YIELD (SIMULATION)")
    print("=" * 70)
    
    no_cache_frac = 50 / 377  # from live session: 50 typed commercial of 377 total
    
    print("\nFactors applied to NOAA baseline:")
    print(f"  (a) Live detection: x0.34 (measured pipeline ratio)")
    print(f"  (b) Type cache (no seeding): x{no_cache_frac:.3f} (50 commercial typed out of 377 live)")
    print(f"  (c) Type cache (NOAA-seeded): x{commercial_fraction:.3f} ({len(found_commercial)} of {len(live_mmsis_int)} live MMSIs found commercial in NOAA)")
    print("  (d) Recording discipline: x(days_per_week / 7)")
    print()

    for c in configs:
        print(f"  Horizon={c['horizon_h']}h, Session={c['sess_h']}h, Start={c['start_h']:02d}:00 UTC")
        print(f"    NOAA baseline: {c['noaa_rate']:.2f}/day ({c['noaa_total']} in {c['noaa_days']} days)")
        
        for cache_label, type_frac in [("no_cache", no_cache_frac), ("NOAA_seeded", commercial_fraction)]:
            for days_wk in [4, 7]:
                rate = c["noaa_rate"] * 0.34 * type_frac * (days_wk / 7.0)
                d30 = math.ceil(30 / rate) if rate > 0.001 else 9999
                print(f"    {cache_label:>12}, {days_wk}d/wk: {rate:.3f}/day -> N>=30 in ~{d30} calendar days")
        print()

    # Best-case summary
    best = max(configs, key=lambda c: c["noaa_rate"])
    print("=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)
    print(f"  Horizon: {best['horizon_h']}h, Session: {best['sess_h']}h, Start: {best['start_h']:02d}:00 UTC")
    print(f"  NOAA baseline: {best['noaa_rate']:.2f} arrivals/day")
    
    worst_rate = best["noaa_rate"] * 0.34 * no_cache_frac * (4/7)
    best_rate = best["noaa_rate"] * 0.34 * commercial_fraction * (7/7)
    worst_d30 = math.ceil(30 / worst_rate) if worst_rate > 0.001 else 9999
    best_d30 = math.ceil(30 / best_rate) if best_rate > 0.001 else 9999
    
    print(f"  Worst case (no type cache, 4d/wk): {worst_rate:.3f}/day -> {worst_d30} days")
    print(f"  Best case (NOAA-seeded, 7d/wk):    {best_rate:.3f}/day -> {best_d30} days")
    print(f"  -> N>=30 takes approximately {best_d30} to {worst_d30} calendar days")

    # Save
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "noaa_type_table_size": len(noaa_type_map),
        "noaa_commercial_count": len(noaa_commercial),
        "live_mmsis": len(live_mmsis_int),
        "live_found_in_noaa": len(found_in_noaa),
        "live_found_commercial": len(found_commercial),
        "type_cache_hit_rate": type_cache_fraction,
        "commercial_fraction_of_live": commercial_fraction,
        "arrivals_72h_censor": len(arrivals),
        "configs": configs,
        "recommended": {
            "horizon_h": best["horizon_h"],
            "session_h": best["sess_h"],
            "start_utc": best["start_h"],
            "noaa_rate": best["noaa_rate"],
            "worst_case_days_to_30": worst_d30,
            "best_case_days_to_30": best_d30,
        }
    }
    out_path = RESULTS_DIR / "m0_closeout_yield.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
