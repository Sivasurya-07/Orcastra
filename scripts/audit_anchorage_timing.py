"""
Audit Anchorage Timing: Before vs After t0 & Prior-Track Feature Evaluation.
"""

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

OUTPUT_DIR = Path(r"D:\ORCASTRA_DATA\historical\noaa_3mo")
RESULTS_DIR = Path(r"D:\ORCASTRA_DATA\historical\results")

SLIM_COLUMNS = [
    "mmsi", "base_date_time", "sog", "cog", "heading",
    "vessel_type", "status", "length", "width", "draft", "cargo",
    "lat", "lon",
]

ENTRANCE_ZONE = {"lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65}
DESTINATION_ZONE = {"lat_min": 29.65, "lat_max": 29.80, "lon_min": -95.15, "lon_max": -94.95}
BOLIVAR_ROADS = {"lat_min": 29.34, "lat_max": 29.44, "lon_min": -94.80, "lon_max": -94.70}
GULF_FAIRWAY = {"lat_min": 29.10, "lat_max": 29.32, "lon_min": -94.85, "lon_max": -94.55}
COMMERCIAL_TYPES = set(range(70, 90))


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def main():
    print("=" * 70)
    print("ANCHORAGE TIMING AUDIT: BEFORE vs AFTER t0")
    print("=" * 70)

    files = sorted(OUTPUT_DIR.glob("houston_2024-*.parquet"))
    df_raw = pl.read_parquet(files, columns=SLIM_COLUMNS)
    df = df_raw.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])

    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    left_bound = min_ts + timedelta(hours=6)
    right_bound = max_ts - timedelta(hours=72)

    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    normalized = {int(k[0] if isinstance(k, tuple) else k): v for k, v in mmsi_groups.items()}

    # Extract all completed arrivals with both crossing definition and arrival
    voyages = []
    for mmsi, vdf in normalized.items():
        recs = vdf.sort("dt").to_dicts()
        n = len(recs)
        was_inside = False
        i = 0
        while i < n:
            rec = recs[i]
            inside = in_b(rec["lat"], rec["lon"])
            if inside and not was_inside:
                t0 = rec["dt"]
                sog0 = rec["sog"] if rec["sog"] is not None and not (abs(rec["sog"] - 102.3) < 0.01) else 0.0
                if t0 <= left_bound or sog0 < 3.0 or t0 > right_bound:
                    was_inside = True
                    i += 1
                    continue
                found = False
                j = i + 1
                while j < n:
                    rn = recs[j]
                    dur = (rn["dt"] - t0).total_seconds() / 3600.0
                    if dur > 72.0:
                        break
                    if in_dest(rn["lat"], rn["lon"]):
                        ns = rn["sog"] if rn["sog"] is not None and not (abs(rn["sog"] - 102.3) < 0.01) else 0.0
                        if ns < 1.5 and dur >= 1.0:
                            cog_val = rec["cog"] if rec["cog"] is not None and not (abs(rec["cog"] - 360.0) < 0.01) else None
                            is_inbound = cog_val is not None and ((270 <= cog_val <= 360) or (0 <= cog_val <= 30))
                            voyages.append({
                                "mmsi": mmsi,
                                "t0": t0,
                                "t_arr": rn["dt"],
                                "duration_h": dur,
                                "sog_t0": sog0,
                                "cog_t0": cog_val,
                                "is_inbound": is_inbound,
                                "start_idx": i,
                                "end_idx": j,
                            })
                            found = True
                            break
                    j += 1
                was_inside = True
                i = max(j if found else i + 1, i + 1)
            else:
                was_inside = inside
                i += 1

    print(f"Total analyzed voyages: {len(voyages)}")

    timing_counts = {
        "wait_before_t0_only": 0,
        "wait_after_t0_only": 0,
        "wait_both_before_and_after": 0,
        "direct_no_wait": 0,
    }

    # Also evaluate duration > 12h subset
    timing_over_12h = {
        "wait_before_t0_only": 0,
        "wait_after_t0_only": 0,
        "wait_both_before_and_after": 0,
        "direct_no_wait": 0,
    }

    prior_track_stats = []

    for v in voyages:
        recs = normalized[v["mmsi"]].sort("dt").to_dicts()
        t0 = v["t0"]
        t_arr = v["t_arr"]

        # Prior track: pings in [t0 - 6h, t0)
        pings_prior_6h = [p for p in recs if (t0 - timedelta(hours=6)) <= p["dt"] < t0]
        pings_prior_2h = [p for p in recs if (t0 - timedelta(hours=2)) <= p["dt"] < t0]
        pings_prior_1h = [p for p in recs if (t0 - timedelta(hours=1)) <= p["dt"] < t0]

        # Post track: pings in [t0, t_arr]
        pings_post = recs[v["start_idx"] : v["end_idx"] + 1]

        # Wait definition: dwelling at SOG < 1.0 kn for >= 0.5 hours
        def get_dwell_hours(ping_list):
            stationary = [p for p in ping_list if p["sog"] is not None and p["sog"] < 1.0 and abs(p["sog"] - 102.3) > 0.01]
            if len(stationary) < 2:
                return 0.0
            return (max(p["dt"] for p in stationary) - min(p["dt"] for p in stationary)).total_seconds() / 3600.0

        dwell_before = get_dwell_hours(pings_prior_6h)
        waited_before = dwell_before >= 0.5

        dwell_after = get_dwell_hours(pings_post)
        waited_after = dwell_after >= 0.5

        if waited_before and not waited_after:
            cat = "wait_before_t0_only"
        elif not waited_before and waited_after:
            cat = "wait_after_t0_only"
        elif waited_before and waited_after:
            cat = "wait_both_before_and_after"
        else:
            cat = "direct_no_wait"

        timing_counts[cat] += 1
        if v["duration_h"] > 12.0:
            timing_over_12h[cat] += 1

        # Calculate prior-track kinematic features
        sogs_prior_2h = [p["sog"] for p in pings_prior_2h if p["sog"] is not None and abs(p["sog"] - 102.3) > 0.01]
        mean_sog_prior_2h = float(np.mean(sogs_prior_2h)) if sogs_prior_2h else v["sog_t0"]
        min_sog_prior_2h = float(np.min(sogs_prior_2h)) if sogs_prior_2h else v["sog_t0"]

        prior_track_stats.append({
            "mmsi": v["mmsi"],
            "duration_h": v["duration_h"],
            "dwell_before_h": dwell_before,
            "dwell_after_h": dwell_after,
            "mean_sog_prior_2h": mean_sog_prior_2h,
            "min_sog_prior_2h": min_sog_prior_2h,
            "is_inbound": v["is_inbound"],
        })

    print(f"\nAnchorage Timing Breakdown (All {len(voyages)} Arrivals):")
    for k, count in timing_counts.items():
        print(f"  {k}: {count} ({100*count/len(voyages):.1f}%)")

    print(f"\nAnchorage Timing for Long Trips (Duration > 12h, N = {sum(timing_over_12h.values())}):")
    for k, count in timing_over_12h.items():
        print(f"  {k}: {count} ({100*count/sum(timing_over_12h.values()):.1f}%)")

    # Correlation between prior-track features and transit duration
    durs = [p["duration_h"] for p in prior_track_stats]
    dwell_befores = [p["dwell_before_h"] for p in prior_track_stats]
    mean_sog_priors = [p["mean_sog_prior_2h"] for p in prior_track_stats]

    corr_dwell = np.corrcoef(dwell_befores, durs)[0, 1]
    corr_sog = np.corrcoef(mean_sog_priors, durs)[0, 1]
    print(f"\nPrior-Track Feature Correlation with Transit Duration:")
    print(f"  Pre-t0 Dwell Hours correlation with Duration: r = {corr_dwell:.3f}")
    print(f"  Pre-t0 Mean SOG (2h) correlation with Duration: r = {corr_sog:.3f}")

    results_out = {
        "timing_counts_all": timing_counts,
        "timing_counts_over_12h": timing_over_12h,
        "correlations": {
            "pre_t0_dwell_vs_duration": float(corr_dwell),
            "pre_t0_mean_sog_vs_duration": float(corr_sog),
        }
    }
    with open(RESULTS_DIR / "anchorage_timing_results.json", "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'anchorage_timing_results.json'}")


def in_b(lat, lon):
    return in_box(lat, lon, ENTRANCE_ZONE)

def in_dest(lat, lon):
    return in_box(lat, lon, DESTINATION_ZONE)


if __name__ == "__main__":
    main()
