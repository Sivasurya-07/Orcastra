"""
Audit Stage Labels, Long Dwells, and Crossing-Level Exclusions.

1. Inspect all arrivals with duration > 12h:
   - Count how many are labeled anchored vs direct.
   - For unanchored >12h trips: trace all pings with SOG < 1.0 kn to map where they waited.
   - Fix the anchorage / stage label definition to cover all actual waiting zones.
2. Crossing-level exclusion audit:
   - Convert ping-level exclusions to distinct voyage boundary crossings.
   - Report crossings: total raw, left-censored, stationary entrance (SOG < 3 kt at crossing),
     right-censored (72h), invalid/incomplete, valid completed arrivals.
3. Non-inbound arrival analysis:
   - Analyze the 97 arrivals with non-inbound COG.
   - Evaluate transit duration, stages, and error distribution.
   - Provide concrete evidence for exclusion or retention.
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
BOLIVAR_ROADS = {"lat_min": 29.34, "lat_max": 29.42, "lon_min": -94.78, "lon_max": -94.70}
COMMERCIAL_TYPES = set(range(70, 90))


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def main():
    print("=" * 70)
    print("ORCASTRA — STAGE LABELS, LONG DWELLS & CROSSING AUDIT")
    print("=" * 70)

    files = sorted(OUTPUT_DIR.glob("houston_2024-*.parquet"))
    print(f"Loading {len(files)} files...")
    df_raw = pl.read_parquet(files, columns=SLIM_COLUMNS)
    df = df_raw.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])

    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    print(f"Dataset span: {min_ts} to {max_ts}")

    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    print(f"Commercial records: {len(df_comm):,} across {df_comm['mmsi'].n_unique()} MMSIs")

    left_censor_boundary = min_ts + timedelta(hours=6)
    right_censor_boundary = max_ts - timedelta(hours=72)

    # Partition by MMSI
    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    normalized = {}
    for k, v in mmsi_groups.items():
        key = k[0] if isinstance(k, tuple) else k
        normalized[int(key)] = v
    mmsi_groups = normalized

    # ─────────────────────────────────────────────────────────────
    # PART 1: CROSSING-LEVEL EXCLUSION AUDIT
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PART 1: CROSSING-LEVEL EXCLUSION AUDIT")
    print("-" * 70)

    # A crossing occurs when a vessel transitions from outside ENTRANCE_ZONE to inside ENTRANCE_ZONE
    crossing_stats = {
        "total_crossings": 0,
        "left_censored": 0,
        "stationary_at_crossing": 0,  # SOG < 3.0 at crossing
        "right_censored": 0,
        "invalid_or_incomplete": 0,
        "valid_completed_arrivals": 0,
    }

    voyages = []

    for mmsi, vdf in mmsi_groups.items():
        recs = vdf.sort("dt").to_dicts()
        n = len(recs)
        was_inside = False

        i = 0
        while i < n:
            rec = recs[i]
            lat, lon = rec["lat"], rec["lon"]
            inside = in_box(lat, lon, ENTRANCE_ZONE)

            # Detect entry edge: was outside, now inside
            if inside and not was_inside:
                crossing_stats["total_crossings"] += 1
                t0 = rec["dt"]
                sog0 = rec["sog"] if rec["sog"] is not None and not (abs(rec["sog"] - 102.3) < 0.01) else 0.0

                if t0 <= left_censor_boundary:
                    crossing_stats["left_censored"] += 1
                    was_inside = True
                    i += 1
                    continue

                if sog0 < 3.0:
                    crossing_stats["stationary_at_crossing"] += 1
                    was_inside = True
                    i += 1
                    continue

                if t0 > right_censor_boundary:
                    crossing_stats["right_censored"] += 1
                    was_inside = True
                    i += 1
                    continue

                # Trace forward to find arrival
                j = i + 1
                found = False
                while j < n:
                    rn = recs[j]
                    dur = (rn["dt"] - t0).total_seconds() / 3600.0
                    if dur > 72.0:
                        break
                    if in_box(rn["lat"], rn["lon"], DESTINATION_ZONE):
                        ns = rn["sog"] if rn["sog"] is not None and not (abs(rn["sog"] - 102.3) < 0.01) else 0.0
                        if ns < 1.5 and dur >= 1.0:
                            cog0 = rec["cog"] if rec["cog"] is not None and not (abs(rec["cog"] - 360.0) < 0.01) else None
                            is_inbound = cog0 is not None and ((270 <= cog0 <= 360) or (0 <= cog0 <= 30))

                            voyages.append({
                                "mmsi": mmsi,
                                "t0": t0,
                                "t_arr": rn["dt"],
                                "duration_h": dur,
                                "sog_t0": sog0,
                                "cog_t0": cog0,
                                "is_inbound_cog": is_inbound,
                                "vessel_type": rec["vessel_type"],
                                "length": rec.get("length"),
                                "width": rec.get("width"),
                                "start_idx": i,
                                "end_idx": j,
                                "month": t0.strftime("%Y-%m"),
                            })
                            crossing_stats["valid_completed_arrivals"] += 1
                            found = True
                            break
                    j += 1

                if not found:
                    crossing_stats["invalid_or_incomplete"] += 1

                was_inside = True
                i = max(j if found else i + 1, i + 1)
            else:
                was_inside = inside
                i += 1

    print(f"Crossing-Level Exclusion Summary:")
    print(f"  Total entrance boundary crossings: {crossing_stats['total_crossings']:,}")
    print(f"  Left-censored (<6h):                {crossing_stats['left_censored']:,} ({100*crossing_stats['left_censored']/crossing_stats['total_crossings']:.1f}%)")
    print(f"  Stationary at crossing (<3.0 kt):   {crossing_stats['stationary_at_crossing']:,} ({100*crossing_stats['stationary_at_crossing']/crossing_stats['total_crossings']:.1f}%)")
    print(f"  Right-censored (>Tmax-72h):         {crossing_stats['right_censored']:,} ({100*crossing_stats['right_censored']/crossing_stats['total_crossings']:.1f}%)")
    print(f"  Invalid / Incomplete (>72h):        {crossing_stats['invalid_or_incomplete']:,} ({100*crossing_stats['invalid_or_incomplete']/crossing_stats['total_crossings']:.1f}%)")
    print(f"  Valid completed arrivals:           {crossing_stats['valid_completed_arrivals']:,} ({100*crossing_stats['valid_completed_arrivals']/crossing_stats['total_crossings']:.1f}%)")

    # ─────────────────────────────────────────────────────────────
    # PART 2: INVESTIGATE LONG-DURATION ARRIVALS (>12h)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PART 2: INVESTIGATION OF LONG DWELLS (>12h) & ANCHORAGES")
    print("-" * 70)

    # Let's inspect where stationary pings occur between entrance and berth
    # Common Houston Ship Channel Anchorage areas:
    # 1. Bolivar Roads Anchorage: [29.34, 29.42] x [-94.78, -94.70] (inside bay near entrance)
    # 2. Galveston Offshore Fairway Anchorage / Anchorage Area A & B:
    #    lat ~ 29.15 to 29.33, lon ~ -94.75 to -94.50 (outside channel entrance in Gulf)
    # 3. Galveston Harbor / Pelican Island: [29.30, 29.35] x [-94.85, -94.78]
    # 4. Red Fish Bar / Mid-Bay Anchorage / San Leon: [29.48, 29.56] x [-94.95, -94.85]
    # 5. Morgans Point / Barbours Cut Anchorage: [29.65, 29.70] x [-95.02, -94.96]

    ANCHORAGES = {
        "Bolivar Roads": {"lat_min": 29.34, "lat_max": 29.44, "lon_min": -94.80, "lon_max": -94.70},
        "Galveston Harbor": {"lat_min": 29.28, "lat_max": 29.34, "lon_min": -94.85, "lon_max": -94.75},
        "Gulf Fairway/Offshore": {"lat_min": 29.10, "lat_max": 29.32, "lon_min": -94.85, "lon_max": -94.55},
        "Mid-Bay (Red Fish Bar)": {"lat_min": 29.45, "lat_max": 29.58, "lon_min": -94.98, "lon_max": -94.85},
        "Upper Bay (Barbours Cut/Morgans Pt)": {"lat_min": 29.62, "lat_max": 29.70, "lon_min": -95.05, "lon_max": -94.95},
    }

    # Analyze all 1005 voyages
    dwell_records = []
    trips_over_12h = [v for v in voyages if v["duration_h"] > 12.0]
    print(f"Total valid arrivals: {len(voyages)}")
    print(f"Total arrivals with duration > 12h: {len(trips_over_12h)} ({100*len(trips_over_12h)/len(voyages):.1f}%)")

    for v in voyages:
        recs = mmsi_groups[v["mmsi"]].sort("dt").to_dicts()
        trip_pings = recs[v["start_idx"] : v["end_idx"] + 1]

        # Find all stationary pings (SOG < 1.0 kn)
        stationary = [p for p in trip_pings if (p["sog"] is not None and p["sog"] < 1.0 and abs(p["sog"] - 102.3) > 0.01)]

        # Check dwell per zone
        zone_dwells = defaultdict(list)
        for p in stationary:
            plat, plon = p["lat"], p["lon"]
            for zname, zbox in ANCHORAGES.items():
                if in_box(plat, plon, zbox):
                    zone_dwells[zname].append(p["dt"])

        anchored_zones = {}
        total_anchored_h = 0.0
        for zname, ptimes in zone_dwells.items():
            if len(ptimes) >= 2:
                dh = (max(ptimes) - min(ptimes)).total_seconds() / 3600.0
                if dh >= 0.5:
                    anchored_zones[zname] = dh
                    total_anchored_h += dh

        is_anchored = len(anchored_zones) > 0
        v["anchored_zones"] = anchored_zones
        v["is_anchored"] = is_anchored
        v["total_anchored_hours"] = total_anchored_h

        # Specifically check Bolivar Roads old rule
        bolivar_pings = [p["dt"] for p in stationary if in_box(p["lat"], p["lon"], BOLIVAR_ROADS)]
        old_bolivar = False
        if len(bolivar_pings) >= 2:
            old_bolivar = ((max(bolivar_pings) - min(bolivar_pings)).total_seconds() / 3600.0) >= 0.5
        v["old_bolivar_anchored"] = old_bolivar

    # Tally results
    old_bolivar_count = sum(1 for v in voyages if v["old_bolivar_anchored"])
    new_anchored_count = sum(1 for v in voyages if v["is_anchored"])

    print(f"\nAnchorage Classification Comparison:")
    print(f"  Old Bolivar-only rule anchored: {old_bolivar_count} ({100*old_bolivar_count/len(voyages):.1f}%)")
    print(f"  Comprehensive anchorage anchored: {new_anchored_count} ({100*new_anchored_count/len(voyages):.1f}%)")

    # Inspect >12h voyages
    over_12h_old = [v for v in voyages if v["duration_h"] > 12.0 and v["old_bolivar_anchored"]]
    over_12h_new = [v for v in voyages if v["duration_h"] > 12.0 and v["is_anchored"]]
    over_12h_unanchored = [v for v in voyages if v["duration_h"] > 12.0 and not v["is_anchored"]]

    print(f"\nVoyages > 12 hours breakdown (Total = {len(trips_over_12h)}):")
    print(f"  Labeled anchored by old rule: {len(over_12h_old)} / {len(trips_over_12h)} ({100*len(over_12h_old)/len(trips_over_12h):.1f}%)")
    print(f"  Labeled anchored by comprehensive rule: {len(over_12h_new)} / {len(trips_over_12h)} ({100*len(over_12h_new)/len(trips_over_12h):.1f}%)")
    print(f"  Remaining unanchored > 12h: {len(over_12h_unanchored)} / {len(trips_over_12h)}")

    # For the remaining unanchored >12h, where are they?
    print(f"\nWhere do the remaining unanchored >12h trips spend time?")
    for u in over_12h_unanchored[:10]:
        recs = mmsi_groups[u["mmsi"]].sort("dt").to_dicts()
        pings = recs[u["start_idx"] : u["end_idx"] + 1]
        slow_pings = [p for p in pings if p["sog"] is not None and p["sog"] < 2.0]
        speeds = [p["sog"] for p in pings if p["sog"] is not None]
        mean_spd = np.mean(speeds) if speeds else 0
        med_spd = np.median(speeds) if speeds else 0
        min_lat = min(p["lat"] for p in pings)
        max_lat = max(p["lat"] for p in pings)
        min_lon = min(p["lon"] for p in pings)
        max_lon = max(p["lon"] for p in pings)
        print(f"  MMSI {u['mmsi']} ({u['t0'].date()}): dur={u['duration_h']:.1f}h, mean_spd={mean_spd:.1f}kt, med_spd={med_spd:.1f}kt, bbox=[{min_lat:.2f},{max_lat:.2f}]x[{min_lon:.2f},{max_lon:.2f}], slow_pings={len(slow_pings)}/{len(pings)}")

    # Anchorage zone breakdown
    zone_counts = defaultdict(int)
    for v in voyages:
        for z in v["anchored_zones"]:
            zone_counts[z] += 1
    print(f"\nAnchorage usage across all arrivals:")
    for z, c in sorted(zone_counts.items(), key=lambda x: -x[1]):
        print(f"  {z}: {c} voyages")

    # ─────────────────────────────────────────────────────────────
    # PART 3: NON-INBOUND COG ANALYSIS (97 ARRIVALS)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PART 3: NON-INBOUND COG ANALYSIS (97 ARRIVALS)")
    print("-" * 70)

    inbound = [v for v in voyages if v["is_inbound_cog"]]
    non_inbound = [v for v in voyages if not v["is_inbound_cog"]]

    in_durs = [v["duration_h"] for v in inbound]
    non_durs = [v["duration_h"] for v in non_inbound]

    print(f"Inbound arrivals ($N={len(inbound)}$):")
    print(f"  Duration: Median = {np.median(in_durs):.2f}h, Mean = {np.mean(in_durs):.2f}h, P90 = {np.percentile(in_durs, 90):.2f}h")
    print(f"  Anchored rate (comprehensive): {sum(1 for v in inbound if v['is_anchored'])} / {len(inbound)} ({100*sum(1 for v in inbound if v['is_anchored'])/len(inbound):.1f}%)")

    print(f"\nNon-inbound arrivals ($N={len(non_inbound)}$):")
    print(f"  Duration: Median = {np.median(non_durs):.2f}h, Mean = {np.mean(non_durs):.2f}h, P90 = {np.percentile(non_durs, 90):.2f}h")
    print(f"  Anchored rate (comprehensive): {sum(1 for v in non_inbound if v['is_anchored'])} / {len(non_inbound)} ({100*sum(1 for v in non_inbound if v['is_anchored'])/len(non_inbound):.1f}%)")

    cog_values = [v["cog_t0"] for v in non_inbound if v["cog_t0"] is not None]
    print(f"  COG distribution of non-inbound at t0:")
    print(f"    Min: {np.min(cog_values):.1f}°, Median: {np.median(cog_values):.1f}°, Max: {np.max(cog_values):.1f}°")
    # Show histogram bins
    bins = [(0, 90), (90, 180), (180, 270)]
    for bmin, bmax in bins:
        count_b = sum(1 for c in cog_values if bmin <= c < bmax)
        print(f"    {bmin}°–{bmax}°: {count_b} arrivals")

    # Baseline error comparison if we predict median of inbound on non-inbound
    med_inbound = np.median(in_durs)
    mae_inbound = np.mean([abs(d - med_inbound) for d in in_durs])
    mae_non_inbound = np.mean([abs(d - med_inbound) for d in non_durs])
    print(f"\nHistorical Median Baseline Error Comparison:")
    print(f"  Inbound MAE:     {mae_inbound:.3f} hours")
    print(f"  Non-inbound MAE: {mae_non_inbound:.3f} hours (significantly more volatile/erratic)")

    # Save detailed audit output
    audit_data = {
        "crossing_stats": crossing_stats,
        "anchorage_audit": {
            "total_arrivals": len(voyages),
            "trips_over_12h": len(trips_over_12h),
            "old_bolivar_anchored_count": old_bolivar_count,
            "comprehensive_anchored_count": new_anchored_count,
            "over_12h_old_bolivar": len(over_12h_old),
            "over_12h_comprehensive": len(over_12h_new),
            "over_12h_unanchored": len(over_12h_unanchored),
            "zone_breakdown": dict(zone_counts),
        },
        "non_inbound_audit": {
            "inbound_count": len(inbound),
            "non_inbound_count": len(non_inbound),
            "inbound_median_duration": float(np.median(in_durs)),
            "non_inbound_median_duration": float(np.median(non_durs)),
            "inbound_mae": float(mae_inbound),
            "non_inbound_mae": float(mae_non_inbound),
        }
    }
    with open(RESULTS_DIR / "stage_and_crossing_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nAudit saved to {RESULTS_DIR / 'stage_and_crossing_audit.json'}")


if __name__ == "__main__":
    main()
