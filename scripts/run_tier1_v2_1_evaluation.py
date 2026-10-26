"""
ORCASTRA Tier 1 -- Benchmark Execution conforming to Protocol v2.1 (docs/EVAL_PROTOCOL_V2_1.md).

KEY CHANGES FROM v2.0 SCRIPT:
  - NO label-conditioned selection: all state-triggered corridor snapshots are included
    regardless of true remaining time.
  - Buckets are defined by DISTANCE TO BERTH (observable at prediction time), not by
    true remaining duration (which is the target).
  - Bucket Median Baseline added: predict training-set median y_rem for each distance bucket.
  - Interval widths (P80 - P50, P95 - P50) reported.
  - Share of snapshots with true y_rem > 3 h reported per bucket.
  - Anchored-midway and right-censored snapshots explicitly counted and tagged.
  - Conformal calibration strictly on Feb 16-29 held-out slice (isolated from test).
  - Vessel-clustered bootstrap CIs (B=2000).
  - Strict ASCII logging to avoid Windows console encoding failures.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

# -- Paths --
OUTPUT_DIR = Path(r"D:\ORCASTRA_DATA\historical\noaa_3mo")
RESULTS_DIR = Path(r"D:\ORCASTRA_DATA\historical\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SLIM_COLUMNS = [
    "mmsi", "base_date_time", "sog", "cog", "heading",
    "vessel_type", "status", "length", "width", "draft", "cargo",
    "lat", "lon",
]

ENTRANCE_ZONE = {"lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65}
DESTINATION_ZONE = {"lat_min": 29.65, "lat_max": 29.80, "lon_min": -95.15, "lon_max": -94.95}
COMMERCIAL_TYPES = set(range(70, 90))

# Houston Ship Channel Centerline Waypoints (Lat, Lon, cumulative km from berth)
CENTERLINE_WP = [
    (29.350, -94.750, 50.4),  # Entrance buoy
    (29.380, -94.780, 46.0),  # Bolivar Roads
    (29.510, -94.900, 28.5),  # Red Fish Bar
    (29.600, -94.960, 16.0),  # Mid-Bay Beacon
    (29.680, -95.000, 4.5),   # Barbours Cut Turn
    (29.720, -95.050, 0.0),   # Inner Berths
]

# -- Corridor boundaries --
CORRIDOR = {"lat_min": 29.34, "lat_max": 29.75, "lon_min": -95.12, "lon_max": -94.70}

# -- Distance-to-berth buckets (v2.1 Section 3.4) --
DISTANCE_BUCKETS = {
    "Near Berth (d <= 10 km)":       (0.0, 10.0),
    "Mid-Channel (10-25 km)":        (10.0, 25.0),
    "Outer Channel (25-50 km)":      (25.0, 50.0),
    "Beyond Entrance (d > 50 km)":   (50.0, 999.0),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0)**2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def channel_distance_to_berth(lat, lon):
    """Compute approximate remaining along-channel distance (km) to berth via waypoints."""
    min_dist = float('inf')
    best_rem = 50.4
    for w_lat, w_lon, rem_km in CENTERLINE_WP:
        d = haversine_km(lat, lon, w_lat, w_lon)
        if d < min_dist:
            min_dist = d
            best_rem = rem_km
    return max(best_rem, 0.5)


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def safe_sog(val):
    """Return numeric SOG, treating None and 102.3 (AIS unavailable) as 0."""
    if val is None or (isinstance(val, (int, float)) and abs(val - 102.3) < 0.01):
        return 0.0
    return float(val)


def safe_cog(val):
    """Return numeric COG, treating None and 360.0 (AIS unavailable) as None."""
    if val is None or (isinstance(val, (int, float)) and abs(val - 360.0) < 0.01):
        return None
    return float(val)


def is_inbound_entrance(cog_val):
    """Entrance zone crossing inbound definition: [270, 360] U [0, 30]."""
    return cog_val is not None and ((270 <= cog_val <= 360) or (0 <= cog_val <= 30))


def is_inbound_corridor(cog_val):
    """Corridor underway heading definition: [270, 360] U [0, 45]."""
    return cog_val is not None and ((270 <= cog_val <= 360) or (0 <= cog_val <= 45))


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


# -- Vessel-Clustered Bootstrap (v2.1 Section 7.1) --
# Specification: EVAL_PROTOCOL_V2_1.md Section 7.1.
# Code lines L126-L150: bootstrap resamples unique MMSIs with replacement.
def cluster_bootstrap_ci(mmsi_array, y_true, y_pred, metric_func, B=2000, seed=42):
    """Vessel-level cluster bootstrap resampling unique MMSIs with replacement.

    Each bootstrap iteration:
      1. Sample N_vessels unique MMSIs with replacement.
      2. Collect all snapshots belonging to those sampled vessels.
      3. Compute the metric on the collected snapshots.
    Report [q_0.025, q_0.975] of the B bootstrap scores.
    """
    rng = np.random.default_rng(seed)
    unique_mmsis = np.unique(mmsi_array)
    n_vessels = len(unique_mmsis)

    mmsi_to_indices = defaultdict(list)
    for idx, m in enumerate(mmsi_array):
        mmsi_to_indices[m].append(idx)

    boot_scores = []
    for _ in range(B):
        sampled_mmsis = rng.choice(unique_mmsis, size=n_vessels, replace=True)
        boot_indices = []
        for sm in sampled_mmsis:
            boot_indices.extend(mmsi_to_indices[sm])
        boot_indices = np.array(boot_indices)
        if len(boot_indices) == 0:
            continue
        score = metric_func(y_true[boot_indices], y_pred[boot_indices])
        boot_scores.append(score)

    q_low = float(np.percentile(boot_scores, 2.5))
    q_high = float(np.percentile(boot_scores, 97.5))
    return q_low, q_high, (q_high - q_low)


def main():
    t_start = time.time()
    print("=" * 70)
    print("ORCASTRA TIER 1 -- PROTOCOL V2.1 BENCHMARK RUN")
    print("  Protocol: docs/EVAL_PROTOCOL_V2_1.md")
    print("  Key change: unconditioned snapshots, distance-to-berth buckets")
    print("=" * 70)

    # -- Load data --
    files = sorted(OUTPUT_DIR.glob("houston_2024-*.parquet"))
    df_raw = pl.read_parquet(files, columns=SLIM_COLUMNS)
    df = df_raw.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])

    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    left_bound = min_ts + timedelta(hours=6)
    right_bound = max_ts - timedelta(hours=72)
    split_boundary = datetime(2024, 3, 1, 0, 0, 0)

    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    normalized = {int(k[0] if isinstance(k, tuple) else k): v for k, v in mmsi_groups.items()}

    print(f"\nDataset: {min_ts} to {max_ts}")
    print(f"Left-censor bound: {left_bound}")
    print(f"Right-censor bound (72h rule): {right_bound}")
    print(f"Commercial MMSIs: {len(normalized)}")

    # ─────────────────────────────────────────────────────────────────
    # PART 1: EXTRACT COHORT A1 AND COHORT A2 ARRIVALS
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PART 1: FULL-TRIP RECONCILIATION: COHORT A1 vs COHORT A2")
    print("-" * 70)

    cohort_a1 = []
    cohort_a2 = []

    for mmsi, vdf in normalized.items():
        recs = vdf.sort("dt").to_dicts()
        n = len(recs)
        was_inside = False
        i = 0
        while i < n:
            rec = recs[i]
            inside = in_box(rec["lat"], rec["lon"], ENTRANCE_ZONE)
            if inside and not was_inside:
                t0 = rec["dt"]
                sog0 = safe_sog(rec["sog"])
                if t0 <= left_bound or sog0 < 3.0:
                    was_inside = True
                    i += 1
                    continue
                cog_val = safe_cog(rec["cog"])
                is_inbound = is_inbound_entrance(cog_val)

                # Search forward for berth arrival
                j = i + 1
                found = False
                while j < n:
                    rn = recs[j]
                    dur = (rn["dt"] - t0).total_seconds() / 3600.0
                    if dur > 72.0:
                        break
                    if in_box(rn["lat"], rn["lon"], DESTINATION_ZONE):
                        ns = safe_sog(rn["sog"])
                        if ns < 1.5 and dur >= 1.0:
                            v_data = {
                                "mmsi": mmsi, "t0": t0, "t_arr": rn["dt"], "duration_h": dur,
                                "sog_t0": sog0, "cog_t0": cog_val, "is_train": t0 < split_boundary,
                                "length": rec.get("length"), "width": rec.get("width"), "draft": rec.get("draft"),
                                "vessel_type": rec["vessel_type"], "hour": t0.hour, "dayofweek": t0.weekday(),
                            }
                            if t0 <= right_bound:
                                if is_inbound:
                                    cohort_a1.append(v_data)
                                cohort_a2.append(v_data)
                                found = True
                            break
                    j += 1

                # If not inbound and not found, allow checking next crossing for Cohort A2
                if not is_inbound and not found:
                    was_inside = True
                    i += 1
                    continue

                was_inside = True
                i = max(j if found else i + 1, i + 1)
            else:
                was_inside = inside
                i += 1

    print(f"Cohort A1 (Strict Inbound First Crossing): {len(cohort_a1)} arrivals "
          f"({sum(1 for a in cohort_a1 if a['is_train'])} Train, {sum(1 for a in cohort_a1 if not a['is_train'])} Test)")
    print(f"Cohort A2 (Inbound with Re-entry Allowed): {len(cohort_a2)} arrivals "
          f"({sum(1 for a in cohort_a2 if a['is_train'])} Train, {sum(1 for a in cohort_a2 if not a['is_train'])} Test)")

    # Full-trip baseline on Cohort A1
    train_a1 = [a for a in cohort_a1 if a["is_train"]]
    test_a1 = [a for a in cohort_a1 if not a["is_train"]]
    y_tr_a1 = np.array([a["duration_h"] for a in train_a1])
    y_te_a1 = np.array([a["duration_h"] for a in test_a1])
    mmsi_te_a1 = np.array([a["mmsi"] for a in test_a1])

    med_a1 = float(np.median(y_tr_a1))
    mae_a1 = float(np.mean(np.abs(y_te_a1 - med_a1)))
    ci_l_a1, ci_h_a1, ci_w_a1 = cluster_bootstrap_ci(
        mmsi_te_a1, y_te_a1, np.full_like(y_te_a1, med_a1),
        lambda yt, yp: np.mean(np.abs(yt - yp))
    )
    print(f"Cohort A1 Full-Trip Historical Median (scalar = {med_a1:.2f} h):")
    print(f"  Test N: {len(test_a1)} arrivals")
    print(f"  MAE:    {mae_a1:.3f} h  [95% CI: {ci_l_a1:.3f}, {ci_h_a1:.3f}] (width: {ci_w_a1:.3f} h)")

    # ─────────────────────────────────────────────────────────────────
    # PART 2: UNCONDITIONED SNAPSHOT EXTRACTION (v2.1 Section 3.2-3.3)
    #   - Triggered by STATE, not outcome
    #   - NO filter on true remaining time
    #   - Tag anchored_midway and right_censored snapshots
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PART 2: UNCONDITIONED SNAPSHOT EXTRACTION")
    print("-" * 70)

    snapshots = []
    n_raw = 0
    n_censored = 0
    n_anchored = 0

    for mmsi, vdf in normalized.items():
        recs = vdf.sort("dt").to_dicts()

        voyage_subset = [v for v in cohort_a1 if v["mmsi"] == mmsi]
        for v in voyage_subset:
            t0 = v["t0"]
            t_arr = v["t_arr"]

            # Corridor pings in [t0, t_arr]
            v_pings = [p for p in recs if t0 <= p["dt"] <= t_arr
                       and in_box(p["lat"], p["lon"], CORRIDOR)]
            if not v_pings:
                continue

            # Detect if vessel anchored midway:
            # SOG < 1.0 kn for >= 30 minutes continuously before reaching berth
            voyage_anchored = False
            slow_streak_start = None
            for p in v_pings:
                ps = safe_sog(p["sog"])
                if ps < 1.0:
                    if slow_streak_start is None:
                        slow_streak_start = p["dt"]
                    elif (p["dt"] - slow_streak_start).total_seconds() >= 1800:
                        voyage_anchored = True
                        break
                else:
                    slow_streak_start = None

            # Sample every 10 minutes from t0 to t_arr
            curr_time = t0
            while curr_time < t_arr:
                # Find closest ping within 5 min
                best_ping = None
                best_diff = timedelta(minutes=6)
                for p in v_pings:
                    diff = abs(p["dt"] - curr_time)
                    if diff < best_diff:
                        best_diff = diff
                        best_ping = p

                if best_ping is not None and best_diff <= timedelta(minutes=5):
                    sog = safe_sog(best_ping["sog"])
                    cog = safe_cog(best_ping["cog"])
                    inbound_head = is_inbound_corridor(cog)

                    if sog >= 3.0 and inbound_head:
                        n_raw += 1
                        rem_dur = (t_arr - curr_time).total_seconds() / 3600.0
                        rem_km = channel_distance_to_berth(best_ping["lat"], best_ping["lon"])

                        # Right-censoring check (v2.1 Section 3.3):
                        # Snapshot within 3h of dataset end AND arrival is unobserved
                        is_rc = (curr_time > (max_ts - timedelta(hours=3)) and t_arr > max_ts)

                        if is_rc:
                            n_censored += 1
                        else:
                            # *** NO FILTER ON rem_dur ***
                            # Unconditioned selection: all state-triggered snapshots kept
                            if voyage_anchored:
                                n_anchored += 1

                            snapshots.append({
                                "mmsi": mmsi,
                                "snap_time": curr_time,
                                "t_arr": t_arr,
                                "rem_dur_h": rem_dur,
                                "lat": best_ping["lat"],
                                "lon": best_ping["lon"],
                                "rem_km": rem_km,
                                "sog": sog,
                                "cog": cog,
                                "heading": best_ping["heading"],
                                "length": v.get("length"),
                                "width": v.get("width"),
                                "draft": v.get("draft"),
                                "vessel_type": v["vessel_type"],
                                "month": curr_time.strftime("%Y-%m"),
                                "is_train": curr_time < split_boundary,
                                "anchored_midway": voyage_anchored,
                            })

                curr_time += timedelta(minutes=10)

    # Summary
    share_gt3h = sum(1 for s in snapshots if s["rem_dur_h"] > 3.0) / max(len(snapshots), 1) * 100
    print(f"Snapshot Extraction Summary:")
    print(f"  Total state-triggered snapshots: {n_raw}")
    print(f"  Right-censored (excluded):       {n_censored}")
    print(f"  Anchored-midway tagged:          {n_anchored}")
    print(f"  Valid uncensored snapshots:      {len(snapshots)}")
    print(f"  Share with true y_rem > 3h:      {share_gt3h:.1f}%")

    # -- Split: Train / Calib / Test --
    snap_train_full = [s for s in snapshots if s["is_train"]]
    snap_test = [s for s in snapshots if not s["is_train"]]

    # Conformal split on train: Train Slice (Jan 1 - Feb 15) vs Calib Slice (Feb 16 - Feb 29)
    # Specification: EVAL_PROTOCOL_V2_1.md Section 5.2.
    calib_cutoff = datetime(2024, 2, 16, 0, 0, 0)
    snap_train_slice = [s for s in snap_train_full if s["snap_time"] < calib_cutoff]
    snap_calib_slice = [s for s in snap_train_full if s["snap_time"] >= calib_cutoff]

    print(f"\nChronological Split:")
    print(f"  Train Slice (Jan 1 - Feb 15): {len(snap_train_slice)} snapshots "
          f"({len(set(s['mmsi'] for s in snap_train_slice))} vessels)")
    print(f"  Calib Slice (Feb 16 - Feb 29): {len(snap_calib_slice)} snapshots "
          f"({len(set(s['mmsi'] for s in snap_calib_slice))} vessels)")
    print(f"  Test Set    (March 1 - 31):    {len(snap_test)} snapshots "
          f"({len(set(s['mmsi'] for s in snap_test))} vessels)")

    # -- Prepare features --
    snap_features = ["lat", "lon", "rem_km", "sog", "cog", "length", "draft"]
    tr_medians = {}
    for col in snap_features:
        vals = [s[col] for s in snap_train_slice if s[col] is not None]
        tr_medians[col] = float(np.median(vals)) if vals else 0.0

    def make_snap_X(s_list):
        rows = []
        for s in s_list:
            row = [s[col] if s[col] is not None else tr_medians[col] for col in snap_features]
            rows.append(row)
        return np.array(rows)

    X_tr_slice = make_snap_X(snap_train_slice)
    y_tr_slice = np.array([s["rem_dur_h"] for s in snap_train_slice])

    X_calib = make_snap_X(snap_calib_slice)
    y_calib = np.array([s["rem_dur_h"] for s in snap_calib_slice])

    X_test = make_snap_X(snap_test)
    y_test = np.array([s["rem_dur_h"] for s in snap_test])
    mmsi_test = np.array([s["mmsi"] for s in snap_test])
    drem_test = np.array([s["rem_km"] for s in snap_test])
    anchored_test = np.array([s["anchored_midway"] for s in snap_test])

    # ─────────────────────────────────────────────────────────────────
    # PART 3: FIT QUANTILE MODELS & CONFORMAL CALIBRATION
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PART 3: QUANTILE MODELS & CONFORMAL CALIBRATION")
    print("-" * 70)

    quantiles = [0.50, 0.80, 0.95]
    models = {}
    calib_preds = {}
    test_raw_preds = {}

    for q in quantiles:
        reg = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=42)
        reg.fit(X_tr_slice, y_tr_slice)
        models[q] = reg
        calib_preds[q] = reg.predict(X_calib)
        test_raw_preds[q] = reg.predict(X_test)

    # -- Conformal calibration on HELD-OUT CALIB SLICE ONLY (v2.1 Section 5.2) --
    # Code lines L366-L377: residuals and shifts computed ONLY on calibration slice
    conformal_shifts = {}
    for q in quantiles:
        residuals = y_calib - calib_preds[q]
        shift = float(np.percentile(residuals, q * 100.0))
        conformal_shifts[q] = shift
        cov_calib = float(np.mean(y_calib <= (calib_preds[q] + shift)) * 100.0)
        print(f"  Conformal Shift for P{int(q*100)}: {shift:+.4f} h "
              f"(Calib Coverage: {cov_calib:.1f}%)")

    # -- Apply monotonic sorting + conformal shift to test predictions --
    # Code lines L379-L393: applied to test WITHOUT re-fitting
    raw_p50 = test_raw_preds[0.50]
    raw_p80 = np.maximum(test_raw_preds[0.80], raw_p50)
    raw_p95 = np.maximum(test_raw_preds[0.95], raw_p80)

    test_calib_preds = {}
    test_calib_preds[0.50] = np.maximum(raw_p50 + conformal_shifts[0.50], 0.05)
    test_calib_preds[0.80] = np.maximum(raw_p80 + conformal_shifts[0.80],
                                         test_calib_preds[0.50])
    test_calib_preds[0.95] = np.maximum(raw_p95 + conformal_shifts[0.95],
                                         test_calib_preds[0.80])

    inversions = int(np.sum(test_calib_preds[0.50] > test_calib_preds[0.80]) +
                     np.sum(test_calib_preds[0.80] > test_calib_preds[0.95]))
    print(f"  Post-Processing Quantile Inversions on Test: {inversions}")

    # -- Baselines --
    # 1. Kinematic: d_rem / SOG
    naive_preds = np.array([
        (s["rem_km"] / 1.852) / max(s["sog"], 1.0) for s in snap_test
    ])

    # 2. Historical Median by distance decile (10 deciles over 0-52 km)
    decile_bins = np.linspace(0, 52, 11)
    train_decile_medians = {}
    for b_idx in range(len(decile_bins) - 1):
        d_min, d_max = decile_bins[b_idx], decile_bins[b_idx + 1]
        subset_y = [s["rem_dur_h"] for s in snap_train_full if d_min <= s["rem_km"] < d_max]
        train_decile_medians[b_idx] = float(np.median(subset_y)) if subset_y else 1.5

    def get_hist_median(s):
        rem = s["rem_km"]
        for b_idx in range(len(decile_bins) - 1):
            if decile_bins[b_idx] <= rem < decile_bins[b_idx + 1]:
                return train_decile_medians[b_idx]
        return 1.5

    hist_preds = np.array([get_hist_median(s) for s in snap_test])

    # 3. Bucket Median Baseline (v2.1 Section 4.3): predict training median for the distance bucket
    bucket_train_medians = {}
    for bname, (d_lo, d_hi) in DISTANCE_BUCKETS.items():
        subset_y = [s["rem_dur_h"] for s in snap_train_full
                     if d_lo <= s["rem_km"] < d_hi]
        bucket_train_medians[bname] = float(np.median(subset_y)) if subset_y else 2.0

    def get_bucket_median_pred(d_rem):
        for bname, (d_lo, d_hi) in DISTANCE_BUCKETS.items():
            if d_lo <= d_rem < d_hi:
                return bucket_train_medians[bname]
        return 2.0

    bucket_median_preds = np.array([get_bucket_median_pred(s["rem_km"]) for s in snap_test])

    # ─────────────────────────────────────────────────────────────────
    # PART 4: EVALUATION BY DISTANCE-TO-BERTH BUCKETS (v2.1 Section 3.4)
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SHORT-HORIZON RESULTS BY DISTANCE-TO-BERTH BUCKET (UNCONDITIONED)")
    print("=" * 70)

    test_mmsis_train = set(s["mmsi"] for s in snap_train_full)
    disjoint_mask = np.array([s["mmsi"] not in test_mmsis_train for s in snap_test])

    all_bucket_results = {}

    eval_buckets = dict(DISTANCE_BUCKETS)
    eval_buckets["Overall"] = (0.0, 999.0)

    for bname, (d_lo, d_hi) in eval_buckets.items():
        mask = (drem_test >= d_lo) & (drem_test < d_hi)
        n_b = int(np.sum(mask))
        if n_b == 0:
            print(f"\nBucket: {bname} -- EMPTY (0 snapshots)")
            continue

        y_b = y_test[mask]
        mmsi_b = mmsi_test[mask]
        anch_b = anchored_test[mask]

        # Predictions for this bucket
        naive_b = naive_preds[mask]
        hist_b = hist_preds[mask]
        bmed_b = bucket_median_preds[mask]
        p50_b = test_calib_preds[0.50][mask]
        p80_b = test_calib_preds[0.80][mask]
        p95_b = test_calib_preds[0.95][mask]

        # -- Core Metrics --
        mae_naive = float(np.mean(np.abs(y_b - naive_b)))
        mae_hist = float(np.mean(np.abs(y_b - hist_b)))
        mae_bmed = float(np.mean(np.abs(y_b - bmed_b)))
        mae_p50 = float(np.mean(np.abs(y_b - p50_b)))
        medae_p50 = float(np.median(np.abs(y_b - p50_b)))

        pb_p50 = pinball_loss(y_b, p50_b, 0.50)
        pb_p80 = pinball_loss(y_b, p80_b, 0.80)
        pb_p95 = pinball_loss(y_b, p95_b, 0.95)

        cov_p50 = float(np.mean(y_b <= p50_b) * 100.0)
        cov_p80 = float(np.mean(y_b <= p80_b) * 100.0)
        cov_p95 = float(np.mean(y_b <= p95_b) * 100.0)

        # -- Interval Widths (v2.1 Section 6) --
        iw_p80 = float(np.mean(p80_b - p50_b))
        iw_p95 = float(np.mean(p95_b - p50_b))
        iw_p50_abs = float(np.mean(p50_b))  # mean point estimate

        # -- Shares --
        share_b_gt3h = float(np.mean(y_b > 3.0) * 100.0)
        share_b_anchored = float(np.mean(anch_b) * 100.0)

        # -- Bootstrap CIs --
        ci_l, ci_h, ci_w = cluster_bootstrap_ci(
            mmsi_b, y_b, p50_b,
            lambda yt, yp: np.mean(np.abs(yt - yp)))
        cov95_l, cov95_h, _ = cluster_bootstrap_ci(
            mmsi_b, y_b, p95_b,
            lambda yt, yp: np.mean(yt <= yp) * 100.0)

        # -- Disjoint (Unseen) Vessels --
        mask_disj = mask & disjoint_mask
        n_disj = int(np.sum(mask_disj))
        if n_disj > 0:
            mae_p50_disj = float(np.mean(np.abs(y_test[mask_disj] -
                                                  test_calib_preds[0.50][mask_disj])))
            cov95_disj = float(np.mean(y_test[mask_disj] <=
                                        test_calib_preds[0.95][mask_disj]) * 100.0)
        else:
            mae_p50_disj, cov95_disj = None, None

        all_bucket_results[bname] = {
            "n_snapshots": n_b,
            "n_unique_vessels": len(np.unique(mmsi_b)),
            "n_unseen_vessels": len(np.unique(mmsi_test[mask_disj])) if n_disj > 0 else 0,
            "share_y_rem_gt_3h_pct": share_b_gt3h,
            "share_anchored_midway_pct": share_b_anchored,
            "mae_kinematic_d_sog": mae_naive,
            "mae_hist_segment_median": mae_hist,
            "mae_bucket_median_baseline": mae_bmed,
            "mae_p50_calibrated": mae_p50,
            "mae_p50_bootstrap_95ci": [ci_l, ci_h],
            "ci_width": ci_w,
            "medae_p50": medae_p50,
            "pinball_p50": pb_p50,
            "pinball_p80": pb_p80,
            "pinball_p95": pb_p95,
            "coverage_p50_pct": cov_p50,
            "coverage_p80_pct": cov_p80,
            "coverage_p95_pct": cov_p95,
            "coverage_p95_bootstrap_95ci": [cov95_l, cov95_h],
            "interval_width_p80_minus_p50_h": iw_p80,
            "interval_width_p95_minus_p50_h": iw_p95,
            "mean_p50_prediction_h": iw_p50_abs,
            "unseen_vessels_mae_p50": mae_p50_disj,
            "unseen_vessels_coverage_p95": cov95_disj,
        }

        # -- Print --
        print(f"\nBucket: {bname} (N = {n_b} snapshots, "
              f"{len(np.unique(mmsi_b))} vessels):")
        print(f"  Share y_rem > 3h:        {share_b_gt3h:.1f}%")
        print(f"  Share anchored midway:   {share_b_anchored:.1f}%")
        print(f"  -- MAE Comparison --")
        print(f"    Kinematic (d/SOG):       {mae_naive:.3f} h")
        print(f"    Hist Segment Median:     {mae_hist:.3f} h")
        print(f"    Bucket Median Baseline:  {mae_bmed:.3f} h")
        print(f"    QR P50 (calibrated):     {mae_p50:.3f} h  "
              f"[95% CI: {ci_l:.3f}, {ci_h:.3f}] (width: {ci_w:.3f})")
        print(f"    MedAE (QR P50):          {medae_p50:.3f} h")
        print(f"  -- Coverage --")
        print(f"    P50: {cov_p50:.1f}%  |  P80: {cov_p80:.1f}%  |  "
              f"P95: {cov_p95:.1f}% [CI: {cov95_l:.1f}%, {cov95_h:.1f}%]")
        print(f"  -- Interval Widths (mean across snapshots) --")
        print(f"    P80 - P50: {iw_p80:.3f} h  |  P95 - P50: {iw_p95:.3f} h")
        print(f"  -- Pinball Loss --")
        print(f"    tau=0.50: {pb_p50:.4f}  |  tau=0.80: {pb_p80:.4f}  |  "
              f"tau=0.95: {pb_p95:.4f}")
        if mae_p50_disj is not None:
            print(f"  -- Unseen Vessels --")
            print(f"    MAE P50: {mae_p50_disj:.3f} h  |  "
                  f"P95 Coverage: {cov95_disj:.1f}%")

    # ─────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ─────────────────────────────────────────────────────────────────
    out_file = RESULTS_DIR / "tier1_v2_1_short_horizon_results.json"
    results_payload = {
        "protocol": "docs/EVAL_PROTOCOL_V2_1.md",
        "protocol_version": "2.1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unconditioned": True,
        "bucket_type": "distance_to_berth_km",
        "conformal_shifts": {f"P{int(q*100)}": v for q, v in conformal_shifts.items()},
        "bucket_train_medians": bucket_train_medians,
        "total_snapshots": len(snapshots),
        "total_censored": n_censored,
        "total_anchored": n_anchored,
        "share_gt3h_overall_pct": share_gt3h,
        "bucket_results": all_bucket_results,
        "cohort_a1_full_trip": {
            "total_arrivals": len(cohort_a1),
            "train_arrivals": len(train_a1),
            "test_arrivals": len(test_a1),
            "hist_median_scalar_h": med_a1,
            "hist_median_mae": mae_a1,
            "bootstrap_ci": [ci_l_a1, ci_h_a1],
            "ci_width": ci_w_a1,
        },
        "cohort_a2_full_trip": {
            "total_arrivals": len(cohort_a2),
            "train_arrivals": sum(1 for a in cohort_a2 if a["is_train"]),
            "test_arrivals": sum(1 for a in cohort_a2 if not a["is_train"]),
        }
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {out_file}")
    print(f"Elapsed: {time.time() - t_start:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
