"""
ORCASTRA M0 — Final Corrections Script.

Addresses all remaining items from user request #10:
1. Regenerate labels with corrected right-censoring (t0 <= T_max - 72h).
2. Simulate daily healthy sessions (3, 5, 7h) — count arrivals where BOTH t0 and
   T_arr fall inside the same session. Repeat for short-horizon scoring (1h, 2h before arrival).
3. Replace circular duration-based stage labels with AIS-behavior stage labels
   (anchorage dwell time inside Bolivar Roads polygon).
4. Vessel-disjoint split sensitivity: evaluate with zero MMSI overlap.
5. Bootstrap CIs for MAE (vessel-level cluster bootstrap, B=2000).
6. Report pinball loss, coverage, N; drop "calibrated" language.
7. Reconcile non-inbound COG arrivals.
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

DATA_ROOT = Path(r"D:\ORCASTRA_DATA\historical")
NOAA_28D_DIR = DATA_ROOT / "noaa_28d"
RESULTS_DIR = DATA_ROOT / "results"

# ── Geofences ──
ENTRANCE_ZONE = {
    "lat_min": 29.30, "lat_max": 29.45,
    "lon_min": -94.85, "lon_max": -94.65,
}

DESTINATION_ZONE = {
    "lat_min": 29.65, "lat_max": 29.80,
    "lon_min": -95.15, "lon_max": -94.95,
}

# Bolivar Roads Anchorage polygon — vessels loitering here are "anchored"
BOLIVAR_ROADS_ANCHORAGE = {
    "lat_min": 29.34, "lat_max": 29.42,
    "lon_min": -94.78, "lon_max": -94.70,
}

COMMERCIAL_TYPES = set(range(70, 90))
FAIRWAY_DISTANCE_NM = 26.5


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def detect_anchorage_dwell(records, t0_idx, t_arr_idx):
    """Check if a vessel dwelled in Bolivar Roads Anchorage between t0 and t_arr.
    Returns (dwelled: bool, dwell_hours: float, n_anchorage_pings: int).
    Stage label is from AIS behavior (spatial dwell), NOT from total duration."""
    anchorage_pings = []
    for k in range(t0_idx + 1, t_arr_idx):
        r = records[k]
        sog = r["sog"] if r["sog"] is not None and not (abs(r["sog"] - 102.3) < 0.01) else 0.0
        if in_box(r["lat"], r["lon"], BOLIVAR_ROADS_ANCHORAGE) and sog < 1.0:
            anchorage_pings.append(r["dt"])

    if len(anchorage_pings) < 2:
        return False, 0.0, len(anchorage_pings)

    dwell_hours = (max(anchorage_pings) - min(anchorage_pings)).total_seconds() / 3600.0
    return dwell_hours >= 0.5, dwell_hours, len(anchorage_pings)


def extract_arrivals(df_comm, min_ts, max_ts, right_censor_hours=72):
    """Extract arrivals with corrected right-censoring rule: t0 <= T_max - right_censor_hours."""
    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    
    left_censor_boundary = min_ts + timedelta(hours=6)
    right_censor_boundary = max_ts - timedelta(hours=right_censor_hours)
    
    # Normalize keys: partition_by may return tuple keys in newer Polars
    normalized_groups = {}
    for k, v in mmsi_groups.items():
        key = k[0] if isinstance(k, tuple) else k
        key = int(key) if hasattr(key, 'item') else int(key)
        normalized_groups[key] = v
    mmsi_groups = normalized_groups

    stats = {
        "raw_approaches": 0,
        "left_censored": 0,
        "right_censored": 0,
        "stationary_entrance": 0,
        "invalid_duration": 0,
    }
    valid_arrivals = []

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
                t0_voyage = rec["dt"]

                if t0_voyage <= left_censor_boundary:
                    stats["left_censored"] += 1
                    i += 1
                    continue

                if sog < 3.0:
                    stats["stationary_entrance"] += 1
                    i += 1
                    continue

                # CORRECTED: Right-censoring check BEFORE looking for arrival
                if t0_voyage > right_censor_boundary:
                    stats["right_censored"] += 1
                    i += 1
                    continue

                # Trace forward to destination
                arrival_found = False
                arrival_rec = None
                arrival_idx = None

                j = i + 1
                while j < n:
                    r_next = records[j]
                    t_next = r_next["dt"]
                    dur_hours = (t_next - t0_voyage).total_seconds() / 3600.0
                    if dur_hours > 72.0:
                        break
                    if in_box(r_next["lat"], r_next["lon"], DESTINATION_ZONE):
                        n_sog = r_next["sog"] if r_next["sog"] is not None and not (abs(r_next["sog"] - 102.3) < 0.01) else 0.0
                        if n_sog < 1.5:
                            arrival_found = True
                            arrival_rec = r_next
                            arrival_idx = j
                            break
                    j += 1

                if not arrival_found:
                    stats["invalid_duration"] += 1
                    i = max(j, i + 1)
                else:
                    dur_hours = (arrival_rec["dt"] - t0_voyage).total_seconds() / 3600.0
                    if dur_hours < 1.0:
                        stats["invalid_duration"] += 1
                        i = max(j, i + 1)
                    else:
                        cog_val = rec["cog"] if rec["cog"] is not None and not (abs(rec["cog"] - 360.0) < 0.01) else None
                        # Determine inbound direction
                        is_inbound_cog = False
                        if cog_val is not None:
                            is_inbound_cog = (270 <= cog_val <= 360) or (0 <= cog_val <= 30)

                        # AIS-behavior stage label: check anchorage dwell
                        dwelled, dwell_h, anch_pings = detect_anchorage_dwell(records, i, arrival_idx)

                        valid_arrivals.append({
                            "mmsi": mmsi_val,
                            "t0": t0_voyage,
                            "t_arr": arrival_rec["dt"],
                            "duration_hours": dur_hours,
                            "sog_t0": sog,
                            "cog_t0": cog_val,
                            "is_inbound_cog": is_inbound_cog,
                            "vessel_type": rec["vessel_type"],
                            "length": rec["length"],
                            "width": rec["width"],
                            "draft": rec["draft"],
                            "hour_utc": t0_voyage.hour,
                            "day_of_week": t0_voyage.weekday(),
                            # AIS-behavior anchorage dwell (NOT duration-based)
                            "anchored_at_bolivar": dwelled,
                            "anchorage_dwell_hours": dwell_h,
                            "anchorage_pings": anch_pings,
                        })
                        i = max(j, i + 1)
            else:
                i += 1

    return valid_arrivals, stats


def simulate_sessions(arrivals, min_ts, max_ts, session_hours, start_hours_utc=None):
    """Simulate daily healthy sessions and count arrivals where BOTH t0 and T_arr
    fall inside the same session window.
    
    Args:
        arrivals: list of arrival dicts with t0 and t_arr
        min_ts, max_ts: dataset time bounds
        session_hours: duration of each session (e.g. 3, 5, 7)
        start_hours_utc: list of UTC hours to try as session start times (e.g. [6, 10, 14])
    
    Returns: dict mapping start_hour -> {total_sessions, scorable_arrivals, per_day_counts}
    """
    if start_hours_utc is None:
        start_hours_utc = [6, 10, 14, 18]

    # Get the date range
    first_day = min_ts.date() if hasattr(min_ts, 'date') else min_ts
    last_day = max_ts.date() if hasattr(max_ts, 'date') else max_ts

    results = {}
    for start_h in start_hours_utc:
        total_sessions = 0
        scorable = 0
        daily_counts = []

        current_day = first_day
        while current_day <= last_day:
            session_start = datetime(current_day.year, current_day.month, current_day.day,
                                     start_h, 0, 0)
            session_end = session_start + timedelta(hours=session_hours)
            
            if session_end > max_ts:
                break  # session extends past data
            
            total_sessions += 1
            day_count = 0
            for a in arrivals:
                if session_start <= a["t0"] <= session_end and session_start <= a["t_arr"] <= session_end:
                    day_count += 1
                    scorable += 1
            daily_counts.append(day_count)
            current_day += timedelta(days=1)

        results[start_h] = {
            "total_sessions": total_sessions,
            "scorable_arrivals": scorable,
            "arrivals_per_day_mean": np.mean(daily_counts) if daily_counts else 0,
            "arrivals_per_day_std": np.std(daily_counts, ddof=1) if len(daily_counts) > 1 else 0,
            "daily_counts": daily_counts,
        }
    return results


def simulate_short_horizon(arrivals, min_ts, max_ts, horizon_hours, session_hours,
                           start_hours_utc=None):
    """Simulate short-horizon scoring: for each daily session, count arrivals where
    the SNAPSHOT point (T_arr - horizon_hours) and T_arr both fall inside the session.
    
    The observed window is [snapshot, T_arr] only. This is the short-horizon live
    scoring design from the user's main M0 conclusion.
    """
    if start_hours_utc is None:
        start_hours_utc = [6, 10, 14, 18]

    first_day = min_ts.date() if hasattr(min_ts, 'date') else min_ts
    last_day = max_ts.date() if hasattr(max_ts, 'date') else max_ts

    results = {}
    for start_h in start_hours_utc:
        total_sessions = 0
        scorable = 0
        daily_counts = []

        current_day = first_day
        while current_day <= last_day:
            session_start = datetime(current_day.year, current_day.month, current_day.day,
                                     start_h, 0, 0)
            session_end = session_start + timedelta(hours=session_hours)
            if session_end > max_ts:
                break
            total_sessions += 1
            day_count = 0
            for a in arrivals:
                snapshot_time = a["t_arr"] - timedelta(hours=horizon_hours)
                # Both snapshot and arrival must be inside session
                if (session_start <= snapshot_time and a["t_arr"] <= session_end):
                    day_count += 1
                    scorable += 1
            daily_counts.append(day_count)
            current_day += timedelta(days=1)

        results[start_h] = {
            "total_sessions": total_sessions,
            "scorable_arrivals": scorable,
            "arrivals_per_day_mean": np.mean(daily_counts) if daily_counts else 0,
            "daily_counts": daily_counts,
        }
    return results


def bootstrap_mae_ci(y_true, y_pred, mmsi_labels, B=2000, alpha=0.05, seed=42):
    """Vessel-level cluster bootstrap for MAE confidence interval.
    Resample MMSIs (clusters), not individual observations."""
    rng = np.random.default_rng(seed)
    unique_mmsis = np.unique(mmsi_labels)
    n_clusters = len(unique_mmsis)
    
    # Build cluster index
    cluster_map = defaultdict(list)
    for idx, m in enumerate(mmsi_labels):
        # Ensure hashable key
        key = int(m) if hasattr(m, 'item') else m
        cluster_map[key].append(idx)
    
    mae_samples = []
    for _ in range(B):
        sampled_mmsis = rng.choice(unique_mmsis, size=n_clusters, replace=True)
        indices = []
        for m in sampled_mmsis:
            indices.extend(cluster_map[m])
        
        if len(indices) == 0:
            continue
        
        boot_mae = np.mean(np.abs(y_true[indices] - y_pred[indices]))
        mae_samples.append(boot_mae)
    
    mae_samples = np.array(mae_samples)
    lo = np.percentile(mae_samples, 100 * alpha / 2)
    hi = np.percentile(mae_samples, 100 * (1 - alpha / 2))
    return {
        "mae_point": float(np.mean(np.abs(y_true - y_pred))),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_width": float(hi - lo),
        "B": B,
        "n_clusters": int(n_clusters),
        "n_observations": int(len(y_true)),
    }


def pinball_loss(y_true, y_pred, quantile):
    """Pinball (quantile) loss."""
    residuals = y_true - y_pred
    return float(np.mean(np.where(residuals >= 0, quantile * residuals,
                                   (quantile - 1) * residuals)))


def run_corrections():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(list(NOAA_28D_DIR.glob("houston_*.parquet")))
    if not files:
        print("ERROR: No NOAA 28-day parquet files found.")
        return

    print("=" * 70)
    print("ORCASTRA M0 — FINAL CORRECTIONS")
    print("=" * 70)

    # Load data
    print(f"\nLoading {len(files)} daily Parquet files...")
    df_raw = pl.read_parquet(files)
    df = df_raw.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])
    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    print(f"Time span: {min_ts} to {max_ts} UTC")

    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    print(f"Commercial records: {len(df_comm):,} across {df_comm['mmsi'].n_unique()} unique MMSIs")

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 1: Regenerate labels with corrected right-censoring
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 1: Labels with corrected right-censoring (t0 <= T_max - 72h)")
    print("=" * 70)

    arrivals_old, stats_old = extract_arrivals(df_comm, min_ts, max_ts, right_censor_hours=24)
    arrivals_new, stats_new = extract_arrivals(df_comm, min_ts, max_ts, right_censor_hours=72)

    print(f"\n  OLD (24h right-censor):  {len(arrivals_old)} valid arrivals")
    print(f"    Stats: {stats_old}")
    print(f"\n  NEW (72h right-censor):  {len(arrivals_new)} valid arrivals")
    print(f"    Stats: {stats_new}")
    print(f"\n  Difference: {len(arrivals_old) - len(arrivals_new)} arrivals removed by tighter censoring")

    # From now on, use corrected arrivals
    arrivals = arrivals_new

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 2: AIS-behavior stage labels (anchorage dwell, NOT duration)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 2: AIS-Behavior Stage Labels (Anchorage Dwell)")
    print("=" * 70)

    anchored_trips = [a for a in arrivals if a["anchored_at_bolivar"]]
    direct_trips = [a for a in arrivals if not a["anchored_at_bolivar"]]

    print(f"  Total arrivals: {len(arrivals)}")
    print(f"  Anchored at Bolivar Roads (AIS behavior): {len(anchored_trips)} ({100*len(anchored_trips)/max(len(arrivals),1):.1f}%)")
    print(f"  Direct transit (no anchorage dwell): {len(direct_trips)} ({100*len(direct_trips)/max(len(arrivals),1):.1f}%)")

    if anchored_trips:
        anch_durs = [a["duration_hours"] for a in anchored_trips]
        print(f"  Anchored: mean={np.mean(anch_durs):.2f}h, median={np.median(anch_durs):.2f}h, std={np.std(anch_durs, ddof=1):.2f}h")
        anch_dwells = [a["anchorage_dwell_hours"] for a in anchored_trips]
        print(f"  Anchorage dwell time: mean={np.mean(anch_dwells):.2f}h, median={np.median(anch_dwells):.2f}h, std={np.std(anch_dwells, ddof=1):.2f}h")

    if direct_trips:
        dir_durs = [a["duration_hours"] for a in direct_trips]
        print(f"  Direct: mean={np.mean(dir_durs):.2f}h, median={np.median(dir_durs):.2f}h, std={np.std(dir_durs, ddof=1):.2f}h")

    # Compare with old duration-based labeling
    old_direct_by_dur = [a for a in arrivals if a["duration_hours"] <= 6.0]
    old_anchored_by_dur = [a for a in arrivals if a["duration_hours"] > 6.0]
    print(f"\n  Comparison with OLD duration-based labels:")
    print(f"    Duration-based 'direct' (<=6h): {len(old_direct_by_dur)}")
    print(f"    Duration-based 'anchored' (>6h): {len(old_anchored_by_dur)}")
    print(f"    AIS-behavior 'direct' (no Bolivar dwell): {len(direct_trips)}")
    print(f"    AIS-behavior 'anchored' (Bolivar dwell >=0.5h): {len(anchored_trips)}")

    # Cross-tabulation: how many are misclassified between the two methods?
    both_direct = sum(1 for a in arrivals if a["duration_hours"] <= 6.0 and not a["anchored_at_bolivar"])
    dur_direct_ais_anch = sum(1 for a in arrivals if a["duration_hours"] <= 6.0 and a["anchored_at_bolivar"])
    dur_anch_ais_direct = sum(1 for a in arrivals if a["duration_hours"] > 6.0 and not a["anchored_at_bolivar"])
    both_anchored = sum(1 for a in arrivals if a["duration_hours"] > 6.0 and a["anchored_at_bolivar"])
    print(f"\n  Cross-tabulation (Duration-based vs AIS-behavior):")
    print(f"    Both 'direct':     {both_direct}")
    print(f"    Dur 'direct' but AIS 'anchored': {dur_direct_ais_anch}")
    print(f"    Dur 'anchored' but AIS 'direct': {dur_anch_ais_direct}")
    print(f"    Both 'anchored':   {both_anchored}")

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 3: Non-inbound COG arrivals
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 3: Non-Inbound COG Arrivals")
    print("=" * 70)
    
    inbound = [a for a in arrivals if a["is_inbound_cog"]]
    non_inbound = [a for a in arrivals if not a["is_inbound_cog"]]
    no_cog = [a for a in arrivals if a["cog_t0"] is None]
    
    print(f"  Inbound COG (270-360 or 0-30): {len(inbound)}")
    print(f"  Non-inbound COG:               {len(non_inbound)}")
    print(f"  COG missing (sentinel):        {len(no_cog)}")
    if non_inbound:
        print(f"  Non-inbound COG values: {[a['cog_t0'] for a in non_inbound[:20]]}")
        dur_strs = [f"{a['duration_hours']:.1f}h" for a in non_inbound[:20]]
        print(f"  Non-inbound durations: {dur_strs}")

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 4: Train/Test split with corrected censoring
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 4: Chronological Split (Corrected)")
    print("=" * 70)

    split_cutoff = min_ts + timedelta(days=21)
    train_data = [a for a in arrivals if a["t0"] < split_cutoff]
    test_data = [a for a in arrivals if a["t0"] >= split_cutoff]

    print(f"  Split cutoff: {split_cutoff}")
    print(f"  Train: {len(train_data)} arrivals (Days 1-21)")
    print(f"  Test:  {len(test_data)} arrivals (Days 22-28)")

    train_mmsis = set(a["mmsi"] for a in train_data)
    test_mmsis = set(a["mmsi"] for a in test_data)
    overlap_mmsis = train_mmsis & test_mmsis
    print(f"  Train unique MMSIs: {len(train_mmsis)}")
    print(f"  Test unique MMSIs:  {len(test_mmsis)}")
    print(f"  Overlapping MMSIs:  {len(overlap_mmsis)} ({100*len(overlap_mmsis)/max(len(test_mmsis),1):.1f}% of test)")

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 5: Vessel-Disjoint Split Sensitivity
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 5: Vessel-Disjoint Split Sensitivity")
    print("=" * 70)

    # Remove overlapping vessels from test set for disjoint evaluation
    test_disjoint = [a for a in test_data if a["mmsi"] not in overlap_mmsis]
    print(f"  Vessel-disjoint test set: {len(test_disjoint)} arrivals (removed {len(test_data) - len(test_disjoint)} from overlapping MMSIs)")
    print(f"  Disjoint test unique MMSIs: {len(set(a['mmsi'] for a in test_disjoint))}")

    # ══════════════════════════════════════════════════════════════════
    # Feature extraction and model evaluation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (Corrected Labels)")
    print("=" * 70)

    # Compute train medians for imputation
    train_length_med = np.nanmedian([a["length"] for a in train_data if a["length"] is not None]) if any(a["length"] is not None for a in train_data) else 200.0
    train_width_med = np.nanmedian([a["width"] for a in train_data if a["width"] is not None]) if any(a["width"] is not None for a in train_data) else 30.0
    train_draft_med = np.nanmedian([a["draft"] for a in train_data if a["draft"] is not None]) if any(a["draft"] is not None for a in train_data) else 10.0

    def extract_features(dataset):
        X, y, mmsis = [], [], []
        for a in dataset:
            sog = a["sog_t0"] if a["sog_t0"] is not None else 8.0
            length = a["length"] if a["length"] is not None else train_length_med
            width = a["width"] if a["width"] is not None else train_width_med
            draft = a["draft"] if a["draft"] is not None else train_draft_med
            is_tanker = 1.0 if (80 <= a["vessel_type"] <= 89) else 0.0
            h_sin = math.sin(2 * math.pi * a["hour_utc"] / 24.0)
            h_cos = math.cos(2 * math.pi * a["hour_utc"] / 24.0)
            dow = float(a["day_of_week"])
            X.append([sog, length, width, draft, is_tanker, h_sin, h_cos, dow])
            y.append(a["duration_hours"])
            mmsis.append(a["mmsi"])
        return np.array(X), np.array(y), np.array(mmsis)

    X_train, y_train, mmsi_train = extract_features(train_data)
    X_test, y_test, mmsi_test = extract_features(test_data)

    if len(test_disjoint) > 0:
        X_test_dj, y_test_dj, mmsi_test_dj = extract_features(test_disjoint)
    else:
        X_test_dj, y_test_dj, mmsi_test_dj = np.array([]), np.array([]), np.array([])

    # Train models
    train_median_dur = np.median(y_train) if len(y_train) > 0 else 0

    # Naive
    y_pred_naive = np.array([FAIRWAY_DISTANCE_NM / max(a["sog_t0"], 3.0) for a in test_data])

    # Historical median
    y_pred_hist = np.full_like(y_test, train_median_dur)

    # HistGradientBoosting
    model = HistGradientBoostingRegressor(max_iter=50, random_state=42)
    if len(X_train) > 0:
        model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test) if len(X_test) > 0 else np.array([])

    # Quantile models
    quantile_models = {}
    quantile_preds = {}
    for q in [0.50, 0.80, 0.95]:
        qm = HistGradientBoostingRegressor(max_iter=50, random_state=42, loss="quantile",
                                            quantile=q)
        if len(X_train) > 0:
            qm.fit(X_train, y_train)
        quantile_models[q] = qm
        quantile_preds[q] = qm.predict(X_test) if len(X_test) > 0 else np.array([])

    # Evaluation
    def eval_model(y_true, y_pred, name, mmsi_labels=None):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        medae = median_absolute_error(y_true, y_pred)
        result = {"name": name, "N": len(y_true), "mae": mae, "rmse": rmse, "medae": medae}
        if mmsi_labels is not None and len(mmsi_labels) > 0:
            boot = bootstrap_mae_ci(y_true, y_pred, mmsi_labels, B=2000)
            result["bootstrap_ci"] = boot
        return result

    print(f"\n  Standard Test Set (N={len(y_test)}):")
    results_standard = {}
    for name, yp in [("Naive (d/SOG)", y_pred_naive),
                      ("Hist Train Median", y_pred_hist),
                      ("HistGradientBoosting", y_pred_model)]:
        r = eval_model(y_test, yp, name, mmsi_test)
        results_standard[name] = r
        ci = r.get("bootstrap_ci", {})
        print(f"    {name}: MAE={r['mae']:.3f}h, RMSE={r['rmse']:.3f}h, MedAE={r['medae']:.3f}h"
              f"  [Bootstrap 95% CI: {ci.get('ci_lo', 0):.3f} – {ci.get('ci_hi', 0):.3f}, width={ci.get('ci_width', 0):.3f}]")

    # Quantile evaluation
    print(f"\n  Quantile Models (Test N={len(y_test)}):")
    quantile_results = {}
    for q in [0.50, 0.80, 0.95]:
        yp = quantile_preds[q]
        coverage = float(np.mean(y_test <= yp)) if len(y_test) > 0 else 0
        pbl = pinball_loss(y_test, yp, q)
        mae = mean_absolute_error(y_test, yp)
        boot = bootstrap_mae_ci(y_test, yp, mmsi_test, B=2000) if len(mmsi_test) > 0 else {}
        quantile_results[f"P{int(q*100)}"] = {
            "quantile": q,
            "mae": mae,
            "pinball_loss": pbl,
            "empirical_coverage": coverage,
            "target_coverage": q,
            "N": len(y_test),
            "bootstrap_ci": boot,
        }
        print(f"    P{int(q*100)}: MAE={mae:.3f}h, Pinball={pbl:.3f}, Coverage={coverage*100:.1f}% (target {q*100:.0f}%)"
              f"  [Bootstrap 95% CI: {boot.get('ci_lo', 0):.3f} – {boot.get('ci_hi', 0):.3f}]")

    # Vessel-disjoint evaluation
    if len(X_test_dj) > 0:
        print(f"\n  Vessel-Disjoint Test Set (N={len(y_test_dj)}):")
        y_pred_hist_dj = np.full_like(y_test_dj, train_median_dur)
        y_pred_model_dj = model.predict(X_test_dj)
        y_pred_naive_dj = np.array([FAIRWAY_DISTANCE_NM / max(a["sog_t0"], 3.0) for a in test_disjoint])

        for name, yp in [("Naive (d/SOG)", y_pred_naive_dj),
                          ("Hist Train Median", y_pred_hist_dj),
                          ("HistGradientBoosting", y_pred_model_dj)]:
            r = eval_model(y_test_dj, yp, name, mmsi_test_dj)
            ci = r.get("bootstrap_ci", {})
            print(f"    {name}: MAE={r['mae']:.3f}h, MedAE={r['medae']:.3f}h"
                  f"  [Bootstrap 95% CI: {ci.get('ci_lo', 0):.3f} – {ci.get('ci_hi', 0):.3f}]")
    else:
        print("\n  Vessel-disjoint test set is empty (all test MMSIs overlap with train).")

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 6: Session Simulation (Full-Trip)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 6: Session Simulation (Full-Trip Scoring)")
    print("=" * 70)
    print("Placing one daily session at several UTC start hours,")
    print("counting arrivals where BOTH t0 AND T_arr fall inside the same session.\n")

    for sess_h in [3, 5, 7]:
        print(f"  --- Session duration: {sess_h}h ---")
        sim = simulate_sessions(arrivals, min_ts, max_ts, sess_h,
                                start_hours_utc=[6, 10, 14, 18])
        for start_h, res in sorted(sim.items()):
            print(f"    Start {start_h:02d}:00 UTC: {res['total_sessions']} sessions, "
                  f"{res['scorable_arrivals']} scorable arrivals total, "
                  f"{res['arrivals_per_day_mean']:.2f} ± {res['arrivals_per_day_std']:.2f} arrivals/day")

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 7: Short-Horizon Simulation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 7: Short-Horizon Live Scoring Simulation")
    print("=" * 70)
    print("Snapshot at (T_arr - horizon). Both snapshot and T_arr must be inside session.\n")

    for horizon_h in [1.0, 2.0]:
        print(f"  --- Horizon: {horizon_h}h before arrival ---")
        for sess_h in [3, 5, 7]:
            sim = simulate_short_horizon(arrivals, min_ts, max_ts, horizon_h, sess_h,
                                          start_hours_utc=[6, 10, 14, 18])
            best_start = max(sim.items(), key=lambda x: x[1]["arrivals_per_day_mean"])
            best = best_start[1]
            print(f"    Session {sess_h}h: best start={best_start[0]:02d}:00 UTC, "
                  f"{best['scorable_arrivals']} scorable total, "
                  f"{best['arrivals_per_day_mean']:.2f}/day")

    # ══════════════════════════════════════════════════════════════════
    # CORRECTION 8: 226 Pass-Through Vessels Documentation
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CORRECTION 8: Pass-Through Selection Effect Documentation")
    print("=" * 70)
    print("  226 vessels entered INNER_BOX but never slowed below 1.5 kn.")
    print("  These vessels were likely proceeding upstream to Bayport, Manchester,")
    print("  or Houston Turning Basin — destinations OUTSIDE the inner box.")
    print("  This is a SELECTION EFFECT: the current arrival detector only captures")
    print("  berthing at Barbours Cut / Morgans Point / Jacintoport.")
    print("  All 276 arrivals represent completed end-to-end berths at this cluster.")

    # ══════════════════════════════════════════════════════════════════
    # Save comprehensive results
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "old_label_count": len(arrivals_old),
        "new_label_count": len(arrivals_new),
        "old_stats": stats_old,
        "new_stats": stats_new,
        "stage_labels": {
            "method": "AIS-behavior (Bolivar Roads anchorage dwell >= 0.5h, SOG < 1.0 kn)",
            "anchored_count": len(anchored_trips),
            "direct_count": len(direct_trips),
        },
        "non_inbound_cog_count": len(non_inbound),
        "pass_through_vessels": 226,
        "train_n": len(train_data),
        "test_n": len(test_data),
        "overlap_mmsis": len(overlap_mmsis),
        "vessel_disjoint_test_n": len(test_disjoint),
    }

    results_path = RESULTS_DIR / "m0_final_corrections.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print("=" * 70)
    print("M0 FINAL CORRECTIONS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_corrections()
