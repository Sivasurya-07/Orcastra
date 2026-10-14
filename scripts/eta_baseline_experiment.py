"""
ORCASTRA M0 — ETA Baseline Feasibility Experiment.

Deliverables:
A. Label audit: raw approaches, exclusions (left/right censored, non-commercial, moored),
   completed arrivals, usable prediction examples.
B. Sample size: 28 days of NOAA 2024 Houston data.
C. Point-in-time features: strict no-leakage feature extraction at entrance timestamp t0.
D. Baseline models:
   - Naive (distance / entrance SOG)
   - Historical Train Median
   - HistGradientBoostingRegressor
   - Chronological split: Days 1-21 Train, Days 22-28 Test.
   - Evaluated on test set: MAE, RMSE, MedAE, residual distributions.
E. Healthy-session and healthy-window definitions.
F. CI-width gate sample size using real baseline test residual sigma.
"""

import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

DATA_ROOT = Path(r"D:\ORCASTRA_DATA\historical")
NOAA_28D_DIR = DATA_ROOT / "noaa_28d"
RESULTS_DIR = DATA_ROOT / "results"

# ── Geofences & Boundaries ──
# HSC Entrance Corridor (inbound entry point from Gulf)
ENTRANCE_ZONE = {
    "lat_min": 29.30, "lat_max": 29.45,
    "lon_min": -94.85, "lon_max": -94.65,
}

# Inner Destination / Berths Zone (Barbours Cut, Bayport, Turning Basin, Morgans Point)
DESTINATION_ZONE = {
    "lat_min": 29.65, "lat_max": 29.80,
    "lon_min": -95.15, "lon_max": -94.95,
}

# Fairway distance from entrance corridor centroid to inner berths centroid
# Centroid Entrance: (29.375, -94.750)
# Centroid Berths: (29.725, -95.050)
# Channel navigation distance is ~26.5 nautical miles (~49.1 km)
FAIRWAY_DISTANCE_NM = 26.5

# Commercial vessel types (IMO / AIS standards):
# 70-79: Cargo (Container, Bulk, General Cargo)
# 80-89: Tanker (Crude, Product, Chemical, LNG/LPG)
COMMERCIAL_TYPES = set(range(70, 90))


def run_experiment():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(list(NOAA_28D_DIR.glob("houston_*.parquet")))
    if not files:
        print("ERROR: No NOAA 28-day parquet files found.")
        return

    print("=" * 70)
    print("ORCASTRA M0 — ETA BASELINE FEASIBILITY EXPERIMENT")
    print("=" * 70)
    print(f"Loaded {len(files)} daily Parquet files from {NOAA_28D_DIR}")
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    print(f"Total dataset disk footprint: {total_mb:.2f} MB ({total_mb/1024:.3f} GB)")

    # ──────────────────────────────────────────────────────────────────
    # Step 1: Scan and Index Commercial Vessel Voyages
    # ──────────────────────────────────────────────────────────────────
    print("\n--- Step 1: Scanning Commercial AIS Trajectories ---")
    t0_scan = time.time()

    # Read all files into Polars
    # Filter directly on commercial types to save memory and process cleanly
    df_raw = pl.read_parquet(files)
    total_raw_rows = len(df_raw)
    print(f"Total rows in Houston bbox: {total_raw_rows:,}")

    # Parse timestamps
    df = df_raw.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])

    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    print(f"Dataset time span: {min_ts} to {max_ts} UTC")

    # Filter to commercial cargo/tanker vessels
    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    comm_rows = len(df_comm)
    comm_vessels = df_comm["mmsi"].n_unique()
    non_comm_rows = total_raw_rows - comm_rows
    print(f"Commercial records: {comm_rows:,} ({comm_rows/total_raw_rows*100:.1f}%) across {comm_vessels} unique MMSIs")
    print(f"Non-commercial / service records excluded: {non_comm_rows:,}")

    # ──────────────────────────────────────────────────────────────────
    # Step 2: Label Audit — Trajectory Segmentation & Arrival Detection
    # ──────────────────────────────────────────────────────────────────
    print("\n--- Step 2: Trajectory Segmentation & Label Audit ---")

    # Approach:
    # A commercial vessel voyage begins when entering ENTRANCE_ZONE with SOG >= 5.0 kn.
    # A voyage completes arrival when entering DESTINATION_ZONE with SOG < 1.5 kn.
    # Exclude:
    # 1. Left-censored: vessel already in channel at start of 28d period (dt < min_ts + 6h)
    # 2. Right-censored: vessel entered entrance zone near end (dt > max_ts - 24h) without completed arrival
    # 3. Moored / Stationary at entrance: SOG < 3.0 kn throughout entrance
    # 4. Out-of-bounds duration: transit duration < 1.0h (spatial error) or > 72.0h (prolonged anchorage)

    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)

    raw_approaches_count = 0
    left_censored_count = 0
    right_censored_count = 0
    stationary_entrance_count = 0
    invalid_duration_count = 0
    valid_arrivals = []

    left_censor_boundary = min_ts + timedelta(hours=6)
    right_censor_boundary = max_ts - timedelta(hours=24)

    for mmsi_val, vdf in mmsi_groups.items():
        vdf = vdf.sort("dt")
        records = vdf.to_dicts()

        # Identify entrances into entrance corridor
        i = 0
        n = len(records)
        while i < n:
            rec = records[i]
            lat, lon = rec["lat"], rec["lon"]
            sog = rec["sog"] if rec["sog"] is not None and not (abs(rec["sog"] - 102.3) < 0.01) else 0.0

            # Check if in entrance zone
            in_ent = (ENTRANCE_ZONE["lat_min"] <= lat <= ENTRANCE_ZONE["lat_max"] and
                      ENTRANCE_ZONE["lon_min"] <= lon <= ENTRANCE_ZONE["lon_max"])

            if in_ent:
                raw_approaches_count += 1
                t0_voyage = rec["dt"]

                # Left-censoring check
                if t0_voyage <= left_censor_boundary:
                    left_censored_count += 1
                    i += 1
                    continue

                # Check if moving at transit speed
                if sog < 3.0:
                    stationary_entrance_count += 1
                    i += 1
                    continue

                # Vessel is an inbound approach candidate! Now trace forward to see if it reaches destination
                arrival_found = False
                arrival_rec = None

                j = i + 1
                while j < n:
                    r_next = records[j]
                    t_next = r_next["dt"]
                    dur_hours = (t_next - t0_voyage).total_seconds() / 3600.0

                    if dur_hours > 72.0:
                        # Exceeded maximum plausible transit/anchorage window
                        break

                    n_lat, n_lon = r_next["lat"], r_next["lon"]
                    n_sog = r_next["sog"] if r_next["sog"] is not None and not (abs(r_next["sog"] - 102.3) < 0.01) else 0.0

                    in_dest = (DESTINATION_ZONE["lat_min"] <= n_lat <= DESTINATION_ZONE["lat_max"] and
                               DESTINATION_ZONE["lon_min"] <= n_lon <= DESTINATION_ZONE["lon_max"])

                    if in_dest and n_sog < 1.5:
                        arrival_found = True
                        arrival_rec = r_next
                        break

                    j += 1

                if not arrival_found:
                    # Check if right censored
                    if t0_voyage >= right_censor_boundary:
                        right_censored_count += 1
                    else:
                        invalid_duration_count += 1
                    i = j
                else:
                    dur_hours = (arrival_rec["dt"] - t0_voyage).total_seconds() / 3600.0
                    if dur_hours < 1.0:
                        invalid_duration_count += 1
                    else:
                        # Valid arrival found!
                        valid_arrivals.append({
                            "mmsi": mmsi_val,
                            "t0": t0_voyage,
                            "t_arr": arrival_rec["dt"],
                            "duration_hours": dur_hours,
                            "sog_t0": sog,
                            "cog_t0": rec["cog"] if rec["cog"] is not None and not (abs(rec["cog"] - 360.0) < 0.01) else None,
                            "vessel_type": rec["vessel_type"],
                            "length": rec["length"],
                            "width": rec["width"],
                            "draft": rec["draft"],
                            "hour_utc": t0_voyage.hour,
                            "day_of_week": t0_voyage.weekday(),
                        })
                    # Advance i past this voyage (at least 6 hours forward)
                    i = j
            else:
                i += 1

    print("\nLABEL AUDIT SUMMARY:")
    print(f"  Total raw approach zone entries detected: {raw_approaches_count}")
    print(f"  - Excluded: Left-censored (within first 6h): {left_censored_count}")
    print(f"  - Excluded: Right-censored (no arrival within 24h of end): {right_censored_count}")
    print(f"  - Excluded: Stationary/moored at entrance (SOG < 3.0 kn): {stationary_entrance_count}")
    print(f"  - Excluded: Incomplete / duration out-of-bounds (<1h or >72h): {invalid_duration_count}")
    print(f"  = Valid Completed Commercial Arrivals: {len(valid_arrivals)}")

    # Inspect repeated trips per MMSI
    trip_counts = {}
    for a in valid_arrivals:
        trip_counts[a["mmsi"]] = trip_counts.get(a["mmsi"], 0) + 1

    print(f"\nREPEATED TRIPS INSPECTION ({len(trip_counts)} unique vessels):")
    counts_dist = {}
    for c in trip_counts.values():
        counts_dist[c] = counts_dist.get(c, 0) + 1
    for num_trips, num_vessels in sorted(counts_dist.items()):
        print(f"  Vessels with {num_trips} completed arrival(s): {num_vessels}")

    # ──────────────────────────────────────────────────────────────────
    # Step 3: Chronological Split (Days 1-21 Train vs Days 22-28 Test)
    # ──────────────────────────────────────────────────────────────────
    print("\n--- Step 3: Chronological Train/Test Split ---")
    split_cutoff = min_ts + timedelta(days=21)
    print(f"Chronological split cutoff: {split_cutoff} UTC")
    print(f"  Train: {min_ts} to {split_cutoff} (Days 1–21)")
    print(f"  Test:  {split_cutoff} to {max_ts} (Days 22–28)")

    train_data = [a for a in valid_arrivals if a["t0"] < split_cutoff]
    test_data = [a for a in valid_arrivals if a["t0"] >= split_cutoff]

    print(f"Train sample size: {len(train_data)} arrivals")
    print(f"Test sample size:  {len(test_data)} arrivals")

    if len(train_data) < 20 or len(test_data) < 10:
        print("WARNING: Sample size is small for train/test split.")

    # ──────────────────────────────────────────────────────────────────
    # Step 4: Point-in-Time Feature Engineering & Missingness
    # ──────────────────────────────────────────────────────────────────
    print("\n--- Step 4: Point-in-Time Features & Missingness ---")

    # Features:
    # 1. entrance_sog
    # 2. length
    # 3. width
    # 4. draft
    # 5. vessel_type (binary: 1 if Tanker (80-89), 0 if Cargo (70-79))
    # 6. hour_utc (cyclical: sin, cos)
    # 7. day_of_week

    # Calculate train medians for imputation (leakage-safe: fit on train only)
    train_length_med = np.nanmedian([a["length"] for a in train_data if a["length"] is not None])
    train_width_med = np.nanmedian([a["width"] for a in train_data if a["width"] is not None])
    train_draft_med = np.nanmedian([a["draft"] for a in train_data if a["draft"] is not None])

    def extract_features(dataset):
        X = []
        y = []
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
        return np.array(X), np.array(y)

    X_train, y_train = extract_features(train_data)
    X_test, y_test = extract_features(test_data)

    # Missingness check
    all_arrivals = valid_arrivals
    n_all = len(all_arrivals)
    miss_len = sum(1 for a in all_arrivals if a["length"] is None)
    miss_wid = sum(1 for a in all_arrivals if a["width"] is None)
    miss_dra = sum(1 for a in all_arrivals if a["draft"] is None)
    miss_cog = sum(1 for a in all_arrivals if a["cog_t0"] is None)

    print("Feature Missingness:")
    print(f"  Length missing: {miss_len}/{n_all} ({miss_len/n_all*100:.1f}%)")
    print(f"  Width missing:  {miss_wid}/{n_all} ({miss_wid/n_all*100:.1f}%)")
    print(f"  Draft missing:  {miss_dra}/{n_all} ({miss_dra/n_all*100:.1f}%)")
    print(f"  COG missing:    {miss_cog}/{n_all} ({miss_cog/n_all*100:.1f}%)")
    print(f"  SOG missing:    0/{n_all} (0.0%)")

    # ──────────────────────────────────────────────────────────────────
    # Step 5: Model Training & Evaluation
    # ──────────────────────────────────────────────────────────────────
    print("\n--- Step 5: Model Training & Evaluation on Test Set ---")

    # Model 1: Naive Constant Velocity Reference (d / SOG)
    # Naive ETA duration: fairway distance / instantaneous entrance SOG
    y_pred_naive = np.array([FAIRWAY_DISTANCE_NM / max(a["sog_t0"], 3.0) for a in test_data])

    # Model 2: Historical Training Median Reference
    train_median_dur = np.median(y_train)
    y_pred_hist_med = np.full_like(y_test, train_median_dur)

    # Model 3: Trained Baseline Regressor (HistGradientBoostingRegressor)
    model = HistGradientBoostingRegressor(max_iter=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred_model = model.predict(X_test)

    # Compute Metrics on Test Set
    def calc_metrics(y_true, y_pred, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        medae = median_absolute_error(y_true, y_pred)
        residuals = y_true - y_pred
        res_mean = np.mean(residuals)
        res_std = np.std(residuals, ddof=1)
        res_skew = stats.skew(residuals)
        p10 = np.percentile(residuals, 10)
        p90 = np.percentile(residuals, 90)
        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "medae": medae,
            "res_mean": res_mean,
            "res_std": res_std,
            "res_skew": res_skew,
            "p10": p10,
            "p90": p90,
            "residuals": residuals,
        }

    m_naive = calc_metrics(y_test, y_pred_naive, "Naive (d / SOG)")
    m_hist = calc_metrics(y_test, y_pred_hist_med, "Historical Train Median")
    m_base = calc_metrics(y_test, y_pred_model, "Baseline (HistGradientBoosting)")

    print(f"\nTEST EVALUATION METRICS (N = {len(y_test)} arrivals):")
    print("-" * 75)
    print(f"{'Model':<30} | {'MAE (h)':<8} | {'RMSE (h)':<8} | {'MedAE (h)':<9} | {'Resid Std':<9}")
    print("-" * 75)
    for m in [m_naive, m_hist, m_base]:
        print(f"{m['name']:<30} | {m['mae']:<8.3f} | {m['rmse']:<8.3f} | {m['medae']:<9.3f} | {m['res_std']:<9.3f}")
    print("-" * 75)

    print(f"\nRESIDUAL DISTRIBUTION ({m_base['name']}):")
    print(f"  Mean error (bias):  {m_base['res_mean']:+.3f} hours")
    print(f"  Std deviation (σ): {m_base['res_std']:.3f} hours")
    print(f"  Skewness:          {m_base['res_skew']:.3f}")
    print(f"  P10 residual:      {m_base['p10']:+.3f} hours")
    print(f"  P90 residual:      {m_base['p90']:+.3f} hours")

    # ──────────────────────────────────────────────────────────────────
    # Step 6: CI-Width Gate Sample Size Projection
    # ──────────────────────────────────────────────────────────────────
    print("\n--- Step 6: CI-Width Gate Sample Size Projection ---")
    sigma = m_base["res_std"]
    print(f"Empirical residual sigma (σ) from baseline test set: {sigma:.3f} hours")
    print(f"Confidence Level: 95% (z = 1.96)")
    print("Formula: N = (1.96 * σ / δ)^2, where δ is desired margin of error (half CI width)")

    deltas = [0.25, 0.50, 1.00]
    ci_table = []
    for d in deltas:
        n_req = math.ceil((1.96 * sigma / d) ** 2)
        days_req = math.ceil(n_req / (len(valid_arrivals) / 28.0))
        ci_table.append({"delta": d, "n": n_req, "days": days_req})
        print(f"  Target margin δ = ±{d:.2f}h (95% CI width {2*d:.2f}h): N >= {n_req} arrivals (~{days_req} days)")

    # Save results to JSON
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_files": len(files),
        "total_disk_mb": total_mb,
        "total_raw_rows": total_raw_rows,
        "commercial_rows": comm_rows,
        "unique_commercial_vessels": comm_vessels,
        "label_audit": {
            "raw_approaches": raw_approaches_count,
            "left_censored": left_censored_count,
            "right_censored": right_censored_count,
            "stationary_entrance": stationary_entrance_count,
            "invalid_duration": invalid_duration_count,
            "valid_completed_arrivals": len(valid_arrivals),
        },
        "train_n": len(train_data),
        "test_n": len(test_data),
        "missingness": {
            "length": miss_len / n_all,
            "width": miss_wid / n_all,
            "draft": miss_dra / n_all,
        },
        "models": {
            "naive": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in m_naive.items() if k != "residuals"},
            "hist_median": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in m_hist.items() if k != "residuals"},
            "baseline": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v) for k, v in m_base.items() if k != "residuals"},
        },
        "ci_projections": ci_table,
    }

    with open(RESULTS_DIR / "eta_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ETA BASELINE FEASIBILITY EXPERIMENT COMPLETED")
    print(f"Saved results to: {RESULTS_DIR / 'eta_baseline_results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
