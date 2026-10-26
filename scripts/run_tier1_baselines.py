"""
ORCASTRA Tier 1 — Frozen Baseline Evaluation.

Conforms strictly to docs/EVAL_PROTOCOL.md:
- Train: Jan 1 – Feb 29, 2024 (60 days)
- Test: March 1 – March 31, 2024 (31 days)
- Test eligible: entrants with t0 <= Tmax - 72h (t0 <= 2024-03-28 23:59:59)
- Commercial types 70-89, SOG >= 3.0 kt at t0, Inbound COG [270-360] U [0-30]
- Baselines: Historical Train Median, Naive Kinematic (d/SOG), Quantile Regression (P50, P80, P95)
- Metrics: MAE, MedAE, Pinball Loss, Coverage
- Vessel-level cluster bootstrap 95% CIs (B=2,000)
- Vessel-disjoint sensitivity analysis
- Output saved to D:\\ORCASTRA_DATA\\historical\\results\\tier1_baseline_results.json
"""

import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

# Paths
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
NOMINAL_DISTANCE_KM = 50.4  # centerline distance from outer sea buoy to Barbours Cut
NOMINAL_DISTANCE_NM = NOMINAL_DISTANCE_KM / 1.852  # ~27.21 nautical miles


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff))


def cluster_bootstrap_ci(mmsi_array, y_true, y_pred, metric_func, B=2000, seed=42):
    """Vessel-level cluster bootstrap resampling unique MMSIs with replacement."""
    rng = np.random.default_rng(seed)
    unique_mmsis = np.unique(mmsi_array)
    n_vessels = len(unique_mmsis)

    # Pre-group indices by MMSI
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
        score = metric_func(y_true[boot_indices], y_pred[boot_indices])
        boot_scores.append(score)

    q_low = float(np.percentile(boot_scores, 2.5))
    q_high = float(np.percentile(boot_scores, 97.5))
    ci_width = q_high - q_low
    return q_low, q_high, ci_width


def main():
    print("=" * 70)
    print("ORCASTRA TIER 1 — FROZEN BASELINE EVALUATION")
    print("  Protocol: docs/EVAL_PROTOCOL.md")
    print("  Train: Jan 1 – Feb 29 (60 days) | Test: March 1 – March 31 (31 days)")
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

    # Censoring bounds
    left_censor_boundary = min_ts + timedelta(hours=6)
    right_censor_boundary = max_ts - timedelta(hours=72)
    split_boundary = datetime(2024, 3, 1, 0, 0, 0)

    print(f"Left-censor boundary:  {left_censor_boundary}")
    print(f"Right-censor boundary: {right_censor_boundary} (72h before Tmax)")
    print(f"Train/Test split:      {split_boundary}")

    # Partition by MMSI
    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    normalized = {}
    for k, v in mmsi_groups.items():
        key = k[0] if isinstance(k, tuple) else k
        normalized[int(key)] = v
    mmsi_groups = normalized

    # Extract eligible arrivals
    arrivals = []
    excluded_counts = {
        "left_censored": 0,
        "stationary_entrance": 0,
        "non_inbound_cog": 0,
        "right_censored": 0,
        "invalid_duration": 0,
    }

    for mmsi, vdf in mmsi_groups.items():
        recs = vdf.sort("dt").to_dicts()
        n = len(recs)
        was_inside = False

        i = 0
        while i < n:
            rec = recs[i]
            lat, lon = rec["lat"], rec["lon"]
            inside = in_box(lat, lon, ENTRANCE_ZONE)

            if inside and not was_inside:
                t0 = rec["dt"]
                sog0 = rec["sog"] if rec["sog"] is not None and not (abs(rec["sog"] - 102.3) < 0.01) else 0.0

                if t0 <= left_censor_boundary:
                    excluded_counts["left_censored"] += 1
                    was_inside = True
                    i += 1
                    continue

                if sog0 < 3.0:
                    excluded_counts["stationary_entrance"] += 1
                    was_inside = True
                    i += 1
                    continue

                cog0 = rec["cog"] if rec["cog"] is not None and not (abs(rec["cog"] - 360.0) < 0.01) else None
                is_inbound = cog0 is not None and ((270 <= cog0 <= 360) or (0 <= cog0 <= 30))

                if not is_inbound:
                    excluded_counts["non_inbound_cog"] += 1
                    was_inside = True
                    i += 1
                    continue

                if t0 > right_censor_boundary:
                    excluded_counts["right_censored"] += 1
                    was_inside = True
                    i += 1
                    continue

                # Forward search for destination arrival
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
                            heading0 = rec["heading"] if rec["heading"] is not None and not (abs(rec["heading"] - 511) < 0.01) else None
                            length_val = rec.get("length") if rec.get("length") is not None and rec.get("length") > 0 else None
                            width_val = rec.get("width") if rec.get("width") is not None and rec.get("width") > 0 else None
                            draft_val = rec.get("draft") if rec.get("draft") is not None and rec.get("draft") > 0 else None

                            arrivals.append({
                                "mmsi": mmsi,
                                "t0": t0,
                                "t_arr": rn["dt"],
                                "duration_hours": dur,
                                "sog_t0": sog0,
                                "cog_t0": cog0,
                                "heading_t0": heading0,
                                "length": length_val,
                                "width": width_val,
                                "draft": draft_val,
                                "cargo": rec.get("cargo"),
                                "vessel_type": rec["vessel_type"],
                                "hour": t0.hour,
                                "dayofweek": t0.weekday(),
                                "month": t0.strftime("%Y-%m"),
                                "is_train": t0 < split_boundary,
                            })
                            found = True
                            break
                    j += 1

                if not found:
                    excluded_counts["invalid_duration"] += 1

                was_inside = True
                i = max(j if found else i + 1, i + 1)
            else:
                was_inside = inside
                i += 1

    print(f"\nExtraction Results (Inbound Commercial Only):")
    print(f"  Valid arrivals extracted: {len(arrivals)}")
    print(f"  Exclusions: {excluded_counts}")

    train_data = [a for a in arrivals if a["is_train"]]
    test_data = [a for a in arrivals if not a["is_train"]]

    print(f"\nDataset Split:")
    print(f"  Train (Jan 1 – Feb 29): {len(train_data)} arrivals across {len(set(a['mmsi'] for a in train_data))} vessels")
    print(f"  Test  (March 1–31):     {len(test_data)} arrivals across {len(set(a['mmsi'] for a in test_data))} vessels")

    # MMSI Overlap Analysis
    train_mmsis = set(a["mmsi"] for a in train_data)
    test_mmsis = set(a["mmsi"] for a in test_data)
    overlapping_mmsis = train_mmsis.intersection(test_mmsis)
    disjoint_mmsis = test_mmsis - train_mmsis

    overlap_test = [a for a in test_data if a["mmsi"] in overlapping_mmsis]
    disjoint_test = [a for a in test_data if a["mmsi"] in disjoint_mmsis]

    print(f"\nTest MMSI Overlap:")
    print(f"  Total test vessels:       {len(test_mmsis)}")
    print(f"  Overlapping with train:   {len(overlapping_mmsis)} vessels ({len(overlap_test)} arrivals, {100*len(overlap_test)/len(test_data):.1f}%)")
    print(f"  Disjoint (never in train): {len(disjoint_mmsis)} vessels ({len(disjoint_test)} arrivals, {100*len(disjoint_test)/len(test_data):.1f}%)")

    # Arrays
    y_train = np.array([a["duration_hours"] for a in train_data])
    y_test = np.array([a["duration_hours"] for a in test_data])
    test_mmsi_array = np.array([a["mmsi"] for a in test_data])

    print(f"\nTrain Duration Distribution (h):")
    print(f"  Min: {np.min(y_train):.2f}, P10: {np.percentile(y_train, 10):.2f}, Median: {np.median(y_train):.2f}, P90: {np.percentile(y_train, 90):.2f}, Max: {np.max(y_train):.2f}")
    print(f"Test Duration Distribution (h):")
    print(f"  Min: {np.min(y_test):.2f}, P10: {np.percentile(y_test, 10):.2f}, Median: {np.median(y_test):.2f}, P90: {np.percentile(y_test, 90):.2f}, Max: {np.max(y_test):.2f}")

    # Feature preparation for Gradient Boosting
    feature_cols = ["sog_t0", "cog_t0", "heading_t0", "length", "width", "draft", "vessel_type", "hour", "dayofweek"]

    # Impute medians computed on Train only
    train_medians = {}
    for col in feature_cols:
        vals = [a[col] for a in train_data if a[col] is not None]
        train_medians[col] = float(np.median(vals)) if vals else 0.0

    def make_features(data_list):
        rows = []
        for a in data_list:
            row = []
            for col in feature_cols:
                v = a[col]
                row.append(v if v is not None else train_medians[col])
            rows.append(row)
        return np.array(rows)

    X_train = make_features(train_data)
    X_test = make_features(test_data)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/EVAL_PROTOCOL.md",
        "split": {
            "train_arrivals": len(train_data),
            "train_vessels": len(train_mmsis),
            "test_arrivals": len(test_data),
            "test_vessels": len(test_mmsis),
            "test_overlapping_vessels": len(overlapping_mmsis),
            "test_overlapping_arrivals": len(overlap_test),
            "test_disjoint_vessels": len(disjoint_mmsis),
            "test_disjoint_arrivals": len(disjoint_test),
        },
        "train_duration_stats": {
            "min": float(np.min(y_train)),
            "p10": float(np.percentile(y_train, 10)),
            "median": float(np.median(y_train)),
            "p90": float(np.percentile(y_train, 90)),
            "max": float(np.max(y_train)),
        },
        "test_duration_stats": {
            "min": float(np.min(y_test)),
            "p10": float(np.percentile(y_test, 10)),
            "median": float(np.median(y_test)),
            "p90": float(np.percentile(y_test, 90)),
            "max": float(np.max(y_test)),
        },
        "models": {},
        "disjoint_sensitivity": {},
    }

    print("\n" + "=" * 70)
    print("EVALUATING FROZEN BASELINES ON TEST SET (N = %d)" % len(test_data))
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────
    # BASELINE 1: HISTORICAL TRAIN MEDIAN
    # ─────────────────────────────────────────────────────────────
    hist_median_val = float(np.median(y_train))
    y_pred_hist = np.full_like(y_test, hist_median_val)

    mae_hist = float(np.mean(np.abs(y_test - y_pred_hist)))
    medae_hist = float(np.median(np.abs(y_test - y_pred_hist)))
    ci_low, ci_high, ci_w = cluster_bootstrap_ci(
        test_mmsi_array, y_test, y_pred_hist, lambda yt, yp: np.mean(np.abs(yt - yp))
    )

    results["models"]["Historical Train Median"] = {
        "prediction_scalar": hist_median_val,
        "mae": mae_hist,
        "medae": medae_hist,
        "bootstrap_95ci_mae": [ci_low, ci_high],
        "ci_width": ci_w,
    }
    print(f"1. Historical Train Median (constant = {hist_median_val:.2f}h):")
    print(f"   MAE:   {mae_hist:.3f} h  [95% CI: {ci_low:.3f}, {ci_high:.3f}] (width: {ci_w:.3f}h)")
    print(f"   MedAE: {medae_hist:.3f} h")

    # ─────────────────────────────────────────────────────────────
    # BASELINE 2: NAIVE KINEMATIC REFERENCE (d / SOG)
    # ─────────────────────────────────────────────────────────────
    # sog is in knots (nautical miles per hour), distance in NM
    y_pred_naive = np.array([
        NOMINAL_DISTANCE_NM / max(a["sog_t0"], 1.0) for a in test_data
    ])
    mae_naive = float(np.mean(np.abs(y_test - y_pred_naive)))
    medae_naive = float(np.median(np.abs(y_test - y_pred_naive)))
    ci_low_n, ci_high_n, ci_w_n = cluster_bootstrap_ci(
        test_mmsi_array, y_test, y_pred_naive, lambda yt, yp: np.mean(np.abs(yt - yp))
    )

    results["models"]["Naive Kinematic (d/SOG)"] = {
        "mae": mae_naive,
        "medae": medae_naive,
        "bootstrap_95ci_mae": [ci_low_n, ci_high_n],
        "ci_width": ci_w_n,
    }
    print(f"\n2. Naive Kinematic Reference (d / SOG(t0)):")
    print(f"   MAE:   {mae_naive:.3f} h  [95% CI: {ci_low_n:.3f}, {ci_high_n:.3f}] (width: {ci_w_n:.3f}h)")
    print(f"   MedAE: {medae_naive:.3f} h")

    # ─────────────────────────────────────────────────────────────
    # BASELINE 3: QUANTILE REGRESSION (P50, P80, P95)
    # ─────────────────────────────────────────────────────────────
    quantiles = [0.50, 0.80, 0.95]
    q_models = {}
    q_preds = {}

    for q in quantiles:
        reg = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=42)
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        q_models[q] = reg
        q_preds[q] = preds

        mae_q = float(np.mean(np.abs(y_test - preds)))
        medae_q = float(np.median(np.abs(y_test - preds)))
        pb_loss = float(pinball_loss(y_test, preds, q))
        coverage = float(np.mean(y_test <= preds) * 100.0)

        # Bootstrap CIs for pinball and coverage
        pb_low, pb_high, pb_w = cluster_bootstrap_ci(
            test_mmsi_array, y_test, preds, lambda yt, yp: pinball_loss(yt, yp, q)
        )
        cov_low, cov_high, cov_w = cluster_bootstrap_ci(
            test_mmsi_array, y_test, preds, lambda yt, yp: np.mean(yt <= yp) * 100.0
        )
        mae_low, mae_high, mae_w = cluster_bootstrap_ci(
            test_mmsi_array, y_test, preds, lambda yt, yp: np.mean(np.abs(yt - yp))
        )

        model_name = f"Quantile P{int(q*100)}"
        results["models"][model_name] = {
            "quantile": q,
            "mae": mae_q,
            "bootstrap_95ci_mae": [mae_low, mae_high],
            "medae": medae_q,
            "pinball_loss": pb_loss,
            "bootstrap_95ci_pinball": [pb_low, pb_high],
            "empirical_coverage_pct": coverage,
            "bootstrap_95ci_coverage": [cov_low, cov_high],
        }
        print(f"\n3.{int(q*100)}. Quantile Regression P{int(q*100)}:")
        print(f"   MAE:          {mae_q:.3f} h  [95% CI: {mae_low:.3f}, {mae_high:.3f}] (width: {mae_w:.3f}h)")
        print(f"   MedAE:        {medae_q:.3f} h")
        print(f"   Pinball Loss: {pb_loss:.3f} h  [95% CI: {pb_low:.3f}, {pb_high:.3f}]")
        print(f"   Coverage:     {coverage:.1f}% (target: {int(q*100)}%)  [95% CI: {cov_low:.1f}%, {cov_high:.1f}%]")

    # Check quantile monotonicity (P50 <= P80 <= P95)
    inversions_50_80 = int(np.sum(q_preds[0.50] > q_preds[0.80]))
    inversions_80_95 = int(np.sum(q_preds[0.80] > q_preds[0.95]))
    results["models"]["quantile_inversions"] = {
        "P50_gt_P80": inversions_50_80,
        "P80_gt_P95": inversions_80_95,
        "monotonic_valid": (inversions_50_80 == 0 and inversions_80_95 == 0),
    }
    print(f"\n   Quantile Monotonicity: {'PASS (0 inversions)' if inversions_50_80 == 0 and inversions_80_95 == 0 else f'INVERSIONS: {inversions_50_80} / {inversions_80_95}'}")

    # ─────────────────────────────────────────────────────────────
    # PART 4: VESSEL-DISJOINT SENSITIVITY TEST
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VESSEL-DISJOINT SENSITIVITY TEST (Test set partitioned by train overlap)")
    print("=" * 70)

    # Overlapping cohort indices
    overlap_idx = np.array([idx for idx, a in enumerate(test_data) if a["mmsi"] in overlapping_mmsis])
    disjoint_idx = np.array([idx for idx, a in enumerate(test_data) if a["mmsi"] in disjoint_mmsis])

    cohorts = {
        "Overlapping (seen in train)": overlap_idx,
        "Disjoint (never in train)": disjoint_idx,
    }

    for cname, cidx in cohorts.items():
        if len(cidx) == 0:
            continue
        y_c = y_test[cidx]
        pred_hist_c = y_pred_hist[cidx]
        pred_p50_c = q_preds[0.50][cidx]
        pred_p80_c = q_preds[0.80][cidx]
        pred_p95_c = q_preds[0.95][cidx]

        mae_h = float(np.mean(np.abs(y_c - pred_hist_c)))
        mae_p50 = float(np.mean(np.abs(y_c - pred_p50_c)))
        cov_p50 = float(np.mean(y_c <= pred_p50_c) * 100.0)
        cov_p80 = float(np.mean(y_c <= pred_p80_c) * 100.0)
        cov_p95 = float(np.mean(y_c <= pred_p95_c) * 100.0)

        results["disjoint_sensitivity"][cname] = {
            "n_arrivals": len(cidx),
            "hist_median_mae": mae_h,
            "p50_mae": mae_p50,
            "p50_coverage": cov_p50,
            "p80_coverage": cov_p80,
            "p95_coverage": cov_p95,
        }
        print(f"\nCohort: {cname} (N = {len(cidx)} arrivals):")
        print(f"  Hist Train Median MAE: {mae_h:.3f} h")
        print(f"  P50 Regressor MAE:     {mae_p50:.3f} h")
        print(f"  Coverage: P50 = {cov_p50:.1f}%, P80 = {cov_p80:.1f}%, P95 = {cov_p95:.1f}%")

    # Decision Criteria evaluation
    p50_mae_val = results["models"]["Quantile P50"]["mae"]
    p95_cov_val = results["models"]["Quantile P95"]["empirical_coverage_pct"]
    p50_ci_w = results["models"]["Quantile P50"]["bootstrap_95ci_mae"][1] - results["models"]["Quantile P50"]["bootstrap_95ci_mae"][0]
    beats_naive = p50_mae_val < mae_naive
    monotonic = results["models"]["quantile_inversions"]["monotonic_valid"]

    gate_continue = beats_naive and monotonic and (p95_cov_val >= 85.0)

    results["gate_evaluation"] = {
        "beats_naive_reference": beats_naive,
        "quantile_monotonicity": monotonic,
        "p95_coverage_sufficient": (p95_cov_val >= 85.0),
        "bootstrap_ci_width_hours": p50_ci_w,
        "recommendation": "CONTINUE" if gate_continue else "STOP/REVISE",
    }

    out_path = RESULTS_DIR / "tier1_baseline_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print(f"PROTOCOL DECISION: {results['gate_evaluation']['recommendation']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
