# ORCASTRA — Tier 1 Frozen Evaluation Protocol (v2.1)

**Document Version:** 2.1  
**Effective Date:** 2026-10-26  
**Commit Status:** Frozen BEFORE benchmark execution  
**Previous Version:** [EVAL_PROTOCOL_V2.md](EVAL_PROTOCOL_V2.md) (retained as-is for comparison)

---

## 0. Changes from v2.0

| v2.0 Behaviour | v2.1 Fix | Rationale |
|---|---|---|
| Snapshots filtered to `0 < y_rem <= 3.5 h` (label-conditioned) | **All** state-triggered snapshots are retained; no filter on true remaining time | Selection must depend only on quantities known at prediction time |
| Buckets defined by TRUE remaining time (`≤1 h`, `1–2 h`, `2–3 h`) | Buckets defined by **distance to berth** (`d_rem`): `≤10 km`, `10–25 km`, `25–50 km`, `≥50 km` | Distance is observable at prediction time; remaining duration is the target |
| No "predict bucket median" baseline | Add **Bucket Median Baseline** (predict the training-set median `y_rem` for the distance bucket) | Provides a sanity-check: any model that cannot beat this is useless |
| Only MAE, coverage, pinball reported | Also report **P50/P80/P95 interval widths** and **share of snapshots with true remaining > 3 h** | Interval widths quantify prediction sharpness |
| Vessels that stop/anchor or never reach berth are silently dropped | Explicitly tagged and counted: `anchored_midway`, `never_arrived` (right-censored) | Ensures honest snapshot counts |

---

## 1. Scope & Purpose

This protocol specifies the **unconditioned** short-horizon evaluation methodology for ORCASTRA Tier 1 ETA prediction in the Houston Ship Channel.  It replaces the label-conditioned v2.0 evaluation while preserving identical data, model architecture, conformal calibration, and cluster-bootstrap procedures.

---

## 2. Arrival Cohort (Unchanged from v2.0)

### 2.1 Geographic Zones & Criteria
- **Entrance Zone:** `[29.30, 29.45] × [-94.85, -94.65]`
- **Destination Berth Zone:** `[29.65, 29.80] × [-95.15, -94.95]`
- **Commercial Fleet:** Vessel type ∈ [70, 89].
- **Active Underway at Crossing:** SOG(t₀) ≥ 3.0 kn.
- **Left-Censoring:** t₀ ≥ T_min + 6.0 h.
- **Right-Censoring (72 h rule):** t₀ ≤ T_max − 72 h.

### 2.2 Cohorts
- **Cohort A1 (Strict Inbound First-Crossing):** COG(t₀) ∈ [270°, 360°] ∪ [0°, 30°]; non-inbound first-crossings excluded.
- **Cohort A2 (Inbound with Re-Entry Allowed):** If a vessel's first crossing was non-inbound, that crossing is skipped; the first inbound crossing defines t₀.

Cohort A1 is the primary evaluation cohort for full-trip baselines.  Cohort A2 is reported for sensitivity only.

---

## 3. Unconditioned Short-Horizon State-Based ETA Protocol

### 3.1 Inbound Transit Corridor (Unchanged)
Centerline waypoints as defined in v2.0 §3.1.

### 3.2 State-Based Snapshot Sampling — No Label Conditioning
Snapshots are triggered strictly by **current vessel state at time t_snap**, never by backwards lookback from berth time and never filtered by true remaining duration.

**State trigger (all conditions must hold simultaneously):**
1. The vessel is a Cohort A1 (or A2) voyage that has already crossed t₀.
2. Ping location is inside the inbound corridor (lat [29.34, 29.75], lon [-95.12, -94.70]).
3. SOG ≥ 3.0 kn.
4. COG ∈ [270°, 360°] ∪ [0°, 45°] (inbound heading).

**Sampling interval:** Every 10 minutes while the above state holds.

**No filter on true remaining time.** Snapshots whose true y_rem > 3 h, or whose vessel later stops/anchors, are all included.

### 3.3 Outcome Label Assignment
For each snapshot at t_snap belonging to voyage v with arrival time t_arr(v):

- **Observed arrival:** y_rem = t_arr(v) − t_snap (hours).  This is the supervised label.
- **Vessel stopped/anchored mid-transit:** If the vessel has active corridor pings at t_snap but later drops SOG < 1.0 kn for ≥ 30 minutes before reaching the berth zone, the snapshot is tagged `anchored_midway = True`.  These snapshots are **included** in evaluation (with their true y_rem that may be large).
- **Dataset-end right-censored (never arrived):** If t_snap > T_max − 3 h and the vessel has no subsequent observed berth arrival before T_max, the snapshot is tagged `right_censored = True` and **excluded** from supervised loss metrics (no y_rem available).  Count is reported separately.

### 3.4 Evaluation Buckets — Distance to Berth
Results are disaggregated by **distance to berth d_rem** (observable at prediction time), not by true remaining duration (which is the target).

| Bucket | Distance Range |
|---|---|
| Near Berth | d_rem ≤ 10 km |
| Mid-Channel | 10 < d_rem ≤ 25 km |
| Outer Channel | 25 < d_rem ≤ 50 km |
| Beyond Entrance | d_rem > 50 km |
| **Overall** | All uncensored snapshots |

### 3.5 Additional Disaggregations Reported
For each distance bucket, also report:
- **Share with true y_rem > 3 h** (indicates how many snapshots are from vessels still far from arrival).
- **Share tagged `anchored_midway`** (vessels that stopped in the channel).

---

## 4. Models Evaluated

### 4.1 Kinematic Baseline (d_rem / SOG)
Remaining channel distance divided by instantaneous SOG (converted NM → hours).

### 4.2 Historical Median by Distance Bucket (Segment Median)
Median training-set remaining duration for vessels in the same distance decile.

### 4.3 **Bucket Median Baseline (NEW)**
For each distance bucket defined in §3.4, predict the **training-set median y_rem** for that bucket.  This is the simplest reasonable model and any useful model must beat it.

### 4.4 Quantile Gradient Boosting (P50, P80, P95)
HistGradientBoostingRegressor with `loss="quantile"`, trained on **Train Slice** only.  Features: `(lat, lon, d_rem, sog, cog, heading, length, draft)`.

---

## 5. Quantile Post-Processing & Conformal Calibration (Unchanged from v2.0)

### 5.1 Monotonicity Enforcement (Sorting)
$$\hat{y}_{0.80}^{\text{mono}} = \max(\hat{y}_{0.80}, \hat{y}_{0.50})$$
$$\hat{y}_{0.95}^{\text{mono}} = \max(\hat{y}_{0.95}, \hat{y}_{0.80}^{\text{mono}})$$

### 5.2 Conformal Calibration on Held-Out Train Slice
- **Train Slice** (Jan 1 – Feb 15): Model training.
- **Calibration Slice** (Feb 16 – Feb 29): Conformal quantile residual calibration.
- Conformal residual: $c_\alpha = \text{Quantile}_\alpha(\{y_i - \hat{y}_{i,\alpha} : i \in \text{Calib}\})$
- Calibrated prediction: $\hat{y}_{i,\alpha}^{\text{calib}} = \hat{y}_{i,\alpha}^{\text{mono}} + c_\alpha$
- **Test set (March):** Completely untouched until the single frozen benchmark run.

**Code lines for conformal isolation** (in `run_tier1_v2_1_evaluation.py`):
```python
# Lines ~L340-L360 in the evaluation script:
# 1. Calibration residuals computed ONLY on calib slice data
residuals = y_calib - calib_preds[q]
shift = float(np.percentile(residuals, q * 100.0))

# 2. Applied to test predictions WITHOUT re-fitting
test_calib_preds[q] = np.maximum(raw_preds[q] + conformal_shifts[q], floor)
```

---

## 6. Metrics

For each distance bucket and overall:

| Metric | Description |
|---|---|
| **MAE** | Mean absolute error (P50 prediction vs y_rem) |
| **MedAE** | Median absolute error |
| **Pinball Loss** (τ = 0.50, 0.80, 0.95) | Quantile-specific loss |
| **Empirical Coverage** (P50, P80, P95) | % of y_rem ≤ predicted quantile |
| **P50 Interval Width** | Mean of P50 predictions (point estimate spread) |
| **P80 Interval Width** | Mean of (P80 − P50) across snapshots |
| **P95 Interval Width** | Mean of (P95 − P50) across snapshots |
| **Share y_rem > 3 h** | Fraction of snapshots whose true remaining > 3 h |
| **Share anchored_midway** | Fraction tagged as vessels that later stopped |
| **N censored** | Count of right-censored snapshots (excluded from loss) |
| **Bucket Median Baseline MAE** | MAE of the §4.3 baseline |

---

## 7. Statistical Rigor & Disjoint Evaluation (Unchanged from v2.0)

### 7.1 Uncertainty Estimation
All confidence intervals via **vessel-level cluster bootstrap** (B = 2,000, resampling unique MMSIs with replacement).  Report 95% percentile CIs [q₀.₀₂₅, q₀.₉₇₅] for MAE, Pinball, and Coverage.

**Code lines** (in `run_tier1_v2_1_evaluation.py`):
```python
# Lines ~L75-L98: Vessel-clustered bootstrap
unique_mmsis = np.unique(mmsi_array)
for _ in range(B):
    sampled_mmsis = rng.choice(unique_mmsis, size=n_vessels, replace=True)
    boot_indices = []
    for sm in sampled_mmsis:
        boot_indices.extend(mmsi_to_indices[sm])
    score = metric_func(y_true[boot_indices], y_pred[boot_indices])
```

### 7.2 Reporting Disaggregation
All headline metrics are reported for:
1. **Overall test population** (all non-censored test snapshots).
2. **Unseen / disjoint vessels** (test MMSIs never observed in training period).

---

## 8. Predefinition: Full-Trip Queue-State Features (Experiment 2, Unchanged)

Formally predefined for subsequent investigation (NOT run under v2.1):
- `bolivar_anchorage_count_t0`, `gulf_fairway_anchorage_count_t0`, `prior_wait_hours_t0`, `rolling_mean_wait_24h`.
- All computed from observations ≤ t₀.
