# ORCASTRA — Tier 1 Frozen Evaluation Protocol

**Document Version:** 1.0  
**Effective Date:** 2026-10-26  
**Commit Status:** Frozen BEFORE baseline modeling  

---

## 1. Objective & Scope

This protocol establishes the immutable evaluation criteria for ORCASTRA Tier 1 ETA prediction in the Houston Ship Channel.
Per project requirements, this protocol is committed **prior to running baseline models** to prevent test-set tuning, metric cherry-picking, or post-hoc threshold adjustment.

---

## 2. Dataset & Chronological Split

### 2.1 Dataset
- **Source:** NOAA MarineCadastre AIS (Houston Ship Channel region: `[29.0, 29.9] x [-95.3, -94.4]`).
- **Temporal span:** 91 consecutive days: 2024-01-01 00:00:00 to 2024-03-31 23:59:59 UTC.
- **Directory:** `D:\ORCASTRA_DATA\historical\noaa_3mo\houston_{date}.parquet`.

### 2.2 Split Structure (Strictly Chronological)
- **Train Set:** Days 1–60 (2024-01-01 to 2024-02-29 UTC, 60 days including leap year).
- **Test Set:** Days 61–91 (2024-03-01 to 2024-03-31 UTC, 31 days).
- **No shuffling:** Split is strictly chronological. No future observations or aggregates may inform training.

---

## 3. Arrival Definition, Geofences & Censoring

### 3.1 Geofences
- **Entrance Zone:** `[29.30, 29.45] x [-94.85, -94.65]` (outer sea buoy to Bolivar Roads fairway).
- **Destination Berth Zone:** `[29.65, 29.80] x [-95.15, -94.95]` (Barbours Cut, Morgans Point, Jacintoport terminals).
- **Anchorage Zones (Dwell Detection):**
  - Bolivar Roads Anchorage: `[29.34, 29.44] x [-94.80, -94.70]`.
  - Galveston Harbor: `[29.28, 29.34] x [-94.85, -94.75]`.
  - Gulf Fairway/Offshore: `[29.10, 29.32] x [-94.85, -94.55]`.
  - Mid-Bay (Red Fish Bar): `[29.45, 29.58] x [-94.98, -94.85]`.
  - Upper Bay / Morgans Point: `[29.62, 29.70] x [-95.05, -94.95]`.

### 3.2 Crossing-Level Eligibility Rules
1. **Commercial Filter:** Vessel type must be commercial ($70 \le \text{vessel\_type} \le 89$, covering cargo, tanker, passenger).
2. **Boundary Crossing:** Evaluated at the **crossing level** (transition from outside to inside Entrance Zone).
3. **Active Underway at Entrance:** SOG at crossing $t_0$ must satisfy $\text{SOG}(t_0) \ge 3.0$ kn.
4. **Directionality:** Course over ground at $t_0$ must be directed inbound into the channel: $270^\circ \le \text{COG}(t_0) \le 360^\circ$ or $0^\circ \le \text{COG}(t_0) \le 30^\circ$. (Non-inbound arrivals are excluded from the primary benchmark).
5. **Left-Censoring Rule:** $t_0 \ge T_{\min} + 6.0$ hours ($t_0 \ge \text{2024-01-01 06:00:00 UTC}$).
6. **Right-Censoring Rule (72h Window):**
   - The maximum transit duration is 72.0 hours.
   - For an entrant to be evaluable, its entire potential 72h voyage must fall within the observed dataset.
   - **Test Set Cutoff:** Only vessels entering with $t_0 \le T_{\max} - 72.0\text{h}$ are eligible for scoring. For March, $T_{\max} = \text{2024-03-31 23:59:59}$, so entrants with $t_0 > \text{2024-03-28 23:59:59}$ are right-censored and excluded.
7. **Arrival Event:** First position report inside Destination Berth Zone with $\text{SOG} < 1.5$ kn and duration $1.0\text{h} \le (T_{\text{arr}} - t_0) \le 72.0\text{h}$.

---

## 4. Models to Evaluate (Baselines Only)

No replacement or complex model training is permitted during this protocol run. Exactly three standard baselines will be fitted on Train and evaluated on Test:

1. **Historical Train Median:**
   $$\hat{y} = \text{Median}(y_{\text{train}})$$
   Predicts the scalar median transit duration from the training set for all test instances.
2. **Naive Kinematic Reference ($d / \text{SOG}$):**
   $$\hat{y} = \frac{d}{\max(\text{SOG}(t_0), 1.0)}$$
   where $d = 50.4$ km (nominal centerline distance from entrance to inner berths).
3. **Quantile Gradient Boosting (P50, P80, P95):**
   - `HistGradientBoostingRegressor(loss='quantile', quantile=alpha)` for $\alpha \in \{0.50, 0.80, 0.95\}$.
   - Point-in-time features strictly at $t_0$:
     - Kinematics: `sog`, `cog`, `heading`.
     - Dimensions: `length`, `width`, `draft`, `cargo`, `vessel_type`.
     - Temporal: `hour_of_day`, `day_of_week`.
   - Feature imputations (medians) computed on **Train only** and applied to Test.

---

## 5. Metrics & Statistical Evaluation

### 5.1 Primary Error Metrics
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
- **Median Absolute Error (MedAE):**
  $$\text{MedAE} = \text{Median}(|y_i - \hat{y}_i|)$$
- **Pinball Loss (Quantile Loss):**
  $$L_\alpha(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N \max\left(\alpha(y_i - \hat{y}_i), (1 - \alpha)(\hat{y}_i - y_i)\right)$$
- **Empirical Coverage ($\%$):**
  $$\text{Coverage}(\alpha) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \le \hat{y}_{i, \alpha}) \times 100\%$$

### 5.2 Uncertainty via Vessel-Level Cluster Bootstrap
- To properly account for correlated repeat voyages by the same vessel, all confidence intervals are computed using a **vessel-level cluster bootstrap**:
  - Sample unique MMSIs with replacement ($B = 2,000$ iterations).
  - All voyages for a selected MMSI are included in that bootstrap replicate.
  - Report 95% Percentile Confidence Intervals $[q_{0.025}, q_{0.975}]$ for MAE, Pinball loss, and Coverage.

### 5.3 Vessel-Disjoint Sensitivity Analysis
- Split Test set into:
  - **Overlapping Cohort:** MMSIs observed in Train.
  - **Disjoint Cohort:** MMSIs never seen in Train.
- Report metrics on both cohorts to detect vessel-memorization bias.

---

## 6. Execution Rules & Test Integrity
- **Single Run:** The baseline experiment is executed ONCE.
- **No Tuning:** Zero hyperparameter tuning or iteration on the test set.
- **Documentation:** All outputs written to `D:\ORCASTRA_DATA\historical\results\tier1_baseline_results.json`.

---

## 7. Decision Criteria: "Continue" vs "Stop"

| Outcome | Metric Condition | Action |
|---|---|---|
| **Continue** | • Pipeline runs without leakage<br>• P50 MAE beats Naive Reference ($d/\text{SOG}$)<br>• Quantile monotonic ordering ($\hat{y}_{0.50} \le \hat{y}_{0.80} \le \hat{y}_{0.95}$)<br>• Disjoint test degradation $< 30\%$ | Proceed to Tier 1 feature engineering (weather, anchorage, traffic state) |
| **Stop / Revise** | • P95 Coverage $< 85\%$ (severe under-coverage)<br>• Bootstrap CI width on MAE $> 12.0$h<br>• Failure to beat naive reference | Block model development; investigate anchorage state representation before retraining |
