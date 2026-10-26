# ORCASTRA — Tier 1 Frozen Evaluation Protocol (v2)

**Document Version:** 2.0  
**Effective Date:** 2026-10-26  
**Commit Status:** Frozen BEFORE baseline execution  

---

## 1. Scope & Purpose

This protocol specifies the evaluation methodology for ORCASTRA Tier 1 ETA prediction in the Houston Ship Channel, incorporating:
1. **Reconciliation of Arrival Cohorts** (strictly inbound first-crossings vs re-entry crossings; exclusion of non-inbound maneuvers).
2. **Short-Horizon State-Based ETA** (the primary task operationalizable on live AISStream feeds).
3. **Quantile Post-Processing & Conformal Calibration** (sorting to eliminate quantile crossing; calibration on held-out slice of train).
4. **Predefinition of Full-Trip Queue Features** (reserved for subsequent experiment).
5. **Vessel-Level Cluster Bootstrap & Disjoint Sensitivity**.

Per project requirements, this document is committed to git **BEFORE** running any models.

---

## 2. Arrival Cohort Reconciliation & Eligibility

### 2.1 Geographic Zones & Criteria
- **Entrance Zone:** `[29.30, 29.45] x [-94.85, -94.65]`
- **Destination Berth Zone:** `[29.65, 29.80] x [-95.15, -94.95]`
- **Commercial Fleet:** Vessel type in range $[70, 89]$ (cargo, tankers, passenger).
- **Active Underway at Crossing:** $\text{SOG}(t_0) \ge 3.0$ kn at boundary entry.
- **Left-Censoring:** $t_0 \ge T_{\min} + 6.0$ hours ($t_0 \ge \text{2024-01-01 06:00:00 UTC}$).
- **Right-Censoring (72h Rule):** Entrants must satisfy $t_0 \le T_{\max} - 72.0\text{h}$ ($t_0 \le \text{2024-03-28 23:59:59 UTC}$).

### 2.2 Reconciled Cohorts
- **Cohort A1 (Strict Inbound First-Crossing, N = 899):**
  - First boundary crossing into the entrance zone has $\text{COG}(t_0) \in [270^\circ, 360^\circ] \cup [0^\circ, 30^\circ]$.
  - Excludes vessels whose initial crossing was non-inbound ($N=109$ non-inbound voyages excluded).
- **Cohort A2 (Inbound with Re-Entry Allowed, N = 985):**
  - If a vessel entered heading non-inbound (e.g. loitering), that non-inbound crossing is skipped; if the vessel subsequently crosses inbound into the channel, that inbound crossing defines $t_0$.
  - Split: 665 Train (Jan-Feb), 320 Test (March).
- **Reporting Requirement:** Full-trip baselines will be reported on **both** Cohort A1 ($N=899$) and Cohort A2 ($N=985$), confirming that non-inbound maneuvers do not contaminate evaluation.

---

## 3. Short-Horizon State-Based ETA Protocol (Core Live Task)

### 3.1 Defined Inbound Transit Corridor
A continuous navigational corridor through Galveston Bay along the Houston Ship Channel:
- Centerline waypoints and channel distances to berth:
  - WP1 (Outer Entrance / Fairway Buoy): $(29.350, -94.750)$ — $d = 50.4$ km ($27.2$ NM)
  - WP2 (Bolivar Roads): $(29.380, -94.780)$ — $d = 46.0$ km ($24.8$ NM)
  - WP3 (Red Fish Bar): $(29.510, -94.900)$ — $d = 28.5$ km ($15.4$ NM)
  - WP4 (Mid-Bay Beacon): $(29.600, -94.960)$ — $d = 16.0$ km ($8.6$ NM)
  - WP5 (Barbours Cut Turn): $(29.680, -95.000)$ — $d = 4.5$ km ($2.4$ NM)
  - WP6 (Inner Berths): $(29.720, -95.050)$ — $d = 0.0$ km

### 3.2 State-Based Snapshot Sampling
- Snapshots are triggered strictly by **current vessel state**, never by backwards lookback from berth time.
- State trigger: A commercial vessel is inside the corridor, actively underway ($\text{SOG} \ge 3.0$ kn), with inbound course ($\text{COG} \in [270^\circ, 360^\circ] \cup [0^\circ, 45^\circ]$).
- Sampling interval: Every **10 minutes** while in this active state.
- Target label: Remaining transit time to berth $y_{\text{rem}} = T_{\text{arr}} - t_{\text{snap}}$ (hours).

### 3.3 Snapshot Right-Censoring Rule
- Snapshots occurring at $t_{\text{snap}} > T_{\max} - 3.0\text{h}$ (within 3 hours of dataset end) whose destination arrival is unobserved before $T_{\max}$ are tagged as `right_censored` and excluded from supervised loss evaluation.

### 3.4 Short-Horizon Evaluation Buckets
Results must be disaggregated by horizon bucket:
1. **$\le 1$ Hour Horizon:** $0 < y_{\text{rem}} \le 1.0$ hour remaining.
2. **1–2 Hour Horizon:** $1.0 < y_{\text{rem}} \le 2.0$ hours remaining.
3. **2–3 Hour Horizon:** $2.0 < y_{\text{rem}} \le 3.0$ hours remaining.

### 3.5 Models Evaluated on Short-Horizon
1. **Kinematic Baseline ($d_{\text{rem}} / \text{SOG}$):**
   Remaining channel distance divided by instantaneous speed.
2. **Historical Median by Corridor Segment:**
   Median remaining duration observed in training for vessels at the same distance-to-go decile.
3. **Quantile Gradient Boosting (P50, P80, P95):**
   Predicts quantiles from point-in-time state $(\text{lat}, \text{lon}, d_{\text{rem}}, \text{sog}, \text{cog}, \text{heading}, \text{length}, \text{width}, \text{draft})$.

---

## 4. Quantile Post-Processing & Conformal Calibration

### 4.1 Monotonicity Enforcement (Sorting)
Quantile crossing is mathematically impossible under sorting:
$$\hat{y}_{0.80}^{\text{mono}} = \max(\hat{y}_{0.80}, \hat{y}_{0.50})$$
$$\hat{y}_{0.95}^{\text{mono}} = \max(\hat{y}_{0.95}, \hat{y}_{0.80}^{\text{mono}})$$

### 4.2 Conformal Calibration on Held-Out Train Slice
- Training Period (Jan 1 – Feb 29, 60 days) is split chronologically:
  - **Train Slice (Jan 1 – Feb 15, 46 days):** Model training.
  - **Calibration Slice (Feb 16 – Feb 29, 14 days):** Held-out for conformal quantile residual calibration.
- Conformal residual adjustment $c_\alpha$:
  $$c_\alpha = \text{Quantile}_\alpha\left(\{y_i - \hat{y}_{i, \alpha} : i \in \text{Calibration}\}\right)$$
- Calibrated Prediction applied to Test set:
  $$\hat{y}_{i, \alpha}^{\text{calib}} = \hat{y}_{i, \alpha}^{\text{mono}} + c_\alpha$$
- **Test Set Integrity:** The March test set is completely untouched until the single frozen benchmark run.

---

## 5. Statistical Rigor & Disjoint Evaluation

### 5.1 Uncertainty Estimation
- All confidence intervals are computed via **vessel-level cluster bootstrap** ($B = 2,000$ iterations, resampling unique MMSIs with replacement).
- Report 95% Percentile Confidence Intervals $[q_{0.025}, q_{0.975}]$ for MAE, Pinball loss, and Empirical Coverage.

### 5.2 Reporting Disaggregation
All headline metrics are reported for:
1. **Overall Test Population** (all test arrivals/snapshots).
2. **Unseen / Disjoint Vessels** (subset of test MMSIs never observed in the training period).

---

## 6. Predefinition: Full-Trip Queue-State Features (Experiment 2)

Full-trip queue features are formally predefined for subsequent investigation:
- `bolivar_anchorage_count_t0`: Number of vessels with $\text{SOG} < 1.0$ kn in Bolivar Roads at $t_0$.
- `gulf_fairway_anchorage_count_t0`: Number of vessels in Gulf Fairway anchorage at $t_0$.
- `prior_wait_hours_t0`: Vessel's own pre-$t_0$ dwell hours in Gulf Fairway.
- `rolling_mean_wait_24h`: Average transit duration of commercial vessels arrived in prior 24 hours.
- **Timing:** All queue features are strictly computed from observations $\le t_0$.
