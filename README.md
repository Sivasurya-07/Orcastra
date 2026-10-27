# ORCASTRA — Operational Real-time Channel & Ship Transit Risk Analyzer

Short-horizon ETA prediction for commercial vessels transiting the Houston Ship Channel, using AIS position data from NOAA historical archives and AISStream live feeds.

## Problem Statement

Predicting when a commercial vessel will reach its berth after entering the Houston Ship Channel is operationally valuable but statistically hard. Roughly 13% of inbound voyages anchor mid-channel for 12–40+ hours awaiting pilot dispatch or berth clearance, creating a heavy-tailed distribution that no simple model can reliably point-predict. Full-trip ETA (entrance-to-berth) has a historical median baseline MAE of 6.3 hours — driven almost entirely by the unpredictable anchorage tail.

**ORCASTRA focuses on short-horizon ETA**: predicting remaining transit time for vessels already actively underway inside the channel corridor. For this task, a quantile regression model achieves a Median Absolute Error of **12.7 minutes overall** and **8.7 minutes near the berth**, with calibrated 95% prediction intervals that achieve 97.9% empirical coverage on a held-out chronological test set.

### Scope and Limitations

This is a research prototype, not a production system. Specifically:

- **The quantile regression model does not beat the distance-bucket median baseline on MAE** (2.725h vs 2.708h overall). The model's value is in providing calibrated prediction _intervals_, not better point estimates.
- **Prediction intervals are very wide** (P95 − P50 ≈ 38–56 hours) because the model honestly accounts for the possibility of mid-channel anchorage delays.
- **Only voyages that reach berth within 72 hours are included.** Vessels that leave, turn around, or never berth are excluded.
- **Historical data covers one winter season** (January–March 2024). Seasonal and year-over-year variation is unquantified.
- **Live AISStream coverage is thinner than NOAA** (~1/3 the vessel density), and 73% of live vessels lack vessel type data.
- **The live scoreboard is Collecting (n = 0)** with no live-scored arrivals to date.
- **Results do not establish long-run collector reliability or production-ready ETA accuracy.**

## Data Sources

| Source | Role | Coverage | Volume |
|---|---|---|---|
| **NOAA MarineCadastre AIS** | Training & offline evaluation | Jan 1 – Mar 31, 2024 (91 days) | 873 MB, ~51M rows |
| **AISStream WebSocket** | Live collection (real-time) | Oct 2026 – present (intermittent sessions) | ~0.5 MB collected |

- **NOAA data** provides complete historical coverage with vessel type for every MMSI, enabling arrival labeling, censoring rules, and chronological train/calibration/test splits.
- **AISStream data** is collected by the `orcastra-collector` service via a live WebSocket connection, recorded to Parquet files, and stored in a session ledger. It requires an API key (see [Setup](#setup)).

## Architecture

```
NOAA Archive (Parquet)                    AISStream (WebSocket)
        |                                         |
        v                                         v
  Arrival Labeling                     orcastra-collector
  Censoring (72h rule)                   - Parquet writer
  Corridor Snapshot Extraction           - Session ledger
        |                                - Crash recovery
        v                                - Vessel cache
  Quantile GBM Training                         |
  (Jan 1 – Feb 15)                              v
        |                              Saved Recordings
  Conformal Calibration                  (Parquet parts)
  (Feb 16 – Feb 29)                             |
        |                                       v
        v                              Offline Replay Demo
  Frozen Evaluation                    (scripts/offline_replay_demo.py)
  (March 1 – 31 test set)
```

**Key design decisions:**
- Full-trip ETA (entrance-to-berth) is evaluated on historical replay only — live scoring of full trips would take months to reach statistical significance.
- Short-horizon ETA (corridor position → remaining time) is the primary live-scorable task.
- The model is trained on the Train Slice, conformally calibrated on a held-out Calibration Slice, and evaluated once on the chronological Test Set (March). The test set is never used for training or calibration.

## Definitions

### Arrival Label
A commercial vessel (type 70–89) crosses the Entrance Zone (`[29.30°, 29.45°] × [-94.85°, -94.65°]`) with SOG ≥ 3.0 kn and inbound heading (COG ∈ [270°, 360°] ∪ [0°, 30°]), then reaches the Destination Berth Zone (`[29.65°, 29.80°] × [-95.15°, -94.95°]`) with SOG < 1.5 kn within 72 hours.

### Censoring
- **Left-censoring:** First 6 hours of the dataset excluded (in-progress voyages).
- **Right-censoring (72h rule):** Entrance crossings within 72 hours of dataset end excluded (unobserved outcomes).
- **Snapshot right-censoring:** Snapshots within 3 hours of dataset end whose berth arrival is unobserved are excluded from supervised loss.

### Evaluation Split (Chronological)
- **Train Slice:** Jan 1 – Feb 15, 2024 (model training).
- **Calibration Slice:** Feb 16 – Feb 29, 2024 (conformal quantile calibration).
- **Test Set:** March 1 – 31, 2024 (single frozen evaluation).

### Metrics
- **MAE / MedAE:** Mean and Median Absolute Error of P50 predictions.
- **Pinball Loss:** Quantile-specific asymmetric loss (τ = 0.50, 0.80, 0.95).
- **Empirical Coverage:** % of true values ≤ predicted quantile.
- **Interval Width:** Mean of (P80 − P50) and (P95 − P50) across snapshots.
- **Vessel-Clustered Bootstrap CI:** 95% confidence intervals via B = 2,000 bootstrap iterations resampling unique MMSIs with replacement.

## Measured Results — Protocol v2.1 (Unconditioned)

Protocol v2.1 (`docs/EVAL_PROTOCOL_V2_1.md`) evaluates **all** state-triggered corridor snapshots, bucketed by distance-to-berth (observable at prediction time). No filtering by true remaining duration.

**Test set: 5,067 snapshots from 264 unique vessels (March 2024). 17.4% have true remaining > 3h. 13.0% are from voyages that anchored midway.**

| Distance Bucket | N | Kinematic MAE | Bucket Median MAE | QR P50 MAE [95% CI] | QR P50 MedAE | P95 Coverage [95% CI] |
|---|---|---|---|---|---|---|
| Near Berth (≤ 10 km) | 1,544 | 2.148 h | 1.659 h | 1.768 h [0.919, 2.755] | **0.145 h** (8.7 min) | 98.4% [96.7, 99.6] |
| Mid-Channel (10–25 km) | 904 | 2.681 h | 2.271 h | 2.454 h [1.530, 3.446] | **0.225 h** (13.5 min) | 97.9% [96.5, 99.2] |
| Outer Channel (25–50 km) | 2,149 | 3.275 h | 2.948 h | 2.946 h [1.846, 4.154] | **0.252 h** (15.1 min) | 97.3% [95.5, 99.0] |
| Beyond Entrance (> 50 km) | 470 | 6.370 h | 5.899 h | 5.381 h [3.911, 6.973] | **0.416 h** (25.0 min) | 98.7% [97.5, 99.8] |
| **Overall** | **5,067** | **3.113 h** | **2.708 h** | **2.725 h** [1.785, 3.723] | **0.211 h** (12.7 min) | **97.9%** [96.4, 99.1] |

**Interpretation:**
- MedAE shows that the **typical** (non-anchoring) prediction is accurate to ~9–15 minutes.
- MAE is dominated by the ~13% of voyages that anchor mid-channel for 12–40+ hours.
- The model's MAE (2.725h) does NOT beat the bucket median baseline (2.708h). The model's value is in providing calibrated uncertainty intervals.
- P95 coverage of 97.9% confirms honest uncertainty quantification — the intervals are wide because the underlying distribution has a heavy tail.

### Why Protocol v2 Results Are Not the Headline

An earlier evaluation (Protocol v2, `docs/EVAL_PROTOCOL_V2.md`) reported MAE of ~14 minutes. **Those results suffer from selection bias**: snapshots were filtered to `0 < y_rem ≤ 3.5h` (using the true outcome as a filter) and bucketed by true remaining duration. This artificially excluded vessels that anchored, stopped, or had remaining times > 3.5 hours — exactly the hard cases that dominate real-world error. Protocol v2 results are retained in `docs/EVAL_PROTOCOL_V2.md` and `D:\ORCASTRA_DATA\historical\results\tier1_v2_short_horizon_results.json` as a documented artifact of selection-biased evaluation but are **not** the headline result.

## Repository Structure

```
ORCASTRA/repo/
├── docs/
│   ├── DECISIONS.md                 # All design decisions (D001–D022)
│   ├── EVAL_PROTOCOL.md             # Protocol v1 (superseded)
│   ├── EVAL_PROTOCOL_V2.md          # Protocol v2 (selection-biased, retained as artifact)
│   └── EVAL_PROTOCOL_V2_1.md        # Protocol v2.1 (unconditioned, current)
├── scripts/
│   ├── offline_replay_demo.py       # One-command offline replay demo
│   ├── run_tier1_v2_1_evaluation.py # Frozen v2.1 benchmark script
│   ├── run_tier1_v2_evaluation.py   # Frozen v2 benchmark script
│   ├── pull_noaa_3mo.py             # NOAA data download script
│   ├── autostart_collector.py       # Windows Task Scheduler autostart supervisor
│   └── ...                          # Audit, analysis, and baseline scripts
├── services/collector/
│   ├── src/orcastra_collector/       # AIS collector package
│   │   ├── collector.py             # WebSocket collector
│   │   ├── parser.py                # AIS message parser
│   │   ├── writer.py                # Crash-safe Parquet writer
│   │   ├── ledger.py                # Session ledger (SQLite)
│   │   ├── replay.py                # Parquet replay module
│   │   ├── schema.py                # Canonical AIS event schema
│   │   ├── vessel_cache.py          # Vessel type cache
│   │   └── config.py                # Configuration constants
│   ├── tests/
│   │   ├── test_collector.py        # 52 assertions: parser, dedup, writer, replay
│   │   └── test_forced_kill.py      # 7 assertions: kill + crash recovery
│   ├── pyproject.toml               # Package definition
│   └── orcastra_collector_task.xml  # Windows Task Scheduler XML
├── .env.example                     # Environment variable template
├── .gitignore
├── LICENSE                          # MIT
├── M0_SUMMARY.md                    # Milestone 0 summary
├── PROGRESS.md                      # Progress tracking
└── README.md                        # This file
```

Data files (Parquet, SQLite, recordings) live outside the repository at `D:\ORCASTRA_DATA\` and are not committed.

## Setup

### Prerequisites
- Python 3.12+
- Windows 10/11 (Task Scheduler integration is Windows-specific)

### Environment

```bash
# Clone the repository
git clone <repo-url>
cd ORCASTRA/repo

# Create .env from the template
copy .env.example .env
# Edit .env and paste your AISStream API key (get one at https://aisstream.io)

# Install the collector package
cd services/collector
pip install -e .
cd ../..

# Install analysis dependencies (for evaluation scripts)
pip install polars numpy scikit-learn pyarrow
```

### Running Tests

```bash
# Collector test suite (52 assertions)
python services/collector/tests/test_collector.py

# Forced kill + crash recovery tests (7 assertions)
python services/collector/tests/test_forced_kill.py
```

### Offline Replay Demo

Replays a saved AISStream recording and prints short-horizon ETA predictions from the frozen model. **Requires no network access.** Requires NOAA historical data at `D:\ORCASTRA_DATA\historical\noaa_3mo\` and a saved recording session.

```bash
python scripts/offline_replay_demo.py
```

The demo re-trains the deterministic model (identical parameters, `random_state=42`) from the NOAA historical data and applies it to the recorded session. Output is clearly labelled `OFFLINE REPLAY — RECORDED DATA, NOT LIVE`.

### Live Collection (Not Required for Evaluation)

```bash
# Start a collection session (requires .env with valid API key)
python scripts/autostart_collector.py --duration 60

# Check environment guards only
python scripts/autostart_collector.py --check-guards
```

## Security

**The AISStream API key must be regenerated by the repository owner on [aisstream.io](https://aisstream.io) before the repository is pushed publicly.**

The `.env` file containing the API key is excluded from version control via `.gitignore`. Only `.env.example` (with placeholder values) is tracked. A full secrets audit confirmed the key is not present in any tracked file or git commit history.

## License

[MIT](LICENSE)
