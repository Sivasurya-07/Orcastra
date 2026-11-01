# ORCASTRA — Operational Real-time Channel & Ship Transit Risk Analyzer

Short-horizon ETA prediction for commercial vessels transiting the Houston Ship Channel, using AIS position data from NOAA historical archives and AISStream live feeds.

## Problem Statement

Predicting when a commercial vessel will reach its berth after entering the Houston Ship Channel is operationally valuable but statistically hard. Roughly 13% of inbound voyages anchor mid-channel for 12–40+ hours awaiting pilot dispatch or berth clearance, creating a heavy-tailed distribution that no simple model can reliably point-predict. Full-trip ETA (entrance-to-berth) has a historical median baseline MAE of 6.3 hours — driven almost entirely by the unpredictable anchorage tail.

**ORCASTRA focuses on short-horizon ETA**: predicting remaining transit time for vessels already actively underway inside the channel corridor. For this task, a quantile regression model achieves a Median Absolute Error of **12.7 minutes overall** and **8.7 minutes near the berth**, with calibrated 95% prediction intervals that achieve 97.9% empirical coverage on a held-out chronological test set.

### Scope and Limitations

This is a research prototype, not a production system. Specifically:

- **Point estimates across simple baselines and QR are statistically indistinguishable**: Segment median is 2.681 h, distance-bucket median is 2.708 h, and QR P50 is 2.725 h. Their vessel-clustered bootstrap confidence intervals overlap heavily, and no model clearly beats the simple baselines. The QR model's primary contribution is in providing calibrated prediction _intervals_, not superior point estimates.
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
- Segment median (2.681 h), bucket median (2.708 h), and QR P50 (2.725 h) have heavily overlapping bootstrap confidence intervals and are statistically indistinguishable; no model clearly beats the simple baselines. The QR model's value is in providing calibrated uncertainty intervals.
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
- ~1 GB free disk space for NOAA historical data (on D: by default, or set `ORCASTRA_DATA_DIR`)

### Environment

```bash
# Clone the repository
git clone https://github.com/Sivasurya-07/Orcastra.git
cd Orcastra/repo

# Create .env from the template
copy .env.example .env
# Edit .env and paste your AISStream API key (get one at https://aisstream.io)

# Install the collector package
cd services/collector
pip install -e .
cd ../..

# Install analysis dependencies (for evaluation scripts and full model)
pip install polars numpy scikit-learn pyarrow
```

### Downloading NOAA Historical Data

The evaluation scripts and full-model replay require NOAA MarineCadastre AIS data for Jan–Mar 2024:

```bash
# Downloads 91 daily Parquet files (~873 MB total) to D:\ORCASTRA_DATA\historical\noaa_3mo\
python scripts/pull_noaa_3mo.py
```

- **Default path:** `D:\ORCASTRA_DATA\historical\noaa_3mo\` (set via `ORCASTRA_DATA_DIR` env var or edit `scripts/pull_noaa_3mo.py`)
- **Expected size:** ~873 MB (91 daily files, ~560K rows/day, bbox-filtered to Houston Ship Channel)
- **Resumable:** Skips already-downloaded files with SHA-256 checksum verification
- The offline replay demo works **without this data** using a kinematic fallback, but model-based predictions require it.

### Running Tests

```bash
# Run all tests using project uv environment (Python 3.12):
uv run --project services/collector pytest services/collector/tests/ -v

# Or run individual test suites directly:
uv run --project services/collector python services/collector/tests/test_collector.py
uv run --project services/collector python services/collector/tests/test_forced_kill.py
uv run --project services/collector python services/collector/tests/test_live_eta_and_dashboard.py
```

### Live ETA & Operations Dashboard

Streamlit + Plotly interactive operations dashboard featuring dual modes:

```bash
# Launch dashboard in project uv environment (Python 3.12):
uv run --project services/collector streamlit run scripts/dashboard.py
```

**Features & Architecture:**
- **Dual Modes:**
  - `LIVE`: Connects to atomic `latest_state.json` written by the collector subprocess. Start/Stop buttons manage the collector as an independent child subprocess. Does not auto-start at launch.
  - `OFFLINE REPLAY`: Clearly labelled with a persistent red banner (`🔴 OFFLINE REPLAY — RECORDED DATA, NOT LIVE`). Replays saved recordings or the bundled test fixture from disk with **zero network access** and no API key.
- **Graceful Offline Map Degradation:**
  - The Houston Ship Channel radar map degrades gracefully with no internet: if tile servers are unreachable, it renders on a clean dark nautical coordinate plane with full axis grids, boundary polygons (entrance, corridor, destination berth box), waypoints, and vessel position markers still accurately plotted.
- **Protocol v2.1 Live Dual ETA:**
  - Displays **BOTH** the **Distance-Bucket Median ETA** and the **Kinematic ETA ($d / \text{SOG}$)**, each labelled, with the operational note: *"Bucket median has lower average error because vessel holds dominate; kinematic is shown for comparison."*
  - Point estimate baselines are statistically indistinguishable (Segment median 2.681 h, bucket median 2.708 h, QR P50 2.725 h with heavily overlapping bootstrap CIs; no model clearly beats the simple baselines).
  - Bucket median baseline values:
    - *Near Berth ($d \le 10$ km):* 1.010 h (~60.6 min)
    - *Mid-Channel ($10 < d \le 25$ km):* 1.759 h (~105.5 min)
    - *Outer Channel ($25 < d \le 50$ km):* 2.591 h (~155.5 min)
    - *Beyond Entrance ($d > 50$ km):* 3.539 h (~212.3 min)
  - Displays P50 point estimates; UI explicitly states that the **P95 uncertainty interval is ~40 h wide** (~41.8 h per Protocol v2.1).
  - Enforces D017 Cohort A live eligibility: untyped vessels are treated as unknown commercial candidates, while known non-commercial craft (tugs, pleasure craft, pilots, fishing) are strictly excluded from scoring.
  - Staleness cutoff: positions older than 5 minutes or with missing coordinates are marked "unavailable" (never shown as actively present).
- **Process Lock & Single Instance:**
  - An atomic lock file (`collector.lock`) prevents concurrent collector runs between Task Scheduler and the dashboard UI. Displays clear error with PID if an instance is already active.
- **Immutable Prediction Logger & Live Scoreboard:**
  - Predictions are immutably logged to `D:\ORCASTRA_DATA\prediction_log.sqlite`.
  - When tracked vessels reach the berth box ($SOG < 1.5$ kn), actual arrival duration and absolute error are computed. The live scoreboard reports `"collecting (n = X)"` until $n \ge 30$, covering UTC dates and operational hours.
### Hosted Demo (Streamlit Community Cloud)

Hosted demo (recorded replay): https://orcastra-gwdgnnyxtdl4fcnkqm5uwr.streamlit.app

Live mode runs locally (see Setup); free hosting cannot run a persistent collector.

**Deployment Configuration:**
- **Repository:** `Sivasurya-07/Orcastra`
- **Branch:** `main`
- **Main file path:** `scripts/dashboard.py`
- **Python version:** `3.12` (specified in `.python-version` and repository dependencies in `requirements.txt`)
- **Environment variables (optional):** `DEPLOY_MODE=1` (automatically active if `D:\ORCASTRA_DATA` is absent)

**Hosted Demo Behavior (`DEPLOY_MODE`):**
- **Offline Replay with Real AIS Data:** Replays public AIS recording events from the bundled 75-minute session (`data/demo/recorded_session_20261008.parquet`, 4,057 events, 186 KB) instead of synthetic or minimal fixtures.
- **Public AIS Recording Provenance:** Captured from public broadcast AIS data on October 8, 2026 across the Houston Ship Channel bounding box (29.0°N–29.9°N, 95.3°W–94.4°W) over a 75-minute observation window (16:02–17:17 UTC).
- **Safety & Isolation:** Live collection controls (Start/Stop Live) and live streaming mode options are completely hidden. Never reads `.env` or opens WebSocket connections.
- **Protocol v2.1 Benchmarks:** Benchmark comparison charts and Protocol v2.1 baseline results load from bundled JSON artifacts in `data/demo/` with no local NOAA dataset required.
- **Prominent Banner:** Displays `"Hosted demo: recorded data only. Run locally for live mode."`

### Offline Replay Demo

Replays a saved AIS recording through the ETA prediction pipeline. **No network access.** No live session created.

```bash
# Uses the bundled sample fixture (20 anonymized events, 13 KB):
python scripts/offline_replay_demo.py

# Or point to a real recording session:
python scripts/offline_replay_demo.py --session D:\ORCASTRA_DATA\recordings\session_20261008_160107
```

**What it does:**
1. Loads AIS events from Parquet files in the session directory.
2. If NOAA data is available: re-trains the deterministic quantile model (identical parameters, `random_state=42`) and produces calibrated ETA intervals.
3. If NOAA data is absent: falls back to kinematic `d_remaining / SOG` estimates.
4. Prints predictions for every inbound vessel in the corridor.

A tiny **bundled sample** recording ships with the repo at `tests/fixtures/sample_session/` (20 anonymized events, 13 KB) so the demo runs immediately after a fresh clone — no recording session or data download needed.

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
