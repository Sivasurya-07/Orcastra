# ORCASTRA — Progress Log

## M0: Feasibility and Recording

**Status:** In progress
**Started:** 2026-10-07

### Phase 1: C: Space Recovery & Tooling ✅

| Action | Result |
|---|---|
| pip cache purged | 1413 files removed (~3.0 GB) |
| npm cache purged | Cleared (~4.3 GB) |
| HF cache moved to D:\Caches\huggingface | 1431 MB moved, verified with real download |
| UV_CACHE_DIR set | D:\Caches\uv |
| UV_PYTHON_INSTALL_DIR set | D:\Caches\uv\python |
| HF_HOME set | D:\Caches\huggingface |
| Python 3.12.15 installed via uv | D:\Caches\uv\python\cpython-3.12.15-windows-x86_64-none |
| uv 0.12.23 installed | C:\Users\Sivasurya\.local\bin |
| **C: free space** | **15.61 GB** (was 7.16 GB) |
| **D: free space** | **95.12 GB** |

### Phase 2: Region Selection & Data Sources

| Item | Result |
|---|---|
| AISStream verified (D002) | Free, no card, multi-bbox, WebSocket |
| DMA (Danish Straits) | Not evaluated / source inaccessible (D008) |
| NOAA MarineCadastre (Houston) | Verified, free, GeoParquet on Azure (D004) |
| Digitraffic (Finland) | Backup only, no historical (D005) |
| Rotterdam | Eliminated (no free historical source) |
| **Region selected** | **Houston Ship Channel (provisional)** |

### Phase 3: Historical Validation

| Item | Result |
|---|---|
| NOAA 3-day sample (Jan 15-17, 2024) | 1,519,892 records, 916 vessels, 351.8 msgs/min |
| Approach rate (filtered) | **1.53/hr [1.24, 1.82] 95% CI** |
| Raw 7.75/hr rate | Rejected (tugs, boundary artifacts, moored vessels) |
| Data quality | Excellent (0.01% SOG null, 0 sentinels) |
| CI-width gate | **Unevaluable** (needs real sigma from pilot residuals) |

### Phase 4: Collector Build & Corrections ✅
 
| Item | Result |
|---|---|
| Canonical schema | 33 fields (`aisstream_received_time`, `ais_payload_timestamp`), 4 distinct timestamps |
| Parser | Class A/B position + ShipStaticData + StaticDataReport (type 24) |
| Writer | Crash-safe: write-to-tmp, atomic rename, part immutability verified |
| Ledger | SQLite WAL: sessions, heartbeats, files, quality counters |
| Vessel cache | SQLite, persists across sessions, 103 typed vessels accumulated |
| Dedup redesign | Key includes `message_type` + payload timestamp; recomputed live duplicates: 1 |
| Frame counter | 4,058 frame reconciliation labeled retroactive; independent counter active from next session |
| Tests | **59 passed, 0 failed** (52 unit across 13 methods + 7 forced-kill) |
| Forced-kill test | Verified on isolated worker subprocess; live collector was not killed |
| Local backup/restore | Verified via `backup_local.py`: 32 files, 645 KB archive, SHA-256 and SQLite verified |
| Sleep detection | Ledger heartbeat gap monitored; max gap in 75-min run was 60.1s (0 sleep events) |
| Scoreboard | Restored to prediction scoring; marked BLOCKED until ETA model exists |
| Healthy thresholds | Provisional: 60 min, 100 vessels, 1% quarantine; healthy lookback window rule defined |

### Phase 5: ETA Baseline Feasibility Experiment ✅

| Item | Result |
|---|---|
| 28-day NOAA sample (Jan 1–28 2024) | 14,760,029 rows, **283.65 MB (0.277 GB)** on D: (under 2 GB limit) |
| Seasonal scope | Single winter season limitation documented (sea fog, winter cold fronts) |
| Label audit | 40,513 raw entries $\rightarrow$ **276 valid completed commercial arrivals** across 239 unique vessels |
| Exclusions | 1,763 left-censored, 44 right-censored, 37,649 stationary/moored at entrance, 781 out-of-bounds duration |
| Chronological split | Days 1–21 Train (216 arrivals), Days 22–28 Test (60 arrivals); zero leakage |
| Point-in-time features | Instantaneous kinematics + vessel dimensions at $t_0$; 0% missingness on tested features |
| Naive reference ($d / \text{SOG}$) | Test MAE = 6.771h, RMSE = 16.165h, MedAE = 1.952h, $\sigma = 14.802$h |
| Historical Train Median | Test MAE = 5.550h, RMSE = 15.599h, **MedAE = 0.559h**, $\sigma = 14.875$h |
| Baseline (HistGradientBoosting) | Test MAE = 9.831h, RMSE = 15.847h, MedAE = 4.613h, $\sigma = 15.909$h |
| Root cause of $\sigma \approx 15.9$h | Anchorage waiting time (12–48h) at Bolivar Roads awaiting pilots/berths |
| CI-width gate | Evaluated with real $\sigma = 15.91$h: $N \ge 3,890$ arrivals (~395 days) for $\delta = \pm 0.5$h |
| CI-width gate status | **BLOCKED** for narrow margins until anchorage/congestion is modeled in Tier 1 |

### Phase 5: ETA Baseline & Audit Evidence ✅

| Item | Result |
|---|---|
| 28-day NOAA sample (Jan 1–28 2024) | 14,760,029 rows, **283.65 MB (0.277 GB)** on D: (under 2 GB limit) |
| Single-season scope | Winter 2024 limitation documented (sea fog, cold fronts) |
| Label audit | 40,513 raw entries (37,649 slow entries verified as ping-level jitter) $\rightarrow$ **276 completed arrivals** across 239 vessels |
| Right-censoring | Cutoff corrected to 72h; sample edge cases documented |
| Chronological split | Days 1–21 Train (216 arrivals), Days 22–28 Test (60 arrivals); 13 overlapping MMSIs across separate voyages; 0 leakage |
| Models on test set | Naive MAE 6.77h, Hist Train Median MAE 5.55h (MedAE 0.56h), Baseline MAE 9.83h |
| Quantile Regression (loss=quantile) | P50 (MAE 5.72h, MedAE 0.81h, cov 40%), P80 (cov 76.7%), P95 (cov 91.7%) |
| Empirical Stage Decomposition | Direct transits ($\le 6$h, 78.6%): mean 3.75h, **std 0.63h**; Anchored transits ($>6$h, 21.4%): mean 39.35h, **std 18.78h** |
| Live Observability Horizon | Median transit 3.97h; at 3h only 11.6% scorable; at 5h 77.2% scorable; scoreboard horizon strictly bounded by continuous session length |
| CI-width gate status | Asymptotic sample size illustrative for mean residual only; final gate applies to MAE using vessel-level bootstrap CIs; remains **BLOCKED** |

### Phase 6: Final Corrections ✅

| Item | Result |
|---|---|
| Right-censoring | Corrected to $t_0 \le T_{\max} - 72$h; arrivals 270 → **257**, test 60 → **41** |
| Stage labels | **AIS-behavior** (Bolivar Roads dwell): 239 direct (93%) + 18 anchored (7%); old duration labels DROPPED |
| "σ = 0.63h" claim | **DROPPED** (circular: defined by picking fast outcomes) |
| Full-trip session sim | 3h: 0.07/day, 5h: 0.61/day, 7h: 1.52/day — old "scorable at 5h: 77.2%" was overstated ~12× |
| Short-horizon sim | 1h horizon / 7h session: **3.44/day** → N≥30 in ~9 days |
| Vessel-disjoint split | N=30 disjoint test: Hist Median MAE=4.381h (no upward bias from overlap) |
| Bootstrap CIs | Hist Median MAE=6.473h [2.356, 11.271]; CI-width gate BLOCKED |
| Quantile eval | P50 cov=36.6%, P80 cov=70.7%, P95 cov=90.2% (all under-cover) |
| Non-inbound COG | 23 of 257 arrivals; kept with documentation |
| Pass-through vessels | 226 documented as selection effect (upstream terminal traffic) |

### M0 Closeout Status (2026-10-14) — CLOSED ✅
- Houston Ship Channel confirmed as target operational region.
- Hybrid data design confirmed: NOAA historical for training, AISStream for live evaluation.
- Full-trip ETA designated historical replay only.
- Live scoreboard scoped to short-horizon ETA (last 1–2 hours before berthing).
- Total M0 agent-hours: ~12h of 32h hard stop.
- Git commit: `22e8d99` (30 files, 7,293 lines).

---

## Tier 1: Approved Preparation Tasks (2026-10-26)

### Task 1: NOAA Jan–Mar 2024 Data Pull & Full Arrival Audit ✅

| Item | Result |
|---|---|
| Temporal scope | 91 consecutive days: 2024-01-01 to 2024-03-31 (full Q1 2024) |
| Geographic coverage | Houston bounding box `[29.0, 29.9] x [-95.3, -94.4]` (0 violations) |
| Columns retained | Slim schema (13 columns): dropped vessel_name, imo, call_sign, transceiver |
| Storage footprint | **873.04 MB** on D: (under 1,024 MB hard budget). Jan 1–28 hardlinked from `noaa_28d` (0 extra blocks). |
| Existing Jan 28-day data | Preserved untouched in `D:\ORCASTRA_DATA\historical\noaa_28d` (283.65 MB) |
| Resumability & integrity | SHA-256 checksums recorded for all 91 daily files in `manifest.json` |
| Total records | **51,058,803 rows** across 91 parquet files |
| Data hygiene | 1,444 internal duplicates (0.0028%), 0 geo violations, 0 timestamp violations |
| Monthly rows & vessels | • 2024-01: 16,356,580 rows, 2,211 unique vessels<br>• 2024-02: 16,661,549 rows, 2,327 unique vessels<br>• 2024-03: 18,040,674 rows, 2,448 unique vessels |
| Commercial subset | 7,137,652 records across 1,608 unique commercial MMSIs |
| **72h Arrival Audit (Full 91d)** | • Raw entrance approaches (pings): 154,563<br>• **Discrete boundary crossings:** 4,672 (19 left-censored, 28 stationary at crossing, 190 right-censored, 3,427 upstream/through-traffic, 1,008 completed)<br>• **Valid completed arrivals:** 1,008 (~11.0/day) |
| Monthly completed arrivals | • 2024-01: **320 arrivals** (270 unique vessels)<br>• 2024-02: **357 arrivals** (291 unique vessels)<br>• 2024-03: **328 arrivals** (270 unique vessels) |
| Unique commercial vessels | **679 unique vessels** with completed arrivals (483 single-trip, 196 multi-trip) |
| Stage labels & >12h dwells | • 228 trips > 12h: Old Bolivar-only captured 73 (32.0%).<br>• Investigation revealed vessels waited in **Gulf Fairway / Offshore Anchorage** (86 voyages) and **Galveston Harbor** (21 voyages).<br>• Comprehensive anchorage captures **154 voyages (63.6% of >12h trips)**. |
| Directionality & Exclusion | Inbound: 899 arrivals (median duration 3.85h, MAE 6.69h).<br>Non-inbound: 109 arrivals (median duration **41.36h**, MAE **36.58h**, heading SSE away from port). **Strictly excluded from benchmark.** |
| Duration distribution (h) | Min 2.36h, P10 2.81h, P25 3.42h, **Median 3.89h**, P75 4.67h, P90 42.18h, Max 71.82h |

### Task 2: Recording Auto-Start & Supervisor Service (Fixed & Verified) ✅

| Item | Result |
|---|---|
| Scheduled task | `\ORCASTRA_Collector` registered in Windows Task Scheduler |
| Schedule & trigger | **Logon trigger** (`<LogonTrigger>` with `<UserId>LAPTOP-EHI6MKH4\Sivasurya</UserId>`); runs-missed-task-ASAP (`StartWhenAvailable=true`); runs only on AC power (`DisallowStartIfOnBatteries=true`) |
| Maximum session | 6 hours per session window (`ExecutionTimeLimit=PT6H`) |
| Timezone conversion | System operates on Indian Standard Time (IST, UTC+05:30). $\text{UTC} = \text{IST} - 5\text{h}30\text{m}$. (12:00 IST = 06:30 UTC). Dual timestamps logged per session. |
| Disk guards | Hard checks: D: $\ge 5.0$ GB, C: $\ge 3.0$ GB pre-flight and every 60s. Halts on violation. |
| Sleep detection (90s) | Threshold raised to **90.0s**. Cleanly ends session on sleep jump, starts fresh session on wake. |
| Heavy flush resilience | Tested via `--test-slow-flush`: 15s simulated slow flush did **NOT** trigger sleep threshold; session completed normally without interruption. |
| Inter-session semantics | Consecutive sessions across sleep do not merge into a healthy window (AIS messages permanently lost during sleep). Short-horizon ETA (1–2h lookback) only requires falling within one continuous post-wake session. |
| Crash/Kill recovery | Tested via `--test-kill`: unclosed sessions in `ledger.sqlite` automatically audited and updated to `crashed: Recovered on restart`. |
| Live Cohort A definition | Operationalized with strictly live fields: $\ge 3$ consecutive reports in 15 min, mean $\text{SOG} \ge 3.0$ kn, median $\text{COG} \in [270, 360] \cup [0, 30]$, length $\ge 50$m *only if reported*. |

---

## Tier 1: Frozen Baseline Protocol & Single Evaluation Run ✅

### Protocol Freeze (docs/EVAL_PROTOCOL.md)
- Committed to git prior to modeling: commit `1a57426`.
- Scope: Train Jan 1 – Feb 29 (60 days), Test March 1 – March 31 (31 days).
- Eligible test arrivals: strictly $t_0 \le T_{\max} - 72\text{h}$ (March 1 to March 28). Entrants after March 28 right-censored.
- Direction: Inbound COG only. Non-inbound excluded.

### Baseline Evaluation Results (Train $N=665$, Test $N=320$ arrivals)

| Model | MAE (hours) | 95% Bootstrap CI (MAE) | CI Width | MedAE (hours) | Pinball Loss | Empirical Coverage |
|---|---|---|---|---|---|---|
| **Historical Train Median** (3.87h) | **5.871 h** | [4.355, 7.490] | 3.135 h | 0.451 h | — | — |
| **Naive Kinematic** ($d/\text{SOG}$) | 6.851 h | [5.312, 8.524] | 3.212 h | 1.542 h | — | — |
| **Quantile Regression P50** | 5.993 h | [4.526, 7.598] | 3.072 h | **0.393 h** | 2.996 h | 60.0% (target 50%) [54.2%, 65.3%] |
| **Quantile Regression P80** | 12.603 h | [11.153, 13.968] | 2.814 h | 8.078 h | 4.682 h | 78.1% (target 80%) [73.8%, 82.4%] |
| **Quantile Regression P95** | 23.487 h | [21.824, 25.141] | 3.316 h | 22.946 h | 3.018 h | 87.2% (target 95%) [83.5%, 90.6%] |

- **Quantile Monotonicity:** 29 inversions out of 320 test instances in v1 (independent tree fitting).
- **Vessel-Disjoint Sensitivity Analysis:**
  - **Overlapping cohort** ($N=111$ arrivals, seen in train): Hist Median MAE = **3.364h**, P50 MAE = **3.504h**, P95 coverage = 89.2%. (Repeating liner vessels have predictable direct transits).
  - **Disjoint cohort** ($N=209$ arrivals, novel MMSIs): Hist Median MAE = **7.202h**, P50 MAE = **7.314h**, P95 coverage = 86.1%.
- **Protocol Gate Outcome:** **STOP/REVISE.**
  - Full-trip entrance kinematics cannot predict downstream anchorage delays without queue features. Led directly to Protocol v2.

---

## Tier 1: Protocol v2 Execution & Short-Horizon Benchmark ✅

### Protocol v2 Freeze (docs/EVAL_PROTOCOL_V2.md)
- Committed to git prior to v2 run: commit `55ca9c3`.

### 1. Reconciled Full-Trip Benchmarks (Inbound Only)
- **Cohort A1 (Strict First-Crossing Inbound, $N=899$):**
  - Train (Jan-Feb): 609 arrivals | Test (March): 290 arrivals.
  - Historical Train Median (scalar = 3.89h): Test MAE = **6.282 h** [95% CI: 4.655, 8.055].
- **Cohort A2 (Inbound with Re-Entry Allowed, $N=985$):**
  - Train (Jan-Feb): 665 arrivals | Test (March): 320 arrivals.
  - Historical Train Median (scalar = 3.87h): Test MAE = **5.871 h** [95% CI: 4.355, 7.490].
- **Old results preserved:** Prior results preserved in `tier1_baseline_results.json`.

### 2. Anchorage Timing & Pre-Entrance Feature Value
- **Full Population ($N=1,008$ arrivals):** 564 (56.0%) waited before $t_0$ only; 43 (4.3%) after only; 195 (19.3%) both before and after; 206 (20.4%) direct transit.
- **Long Trips ($>12$h, $N=228$):** **99.1% (226 of 228)** had dwells **AFTER $t_0$** (185 waited both before and after, 41 after only).
- **Pre-entrance feature correlations with transit duration:** Pre-$t_0$ dwell hours $r = 0.019$; Pre-$t_0$ mean SOG $r = 0.069$. Pre-entrance kinematics have negligible predictive value for channel duration.

### 3. Short-Horizon State-Based ETA (March Test Set: $N=4,215$ Snapshots, 259 Vessels)
- Conformal calibration fitted on held-out slice (Feb 16–29); test set completely untouched until frozen run.
- Quantile monotonicity strictly enforced via sorting (**0 inversions**).

| Horizon Bucket | Test Snapshots | Naive ($d/\text{SOG}$) MAE | Hist Segment Median MAE | Calibrated P50 MAE | 95% Bootstrap CI (MAE) | CI Width | MedAE | P95 Coverage | Unseen Vessels MAE (P50) | Unseen Vessels P95 Coverage |
|---|---|---|---|---|---|---|---|---|---|---|
| **$\le 1$ hour** | 1,020 | 0.487 h (29.2m) | 0.459 h (27.5m) | **0.262 h (15.7m)** | [0.227, 0.297] | 0.070 h (4.2m) | **0.148 h (8.9m)** | **100.0%** [100.0%, 100.0%] | 0.269 h (16.1m) | 100.0% |
| **1–2 hours** | 1,726 | 0.787 h (47.2m) | 0.441 h (26.5m) | **0.234 h (14.0m)** | [0.206, 0.261] | 0.055 h (3.3m) | **0.148 h (8.9m)** | **99.1%** [98.1%, 99.9%] | 0.239 h (14.3m) | 99.7% |
| **2–3 hours** | 1,469 | 0.928 h (55.7m) | 0.284 h (17.0m) | **0.230 h (13.8m)** | [0.207, 0.256] | 0.049 h (2.9m) | **0.158 h (9.5m)** | **98.6%** [97.2%, 99.7%] | 0.229 h (13.7m) | 99.4% |
| **Overall ($\le 3$h)** | **4,215** | 0.763 h (45.8m) | 0.390 h (23.4m) | **0.239 h (14.3m)** | [0.212, 0.267] | **0.054 h (3.2m)** | **0.151 h (9.0m)** | **99.2%** [98.2%, 99.8%] | **0.243 h (14.6m)** | **99.7%** |

- **Key Takeaway:** Short-horizon ETA along the channel corridor achieves **$\text{MAE} \approx 14$ minutes and $\text{MedAE} \approx 9$ minutes**, unblocking the CI-width gate ($\text{CI width} = 3.2$ minutes) and generalizing to novel vessels with zero degradation ($0.239$h overall vs $0.243$h unseen).

---

### Task Scheduler Real Test Verification ✅
- Triggers active: `<LogonTrigger>`, `<SessionStateChangeTrigger>` (`SessionUnlock`), and `<EventTrigger>` (`Power-Troubleshooter Event ID 1`).
- Real run triggered via `schtasks /Run /TN "ORCASTRA_Collector"`.
- Clean execution verified in `D:\ORCASTRA_DATA\autostart.log` with dual UTC and IST timestamps.
- Forced kill tested: unclosed session `d4014118` audited and recovered to status `crashed` by supervisor.

---

### Environment & Budget Accounting (2026-10-26)

| Metric | Value | Budget / Cap | Status |
|---|---|---|---|
| **C: Free Space** | **28.81 GB** | $\ge 3.0$ GB | PASS (healthy) |
| **D: Free Space** | **88.27 GB** | $\ge 5.0$ GB | PASS (healthy) |
| **Historical Data (D:)** | **873.04 MB** | $\le 1,024$ MB (1.0 GB) | PASS (within 1 GB budget) |
| **M0 Hours Used** | **~12.0 hours** | 32.0 hours hard stop | Completed |
| **Tier 1 Hours Used** | **~2.0 hours** | — | Preparation & v2 benchmark |
| **Total Agent Hours** | **~14.0 hours** | 32.0 hours | Well within safety margin |

### Current Status
- **STOPPED.**
- Protocol v2 frozen and committed (`docs/EVAL_PROTOCOL_V2.md`).
- Short-horizon benchmarks and full-trip reconciliations executed and documented (`tier1_v2_short_horizon_results.json`).
- Experiment 2 (queue features) predefined and deferred per user order.
- No further models run. No live collection beyond scheduled test run.




