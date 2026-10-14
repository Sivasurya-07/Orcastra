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

### Current Environment & Status (2026-10-14)

| Metric | Value |
|---|---|
| C: free space | **~10 GB** |
| D: free space | **~38 GB** |
| NOAA 28-day data (D:) | 283.65 MB |
| Live session data (D:) | 35.5 MB |
| Collector test suite | 59/59 passing |
| **M0 agent-hours used** | **~12h of 32h hard stop** |

### Design Conclusions (M0)

1. **Full-trip ETA** (entrance-to-berth) → **historical replay only.** Live daily sessions cannot capture both endpoints.
2. **Short-horizon ETA** (1–2h before arrival) → **live scoreboard.** Feasible at ~3.4 arrivals/day with 7h sessions.
3. **CI-width gate remains BLOCKED** until anchorage features reduce variance.
4. **Stage labels** from AIS behavior (anchorage dwell), not from duration thresholds.

### Next Steps

- [ ] User review of final corrections
- [ ] Jan–Mar 2024 NOAA data pull (≤1.0 GB on D:, stream-filtered via DuckDB) — approved offline analysis
- [ ] Tier 1 build (awaiting user approval)

