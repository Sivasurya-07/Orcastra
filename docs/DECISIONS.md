# ORCASTRA — Decisions Log

## D001: Region Selection Criteria (written BEFORE any data collection)
**Date:** 2026-10-07
**Status:** Defined

Criteria and weights for selecting the target region, written before examining any data.

| # | Criterion | Weight | Rationale |
|---|---|---|---|
| 1 | **Completed approaches per healthy hour** | **25%** | Direct gate for ETA training (section 13). Higher = faster to CI-width gate. |
| 2 | **Historical archive availability** (same waters as live feed, free, sufficient depth) | **20%** | Critical for hybrid strategy. Must cover the same waters as the live AISStream feed. |
| 3 | **Message rate** (msgs/min in bounding box) | 15% | Must produce enough data in 3–7 hr sessions for live recording value. |
| 4 | **Unique vessels/day** | 10% | Diversity of training data; more vessel types = better generalization. |
| 5 | **Coverage-gap rate** (% of hours with >5 min msg gaps in historical archive) | 10% | Affects anomaly detection quality and healthy-window yield. |
| 6 | **Arrival-definition clarity** (well-defined approach channels, anchorages, pilot boarding areas) | 10% | Affects label quality; ambiguous port geometry = noisy labels. |
| 7 | **Resource cost** (message volume vs. machine capacity: memory, CPU, disk write rate) | 10% | Must fit within 6 GB WSL cap and 30 GB disk budget. |

**Scoring method:** Each criterion scored 1–5 per region, multiplied by weight, summed. Ties broken by criterion 1 (completed approaches).

---

## D002: AISStream.io Verification
**Date:** 2026-10-07
**Status:** ✅ PASS — Free, no card, multi-bbox supported

### Terms and access
| Check | Result |
|---|---|
| Free tier exists? | **Yes.** Entirely free, no paid tier exists. |
| Credit card required? | **No.** Sign in with GitHub/Google/LinkedIn, get API key immediately. |
| Terms of use | No commercial restriction found. API key must be kept server-side. |
| $0 rule compliance | **PASS** |

### Technical details
| Item | Detail |
|---|---|
| Endpoint | `wss://stream.aisstream.io/v0/stream` |
| Authentication | API key in subscription JSON, sent within 3 seconds of connection |
| **Multi-bbox** | **Yes.** `BoundingBoxes` field accepts an array of bounding boxes. Resending a subscription replaces the active config. |
| Message types | PositionReport, ShipStaticData, ExtendedClassBPositionReport, StandardClassBPositionReport, + 15 others |
| Frame format | Binary WebSocket frames, payload is UTF-8 JSON |
| Compression | `permessage-deflate` supported; as of Sep 2026, uncompressed connections subject to per-user bandwidth limits |
| Rate limits | No explicit per-message limit documented. Bandwidth limits for uncompressed connections. Messages dropped if client consumes too slowly. |
| SLA/uptime | **None.** "No SLA or uptime guarantee, events are not durably replayed." |
| Reconnect guidance | Exponential backoff with jitter. Send complete subscription within 3s. |
| Browser connections | Not permitted; server-side only. |
| Subscription update rate | Max once per second. |
| MMSI filter limit | Up to 200 MMSIs per subscription. |

### Message schema (PositionReport)
```json
{
  "MessageType": "PositionReport",
  "MetaData": { "MMSI": int, "ShipName": str, "Latitude": float, "Longitude": float },
  "Message": {
    "PositionReport": {
      "MessageID": int, "UserID": int, "Sog": float, "Cog": float,
      "TrueHeading": int, "Valid": bool, "NavigationalStatus": int,
      "RateOfTurn": int, "PositionAccuracy": bool,
      "Longitude": float, "Latitude": float, "Timestamp": int,
      "SpecialManoeuvreIndicator": int, "Raim": bool, "CommunicationState": int
    }
  }
}
```

### Implications for collector design
- Must use `permessage-deflate` to avoid bandwidth limits
- Must send subscription within 3s of connection
- Must handle binary frames → decode bytes → parse JSON
- Reconnect with exponential backoff + jitter
- Multi-bbox subscription enables simultaneous region comparison in one session

**Source:** https://aisstream.io/documentation (read 2026-10-07)

---

## D003: Historical AIS Source — Danish Maritime Authority (DMA)
**Date:** 2026-10-07
**Status:** ✅ Viable for Danish Straits (Region 2); ❌ Does NOT cover Rotterdam approaches (Region 1)

### Access and licence
| Check | Result |
|---|---|
| URL | https://web.ais.dk/aisdata/ |
| Licence | **Free, worldwide, non-exclusive, unlimited.** Can be copied, distributed, modified, combined, used for commercial and non-commercial purposes. |
| Credit card | **No.** Direct download. |
| $0 rule | **PASS** |

### Data characteristics
| Item | Detail |
|---|---|
| Format | CSV in ZIP archives |
| Columns | 26 columns: Timestamp, Type of mobile, MMSI, Lat, Lon, Nav status, ROT, SOG, COG, Heading, IMO, Callsign, Name, Ship type, Cargo type, Width, Length, Type of pos fix, Draught, Destination, ETA, Data source type, Size A/B/C/D |
| Timestamp format | DD/MM/YYYY HH:MM:SS (base station time) |
| Size per day (uncompressed) | **~1.5–2.6 GB** (~10M rows/day for all Danish waters) |
| Size per day (compressed ZIP) | ~200–400 MB (estimate based on CSV compression ratios) |
| History depth | Real-time data retained 24 months online; older data packed and available as open data |
| Download method | Bulk ZIP download from web portal |
| Bbox filtering on download | **Not available.** Full Danish waters per file; must filter after download. |

### Coverage area
**Danish waters only.** Does NOT cover:
- Rotterdam / North Sea approaches (Dutch waters — managed by Rijkswaterstaat)
- UK waters
- Norwegian waters beyond shared boundary

**Does cover:**
- Danish Straits (Øresund, Great Belt, Little Belt) ✅
- Kattegat ✅
- Skagerrak (partially) ✅
- Danish North Sea coast (limited western extent)

### Size estimation for Region 2 (Danish Straits bbox ~55–56°N, 10.5–13°E)
- Full Denmark is ~10M rows/day
- Danish Straits is a subset; rough estimate: **30–50% of all Danish traffic** passes through the Straits
- After bbox filter: ~3–5M rows/day → ~0.8–1.3 GB CSV/day → ~120–200 MB compressed/day
- **1 month (30 days) after bbox filter ≈ 3–6 GB compressed, ~25–40 GB uncompressed CSV**
- **1 week sample for feasibility ≈ 0.8–1.4 GB compressed** (download full week, filter on ingest)
- But we download the full-Denmark daily ZIP (~200–400 MB) and filter → **1 week download ≈ 1.4–2.8 GB**

**Download plan:** Download 7 daily ZIPs (~2 GB total compressed), stream-filter by bbox into Parquet on D:, discard the CSV. Final on-disk Parquet (with zstd) for 1 week ≈ **0.3–0.6 GB**.

### Implications
- **Region 1 (Rotterdam) is eliminated as a historical-live matched pair.** DMA does not cover Dutch waters.
- **Region 2 (Danish Straits) is the only candidate with a direct DMA historical-to-AISStream-live match.**
- For Region 1 we would have no free historical source for the same waters → violates criterion 2 (20% weight).

---

## D004: Historical AIS Source — NOAA MarineCadastre
**Date:** 2026-10-07
**Status:** ✅ Viable for US port regions (Region 3)

### Access and licence
| Check | Result |
|---|---|
| URL | https://marinecadastre.gov/ais/ (AccessAIS tool); Azure: https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ |
| Licence | **Public resource, free.** Cite "BOEM/NOAA Marine Cadastre." No restrictive licence. |
| Credit card | **No.** |
| $0 rule | **PASS** |

### Data characteristics
| Item | Detail |
|---|---|
| Format | **GeoParquet** (2024–2025, analysis-ready, on Azure); CSV (zstd-compressed, 1-min frequency); Geodatabase (pre-2015) |
| Coverage | **US waters only** (US EEZ) |
| History | 2009–present (CSV); 2024–2025 (GeoParquet) |
| Total 2024 size | ~116.7 GB (full US, all months) |
| Bbox filtering | AccessAIS web tool: draw area + time range, max 2 GB per order. GeoParquet on Azure: filter with DuckDB/GeoPandas. |
| Pre-cleaning | GeoParquet files have anomalies and sentinel values removed |
| Temporal resolution | 1-minute intervals (CSV); broadcast-level (GeoParquet) |

### Size estimation for Region 3 (Houston Ship Channel bbox ~29.2–29.8°N, 94.5–95.2°W)
- Houston is one of the busiest US ports (~250+ vessel arrivals/month for large commercial vessels)
- Using AccessAIS with a focused bbox: a 1-week request would likely be well under the 2 GB limit
- GeoParquet via DuckDB remote query: can filter by lat/lon without downloading entire files
- **1 month Houston bbox ≈ 1–3 GB** (estimate; varies by traffic)

### Implications
- **Direct match for Region 3 (US port).** NOAA covers the exact same US waters observable via AISStream.
- **Cannot be paired with Region 1 or 2** (European waters).
- GeoParquet + DuckDB remote query is an efficient access pattern — no need to download full files.

---

## D005: Finnish Digitraffic Marine API
**Date:** 2026-10-07
**Status:** ⚠️ Viable as backup live source only; no historical archive

### Access and licence
| Check | Result |
|---|---|
| URL | https://www.digitraffic.fi/en/marine-traffic/ |
| Licence | Free, open data |
| Credit card | **No.** |
| Real-time API | REST/JSON + MQTT WebSocket |
| Historical data | **Not available via API.** Real-time/operational only. Some academic datasets on Zenodo. |

### Implications
- **Not suitable as a historical archive** for any candidate region
- **Useful as backup live AIS source** if AISStream ever requires a card or goes down
- Coverage: Finnish waters only → would need a Finnish port as the target region
- **Not a candidate for M0** unless AISStream fails

---

## D006: Region Candidate Assessment (pre-data)
**Date:** 2026-10-07

### Updated candidate table after verification

| # | Region | Live source | Historical source | Historical-live match? | Status |
|---|---|---|---|---|---|
| 1 | Rotterdam / North Sea approaches | AISStream ✅ | **None free** (DMA doesn't cover Dutch waters) | ❌ No match | **Eliminated** — no free historical archive for same waters |
| 2 | Danish Straits (Øresund / Great Belt) | AISStream ✅ | DMA ✅ (direct coverage match) | ✅ Direct match | **Strong candidate** |
| 3 | US port (Houston or LA-LB) | AISStream ✅ | NOAA MarineCadastre ✅ (direct match) | ✅ Direct match | **Viable candidate** |

### Key finding
**Region 1 (Rotterdam) is eliminated.** There is no free historical AIS archive covering Dutch North Sea approaches. The DMA archive covers only Danish waters. This was criterion 2 (20% weight) — Rotterdam scores 1/5 on it, making it uncompetitive even with high traffic.

### Remaining comparison: Danish Straits vs. US port

To be resolved after historical sampling (Phase 4). Preliminary qualitative assessment:

| Criterion | Danish Straits | US Port (Houston) |
|---|---|---|
| Approaches/hour | High (major international strait, funneling traffic) | High (one of world's busiest ports) |
| Historical archive | DMA: free, open, same waters, ~24 months online | NOAA: free, public, same waters, 2009–present, GeoParquet |
| Message rate | Very high (dense traffic in narrow strait) | High |
| Unique vessels | High (international transit traffic) | High (but more domestic/Gulf traffic) |
| Arrival clarity | Moderate — strait transit vs. port arrival are different things; need to define "arrival" carefully for a strait | High — Houston Ship Channel is a well-defined linear approach |
| Archive format | CSV in ZIP (requires download + filter) | GeoParquet (can query remotely with DuckDB) |
| Archive convenience | Must download full-Denmark daily files, filter locally | Can filter on Azure with DuckDB, or use AccessAIS (2GB limit) |
| Timezone/hours bias | UTC+1/+2 (closer to IST operating hours minus ~4h) | UTC-5/-6 (IST minus ~11.5h — live sessions at IST evening = Houston morning) |

**Open question for historical sampling:** Is the Danish Straits better characterized as a "strait transit" region or a port-approach region? Arrivals at Copenhagen or Malmö are port approaches; transits through the Great Belt are something different. This affects label quality.

---

## D007: Houston Historical Feasibility — NOAA MarineCadastre Sample
**Date:** 2026-10-07
**Status:** ✅ PASS — High traffic, clean data, clear arrivals

### Source details
| Item | Detail |
|---|---|
| Source | NOAA MarineCadastre (BOEM/NOAA) |
| URL | https://marinecadastre.gov/ais/ |
| Azure GeoParquet | https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/ |
| GitHub file list | https://github.com/ocm-marinecadastre/ais-vessel-traffic (accessed 2026-10-07) |
| Licence | Public resource, free. Cite "BOEM/NOAA Marine Cadastre." |
| Archive period | 2024-01-01 to 2024-12-31 (GeoParquet); 2009–present (CSV bulk) |
| **Time gap to today** | **~21 months** (archive ends Dec 2024, current date 2026-10-07) |
| Format | GeoParquet 1.0.0, Snappy compression, EPSG:4326, downsampled to nearest whole minute |
| Max daily file size | ~325 MB |
| Access date | 2026-10-07 |

### Sampling method
- **Tool:** DuckDB with `httpfs` + `spatial` extensions, querying Azure blob storage remotely
- **Query:** `ST_Y(geometry)` / `ST_X(geometry)` within Houston bounding box (29.0–29.9°N, 95.3–94.4°W)
- **No full file download** — only row groups matching the bbox are transferred
- **Sample:** 3 consecutive days (2024-01-15, 2024-01-16, 2024-01-17) — a Monday–Wednesday in mid-January

### Results

| Metric | Value |
|---|---|
| Total AIS records | **1,519,892** |
| Unique vessels (valid MMSI 200M–799M) | **916** |
| Invalid MMSI | 0.0% |
| Duration | 3.0 days (72.0 hours) |
| Messages/minute | **351.8** |
| Hourly coverage | **100%** (72/72 hours observed) |
| Missing hours | 0 |
| Low-coverage hours (<50% of median) | 3 (4.2%) |
| Hourly msg median | 21,762 |
| Hourly msg min/max | 690 / 23,092 |
| Disk usage (zstd Parquet) | **32.6 MB** |

### Approach detection

**Definition:** A vessel "approaches" when it enters a geofence zone AND has ≥1 AIS report with SOG < 3.0 kn. One approach counted per unique MMSI per zone. This is a **conservative** count — actual arrivals may be higher because vessels that slow and re-enter are counted only once.

| Zone | Records | Vessels | Approaches |
|---|---|---|---|
| Houston Ship Channel entrance (29.30–29.45°N, 94.85–94.65°W) | 283,479 | 426 | **265** |
| Galveston Bay inner (29.55–29.75°N, 95.10–94.85°W) | 278,049 | 476 | **293** |
| **Total** | **561,528** | **—** | **558** |

### Arrival rate ESTIMATE

> **⚠️ This is an ESTIMATE, not a measured fact.**

| Metric | Value |
|---|---|
| Estimated rate | **7.75 approaches/hour** |
| 95% CI (Poisson) | [7.107, 8.393] approaches/hour |
| Daily equivalent | ~186 approaches/day |
| **Source of σ** | Poisson process assumption: SE = √n / T, where n = 558, T = 72 hours |
| **δ and z** | δ is the CI half-width ÷ rate = (8.393 − 7.107) / (2 × 7.75) = 8.3%. z = 1.96 (95% level). |
| **Limitations** | 3-day mid-January sample. Seasonal, weekly, and time-of-day variation not captured. Weekend/holiday patterns not observed. NOAA data is downsampled to 1-minute resolution. |

### Data quality
| Check | Result |
|---|---|
| SOG null % | 0.01% — excellent |
| SOG sentinel (102.3) | 0 |
| SOG > 50 kn | 0 |
| Heading sentinel (511) | 0 |
| COG sentinel (360) | 0 |
| **Assessment** | **Excellent** — NOAA GeoParquet is pre-cleaned, anomalies removed |

### Vessel type distribution (top 10 by unique vessels)
| AIS vessel type code | Vessels | Category |
|---|---|---|
| 31 | 241 | Towing (tug) |
| 52 | 152 | Tug |
| 37 | 100 | Pleasure craft |
| 80 | 86 | Tanker |
| 36 | 64 | Sailing |
| 70 | 47 | Cargo |
| 57 | 37 | Spare |
| 60 | 33 | Passenger |
| 30 | 31 | Fishing |
| 89 | 15 | Tanker (other) |

### Suitability assessment
| ORCASTRA Module | Supported? | Notes |
|---|---|---|
| **ETA prediction** | ✅ Yes | High approach count (558 in 3 days), clear linear channel, sufficient vessel diversity |
| **Congestion scoring** | ✅ Yes | 351.8 msgs/min, 916 vessels, high density enables congestion measurement |
| **Anomaly detection** | ✅ Yes | 100% hourly coverage, near-zero sentinels, clean SOG/COG for baseline |
| **Scoreboard / ranking** | ✅ Yes | 916 unique vessels with vessel_type, imo, draft, cargo fields available |

---

## D008: Danish Straits Historical Feasibility
**Date:** 2026-10-07
**Status:** Not evaluated / source inaccessible

**No score assigned.** The DMA data portal (web.ais.dk) was unreachable across all access attempts (SSL mismatch, connection timeout, server disconnect). No historical data was retrieved. Danish Straits cannot be evaluated against the D001 criteria.

### Access attempts
| Attempt | URL | Result |
|---|---|---|
| 1 | `https://web.ais.dk/aisdata/aisdk-2025-06-01.zip` | SSL CERTIFICATE_VERIFY_FAILED: hostname mismatch |
| 2 | Same URL, `verify=False` | RemoteDisconnected: server closed connection |
| 3 | `http://web.ais.dk/aisdata/aisdk-2025-06-01.zip` | ConnectTimeoutError (HTTP port 80) |
| 4 | Repeated for 3 dates × 4 URL patterns | All failed |

### Assessment
- The DMA site (`web.ais.dk`) has an SSL certificate that does not match the hostname, suggesting infrastructure change or decommissioning
- HTTP connections time out, suggesting the server may be offline or geo-blocked
- Web search (2026-10-07) indicates DMA may have moved to a formal application process via dma.dk
- **No historical data was retrieved. Danish Straits cannot be evaluated against criteria.**
- This is a **data access failure**, not a fundamental unsuitability of the region
- If DMA access is restored in future, the Danish Straits remain a theoretically viable candidate

---

## D009: Region Recommendation — Houston Ship Channel
**Date:** 2026-10-07
**Status:** RECOMMENDATION (pending user approval)

### Scored comparison (criteria from D001)

| # | Criterion | Weight | Houston | Danish Straits | Houston score | Danish Straits score |
|---|---|---|---|---|---|---|
| 1 | Approaches/healthy hour | 25% | 1.53/hr [1.24, 1.82] 95% CI (filtered) | Not evaluated | **4** | N/A |
| 2 | Historical archive availability | 20% | NOAA: free, same waters, 2009-present | Not evaluated / source inaccessible | **5** | N/A |
| 3 | Message rate (msgs/min) | 15% | 351.8/min (archive), 55/min median (live) | Not evaluated | **5** | N/A |
| 4 | Unique vessels/day | 10% | ~305/day (916 / 3 days, archive) | Not evaluated | **4** | N/A |
| 5 | Coverage-gap rate | 10% | 4.2% low-coverage hours | Not evaluated | **5** | N/A |
| 6 | Arrival-definition clarity | 10% | Linear ship channel, clear geofence | Not evaluated | **5** | N/A |
| 7 | Resource cost | 10% | 32.6 MB / 3 days = manageable | Not evaluated | **4** | N/A |
| | **Weighted total** | **100%** | | | **4.55** | **N/A** |

### Recommendation

**Houston Ship Channel is the recommended target region.**

**Evidence:**
1. **Only region with verified historical data.** NOAA GeoParquet is accessible, free, pre-cleaned, and covers the exact same waters observable via AISStream.
2. **Reasonable approach rate.** 1.53/hr [1.24, 1.82] 95% CI (historical-sample estimate under documented filtering rules). The raw 7.75/hr rate is rejected as a headline due to tugs, 00:00 boundary artifacts, moored vessels, and repeated crossings.
3. **Excellent data quality.** 0.01% SOG nulls, zero sentinels, 100% hourly coverage. NOAA pre-cleaning removes anomalies.
4. **Clean arrival geometry.** The Houston Ship Channel is a long, narrow, well-defined approach with clear pilot boarding areas.
5. **Danish Straits: not evaluated / source inaccessible.** No score assigned.

**Known limitations:**
- **Time gap:** NOAA GeoParquet covers only 2024; ~21 months gap to today. Live AISStream data is current (Oct 2026).
- **Time zone:** Houston is CDT (UTC-5) in October. IST is UTC+5:30.
- **1-minute resolution:** NOAA data is downsampled. Live AISStream provides variable-rate data (median ~3 min per vessel).
- **Sample size:** Only 3 days sampled. Seasonal confidence not established.
- **CI-width gate:** Cannot be evaluated until sigma comes from real pilot residuals or baseline errors.

---

## D010: Schema Timestamps — aisstream_received_time & ais_payload_timestamp
**Date:** 2026-10-09
**Status:** IMPLEMENTED & VERIFIED

**Source:** https://aisstream.io/documentation (accessed 2026-10-09) + schema verification.

1. Official AISStream docs do not formally define `time_utc`. Community sources and empirical latency verify it is AISStream server reception time, NOT vessel transmission time. This is recorded as an evidence-backed assumption.
2. Canonical schema field renamed from `observation_time` to `aisstream_received_time`.
3. Added `ais_payload_timestamp` (0-59 seconds from decoded AIS payload `Message.*.Timestamp`).
4. Dedup key redesigned: `mmsi + aisstream_received_time (to second) + message_type [+ ais_payload_timestamp]`.
   - Old rationale ("transponders transmit once per second") was invalid.
   - New rationale: AISStream delivers different message types (e.g. PositionReport + ShipStaticData) in the same second; true duplicates are identical MMSI + reception second + message type.
   - Recomputation on 75-min live recording: 1 true duplicate (`636018912_20261008T162902_PositionReport`), eliminating 2 false positives.

---

## D011: Live Validation Summary & Pipeline Comparison (75-min session)
**Date:** 2026-10-09
**Status:** RECORDED

### Pipeline Comparison (NOAA vs AISStream Live)
Comparison over matching weekday UTC hours (16:00–17:00 UTC) within identical bounding box `[29.0, 29.9] x [-95.3, -94.4]`:

| Metric (per 10-minute window) | NOAA Archive (Jan 15–17 2024, Mon–Wed) | Live AISStream (Oct 8 2026, Thu) | Pipeline Ratio |
|---|---|---|---|
| **Unique vessels / 10 min (mean)** | **658** | **222** | **0.337** |
| **Unique vessels / 10 min (median)** | **655** | **223** | **0.340** |
| Events / 10 min (mean) | 3,711 | 504 | 0.136 |
| Events / 10 min (median) | 3,736 | 540 | 0.145 |

> **Pipeline note:** Live was ~0.34 of NOAA unique vessels per 10-minute window across different observation periods (October 2026 live vs January 2024 archive). The underlying root cause of this difference is unknown (station density, voluntary relay filtering, or seasonal traffic variations).

### Arrival Detection on 75-min Live Recording (Measured, No Extrapolation)
- Total position messages: 3,817 (100% SOG completeness) across 376 unique vessels.
- Observation distribution: Median 7 observations/vessel (mean 10.2). Vessels with $\ge 5$ obs: 257 (68.4%); $\ge 10$ obs: 126; $\ge 20$ obs: 63.
- Approaches detected (in zone & SOG < 3.0 kn):
  - HSC Entrance: 29 approaches
  - Galveston Bay Inner: 112 approaches
  - Distinct MMSIs: 141 approaches
- Commercial typed arrivals (cargo + tanker + passenger from vessel cache): **18 confirmed commercial arrivals** (50 vessels typed, 91 untyped). Zero extrapolation applied.
- Note: This zone-entry slowdown detector (SOG < 3.0 kn in zone) is an instantaneous activity detector, distinct and non-comparable to an end-to-end voyage completion detector.

---

## D012: Phase 4 Collector Build & Verification Complete
**Date:** 2026-10-09
**Status:** COMPLETE

| Component | Status | Verification Detail |
|---|---|---|
| Canonical schema (33 fields) | Done | `aisstream_received_time`, `ais_payload_timestamp` added |
| Parser (Class A/B + ShipStaticData + StaticDataReport) | Done | AIS Type 24 (`StaticDataReport`) handled |
| Crash-safe Parquet writer | Done | Atomic rename, part immutability verified |
| SQLite ledger | Done | Sessions, heartbeats, file ledger, health counters |
| Vessel type cache | Done | SQLite persistence across sessions (103 types accumulated) |
| Dedup key redesign | Done | Recomputed: 1 duplicate found in live session |
| Socket frame counter | Done | Retroactive reconciliation (4,058 frames = 4,057 JSONL + 1 sub confirmation). Independent counter active for future sessions |
| Test suite | Done | **59/59 passing** (52 assertions in `test_collector.py` across 13 methods, 7 in `test_forced_kill.py`) |
| Forced kill test | Done | Killed isolated test worker subprocess (`taskkill /F`), NOT live collector |
| Local backup/restore | Done | `backup_local.py` verified: 32 files, 645 KB archive, all SHA-256 and SQLite rows matched |
| Sleep/hibernate detection | Done | Ledger heartbeat gaps tracked; max gap in 75-min run was 60.1s (0 sleep events) |

---

## D013: Scoreboard Definition Restored & Blocked
**Date:** 2026-10-09
**Status:** BLOCKED

1. **Scoreboard definition restored:** Model predictions scored against fully observed real-world outcomes (e.g. ETA prediction error $|\widehat{\text{ETA}} - \text{Actual}|$, Brier score for congestion transitions).
2. **"Rank by approach count" removed as scoreboard:** Port traffic ranking is an activity metric, not a model evaluation scoreboard.
3. **Scoreboard status:** BLOCKED until an ETA prediction model exists.
4. **Healthy session thresholds:** Provisional (60 min min duration, 100 unique vessels, 1% max quarantine, max heartbeat gap 120s), derived from a single 75-min session.
5. **Healthy window rule for live scoring:** An arrival evaluation at time $T_{\text{arr}}$ requires that the entire lookback window $[t_0, T_{\text{arr}}]$ had no collector crash, no heartbeat gap $> 120$s, and $\le 1\%$ quarantined messages.

---

## D014: ETA Baseline Feasibility & CI-Width Gate
**Date:** 2026-10-09
**Status:** EXPERIMENT COMPLETED

### 1. Dataset & Single-Season Limitation
- 28 consecutive days from NOAA MarineCadastre January 2024 (2024-01-01 to 2024-01-28).
- Stream-filtered by Houston bbox directly to D: via DuckDB range requests.
- Actual size: **283.65 MB (0.277 GB)**, 14,760,029 rows (well under 2 GB limit).
- **Single-season limitation:** January 2024 represents exclusively winter navigational conditions (sea fog closures, winter frontal passages, seasonal traffic), which do not generalize to summer or hurricane season.

### 2. Label Audit
- Raw entrance zone entries: 40,513 (note: 37,649 slow entries were ping-level loitering reports, not distinct vessel voyage attempts).
- Excluded left-censored (within first 6h of dataset): 1,763
- Excluded right-censored (within 24h cutoff): 44
- Excluded stationary/moored at entrance (SOG < 3.0 kn pings): 37,649
- Excluded invalid duration (<1h cut-throughs or >72h long layups): 781
- **Valid completed commercial arrivals:** **276** across 239 unique commercial vessels.
- Repeated trips: 205 vessels made 1 arrival, 31 made 2 arrivals, 3 made 3 arrivals. Service/tug repeat loops successfully eliminated.

### 3. Chronological Split & Point-in-Time Features
- Split: Days 1–21 Train (216 arrivals) vs Days 22–28 Test (60 arrivals). Strictly chronological, zero leakage.
- Features at entrance $t_0$: instantaneous SOG, length, width, draft, vessel type (cargo vs tanker), hour of day, day of week. Feature missingness: 0.0%.

### 4. Test Set Evaluation Metrics (N = 60 arrivals)
| Model | MAE (h) | RMSE (h) | MedAE (h) | Resid Std ($\sigma$) |
|---|---|---|---|---|
| Naive ($d / \text{SOG}$) | 6.771 | 16.165 | 1.952 | 14.802 |
| Historical Train Median | **5.550** | **15.599** | **0.559** | **14.875** |
| Baseline (`HistGradientBoosting`) | 9.831 | 15.847 | 4.613 | 15.909 |

- **Root Cause Analysis:** MedAE is low (0.56h for historical median), but standard deviation is high ($\sigma \approx 15.9$h). Vessels undergo highly variable anchorage waiting times (12–48h) at Bolivar Roads awaiting pilots or berths. Simple kinematics at $t_0$ cannot predict pilot dispatch queues.

### 5. Illustrative CI-Width Sample Size Projections (Mean Residual Only)
> **Statistical note:** These sample size calculations are purely illustrative asymptotic approximations for the mean residual under assumptions of independent, identically distributed observations and Gaussian errors with zero skew. They do not apply directly to MAE or skewed distributions. Observation days are omitted as live arrival rates will differ.

$$N \ge \left( \frac{1.96 \cdot 15.909}{\delta} \right)^2$$
- Target margin $\delta = \pm 0.25$h ($\pm 15$ min): $N \ge 15,557$ arrivals
- Target margin $\delta = \pm 0.50$h ($\pm 30$ min): $N \ge 3,890$ arrivals
- Target margin $\delta = \pm 1.00$h ($\pm 60$ min): $N \ge 973$ arrivals

**Conclusion:** The CI-width gate remains **BLOCKED** for narrow evaluation margins until Tier 1 incorporates anchorage waiting time and congestion state features. The final gate metric is MAE, to be evaluated via vessel-level bootstrap confidence intervals.

---

## D015: Label Audit Evidence, Live Observability & Quantile Feasibility
**Date:** 2026-10-09
**Status:** EVIDENCE AUDITED; TIER 1 APPROVED FOR PLANNING ONLY

### 1. Censoring & Exclusion Audit Evidence
- **37,649 Slow-Entry Exclusions:** Verified that the loop in `eta_baseline_experiment.py` incremented counts per-AIS-message while inside the entrance zone with SOG < 3.0. There are 162,689 such pings in the dataset. When measured at the **voyage boundary crossing level**, there were only **1,303 distinct boundary entries** (1,290 crossing at SOG $\ge 3$ kn, 13 at SOG $< 3$ kn). The 37,649 count was ping-level geofence jitter from vessels loitering at Bolivar Roads anchorage, not 37,649 separate approach attempts.
- **Right-Censoring Rule (24h vs 72h):** The cutoff `max_ts - 24h` conflicted with the 72h maximum duration. 114 vessels entered between `max_ts - 72h` and `max_ts` without arriving; those entering between 24h and 72h before the end were incorrectly labeled `invalid_duration` instead of `right_censored`. (Sample edge cases: MMSI 205736000 entered at $T_{\text{max}} - 30.34$h; MMSI 215185000 entered at $T_{\text{max}} - 31.15$h). The true right-censoring rule is $t_0 > T_{\text{max}} - 72$h.

### 2. Train/Test Overlap & Leakage Audit
- Train (Days 1–21): 192 unique MMSIs. Test (Days 22–28): 60 unique MMSIs.
- **Overlap:** Exactly **13 MMSIs** (21.7% of test vessels) appear in both sets, each making 1 separate voyage in Train and 1 separate voyage in Test. (Sample: MMSI 319478000 Trip 1 Jan 03 in Train [65.75h], Trip 2 Jan 26 in Test [3.31h]).
- **Leakage check:** All features at $t_0$ use only observations $\le t_0$. Imputation medians were computed on Train only. Test set was not modified. Zero leakage.

### 3. Arrival Definition & Geographic Scope
- **Direction:** 91.3% of arrivals had entrance COG directed inbound (270°–360° or 0°–30°). All 276 arrivals exhibited strictly positive spatial displacement from entrance to inner berths.
- **Pass-through test:** 382 vessels stopped/berthed in the inner box, while **226 vessels entered the inner box but never slowed below 1.5 kn** (passed through toward upstream berths). SOG < 1.5 in this box misses through-traffic proceeding to the upper Houston Ship Channel.
- **Geographic scope of inner box:** `[29.65, 29.80] x [-95.15, -94.95]` contains Barbours Cut (29.688, -95.000), Morgans Point (29.680, -94.985), and Jacintoport (29.740, -95.060). It does **NOT** contain Bayport (29.610, -95.015, which is south) or Houston Turning Basin (29.750, -95.290, which is west).
- **Count reconciliation:** 276 arrivals / 28 days (~9.9/day) represents completed end-to-end berths at Barbours Cut/Morgans Point. Phase 3's ~110 commercial approaches / 72h (~37/day) counted any instantaneous SOG < 3 kn detection in either the entrance or inner bay zones, including anchorage loiterers, through-traffic, and Bayport traffic.

### 4. Live Observability & Scoreboard Horizon (Core M0 Answer)
- Approach-to-arrival duration distribution ($N = 276$): Min 2.39h, P10 2.96h, P25 3.49h, **Median 3.97h**, P75 4.79h, P90 38.17h, Max 71.69h.
- Completed arrivals observable from lookback start through arrival within healthy session window $H$:
  - $H = 3.0$h: **32 / 276 arrivals (11.6%)** (~1.14 arrivals/day).
  - $H = 5.0$h: **213 / 276 arrivals (77.2%)** (~7.61 arrivals/day).
  - $H = 7.0$h: **218 / 276 arrivals (79.0%)** (~7.79 arrivals/day).
- **Conclusion:** A live scoreboard's evaluation horizon is **strictly bounded by continuous healthy session duration**. In 3-hour sessions, only ~11.6% of arrivals (the fastest direct transits) can be scored. Reliable live scoring of direct transits requires continuous 5–6 hour sessions. Anchored voyages (12–48h) cannot be scored live without multi-session persistence.

### 5. Quantile Regression & Stage Decomposition Findings
- **Two-stage decomposition is a hypothesis.** Prior claims of Stage 2 $\sigma \in [0.5, 1.0]$h were unmeasured.
- **Quantile Regression evaluated on Test set ($N = 60$):**
  - **P50:** MAE = 5.722h, MedAE = 0.813h, Pinball Loss = 2.861h, Test Coverage = 40.0% (target 50%).
  - **P80:** MAE = 13.150h, MedAE = 8.942h, Pinball Loss = 4.156h, Test Coverage = 76.7% (target 80%).
  - **P95:** MAE = 30.490h, MedAE = 30.764h, Pinball Loss = 2.073h, Test Coverage = 91.7% (target 95%).
- **Empirical stage decomposition:**
  - Direct transits ($\le 6.0$h, 78.6% of arrivals): Mean = 3.75h, Median = 3.81h, **Std = 0.63h** (measured).
  - Anchored transits ($> 6.0$h, 21.4% of arrivals): Mean = 39.35h, Median = 37.84h, **Std = 18.78h** (measured).
- **Bootstrap CI proposal:** Evaluated on MAE via vessel-level cluster bootstrap (resampling MMSIs with $B=2,000$ iterations).

---

## D016: M0 Final Corrections
**Date:** 2026-10-14
**Status:** CORRECTIONS APPLIED

### 1. Corrected Right-Censoring Rule
- Previous rule: `right_censor_boundary = T_max - 24h`, conflicting with 72h max voyage duration.
- Corrected rule: `right_censor_boundary = T_max - 72h`. Any entrance at $t_0 > T_{\max} - 72$h is right-censored.
- **Effect:** Valid arrivals reduced from 270 → **257** (13 trips removed). Test set reduced from 60 → **41** arrivals.

### 2. AIS-Behavior Stage Labels (Replaces Circular Duration-Based Labels)
- **Old method (DROPPED):** Stage defined by duration threshold (≤6h = "direct", >6h = "anchored"). This was circular — the label used the predicted outcome.
- **New method:** Stage defined by observed AIS behavior — whether the vessel dwelled at the Bolivar Roads Anchorage polygon (SOG < 1.0 kn, dwell ≥ 0.5h) between entrance and arrival.
- **Result:** AIS-behavior gives 239 direct (93.0%) vs 18 anchored (7.0%). Duration-based gave 199 vs 58.
- **40 voyages** were "anchored" by duration but had no observed Bolivar dwell — likely slow transits, not anchorage stops.
- **"σ = 0.63h direct transits" claim DROPPED** — this was circular (defined by picking fast outcomes).

### 3. Session Simulation (Full-Trip Scoring)
- Simulated daily healthy sessions (3, 5, 7h) at several UTC start hours across the 28-day dataset.
- Counted arrivals where BOTH $t_0$ (entrance) and $T_{\text{arr}}$ (berth) fall inside the same session.
- **Full-trip scoring yield (best start time per session):**
  - 3h session: **0.07 arrivals/day** (2 total in 28 days)
  - 5h session: **0.61 arrivals/day** (17 total in 28 days)
  - 7h session: **1.52 arrivals/day** (41 total in 28 days)
- Previous "scorable at 5h: 77.2%" was overstated by ~12×. It only checked if duration ≤ H, not that both endpoints fell inside a session.

### 4. Short-Horizon Live Scoring (Main M0 Design Conclusion)
- Live scoreboard scores **short-horizon ETA** (last 1–2h before arrival). Full-trip ETA stays historical-replay only.
- **Short-horizon yield (best start time 18:00 UTC):**
  - 1h horizon, 7h session: **3.44 arrivals/day** → N ≥ 30 in ~9 days
  - 2h horizon, 5h session: **1.61 arrivals/day** → N ≥ 30 in ~19 days

### 5. Vessel-Disjoint Split
- Standard test: N=41, 11 MMSIs overlap with train (26.8%).
- Vessel-disjoint test (overlapping removed): N=30.
- Disjoint test MAE is **better** than standard (Hist Median: 4.381h vs 6.473h), indicating no upward bias from overlap.

### 6. Model Evaluation (Corrected, N=41)
- **Language:** "Evaluated", not "calibrated."
- Hist Train Median: MAE = 6.473h [Bootstrap 95% CI: 2.356, 11.271]
- HistGradientBoosting: MAE = 9.520h [Bootstrap 95% CI: 5.736, 13.806]
- Quantile P50: MAE = 6.637h, Pinball = 3.319, Coverage = 36.6% (target 50%)
- Quantile P80: MAE = 14.007h, Pinball = 4.717, Coverage = 70.7% (target 80%)
- Quantile P95: MAE = 30.190h, Pinball = 2.215, Coverage = 90.2% (target 95%)
- CI-width gate remains **BLOCKED** (bootstrap CI widths ~8–9h).

### 7. Non-Inbound COG Arrivals
- 234 inbound, 23 non-inbound (mostly COG ~155°, heading SSE at entrance), 0 missing.
- Non-inbound arrivals are kept; they represent vessels that maneuvered at entrance before transiting inbound. Most have long durations (anchorage delays). Excluding them would not materially change median performance.

### 8. Pass-Through Selection Effect (226 Vessels)
- 226 vessels entered the inner box but never slowed below 1.5 kn (proceeding upstream to Bayport, Manchester, Turning Basin).
- This is a **documented selection effect**: the detector captures berthing at Barbours Cut / Morgans Point / Jacintoport only.

### 9. M0 Time Accounting
- Total M0 elapsed: **~12 agent-hours** against 32h hard stop.
- Phases completed: Space recovery, region selection, historical validation, collector build (59/59 tests), ETA baseline experiment, audit & evidence, final corrections.
- Remaining before Tier 1 build: User review of corrected metrics, potential Jan–Mar 2024 data pull.

---

## D017: Live Scoreboard Traffic Eligibility & Non-Commercial Exclusion
**Date:** 2026-10-26
**Status:** ADOPTED FOR SCOREBOARD DESIGN

### 1. Principle: Do Not Score Non-Commercial Traffic with Commercial Model
- The baseline ETA model is trained strictly on commercial vessel voyages (types 70–89: cargo, tanker, passenger).
- Non-commercial vessels (tugs, pushboats, fishing boats, recreational craft, pilot tenders) have fundamentally distinct kinematics, loitering patterns, and docking destinations. Scoring them with a commercial-trained model produces nonsensical errors and distorts evaluation.
- However, in live AISStream feeds, ~73% of vessels lack static type metadata, and seeding from 2024 historical archives covers only ~1.1% of active live MMSIs due to fleet turnover.

### 2. Live Eligibility Dual Definitions
To resolve the untyped vessel challenge without contaminating commercial evaluations, the live scoreboard evaluates two cohorts and reports both:
1. **Behavioral Inbound Eligibility (Cohort A):**
   - Evaluates any vessel regardless of static type that exhibits commercial-like inbound channel transit behavior:
     - Enters Entrance Zone (`[29.30, 29.45] x [-94.85, -94.65]`) at $\text{SOG} \ge 3.0$ kn.
     - Inbound COG: Heading directed into the channel ($270^\circ \le \text{COG} \le 360^\circ$ or $0^\circ \le \text{COG} \le 30^\circ$).
     - Length filter: $\ge 50$m (when reported) or sustained continuous displacement past Bolivar Roads towards inner channel.
2. **Unclassified Commercial / Verified Commercial (Cohort B):**
   - Strictly verified commercial vessels ($70 \le \text{type} \le 89$ via live Type 5/24 static messages or cache) PLUS unclassified vessels ($\text{type} \text{ IS NULL}$) that meet Cohort A's behavioral inbound criteria.
   - Known non-commercial vessels (tugs type 31/52, pleasure craft type 36/37, pilot boats type 50, fishing type 30) are **strictly excluded** from scoring.

### 3. Reporting Requirement
The live scoreboard will present metrics disaggregated by:
- **Verified Commercial Only** (conservative, lower $N$)
- **Behavioral + Unclassified Commercial** (operational, higher $N$, tracks real traffic)

---

## D018: Task 2 Fixes — Logon Schedule, 90s Sleep Resilience, and Inter-Session Semantics
**Date:** 2026-10-26
**Status:** IMPLEMENTED & EMPIRICALLY VERIFIED

### 1. Timezone & Schedule Fix
- **Time Conversion:** The machine operates on Indian Standard Time (IST, UTC+05:30).
  $$\text{Time}_{\text{UTC}} = \text{Time}_{\text{IST}} - 5\text{h}30\text{m}$$
  Earlier document referenced 12:00 local as 18:00 UTC, which was erroneous (12:00 IST is actually 06:30 UTC).
- **Trigger Redesign:** The fixed daily clock trigger was removed from Task Scheduler. The task `\ORCASTRA_Collector` now triggers on **User Logon** (`<LogonTrigger>`), enforces AC power (`DisallowStartIfOnBatteries=true`), starts missed tasks immediately upon wake (`StartWhenAvailable=true`), and caps each run at 6 hours (`ExecutionTimeLimit=PT6H`).
- **UTC Session Logging:** Every collector session logs dual timestamps: `YYYY-MM-DD HH:MM:SS UTC [Local: YYYY-MM-DD HH:MM:SS (IST, UTC+05:30)]`.

### 2. Sleep Detection Threshold (90 Seconds) & Flush Resilience
- Raised the wall-clock jump trigger from 15s to **90.0s**.
- **Empirical Flush Test (`--test-slow-flush`):** Verified that a 15-second heavy I/O / SQLite slow flush does NOT trigger the 90s threshold, preventing spurious session splits during intense writes.
- **Inter-Session Semantics Across Sleep:**
  - AISStream is a real-time ephemeral WebSocket stream without backlog buffering. Any period during which the machine is asleep represents permanently lost observations.
  - Therefore, sessions separated by a sleep event are registered as distinct, non-contiguous sessions in the ledger.
  - A continuous healthy window cannot cross a sleep boundary. However, for short-horizon ETA (e.g. 1–2h before arrival), an arrival is scorable as long as its short-horizon window falls entirely within one post-resume session.

---

## D019: Crossing-Level Audit, Stage Labels & Non-Inbound Exclusion
**Date:** 2026-10-26
**Status:** ADOPTED FOR TIER 1

### 1. Crossing-Level vs Ping-Level Exclusions (91-Day Full Audit)
- Previous "138,314 stationary entrance exclusions" was a ping-level count of position reports while loitering inside the box.
- At the **voyage crossing level** (transitions from outside to inside entrance box):
  - Total entrance crossings: **4,672**
  - Left-censored ($<6$h): **19 (0.4%)**
  - Stationary at crossing ($\text{SOG} < 3$ kn): **28 (0.6%)** — nearly all commercial vessels cross at speed.
  - Right-censored ($> T_{\max} - 72$h): **190 (4.1%)**
  - Invalid / Through-traffic ($> 72$h or unberthed): **3,427 (73.4%)** (mostly proceeding upstream past Jacintoport).
  - Valid completed arrivals: **1,008 (21.6%)**

### 2. Investigation of >12h Trips and Anchorage Geography
- Of 1,008 arrivals, **228 (22.6%)** have duration $> 12$h. The old Bolivar-only box captured only 73 of these 228 trips (32.0%).
- Empirical spatial analysis of stationary pings revealed where the remaining vessels waited:
  - **Gulf Fairway / Offshore Anchorage (Area A & B):** `[29.10, 29.32] x [-94.85, -94.55]` (outside entrance channel in Gulf). **86 voyages** waited here for pilots before entering.
  - **Galveston Harbor / Pelican Island:** `[29.28, 29.34] x [-94.85, -94.75]`. **21 voyages** anchored here.
  - Expanding the anchorage definition to include Gulf Fairway and Galveston Harbor increases anchored trip capture to **154 voyages (63.6% of >12h trips)**.

### 3. Non-Inbound Arrivals Exclusion
- 109 arrivals entered with non-inbound COG (median COG = 156.1°, heading SSE away from port out to sea).
- **Performance:** Inbound arrivals have median duration 3.85h (historical median MAE = 6.686h). Non-inbound arrivals have median duration **41.36h** and historical median MAE = **36.577h**.
- Non-inbound arrivals represent offshore loitering and holding patterns rather than active channel approaches. They are **strictly excluded** from the primary benchmark.

---

## D020: Operational Definition of Live Cohort A
**Date:** 2026-10-26
**Status:** SPECIFIED FOR LIVE SCOREBOARD

To enable live scoring without unobserved historical fields or complex profile tracing:
- **Location:** Position inside Entrance Zone `[29.30, 29.45] x [-94.85, -94.65]`.
- **Observation Density:** At least $N \ge 3$ consecutive AIS reports within the last 15 minutes.
- **Speed:** Mean SOG over the last $N$ reports $\ge 3.0$ kn.
- **Direction:** Median COG over the last $N$ reports directed inbound: $270^\circ \le \text{COG} \le 360^\circ$ or $0^\circ \le \text{COG} \le 30^\circ$.
- **Length Filter:** Length $\ge 50$m *only if reported* in static messages (not required if null/missing).

---

## D021: Protocol v2 Execution — Anchorage Timing, Short-Horizon State ETA & Conformal Calibration
**Date:** 2026-10-26
**Status:** VALIDATED ON MARCH CHRONOLOGICAL TEST SET

### 1. Anchorage Timing: Before vs After $t_0$
- **Full Population ($N = 1,008$ arrivals):**
  - Waited offshore before $t_0$ only: **564 (56.0%)**
  - Waited after $t_0$ only: **43 (4.3%)**
  - Waited both before and after: **195 (19.3%)**
  - Direct transit (no dwell): **206 (20.4%)**
- **Long Voyages ($> 12$h, $N = 228$):**
  - **99.1% (226 of 228)** had dwells **AFTER $t_0$** (185 waited both before and after, 41 after only).
  - Only 2 voyages had dwells before $t_0$ without dwelling after $t_0$.
- **Prior-Track Feature Value:**
  - Correlation of pre-$t_0$ dwell hours with transit duration: $r = 0.019$.
  - Correlation of pre-$t_0$ mean SOG (2h) with transit duration: $r = 0.069$.
  - **Conclusion:** Prior-track offshore kinematics do not predict channel waiting time. Once cleared to enter, direct transits proceed regardless of offshore dwell, while long delays are driven by post-entrance pilot/berth dispatch inside the channel.

### 2. Full-Trip Cohort Reconciliation
- **Cohort A1 (Strict First-Crossing Inbound, $N = 899$):**
  - Train: 609 arrivals | Test: 290 arrivals.
  - Historical Train Median (scalar = 3.89h): Test MAE = **6.282 h** [95% CI: 4.655, 8.055].
- **Cohort A2 (Inbound with Re-Entry Allowed, $N = 985$):**
  - Train: 665 arrivals | Test: 320 arrivals.
  - Historical Train Median (scalar = 3.87h): Test MAE = **5.871 h** [95% CI: 4.355, 7.490].
- **Reconciliation:** The 30 additional arrivals in Cohort A2 were vessels that lingered at boundary crossing and subsequently entered inbound; capturing them at their true inbound entry reduces variance. Both results are preserved.

### 3. Short-Horizon State-Based ETA (The Operational Live Breakthrough)
- Evaluated on **4,215 state-triggered snapshots** (10-minute intervals while actively underway in channel corridor) on the March test set ($N=259$ unique vessels).
- Conformal calibration performed on held-out slice of training period (Feb 16–29), test set completely untouched until frozen run.
- **Performance Across Horizon Buckets:**
  - **$\le 1$h Horizon ($N=1,020$):** P50 MAE = **0.262 h (15.7 min)** [CI: 0.227, 0.297], MedAE = **0.148 h (8.9 min)**, P95 Cov = **100.0%**. Unseen vessels MAE = **0.269 h**.
  - **1–2h Horizon ($N=1,726$):** P50 MAE = **0.234 h (14.0 min)** [CI: 0.206, 0.261], MedAE = **0.148 h (8.9 min)**, P95 Cov = **99.1%**. Unseen vessels MAE = **0.239 h**.
  - **2–3h Horizon ($N=1,469$):** P50 MAE = **0.230 h (13.8 min)** [CI: 0.207, 0.256], MedAE = **0.158 h (9.5 min)**, P95 Cov = **98.6%**. Unseen vessels MAE = **0.229 h**.
  - **Overall $\le 3$h Horizon ($N=4,215$):** P50 MAE = **0.239 h (14.3 min)** [CI: 0.212, 0.267] (CI width = **3.2 min!**), MedAE = **0.151 h (9.0 min)**, P95 Cov = **99.2%**. Unseen vessels MAE = **0.243 h**.
- **Monotonicity:** Exactly **0 quantile inversions** on test set (strict sorting enforced).
- **Core Conclusion:** While full-trip ETA at entrance remains high-variance due to downstream anchorage dispatch, **short-horizon ETA along the channel corridor is exceptionally accurate ($\text{MAE} \approx 14$ min, $\text{MedAE} \approx 9$ min) and generalizes to completely novel vessels with zero degradation.** This confirms the M0 design decision that live scoring must focus on short-horizon ETA.

---

## D022: Protocol v2.1 Execution — Unconditioned State Snapshots & Distance Buckets
**Date:** 2026-10-26  
**Status:** FROZEN AND VALIDATED ON MARCH CHRONOLOGICAL TEST SET  
**Protocol Document:** `docs/EVAL_PROTOCOL_V2_1.md` (committed before execution in git `553543d`)  
**Results Artifact:** `D:\ORCASTRA_DATA\historical\results\tier1_v2_1_short_horizon_results.json`

### 1. Motivation for Protocol v2.1 Fix
- In Protocol v2, snapshots were filtered by `0 < rem_dur <= 3.5h` and evaluated in buckets defined by TRUE remaining duration (`<=1h`, `1-2h`, `2-3h`). This was **label-conditioned selection**: it artificially discarded vessels that slowed down, anchored, or encountered delays after entering the corridor.
- Protocol v2.1 eliminates label-conditioned selection entirely:
  1. **All** state-triggered corridor snapshots are retained regardless of actual duration to berth.
  2. Bucketing is done strictly by **quantities known at prediction time**: distance-to-berth along the channel ($d_{\text{rem}}$).
  3. A new **Bucket Median Baseline** (predicting the training-set median $y_{\text{rem}}$ for each distance bucket) is evaluated alongside Kinematic ($d/\text{SOG}$), Hist Segment Median, and Quantile Regression P50.
  4. P50/P80/P95 interval widths and the share of snapshots with true $y_{\text{rem}} > 3$h are explicitly reported.

### 2. Snapshot Population Breakdown (Unconditioned)
- **Total State-Triggered Snapshots:** 15,768 (10-min sampling while underway $\text{SOG} \ge 3$ kn and inbound heading in corridor).
- **Right-Censored Snapshots (Unobserved Berth Arrival at Dataset End):** 0
- **Anchored-Midway Snapshots (Vessel Later Stopped $\ge 30$ min):** 1,881 (11.9% overall, 13.0% in test)
- **Chronological Split:**
  - Train Slice (Jan 1 – Feb 15): **7,963 snapshots** (373 unique vessels)
  - Calibration Slice (Feb 16 – Feb 29): **2,738 snapshots** (155 unique vessels)
  - March Test Set: **5,067 snapshots** (264 unique vessels)
- **Overall Share with True $y_{\text{rem}} > 3$h:** **20.2%** (17.4% in March test set).

### 3. Conformal Calibration Isolation Verification
- Quantile regressors ($\tau \in [0.50, 0.80, 0.95]$) trained strictly on Train Slice (Jan 1 – Feb 15).
- Conformal residual adjustments calculated strictly on Calibration Slice (Feb 16 – Feb 29):
  - $c_{0.50} = -0.0160$ h (Calibration Coverage = 50.0%)
  - $c_{0.80} = +0.0405$ h (Calibration Coverage = 80.0%)
  - $c_{0.95} = +33.5222$ h (Calibration Coverage = 95.0%)
- Calibrated test predictions: $\hat{y}_{\text{test}, \alpha}^{\text{calib}} = \hat{y}_{\text{test}, \alpha}^{\text{mono}} + c_\alpha$.
- Quantile inversions on test set: **0 (Strict Monotonicity PASS)**.
- Test set remained completely untouched until the single frozen benchmark execution.

### 4. Unconditioned Benchmark Results by Distance Bucket

| Distance Bucket | Snapshots (Vessels) | Share $y > 3$h | Share Anchored | Kinematic MAE | Bucket Median MAE | QR P50 MAE [95% CI] | QR P50 MedAE | P95 Cov [95% CI] | P80$-$P50 Width | P95$-$P50 Width | Unseen P50 MAE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Near Berth ($d \le 10$ km)** | 1,544 (263) | 3.2% | 12.4% | 2.148 h | 1.659 h | **1.768 h** [0.919, 2.755] | **0.145 h** (8.7 min) | 98.4% [96.7, 99.6] | 2.487 h | 38.856 h | 2.471 h |
| **Mid-Channel ($10 < d \le 25$ km)** | 904 (261) | 4.2% | 12.4% | 2.681 h | 2.271 h | **2.454 h** [1.530, 3.446] | **0.225 h** (13.5 min) | 97.9% [96.5, 99.2] | 3.781 h | 39.833 h | 3.284 h |
| **Outer Channel ($25 < d \le 50$ km)** | 2,149 (261) | 21.1% | 12.8% | 3.275 h | 2.948 h | **2.946 h** [1.846, 4.154] | **0.252 h** (15.1 min) | 97.3% [95.5, 99.0] | 3.817 h | 41.676 h | 4.021 h |
| **Beyond Entrance ($d > 50$ km)** | 470 (243) | 71.9% | 17.0% | 6.370 h | 5.899 h | **5.381 h** [3.911, 6.973] | **0.416 h** (25.0 min) | 98.7% [97.5, 99.8] | 5.862 h | 55.918 h | 6.731 h |
| **Overall** | **5,067 (264)** | **17.4%** | **13.0%** | **3.113 h** | **2.708 h** | **2.725 h** [1.785, 3.723] | **0.211 h** (12.7 min) | **97.9%** [96.4, 99.1] | **3.595 h** | **41.809 h** | **3.658 h** |

### 5. Insights & Comparison with Protocol v2
1. **MedAE reveals the true direct transit precision:** Under unconditioned selection, the Median Absolute Error for QR P50 is **8.7 minutes** near berth and **12.7 minutes overall**. For the ~87% of vessels that transit directly without anchoring, state-based ETA is exceptionally accurate.
2. **Mean Absolute Error (MAE) captures the long tail of midway anchorages:** 12.4%–17.0% of snapshots come from voyages that anchor midway inside the bay/channel. Because an anchorage delay adds 12 to 40+ hours, the unconditioned MAE reflects these heavy-tailed residuals (1.77h near berth, 2.73h overall), whereas MedAE isolates direct transit accuracy.
3. **P95 Interval Width ($38.8$ to $55.9$ h) rigorously quantifies uncertainty:** Rather than yielding artificially tight intervals by discarding delayed vessels, the calibrated P95 interval accurately expands to absorb unexpected anchorages, achieving **97.9% empirical coverage** [96.4%, 99.1%] on the test set.
4. **Generalization to Disjoint Vessels:** On completely novel vessels never seen in training ($N=167$ unseen vessels in March), P95 coverage remains well above nominal at **96.8%**, confirming robust out-of-sample uncertainty calibration.
5. **Statistical Equivalence of Point Estimate Baselines:** Overall test MAEs across methods are 2.681 h for Historical Segment Median, 2.708 h for Distance-Bucket Median, and 2.725 h for QR P50 (with 95% bootstrap CI [1.785, 3.723]). Because the vessel-clustered bootstrap confidence intervals overlap heavily, these estimates are statistically indistinguishable. No model clearly beats the simple baselines on point MAE.

---

## D023: Phase 2 Live ETA Display, Map Radar UX, and Non-Dimming Streamlit Architecture

### Context
Phase 2 deploys live ETA capabilities to the Streamlit operational dashboard. Initial implementation required UI improvements to prevent whole-page reloading, clarify baseline equivalence, and compare kinematic transit speeds against empirical hold baselines.

### Decision
1. **Dual ETA Display:** The vessel table, map hover tooltips, and vessel detail panel present BOTH:
   - **Distance-Bucket Median ETA:** Empirical baseline accounting for historical channel holds.
   - **Kinematic ETA ($d / \text{SOG}$):** Direct speed comparator.
   - Accompanying operational note: *"Bucket median has lower average error because vessel holds dominate; kinematic is shown for comparison."*
2. **Neutral Baseline Wording:** Removed assertions that any single method is the "best MAE". The documentation and dashboard explicitly explain that segment median (2.681 h), bucket median (2.708 h), and QR P50 (2.725 h) have heavily overlapping bootstrap CIs and are statistically indistinguishable.
3. **Map UX & Radar Interaction:** Permanent text labels on vessel markers removed to prevent clutter. Markers are colored by operational status (`Eligible / Active`, `Stationary`, `Out of Corridor`, `Non-Commercial`, `Stale`) with an interactive legend. Added a zoom / focus selector ("Focus on Channel Corridor", "Full Bay", "Inner Berth Box", "Entrance") to separate clustered vessels.
4. **Vessel Selection Detail Panel:** A selectbox enables inspecting any tracked vessel's position, speed history, along-channel distance, dual ETAs, and Cohort A eligibility. Selection persists across page refreshes via `st.session_state`.
5. **Non-Dimming Live Refresh:** Wrapped live telemetry panels in `@st.fragment(run_every=10)` to periodically poll `latest_state.json` every ~10s without full-page dimming, preserving immediate button responsiveness for Start/Stop Live controls.

