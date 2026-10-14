# ORCASTRA — M0 Summary

**Status:** M0 CLOSED | **Date:** 2026-10-14 | **Hours used:** ~14 of 32h hard stop

## Region
Houston Ship Channel — Barbours Cut / Morgans Point terminal cluster.

## Design Conclusions
1. **Hybrid architecture:** NOAA MarineCadastre for training; AISStream for live inference.
2. **Full-trip ETA** (entrance-to-berth) → historical replay only.
3. **Short-horizon ETA** (last 1–2h before arrival) → live scoreboard.
4. **Live yield:** ~1 scorable arrival/day with daily 7h sessions (if type filter dropped).
5. **Anchorage variance** (~15h std) dominates. Anchorage queue features required.
6. **CI-width gate** blocked until more data and anchorage-aware models.

## Proven
- Collector: 59/59 tests, 75-min live recording, crash-safe writes.
- NOAA data usable: 257 arrivals in 28 days (72h-censored).
- Historical median beats trained model (MAE 6.47h vs 9.52h, N=41).
- Full-trip scoring infeasible live (only 41/257 arrivals in best 7h daily sessions).

## Unproven
- Quantile coverage inconclusive at N=41.
- Vessel-disjoint split is a sensitivity check, not proof.
- Single-season (Jan 2024) data only.
- NOAA type cache seeding: only 4/377 live MMSIs matched (fleet turnover).

## Data on Disk
- NOAA 28-day: 283.65 MB (D:)
- Live session: 35.5 MB (D:)
- Jan–Mar 2024 pull: NOT YET DONE (first Tier 1 step)

## Next (Tier 1, Approved)
1. Pull Jan–Mar 2024 NOAA data (≤1.0 GB, D: only)
2. ETA quantile models with anchorage features
3. Short-horizon live scoreboard
4. Historical replay harness
