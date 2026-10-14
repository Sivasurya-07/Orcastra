"""
ORCASTRA M0 — Houston Detector Validation (corrections 5-7, 9)

Validates the approach detector against the 3-day sample:
- Inspect representative trajectories
- Check for duplicate/repeated crossings
- Report unique vessels vs total approaches
- Approaches per day and per hour of day
- Break down by AIS ship type
- Report methodology details
"""

import json, math, sys, logging
from pathlib import Path
from datetime import datetime, timezone

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

DATA_DIR = Path(r"D:\ORCASTRA_DATA\historical")
SAMPLE_PATH = DATA_DIR / "noaa" / "houston" / "houston_sample.parquet"
RESULTS_DIR = DATA_DIR / "results"

HOUSTON_BBOX = {"name": "houston", "lat_min": 29.0, "lat_max": 29.9,
                "lon_min": -95.3, "lon_max": -94.4}

APPROACH_ZONES = [
    {"name": "houston_ship_channel_entrance",
     "lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65},
    {"name": "galveston_bay_inner",
     "lat_min": 29.55, "lat_max": 29.75, "lon_min": -95.10, "lon_max": -94.85},
]

ARRIVAL_SPEED_THRESHOLD = 3.0

# AIS vessel type code to category mapping
# See: https://api.vtexplorer.com/docs/ref-aistypes.html
VESSEL_TYPE_CATEGORIES = {
    # Cargo: 70-79
    **{i: "cargo" for i in range(70, 80)},
    # Tanker: 80-89
    **{i: "tanker" for i in range(80, 90)},
    # Passenger: 60-69
    **{i: "passenger" for i in range(60, 70)},
    # Tug/Towing: 31, 32, 52
    31: "tug_towing", 32: "tug_towing", 52: "tug_towing",
    # Fishing: 30
    30: "fishing",
    # Pilot: 50
    50: "pilot",
    # Pleasure/Sailing: 36, 37
    36: "pleasure_sailing", 37: "pleasure_sailing",
    # SAR: 51
    51: "sar",
    # Other: everything else
}


def categorize_vessel_type(vt):
    if vt is None:
        return "unknown"
    return VESSEL_TYPE_CATEGORIES.get(vt, "other")


def detect_approaches_detailed(df: pl.DataFrame, zone: dict) -> list[dict]:
    """
    Detect approaches with full detail for validation.
    
    APPROACH DEFINITION:
    - A vessel "approaches" when it transitions from OUTSIDE the geofence zone
      to INSIDE the zone, AND subsequently has at least one AIS report with
      SOG < 3.0 kn while inside the zone.
    - Multiple approaches per vessel ARE counted if:
      - The vessel leaves the zone entirely (at least one report outside)
      - Then re-enters the zone
      - Then slows below threshold again
    - This allows the same vessel to have multiple port calls across the 3-day window.
    
    MINIMUM TRAJECTORY REQUIREMENTS:
    - Vessel must have >= 3 total AIS reports in the full bbox (not just zone)
    - At least one report must be outside the zone (to detect a transition)
    
    MISSING POSITION EFFECTS:
    - If a vessel's track has a gap (e.g. AIS off), and they appear inside the zone
      with no prior "outside" report, the first appearance is treated as an entry
      (prev_in_zone defaults to False for the first row).
    - This may slightly inflate approach counts for vessels that were already
      in the zone at the start of the sample window.
    """
    zone_df = df.filter(
        (pl.col('lat') >= zone['lat_min']) & (pl.col('lat') <= zone['lat_max']) &
        (pl.col('lon') >= zone['lon_min']) & (pl.col('lon') <= zone['lon_max'])
    )
    if len(zone_df) == 0:
        return []

    zone_mmsis = zone_df['mmsi'].drop_nulls().unique().to_list()
    approaches = []

    for mmsi in zone_mmsis:
        vessel = df.filter(pl.col('mmsi') == mmsi).sort('event_time')
        if len(vessel) < 3:
            continue

        vessel = vessel.with_columns(
            ((pl.col('lat') >= zone['lat_min']) & (pl.col('lat') <= zone['lat_max']) &
             (pl.col('lon') >= zone['lon_min']) & (pl.col('lon') <= zone['lon_max'])).alias('in_zone')
        )
        vessel = vessel.with_columns(pl.col('in_zone').shift(1).alias('prev_in_zone'))

        # Find ALL outside→inside transitions (not just first)
        entries = vessel.filter(
            pl.col('in_zone') & ~pl.col('prev_in_zone').fill_null(False)
        )

        for entry in entries.iter_rows(named=True):
            entry_time = entry['event_time']

            # Find reports inside zone after this entry, up to when vessel leaves
            post = vessel.filter(
                (pl.col('event_time') >= entry_time)
            )
            # Get the subset that's continuously inside the zone from entry
            in_zone_stretch = []
            for row in post.iter_rows(named=True):
                if row['in_zone']:
                    in_zone_stretch.append(row)
                else:
                    break  # Vessel left zone

            if not in_zone_stretch:
                continue

            slow = [r for r in in_zone_stretch
                    if r['sog'] is not None and 0 <= r['sog'] < ARRIVAL_SPEED_THRESHOLD]

            if len(slow) > 0:
                # Get vessel type from any report
                vt_row = vessel.filter(pl.col('vessel_type').is_not_null()).head(1)
                vt = int(vt_row['vessel_type'][0]) if len(vt_row) > 0 else None

                approaches.append({
                    'mmsi': int(mmsi),
                    'entry_time': str(entry_time),
                    'zone': zone['name'],
                    'vessel_type': vt,
                    'vessel_category': categorize_vessel_type(vt),
                    'reports_in_stretch': len(in_zone_stretch),
                    'slow_reports': len(slow),
                    'min_sog_in_zone': min(r['sog'] for r in in_zone_stretch if r['sog'] is not None),
                    'entry_day': str(entry_time)[:10] if entry_time else None,
                    'entry_hour': entry_time.hour if entry_time else None,
                })

    return approaches


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ORCASTRA M0 — Houston Detector Validation")

    df = pl.read_parquet(SAMPLE_PATH)
    log.info(f"Loaded: {len(df):,} rows")

    # Rename and prepare
    df = df.rename({'base_date_time': 'event_time'})
    if df['event_time'].dtype == pl.Utf8:
        df = df.with_columns(pl.col('event_time').str.strptime(pl.Datetime('us'), '%Y-%m-%dT%H:%M:%S', strict=False))

    df = df.filter(pl.col('event_time').is_not_null()).sort('event_time')
    df = df.with_columns([
        pl.col('mmsi').cast(pl.Int64, strict=False),
        pl.col('sog').cast(pl.Float32, strict=False),
        pl.col('lat').cast(pl.Float64, strict=False),
        pl.col('lon').cast(pl.Float64, strict=False),
        pl.col('vessel_type').cast(pl.Int32, strict=False),
    ])

    log.info(f"Time range: {df['event_time'].min()} to {df['event_time'].max()}")
    log.info(f"Days in sample: {df['event_time'].min()} is {df['event_time'].min().strftime('%A') if df['event_time'].min() else 'unknown'}")

    # Day of week for each sample date
    for day_offset in range(3):
        from datetime import timedelta
        dt = datetime(2024, 1, 15) + timedelta(days=day_offset)
        log.info(f"  {dt.strftime('%Y-%m-%d')}: {dt.strftime('%A')}")

    # ── Run detailed approach detection ──
    all_approaches = []
    for zone in APPROACH_ZONES:
        log.info(f"\nDetecting approaches: {zone['name']}")
        approaches = detect_approaches_detailed(df, zone)
        log.info(f"  Approaches detected: {len(approaches)}")
        all_approaches.extend(approaches)

    log.info(f"\nTotal approaches (all zones): {len(all_approaches)}")

    # ── Validation checks ──
    results = {}

    # 1. Unique vessels vs total approaches
    unique_mmsis = set(a['mmsi'] for a in all_approaches)
    results['total_approaches'] = len(all_approaches)
    results['unique_vessels_with_approaches'] = len(unique_mmsis)
    results['avg_approaches_per_vessel'] = round(len(all_approaches) / len(unique_mmsis), 2) if unique_mmsis else 0
    log.info(f"\nUnique vessels with approaches: {len(unique_mmsis)}")
    log.info(f"Avg approaches/vessel: {results['avg_approaches_per_vessel']}")

    # Check for vessels with many approaches (potential duplicates)
    from collections import Counter
    mmsi_counts = Counter(a['mmsi'] for a in all_approaches)
    multi_approach = {m: c for m, c in mmsi_counts.items() if c > 1}
    results['vessels_with_multiple_approaches'] = len(multi_approach)
    results['max_approaches_single_vessel'] = max(mmsi_counts.values()) if mmsi_counts else 0
    log.info(f"Vessels with >1 approach: {len(multi_approach)}")
    log.info(f"Max approaches by one vessel: {results['max_approaches_single_vessel']}")

    # Show top repeat offenders
    top_repeats = sorted(multi_approach.items(), key=lambda x: -x[1])[:5]
    log.info(f"Top repeat vessels: {top_repeats}")
    results['top_repeat_vessels'] = [{'mmsi': m, 'approaches': c} for m, c in top_repeats]

    # 2. Approaches per day
    day_counts = Counter(a['entry_day'] for a in all_approaches)
    results['approaches_per_day'] = dict(sorted(day_counts.items()))
    log.info(f"\nApproaches per day:")
    for day, count in sorted(day_counts.items()):
        log.info(f"  {day}: {count}")

    # 3. Approaches per hour of day
    hour_counts = Counter(a['entry_hour'] for a in all_approaches)
    results['approaches_per_hour'] = {str(h): c for h, c in sorted(hour_counts.items())}
    log.info(f"\nApproaches per hour of day (UTC):")
    for h in range(24):
        c = hour_counts.get(h, 0)
        bar = '#' * (c // 2)
        log.info(f"  {h:02d}:00 — {c:3d}  {bar}")

    # Check if rate is concentrated
    peak_hour = max(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (0, 0)
    min_hour = min(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (0, 0)
    results['peak_hour'] = {'hour': peak_hour[0], 'count': peak_hour[1]}
    results['min_hour'] = {'hour': min_hour[0], 'count': min_hour[1]}
    results['peak_to_min_ratio'] = round(peak_hour[1] / max(min_hour[1], 1), 1)

    # 4. Breakdown by vessel category
    cat_counts = Counter(a['vessel_category'] for a in all_approaches)
    results['approaches_by_category'] = dict(sorted(cat_counts.items(), key=lambda x: -x[1]))
    log.info(f"\nApproaches by vessel category:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_approaches) * 100
        log.info(f"  {cat:20s}: {count:4d} ({pct:.1f}%)")

    # Separate rates for commercial vs support vessels
    commercial_cats = {'cargo', 'tanker', 'passenger'}
    support_cats = {'tug_towing', 'pilot', 'fishing', 'pleasure_sailing'}
    commercial_approaches = sum(c for cat, c in cat_counts.items() if cat in commercial_cats)
    support_approaches = sum(c for cat, c in cat_counts.items() if cat in support_cats)
    other_approaches = len(all_approaches) - commercial_approaches - support_approaches
    hours = 72.0

    results['commercial_rate'] = {
        'approaches': commercial_approaches,
        'rate_per_hour': round(commercial_approaches / hours, 3),
        'categories': list(commercial_cats),
    }
    results['support_rate'] = {
        'approaches': support_approaches,
        'rate_per_hour': round(support_approaches / hours, 3),
        'categories': list(support_cats),
    }
    results['other_rate'] = {
        'approaches': other_approaches,
        'rate_per_hour': round(other_approaches / hours, 3),
    }

    log.info(f"\nCommercial (cargo+tanker+passenger): {commercial_approaches} ({commercial_approaches/hours:.2f}/hr)")
    log.info(f"Support (tug+pilot+fishing+pleasure): {support_approaches} ({support_approaches/hours:.2f}/hr)")
    log.info(f"Other/unknown: {other_approaches} ({other_approaches/hours:.2f}/hr)")

    # 5. Representative trajectory inspection
    log.info(f"\n--- Representative Trajectory Samples ---")
    # Pick 3 vessels with single approach, 2 with multiple
    single_approach_mmsis = [m for m, c in mmsi_counts.items() if c == 1][:3]
    multi_approach_mmsis = [m for m, c in sorted(multi_approach.items(), key=lambda x: -x[1])][:2]

    trajectory_samples = []
    for mmsi in single_approach_mmsis + multi_approach_mmsis:
        vessel = df.filter(pl.col('mmsi') == mmsi).sort('event_time')
        n_reports = len(vessel)
        time_span = vessel['event_time'].max() - vessel['event_time'].min()
        hours_span = time_span.total_seconds() / 3600 if time_span else 0

        # Check in-zone status
        for zone in APPROACH_ZONES:
            in_zone = vessel.filter(
                (pl.col('lat') >= zone['lat_min']) & (pl.col('lat') <= zone['lat_max']) &
                (pl.col('lon') >= zone['lon_min']) & (pl.col('lon') <= zone['lon_max'])
            )
            out_zone = len(vessel) - len(in_zone)

            if len(in_zone) > 0:
                sample = {
                    'mmsi': int(mmsi),
                    'total_reports': n_reports,
                    'hours_span': round(hours_span, 1),
                    'zone': zone['name'],
                    'reports_in_zone': len(in_zone),
                    'reports_outside_zone': out_zone,
                    'approaches_for_vessel': mmsi_counts[mmsi],
                    'sog_range': f"{float(vessel['sog'].min()):.1f} - {float(vessel['sog'].max()):.1f} kn",
                }
                trajectory_samples.append(sample)
                log.info(f"  MMSI {mmsi}: {n_reports} reports over {hours_span:.1f}h, "
                         f"{len(in_zone)} in zone '{zone['name']}', "
                         f"{out_zone} outside, {mmsi_counts[mmsi]} approach(es)")
                break

    results['trajectory_samples'] = trajectory_samples

    # 6. Missing position effects
    # Count vessels that appear only inside a zone (no outside reports)
    zone_only_vessels = 0
    for zone in APPROACH_ZONES:
        zone_df = df.filter(
            (pl.col('lat') >= zone['lat_min']) & (pl.col('lat') <= zone['lat_max']) &
            (pl.col('lon') >= zone['lon_min']) & (pl.col('lon') <= zone['lon_max'])
        )
        zone_mmsis = set(zone_df['mmsi'].drop_nulls().unique().to_list())
        for mmsi in zone_mmsis:
            vessel = df.filter(pl.col('mmsi') == mmsi)
            outside = vessel.filter(
                ~((pl.col('lat') >= zone['lat_min']) & (pl.col('lat') <= zone['lat_max']) &
                  (pl.col('lon') >= zone['lon_min']) & (pl.col('lon') <= zone['lon_max']))
            )
            if len(outside) == 0:
                zone_only_vessels += 1

    results['vessels_only_in_zone'] = zone_only_vessels
    log.info(f"\nVessels with ALL reports inside a zone (no outside track): {zone_only_vessels}")
    log.info(f"  These may be moored/anchored vessels or vessels with gap-inflated first entries")

    # Save detailed results
    results_path = RESULTS_DIR / "validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nValidation results saved: {results_path}")


if __name__ == "__main__":
    main()
