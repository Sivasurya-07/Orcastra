"""
ORCASTRA M0 — Historical AIS Feasibility Sampling v3

Key fixes:
- NOAA: geometry column is GEOMETRY('EPSG:4326'), use ST_Y/ST_X directly
- NOAA columns: mmsi, base_date_time, sog, cog, heading, vessel_name, imo,
  call_sign, vessel_type, status, length, width, draft, cargo, transceiver, geometry
- DMA: web.ais.dk unreachable — document as finding
"""

import os, sys, json, math, logging
from pathlib import Path
from datetime import datetime, timezone

import polars as pl
import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

DATA_DIR = Path(r"D:\ORCASTRA_DATA\historical")
RESULTS_DIR = DATA_DIR / "results"

HOUSTON_BBOX = {"name": "houston", "lat_min": 29.0, "lat_max": 29.9,
                "lon_min": -95.3, "lon_max": -94.4}

HOUSTON_APPROACH_ZONES = [
    {"name": "houston_ship_channel_entrance",
     "lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65},
    {"name": "galveston_bay_inner",
     "lat_min": 29.55, "lat_max": 29.75, "lon_min": -95.10, "lon_max": -94.85},
]

ARRIVAL_SPEED_THRESHOLD = 3.0


def query_noaa_houston(dates: list[str], bbox: dict, output_dir: Path) -> Path | None:
    """Query NOAA 2024 GeoParquet on Azure with correct spatial extraction."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "houston_sample.parquet"

    if parquet_path.exists():
        log.info(f"Already have: {parquet_path}")
        return parquet_path

    base_url = "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024"
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")

    all_rows = []
    for date_str in dates:
        file_url = f"{base_url}/ais-{date_str}.parquet"
        log.info(f"Querying {file_url}...")
        try:
            rows = con.execute(f"""
                SELECT mmsi, base_date_time, sog, cog, heading,
                       vessel_name, imo, call_sign, vessel_type, status,
                       length, width, draft, cargo, transceiver,
                       ST_Y(geometry) as lat, ST_X(geometry) as lon
                FROM read_parquet('{file_url}')
                WHERE ST_Y(geometry) BETWEEN {bbox['lat_min']} AND {bbox['lat_max']}
                  AND ST_X(geometry) BETWEEN {bbox['lon_min']} AND {bbox['lon_max']}
            """).fetchall()
            log.info(f"  {len(rows):,} rows in bbox")
            all_rows.extend(rows)
        except Exception as e:
            log.warning(f"  Failed: {str(e)[:150]}")

    con.close()

    if not all_rows:
        log.error("No NOAA data retrieved")
        return None

    cols = ['mmsi','base_date_time','sog','cog','heading','vessel_name','imo',
            'call_sign','vessel_type','status','length','width','draft','cargo',
            'transceiver','lat','lon']
    df = pl.DataFrame([dict(zip(cols, r)) for r in all_rows])
    df.write_parquet(parquet_path, compression='zstd')
    size_mb = parquet_path.stat().st_size / 1024 / 1024
    log.info(f"Written: {parquet_path} ({size_mb:.1f} MB, {len(df):,} rows)")
    return parquet_path


def detect_approaches(df: pl.DataFrame, zone: dict) -> list[dict]:
    """Detect vessels entering zone and slowing below threshold."""
    zone_df = df.filter(
        (pl.col('lat') >= zone['lat_min']) & (pl.col('lat') <= zone['lat_max']) &
        (pl.col('lon') >= zone['lon_min']) & (pl.col('lon') <= zone['lon_max'])
    )
    if len(zone_df) == 0 or 'mmsi' not in zone_df.columns:
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
        entries = vessel.filter(pl.col('in_zone') & ~pl.col('prev_in_zone').fill_null(False))

        for entry in entries.iter_rows(named=True):
            post = vessel.filter(
                (pl.col('event_time') >= entry['event_time']) & pl.col('in_zone')
            )
            slow = post.filter(
                pl.col('sog').is_not_null() & (pl.col('sog') < ARRIVAL_SPEED_THRESHOLD) & (pl.col('sog') >= 0)
            )
            if len(slow) > 0:
                approaches.append({
                    'mmsi': int(mmsi) if mmsi is not None else 0,
                    'entry_time': str(entry['event_time']),
                    'zone': zone['name'],
                })
                break
    return approaches


def analyze_houston(parquet_path: Path) -> dict:
    """Full analysis of Houston sample."""
    log.info(f"\n{'='*60}\nAnalyzing Houston\n{'='*60}")

    df = pl.read_parquet(parquet_path)
    log.info(f"Loaded: {len(df):,} rows, columns: {df.columns}")

    # Rename to standard names
    rename_map = {'base_date_time': 'event_time'}
    df = df.rename(rename_map)

    # Parse timestamps
    if df['event_time'].dtype == pl.Utf8:
        for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M']:
            try:
                df = df.with_columns(pl.col('event_time').str.strptime(pl.Datetime('us'), fmt, strict=False))
                break
            except Exception:
                continue

    df = df.filter(pl.col('event_time').is_not_null()).sort('event_time')
    df = df.with_columns([
        pl.col('mmsi').cast(pl.Int64, strict=False),
        pl.col('sog').cast(pl.Float32, strict=False),
        pl.col('cog').cast(pl.Float32, strict=False),
        pl.col('lat').cast(pl.Float64, strict=False),
        pl.col('lon').cast(pl.Float64, strict=False),
    ])

    r = {}
    r['total_records'] = len(df)
    r['time_range_start'] = str(df['event_time'].min())
    r['time_range_end'] = str(df['event_time'].max())
    dur = df['event_time'].max() - df['event_time'].min()
    r['duration_hours'] = round(dur.total_seconds() / 3600, 1) if dur else 0
    r['duration_days'] = round(r['duration_hours'] / 24, 2)

    # Valid MMSI
    valid = df.filter((pl.col('mmsi') >= 200_000_000) & (pl.col('mmsi') <= 799_999_999))
    r['unique_vessels'] = int(valid['mmsi'].n_unique())
    r['total_unique_mmsi'] = int(df['mmsi'].n_unique())
    r['invalid_mmsi_pct'] = round((len(df) - len(valid)) / len(df) * 100, 2)

    # Message rate
    if r['duration_hours'] > 0:
        r['msgs_per_minute'] = round(r['total_records'] / r['duration_hours'] / 60, 1)

    # Hourly coverage
    df_hourly = (
        df.with_columns(pl.col('event_time').dt.truncate('1h').alias('hour'))
        .group_by('hour').agg([pl.len().alias('msg_count'),
                               pl.col('mmsi').n_unique().alias('unique_mmsis')])
        .sort('hour')
    )
    total_hrs = max(int(r['duration_hours']), 1)
    r['observed_hours'] = len(df_hourly)
    r['total_calendar_hours'] = total_hrs
    r['coverage_pct'] = round(len(df_hourly) / total_hrs * 100, 1)
    r['missing_hours'] = total_hrs - len(df_hourly)

    if len(df_hourly) > 0:
        mc = df_hourly['msg_count']
        r['hourly_msg_median'] = float(mc.median())
        r['hourly_msg_mean'] = round(float(mc.mean()), 0)
        r['hourly_msg_min'] = int(mc.min())
        r['hourly_msg_max'] = int(mc.max())
        gap_thresh = r['hourly_msg_median'] * 0.5
        gaps = df_hourly.filter(pl.col('msg_count') < gap_thresh)
        r['low_coverage_hours'] = len(gaps)
        r['low_coverage_pct'] = round(len(gaps) / len(df_hourly) * 100, 1)

    # Data quality
    q = {}
    sog = df['sog'].drop_nulls()
    q['sog_null_pct'] = round((len(df) - len(sog)) / len(df) * 100, 2)
    q['sog_sentinel_102_3'] = int(sog.filter(sog == 102.3).len())
    q['sog_over_50kn'] = int(sog.filter(sog > 50).len())
    if 'heading' in df.columns:
        hdg = df['heading'].drop_nulls()
        q['heading_sentinel_511'] = int(hdg.filter(hdg == 511).len())
    if 'cog' in df.columns:
        cog_col = df['cog'].drop_nulls()
        q['cog_sentinel_360'] = int(cog_col.filter(cog_col == 360).len())
    r['data_quality'] = q

    # Vessel type distribution
    if 'vessel_type' in df.columns:
        vt = (df.filter(pl.col('vessel_type').is_not_null())
              .group_by('vessel_type')
              .agg(pl.col('mmsi').n_unique().alias('vessels'))
              .sort('vessels', descending=True).head(10))
        r['top_vessel_types'] = vt.to_dicts()

    # Approach detection
    approach_results = []
    for zone in HOUSTON_APPROACH_ZONES:
        log.info(f"  Detecting approaches: {zone['name']}")
        approaches = detect_approaches(df, zone)
        zone_df = df.filter(
            (pl.col('lat') >= zone['lat_min']) & (pl.col('lat') <= zone['lat_max']) &
            (pl.col('lon') >= zone['lon_min']) & (pl.col('lon') <= zone['lon_max'])
        )
        ar = {
            'zone': zone['name'],
            'records_in_zone': len(zone_df),
            'vessels_in_zone': int(zone_df['mmsi'].n_unique()),
            'approaches_detected': len(approaches),
        }
        approach_results.append(ar)
        log.info(f"    Records: {ar['records_in_zone']:,}, Vessels: {ar['vessels_in_zone']}, "
                 f"Approaches: {ar['approaches_detected']}")

    r['approach_analysis'] = approach_results
    total_approaches = sum(a['approaches_detected'] for a in approach_results)
    r['total_approaches'] = total_approaches

    # Arrival rate estimate
    if total_approaches >= 30 and r['duration_hours'] > 0:
        rate = total_approaches / r['duration_hours']
        se = math.sqrt(total_approaches) / r['duration_hours']
        z = 1.96
        r['approach_rate_per_hour'] = round(rate, 3)
        r['approach_rate_per_day'] = round(rate * 24, 1)
        r['approach_rate_ci_lower'] = round(max(0, rate - z * se), 3)
        r['approach_rate_ci_upper'] = round(rate + z * se, 3)
        r['approach_rate_label'] = "ESTIMATE (Poisson 95% CI)"
        r['approach_rate_note'] = (
            f"Based on {total_approaches} approaches over {r['duration_hours']:.1f}h "
            f"({r['duration_days']:.1f} days). Seasonal/weekly/time-of-day variation not captured. "
            f"Approach defined as: vessel enters geofence zone and has >=1 report with SOG < {ARRIVAL_SPEED_THRESHOLD} kn."
        )
    else:
        r['approach_rate_label'] = f"INSUFFICIENT (n={total_approaches}, need >=30)"
        r['approach_rate_per_hour'] = None
        r['approach_rate_note'] = (
            f"Only {total_approaches} approaches detected in {r['duration_hours']:.1f}h. "
            f"Insufficient for rate estimation (need >=30)."
        )

    return r


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ORCASTRA M0 — Historical AIS Feasibility v3")
    log.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # ── NOAA Houston ──
    log.info("\n" + "="*70 + "\nNOAA MARINECADASTRE — HOUSTON\n" + "="*70)

    noaa_dates = ["2024-01-15", "2024-01-16", "2024-01-17"]
    noaa_dir = DATA_DIR / "noaa" / "houston"
    noaa_path = query_noaa_houston(noaa_dates, HOUSTON_BBOX, noaa_dir)

    houston = None
    if noaa_path:
        houston = analyze_houston(noaa_path)
        houston['source'] = "NOAA MarineCadastre"
        houston['source_url'] = "https://marinecadastre.gov/ais/"
        houston['azure_url'] = "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024/"
        houston['licence'] = "Public resource, free. Cite BOEM/NOAA Marine Cadastre."
        houston['access_date'] = "2026-10-07"
        houston['archive_period'] = "2024-01-01 to 2024-12-31 (GeoParquet on Azure); 2009-present (CSV bulk)"
        houston['time_gap_to_today'] = "~21 months (2024 data, current date 2026-10-07)"
        houston['sample_dates'] = noaa_dates
        houston['sampling_method'] = (
            "DuckDB remote query of NOAA GeoParquet files on Azure blob storage. "
            "Spatial filter: ST_Y(geometry)/ST_X(geometry) within Houston bounding box. "
            "No full file download — only matching row groups transferred."
        )
        houston['bbox'] = HOUSTON_BBOX
        houston['approach_zones'] = HOUSTON_APPROACH_ZONES
        houston['approach_definition'] = (
            f"Vessel enters approach zone geofence AND has >=1 AIS report with SOG < {ARRIVAL_SPEED_THRESHOLD} kn. "
            "One approach counted per unique MMSI per zone. "
            "This is a CONSERVATIVE count — actual arrivals may be higher."
        )
        houston['disk_usage_mb'] = round(noaa_path.stat().st_size / 1024 / 1024, 1)
    else:
        houston = {'error': 'Could not retrieve NOAA GeoParquet', 'access_date': '2026-10-07'}

    # ── DMA Danish Straits (document failure) ──
    danish = {
        'error': 'DMA web.ais.dk unreachable',
        'source': 'Danish Maritime Authority (DMA)',
        'source_url': 'https://web.ais.dk/aisdata/',
        'licence': 'Free, worldwide, non-exclusive, unlimited usage rights (per DMA policy)',
        'access_date': '2026-10-07',
        'access_attempts': [
            'https://web.ais.dk/aisdata/ (HTTPS) — SSL CERTIFICATE_VERIFY_FAILED: hostname mismatch',
            'https://web.ais.dk/aisdata/ (HTTPS, verify=False) — RemoteDisconnected',
            'http://web.ais.dk/aisdata/ (HTTP) — ConnectTimeoutError',
        ],
        'note': (
            'The DMA data portal (web.ais.dk) is currently unreachable due to SSL certificate '
            'hostname mismatch and connection timeouts on both HTTPS and HTTP. '
            'Recent search results suggest DMA may have moved to a formal application process. '
            'This makes Danish Straits historical data inaccessible for automated download as of 2026-10-07. '
            'Danish Straits cannot be evaluated without historical data.'
        ),
        'total_records': 0,
        'unique_vessels': 0,
        'total_approaches': 0,
        'approach_rate_label': 'CANNOT EVALUATE — archive inaccessible',
    }

    # ── Save ──
    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'houston': houston,
        'danish_straits': danish,
    }
    results_path = RESULTS_DIR / "feasibility_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ── Print summary ──
    log.info("\n" + "="*70 + "\nSUMMARY\n" + "="*70)

    if houston and 'error' not in houston:
        log.info("\n--- Houston (NOAA MarineCadastre) ---")
        log.info(f"  Source: {houston['source']}")
        log.info(f"  Archive period: {houston['archive_period']}")
        log.info(f"  Time gap to today: {houston['time_gap_to_today']}")
        log.info(f"  Sample dates: {houston['sample_dates']}")
        log.info(f"  Records: {houston['total_records']:,}")
        log.info(f"  Unique vessels (valid MMSI): {houston['unique_vessels']}")
        log.info(f"  Duration: {houston['duration_days']} days ({houston['duration_hours']}h)")
        log.info(f"  Msgs/min: {houston.get('msgs_per_minute', 'N/A')}")
        log.info(f"  Coverage: {houston.get('coverage_pct', 'N/A')}%")
        log.info(f"  Missing hours: {houston.get('missing_hours', 'N/A')}")
        log.info(f"  Low-coverage hours: {houston.get('low_coverage_hours', 'N/A')} ({houston.get('low_coverage_pct', 'N/A')}%)")
        for az in houston.get('approach_analysis', []):
            log.info(f"  Zone '{az['zone']}': {az['approaches_detected']} approaches, "
                     f"{az['vessels_in_zone']} vessels, {az['records_in_zone']:,} records")
        log.info(f"  Total approaches: {houston['total_approaches']}")
        log.info(f"  Rate: {houston['approach_rate_label']}")
        if houston.get('approach_rate_per_hour') is not None:
            log.info(f"    {houston['approach_rate_per_hour']}/hr "
                     f"[{houston['approach_rate_ci_lower']}, {houston['approach_rate_ci_upper']}] 95% CI")
            log.info(f"    = {houston['approach_rate_per_day']}/day")
        log.info(f"  Data quality: {houston.get('data_quality', {})}")
        log.info(f"  Disk: {houston.get('disk_usage_mb', 0)} MB")
    else:
        log.info(f"\n--- Houston: ERROR: {houston.get('error', 'unknown')}")

    log.info(f"\n--- Danish Straits ---")
    log.info(f"  {danish['note']}")

    log.info(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
