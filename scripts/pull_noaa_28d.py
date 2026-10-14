"""
ORCASTRA — Pull 28-day NOAA AIS Sample for ETA Baseline.

Stream-filtered by Houston bbox directly from NOAA MarineCadastre on Azure.
Range: 2024-01-01 to 2024-01-28 (28 consecutive days).
Output: D:\ORCASTRA_DATA\historical\noaa_28d\houston_{date}.parquet
Hard limit: 2 GB total.
"""

import concurrent.futures
import math
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl

DATA_ROOT = Path(r"D:\ORCASTRA_DATA\historical")
OUTPUT_DIR = DATA_ROOT / "noaa_28d"
BASE_URL = "https://ocmgeodatastor1.blob.core.windows.net/marinecadastre/ais2024"
BBOX = {"lat_min": 29.0, "lat_max": 29.9, "lon_min": -95.3, "lon_max": -94.4}
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB hard limit


def get_dates() -> list[str]:
    start = date(2024, 1, 1)
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(28)]


def download_day(date_str: str) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"houston_{date_str}.parquet"

    # Check if already downloaded
    if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
        size = out_file.stat().st_size
        return {"date": date_str, "status": "cached", "bytes": size, "path": str(out_file)}

    # Also check if available in test dir
    test_file = DATA_ROOT / "noaa_28d_test" / f"houston_{date_str}.parquet"
    if test_file.exists() and test_file.stat().st_size > 1024 * 1024:
        import shutil
        shutil.copy2(test_file, out_file)
        return {"date": date_str, "status": "copied_test", "bytes": out_file.stat().st_size, "path": str(out_file)}

    file_url = f"{BASE_URL}/ais-{date_str}.parquet"
    out_sql_path = str(out_file).replace("\\", "/")

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL spatial; LOAD spatial;")

        sql = f"""
        COPY (
            SELECT mmsi, base_date_time, sog, cog, heading,
                   vessel_name, imo, call_sign, vessel_type, status,
                   length, width, draft, cargo, transceiver,
                   ST_Y(geometry) as lat, ST_X(geometry) as lon
            FROM read_parquet('{file_url}')
            WHERE ST_Y(geometry) BETWEEN {BBOX['lat_min']} AND {BBOX['lat_max']}
              AND ST_X(geometry) BETWEEN {BBOX['lon_min']} AND {BBOX['lon_max']}
        ) TO '{out_sql_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(sql)
    finally:
        con.close()

    size = out_file.stat().st_size if out_file.exists() else 0
    return {"date": date_str, "status": "downloaded", "bytes": size, "path": str(out_file)}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = get_dates()
    print(f"Target: 28 days ({dates[0]} to {dates[-1]})")
    print(f"Bounding box: {BBOX}")
    print(f"Destination: {OUTPUT_DIR}")
    print(f"Hard limit: 2.0 GB ({MAX_TOTAL_BYTES:,} bytes)")
    print(f"Expected size: ~280 MB - 310 MB (~14M - 15.5M rows)")

    t0 = time.time()
    results = []

    # Use 4 parallel workers for HTTP range request downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_date = {executor.submit(download_day, d): d for d in dates}
        for future in concurrent.futures.as_completed(future_to_date):
            d = future_to_date[future]
            try:
                res = future.result()
                results.append(res)
                # Check cumulative size
                current_total = sum(f.stat().st_size for f in OUTPUT_DIR.glob("houston_*.parquet"))
                print(f"[{len(results)}/28] {d}: {res['status']} ({res['bytes']/1024/1024:.2f} MB), Total: {current_total/1024/1024:.2f} MB")
                if current_total > MAX_TOTAL_BYTES:
                    print(f"ABORT: Exceeded 2 GB limit ({current_total/1024/1024:.2f} MB)")
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR on {d}: {e}")
                sys.exit(1)

    elapsed = time.time() - t0
    final_files = list(OUTPUT_DIR.glob("houston_*.parquet"))
    total_bytes = sum(f.stat().st_size for f in final_files)

    print("\n" + "=" * 60)
    print("NOAA 28-DAY PULL COMPLETE")
    print(f"Days completed: {len(final_files)}/28")
    print(f"Total size: {total_bytes / 1024 / 1024:.2f} MB ({total_bytes / 1024 / 1024 / 1024:.3f} GB)")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Hard limit check: {total_bytes / 1024 / 1024:.2f} MB < 2048 MB (PASS)")
    print("=" * 60)


if __name__ == "__main__":
    main()
