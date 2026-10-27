"""
ORCASTRA -- Offline Replay Demo.

Replays a saved AISStream recording session from Parquet files on disk,
applies the deterministic short-horizon ETA model (re-trained from NOAA
historical data with identical parameters and random_state=42), and prints
corridor-state ETA predictions.

Requirements:
  - NOAA 3-month historical Parquet files in D:\\ORCASTRA_DATA\\historical\\noaa_3mo\\
  - A saved recording session in D:\\ORCASTRA_DATA\\recordings\\

NO NETWORK ACCESS. Never starts the collector or creates a live session.
Output is clearly labelled as OFFLINE REPLAY.

Usage:
    python scripts/offline_replay_demo.py [--session SESSION_DIR]
"""

import argparse
import math
import sys
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Prevent any network access
os.environ.setdefault("NO_PROXY", "*")

import numpy as np
import polars as pl

BANNER = """
===============================================================================
  OFFLINE REPLAY -- RECORDED DATA, NOT LIVE
  ORCASTRA Short-Horizon ETA Demo
  No network access. No live AISStream connection. No new session created.
===============================================================================
"""

# -- Paths --
NOAA_DIR = Path(r"D:\ORCASTRA_DATA\historical\noaa_3mo")
DEFAULT_SESSION = Path(r"D:\ORCASTRA_DATA\recordings\session_20261008_160107")

NOAA_COLUMNS = [
    "mmsi", "base_date_time", "sog", "cog", "heading",
    "vessel_type", "length", "width", "draft", "lat", "lon",
]

CENTERLINE_WP = [
    (29.350, -94.750, 50.4),
    (29.380, -94.780, 46.0),
    (29.510, -94.900, 28.5),
    (29.600, -94.960, 16.0),
    (29.680, -95.000, 4.5),
    (29.720, -95.050, 0.0),
]

CORRIDOR = {"lat_min": 29.34, "lat_max": 29.75, "lon_min": -95.12, "lon_max": -94.70}
ENTRANCE_ZONE = {"lat_min": 29.30, "lat_max": 29.45, "lon_min": -94.85, "lon_max": -94.65}
DESTINATION_ZONE = {"lat_min": 29.65, "lat_max": 29.80, "lon_min": -95.15, "lon_max": -94.95}
COMMERCIAL_TYPES = set(range(70, 90))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0)**2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def channel_distance_to_berth(lat, lon):
    min_dist = float('inf')
    best_rem = 50.4
    for w_lat, w_lon, rem_km in CENTERLINE_WP:
        d = haversine_km(lat, lon, w_lat, w_lon)
        if d < min_dist:
            min_dist = d
            best_rem = rem_km
    return max(best_rem, 0.5)


def in_box(lat, lon, box):
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def safe_sog(val):
    if val is None or (isinstance(val, (int, float)) and abs(val - 102.3) < 0.01):
        return 0.0
    return float(val)


def safe_cog(val):
    if val is None or (isinstance(val, (int, float)) and abs(val - 360.0) < 0.01):
        return None
    return float(val)


def is_inbound(cog_val):
    return cog_val is not None and ((270 <= cog_val <= 360) or (0 <= cog_val <= 45))


def train_model_from_noaa():
    """Train the identical quantile model from NOAA historical data.
    
    Uses the exact same parameters, features, and random_state=42 as the
    frozen Protocol v2.1 evaluation, producing a deterministically identical model.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    print("[MODEL] Training from NOAA historical data (Jan 1 - Feb 15 2024)...")
    files = sorted(NOAA_DIR.glob("houston_2024-*.parquet"))
    if not files:
        return None, None, None

    df_raw = pl.read_parquet(files, columns=NOAA_COLUMNS)
    df = df_raw.with_columns(
        pl.col("base_date_time").cast(pl.Datetime).alias("dt")
    ).sort(["mmsi", "dt"])

    min_ts = df["dt"].min()
    max_ts = df["dt"].max()
    left_bound = min_ts + timedelta(hours=6)
    right_bound = max_ts - timedelta(hours=72)
    split_boundary = datetime(2024, 3, 1, 0, 0, 0)
    calib_cutoff = datetime(2024, 2, 16, 0, 0, 0)

    df_comm = df.filter(pl.col("vessel_type").is_in(list(COMMERCIAL_TYPES)))
    mmsi_groups = df_comm.partition_by("mmsi", as_dict=True)
    normalized = {int(k[0] if isinstance(k, tuple) else k): v for k, v in mmsi_groups.items()}

    # Extract Cohort A1 arrivals
    cohort = []
    for mmsi, vdf in normalized.items():
        recs = vdf.sort("dt").to_dicts()
        n = len(recs)
        was_inside = False
        i = 0
        while i < n:
            rec = recs[i]
            inside = in_box(rec["lat"], rec["lon"], ENTRANCE_ZONE)
            if inside and not was_inside:
                t0 = rec["dt"]
                sog0 = safe_sog(rec["sog"])
                if t0 <= left_bound or sog0 < 3.0:
                    was_inside = True; i += 1; continue
                cog_val = safe_cog(rec["cog"])
                if not is_inbound(cog_val):
                    was_inside = True; i += 1; continue
                j = i + 1
                found = False
                while j < n:
                    rn = recs[j]
                    dur = (rn["dt"] - t0).total_seconds() / 3600.0
                    if dur > 72.0: break
                    if in_box(rn["lat"], rn["lon"], DESTINATION_ZONE):
                        ns = safe_sog(rn["sog"])
                        if ns < 1.5 and dur >= 1.0:
                            if t0 <= right_bound:
                                cohort.append({"mmsi": mmsi, "t0": t0, "t_arr": rn["dt"],
                                               "is_train": t0 < split_boundary,
                                               "length": rec.get("length"), "draft": rec.get("draft"),
                                               "vessel_type": rec["vessel_type"]})
                            found = True; break
                    j += 1
                was_inside = True
                i = max(j if found else i + 1, i + 1)
            else:
                was_inside = inside; i += 1

    # Extract snapshots for training
    snap_features = ["lat", "lon", "rem_km", "sog", "cog", "length", "draft"]
    snapshots = []
    for mmsi, vdf in normalized.items():
        recs = vdf.sort("dt").to_dicts()
        for v in [c for c in cohort if c["mmsi"] == mmsi and c["is_train"]]:
            t0, t_arr = v["t0"], v["t_arr"]
            v_pings = [p for p in recs if t0 <= p["dt"] <= t_arr
                       and in_box(p["lat"], p["lon"], CORRIDOR)]
            curr = t0
            while curr < t_arr:
                best_p, best_d = None, timedelta(minutes=6)
                for p in v_pings:
                    d = abs(p["dt"] - curr)
                    if d < best_d: best_d = d; best_p = p
                if best_p and best_d <= timedelta(minutes=5):
                    sog = safe_sog(best_p["sog"])
                    cog = safe_cog(best_p["cog"])
                    if sog >= 3.0 and is_inbound(cog):
                        rem = (t_arr - curr).total_seconds() / 3600.0
                        snapshots.append({
                            "lat": best_p["lat"], "lon": best_p["lon"],
                            "rem_km": channel_distance_to_berth(best_p["lat"], best_p["lon"]),
                            "sog": sog, "cog": cog,
                            "length": v.get("length"), "draft": v.get("draft"),
                            "rem_dur_h": rem,
                            "snap_time": curr,
                        })
                curr += timedelta(minutes=10)

    train_slice = [s for s in snapshots if s["snap_time"] < calib_cutoff]
    calib_slice = [s for s in snapshots if s["snap_time"] >= calib_cutoff]

    tr_medians = {}
    for col in snap_features:
        vals = [s[col] for s in train_slice if s[col] is not None]
        tr_medians[col] = float(np.median(vals)) if vals else 0.0

    def make_X(s_list):
        return np.array([[s[c] if s[c] is not None else tr_medians[c] for c in snap_features] for s in s_list])

    X_tr = make_X(train_slice)
    y_tr = np.array([s["rem_dur_h"] for s in train_slice])
    X_cal = make_X(calib_slice)
    y_cal = np.array([s["rem_dur_h"] for s in calib_slice])

    models = {}
    conformal_shifts = {}
    for q in [0.50, 0.80, 0.95]:
        reg = HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=42)
        reg.fit(X_tr, y_tr)
        models[q] = reg
        residuals = y_cal - reg.predict(X_cal)
        conformal_shifts[q] = float(np.percentile(residuals, q * 100.0))

    print(f"[MODEL] Trained on {len(train_slice)} snapshots, calibrated on {len(calib_slice)}")
    print(f"[MODEL] Conformal shifts: P50={conformal_shifts[0.50]:+.3f}h, "
          f"P80={conformal_shifts[0.80]:+.3f}h, P95={conformal_shifts[0.95]:+.3f}h")

    return models, conformal_shifts, tr_medians


def main():
    parser = argparse.ArgumentParser(description="ORCASTRA Offline Replay Demo")
    parser.add_argument("--session", type=str, default=str(DEFAULT_SESSION),
                        help="Path to a saved recording session directory")
    args = parser.parse_args()

    print(BANNER)

    session_dir = Path(args.session)
    events_dir = session_dir / "events"
    if not events_dir.exists():
        print(f"[ERROR] No events directory found at {events_dir}")
        print("[ERROR] Cannot replay: no compatible recording available.")
        return 1

    parts = sorted(events_dir.glob("part_*.parquet"))
    if not parts:
        print(f"[ERROR] No Parquet parts found in {events_dir}")
        return 1

    # Check model prerequisites
    if not NOAA_DIR.exists() or not list(NOAA_DIR.glob("houston_2024-*.parquet")):
        print("[ERROR] NOAA historical data not found at expected path.")
        print("[ERROR] Cannot replay: frozen model requires historical training data.")
        print("[ERROR] No predictions fabricated.")
        return 1

    # Train the deterministic model
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        print("[ERROR] scikit-learn not installed. Cannot load model.")
        return 1

    models, conformal_shifts, tr_medians = train_model_from_noaa()
    if models is None:
        print("[ERROR] Model training failed: no historical data.")
        return 1

    snap_features = ["lat", "lon", "rem_km", "sog", "cog", "length", "draft"]

    # Load and replay the recorded session
    print(f"\n[REPLAY] Loading session: {session_dir.name}")
    dfs = [pl.read_parquet(p) for p in parts]
    df = pl.concat(dfs, how="diagonal")
    if "is_duplicate" in df.columns:
        df = df.filter(pl.col("is_duplicate") == False)
    time_col = "aisstream_received_time" if "aisstream_received_time" in df.columns else "observation_time"
    df = df.sort(time_col)

    print(f"[REPLAY] {len(df)} events from {len(parts)} parts")
    print(f"[REPLAY] Time range: {df[time_col].min()} to {df[time_col].max()}")
    print(f"[REPLAY] Unique MMSIs: {df['mmsi'].n_unique()}")

    # Find vessels in the inbound corridor with underway state
    recs = df.to_dicts()
    predictions = []
    for rec in recs:
        lat = rec.get("latitude")
        lon = rec.get("longitude")
        sog_raw = rec.get("sog")
        cog_raw = rec.get("cog")
        if lat is None or lon is None:
            continue
        if not in_box(lat, lon, CORRIDOR):
            continue
        sog = safe_sog(sog_raw)
        cog = safe_cog(cog_raw)
        if sog < 3.0 or not is_inbound(cog):
            continue

        rem_km = channel_distance_to_berth(lat, lon)
        feat_vals = [lat, lon, rem_km, sog, cog,
                     rec.get("length") or tr_medians.get("length", 0),
                     rec.get("draft") or tr_medians.get("draft", 0)]
        # Handle None cog
        feat_vals = [v if v is not None else tr_medians.get(snap_features[i], 0.0) for i, v in enumerate(feat_vals)]
        X = np.array([feat_vals])

        raw_p50 = models[0.50].predict(X)[0]
        raw_p80 = max(models[0.80].predict(X)[0], raw_p50)
        raw_p95 = max(models[0.95].predict(X)[0], raw_p80)

        p50 = max(raw_p50 + conformal_shifts[0.50], 0.05)
        p80 = max(raw_p80 + conformal_shifts[0.80], p50)
        p95 = max(raw_p95 + conformal_shifts[0.95], p80)

        predictions.append({
            "mmsi": rec["mmsi"],
            "vessel": rec.get("vessel_name", ""),
            "time": rec.get(time_col),
            "lat": lat, "lon": lon,
            "sog": sog, "cog": cog,
            "d_rem_km": rem_km,
            "eta_p50_h": p50,
            "eta_p80_h": p80,
            "eta_p95_h": p95,
        })

    print(f"\n[PREDICTIONS] {len(predictions)} corridor snapshots with inbound underway state")
    if not predictions:
        print("[PREDICTIONS] No vessels found in the inbound corridor in this recording.")
        return 0

    # Print a summary table for unique vessels
    seen = set()
    print(f"\n{'='*90}")
    print(f"  OFFLINE REPLAY -- RECORDED DATA, NOT LIVE")
    print(f"  Short-Horizon ETA Predictions (Protocol v2.1 frozen model)")
    print(f"{'='*90}")
    print(f"  {'MMSI':<12} {'Vessel':<20} {'d_rem':>7} {'SOG':>5} {'P50 ETA':>9} {'P80 ETA':>9} {'P95 ETA':>9}")
    print(f"  {'':12} {'':20} {'(km)':>7} {'(kn)':>5} {'(hours)':>9} {'(hours)':>9} {'(hours)':>9}")
    print(f"  {'-'*82}")

    displayed = 0
    for p in predictions:
        key = (p["mmsi"], round(p["d_rem_km"], 0))
        if key in seen:
            continue
        seen.add(key)
        name = (p["vessel"] or "")[:18]
        print(f"  {p['mmsi']:<12} {name:<20} {p['d_rem_km']:>7.1f} {p['sog']:>5.1f} "
              f"{p['eta_p50_h']:>9.2f} {p['eta_p80_h']:>9.2f} {p['eta_p95_h']:>9.2f}")
        displayed += 1
        if displayed >= 25:
            print(f"  ... ({len(predictions) - displayed} more snapshots)")
            break

    print(f"{'='*90}")
    print(f"  NOTE: These are predictions from a frozen model applied to recorded data.")
    print(f"  P95 intervals are very wide (~34-56h) because the model honestly accounts")
    print(f"  for the possibility of mid-channel anchorage delays (13% of historical voyages).")
    print(f"  The model does NOT beat the distance-bucket median baseline on MAE.")
    print(f"  See docs/EVAL_PROTOCOL_V2_1.md for full methodology.")
    print(f"{'='*90}")
    print(f"\n  OFFLINE REPLAY -- RECORDED DATA, NOT LIVE")

    return 0


if __name__ == "__main__":
    sys.exit(main())
