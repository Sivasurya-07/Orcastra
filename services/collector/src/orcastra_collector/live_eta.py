"""
ORCASTRA Live ETA Engine & Prediction Logger.

Implements Protocol v2.1 Distance-Bucket Median Baseline:
  - Centerline waypoints & along-channel distance calculation
  - Honest distance-to-berth bucket baseline medians (1.010, 1.759, 2.591, 3.539 h)
  - Cohort A live eligibility rule (D017): untyped vessels treated as unknown, not non-commercial
  - Stale (> 5 min) and invalid coordinate handling: ETA "unavailable"
  - Immutable prediction log & completed arrival tracking in SQLite
"""

import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# ── Geographic Definitions (Houston Ship Channel) ──

CENTERLINE_WP = [
    (29.350, -94.750, 50.4),  # Entrance buoy
    (29.380, -94.780, 46.0),  # Bolivar Roads
    (29.510, -94.900, 28.5),  # Red Fish Bar
    (29.600, -94.960, 16.0),  # Mid-Bay Beacon
    (29.680, -95.000, 4.5),   # Barbours Cut Turn
    (29.720, -95.050, 0.0),   # Inner Berths
]

CORRIDOR = {
    "lat_min": 29.34, "lat_max": 29.75,
    "lon_min": -95.12, "lon_max": -94.70,
}

ENTRANCE_ZONE = {
    "lat_min": 29.30, "lat_max": 29.45,
    "lon_min": -94.85, "lon_max": -94.65,
}

DESTINATION_ZONE = {
    "lat_min": 29.65, "lat_max": 29.80,
    "lon_min": -95.15, "lon_max": -94.95,
}

# Known non-commercial vessel types strictly excluded from scoring
# (tugs 31/52, pleasure 36/37, pilot 50, fishing 30)
KNOWN_NON_COMMERCIAL_TYPES = {30, 31, 36, 37, 50, 52}

# ── Distance-to-Berth Buckets & Protocol v2.1 Frozen Baseline Values ──
# Source: tier1_v2_1_short_horizon_results.json -> "bucket_train_medians"
DISTANCE_BUCKETS = [
    ("Near Berth (d <= 10 km)", 0.0, 10.0, 1.009861111111111),       # 1.010 h (~60.6 min)
    ("Mid-Channel (10-25 km)", 10.0, 25.0, 1.7586111111111111),      # 1.759 h (~105.5 min)
    ("Outer Channel (25-50 km)", 25.0, 50.0, 2.5908333333333333),    # 2.591 h (~155.5 min)
    ("Beyond Entrance (d > 50 km)", 50.0, 999.0, 3.5388888888888888), # 3.539 h (~212.3 min)
]

METHOD_NAME = "Distance-Bucket Median Baseline"
P95_INTERVAL_NOTE = "P95 uncertainty interval is ~40 h wide"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0)**2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def channel_distance_to_berth(lat: float, lon: float) -> float:
    """
    Compute approximate remaining along-channel distance (km) to berth
    via projection onto Houston Ship Channel centerline waypoints.
    """
    min_dist = float("inf")
    best_rem = 50.4
    for w_lat, w_lon, rem_km in CENTERLINE_WP:
        d = haversine_km(lat, lon, w_lat, w_lon)
        if d < min_dist:
            min_dist = d
            best_rem = rem_km
    return max(best_rem, 0.5)


def in_box(lat: float, lon: float, box: dict[str, float]) -> bool:
    """Check if lat/lon is inside a bounding box."""
    return (box["lat_min"] <= lat <= box["lat_max"] and
            box["lon_min"] <= lon <= box["lon_max"])


def is_inbound_heading(cog: Optional[float], in_entrance: bool = False) -> bool:
    """
    Check if heading is inbound towards the port of Houston.
    Corridor: [270, 360] U [0, 45]. Entrance: [270, 360] U [0, 30].
    """
    if cog is None or abs(cog - 360.0) < 0.01:
        return False
    max_angle = 30.0 if in_entrance else 45.0
    return (270.0 <= cog <= 360.0) or (0.0 <= cog <= max_angle)


@dataclass
class VesselLiveEta:
    """Live ETA result for a single vessel observation."""
    mmsi: int
    name: str
    latitude: Optional[float]
    longitude: Optional[float]
    sog: Optional[float]
    cog: Optional[float]
    distance_to_berth_km: Optional[float]
    bucket_name: Optional[str]
    method: str
    predicted_duration_h: Optional[float]
    predicted_eta_utc: Optional[str]
    status: str             # "active", "stale", "invalid", "not_inbound", "stationary", "non_commercial"
    is_eligible: bool       # Eligible for live Cohort A scoring
    is_stale: bool
    data_age_seconds: float
    reason: str


def compute_live_eta(
    vessel_data: dict[str, Any],
    now_utc: Optional[datetime] = None,
    stale_threshold_seconds: float = 300.0,
) -> VesselLiveEta:
    """
    Compute Distance-Bucket Median Baseline ETA for a vessel state record.
    Enforces D017 Cohort A rules and staleness cutoff.
    """
    now = now_utc or datetime.now(timezone.utc)
    mmsi = vessel_data.get("mmsi", 0)
    name = vessel_data.get("name", f"MMSI_{mmsi}")
    lat = vessel_data.get("latitude")
    lon = vessel_data.get("longitude")
    sog = vessel_data.get("sog")
    cog = vessel_data.get("cog")
    vtype = vessel_data.get("vessel_type")
    last_seen_str = vessel_data.get("last_seen_utc")

    # 1. Staleness check
    data_age_s = float("inf")
    if last_seen_str:
        try:
            ls = datetime.fromisoformat(last_seen_str.replace("Z", "+00:00"))
            data_age_s = max(0.0, (now - ls).total_seconds())
        except Exception:
            pass

    is_stale = data_age_s > stale_threshold_seconds

    # 2. Invalid coordinates check
    if lat is None or lon is None or math.isnan(lat) or math.isnan(lon):
        return VesselLiveEta(
            mmsi=mmsi, name=name, latitude=None, longitude=None, sog=sog, cog=cog,
            distance_to_berth_km=None, bucket_name=None, method=METHOD_NAME,
            predicted_duration_h=None, predicted_eta_utc=None,
            status="invalid", is_eligible=False, is_stale=is_stale,
            data_age_seconds=data_age_s, reason="Invalid or missing coordinates",
        )

    # 3. Staleness: ETA is unavailable
    if is_stale:
        return VesselLiveEta(
            mmsi=mmsi, name=name, latitude=lat, longitude=lon, sog=sog, cog=cog,
            distance_to_berth_km=None, bucket_name=None, method=METHOD_NAME,
            predicted_duration_h=None, predicted_eta_utc=None,
            status="stale", is_eligible=False, is_stale=True,
            data_age_seconds=data_age_s, reason=f"Position stale ({data_age_s:.0f}s > {stale_threshold_seconds:.0f}s)",
        )

    # 4. Non-commercial exclusion (D017)
    # Untyped vessels (None or 0) are treated as UNKNOWN, NOT non-commercial!
    if vtype is not None and vtype in KNOWN_NON_COMMERCIAL_TYPES:
        return VesselLiveEta(
            mmsi=mmsi, name=name, latitude=lat, longitude=lon, sog=sog, cog=cog,
            distance_to_berth_km=None, bucket_name=None, method=METHOD_NAME,
            predicted_duration_h=None, predicted_eta_utc=None,
            status="non_commercial", is_eligible=False, is_stale=False,
            data_age_seconds=data_age_s, reason=f"Known non-commercial vessel type {vtype}",
        )

    # 5. Position & Corridor check
    in_corridor = in_box(lat, lon, CORRIDOR)
    in_entrance = in_box(lat, lon, ENTRANCE_ZONE)
    if not (in_corridor or in_entrance):
        return VesselLiveEta(
            mmsi=mmsi, name=name, latitude=lat, longitude=lon, sog=sog, cog=cog,
            distance_to_berth_km=None, bucket_name=None, method=METHOD_NAME,
            predicted_duration_h=None, predicted_eta_utc=None,
            status="out_of_corridor", is_eligible=False, is_stale=False,
            data_age_seconds=data_age_s, reason="Outside channel corridor",
        )

    # 6. Underway speed check (SOG >= 3.0 kn)
    safe_s = 0.0 if (sog is None or abs(sog - 102.3) < 0.01) else float(sog)
    if safe_s < 3.0:
        return VesselLiveEta(
            mmsi=mmsi, name=name, latitude=lat, longitude=lon, sog=safe_s, cog=cog,
            distance_to_berth_km=None, bucket_name=None, method=METHOD_NAME,
            predicted_duration_h=None, predicted_eta_utc=None,
            status="stationary", is_eligible=False, is_stale=False,
            data_age_seconds=data_age_s, reason="Stationary / speed < 3.0 kn",
        )

    # 7. Inbound heading check
    if not is_inbound_heading(cog, in_entrance=in_entrance):
        return VesselLiveEta(
            mmsi=mmsi, name=name, latitude=lat, longitude=lon, sog=safe_s, cog=cog,
            distance_to_berth_km=None, bucket_name=None, method=METHOD_NAME,
            predicted_duration_h=None, predicted_eta_utc=None,
            status="not_inbound", is_eligible=False, is_stale=False,
            data_age_seconds=data_age_s, reason="Heading not inbound towards port",
        )

    # 8. Compute along-channel distance and bucket median
    d_rem = channel_distance_to_berth(lat, lon)
    selected_bucket = None
    pred_h = 2.0
    for bname, d_lo, d_hi, median_val in DISTANCE_BUCKETS:
        if d_lo <= d_rem < d_hi:
            selected_bucket = bname
            pred_h = median_val
            break

    if selected_bucket is None:
        selected_bucket = DISTANCE_BUCKETS[-1][0]
        pred_h = DISTANCE_BUCKETS[-1][3]

    eta_time = now + timedelta(hours=pred_h)
    eta_str = eta_time.strftime("%H:%M UTC (%Y-%m-%d)")

    return VesselLiveEta(
        mmsi=mmsi,
        name=name,
        latitude=lat,
        longitude=lon,
        sog=safe_s,
        cog=cog,
        distance_to_berth_km=round(d_rem, 1),
        bucket_name=selected_bucket,
        method=METHOD_NAME,
        predicted_duration_h=round(pred_h, 3),
        predicted_eta_utc=eta_str,
        status="active",
        is_eligible=True,
        is_stale=False,
        data_age_seconds=data_age_s,
        reason="Active inbound vessel",
    )


class PredictionLogger:
    """
    Immutable SQLite prediction log and arrival scoring engine.
    Stores live predictions and computes true absolute error upon berth arrival.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mmsi INTEGER NOT NULL,
                    vessel_name TEXT,
                    prediction_time_utc TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    sog REAL,
                    cog REAL,
                    distance_to_berth_km REAL,
                    bucket_name TEXT,
                    method TEXT NOT NULL,
                    predicted_duration_h REAL NOT NULL,
                    predicted_eta_utc TEXT NOT NULL,
                    is_eligible_cohort_a INTEGER NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS completed_arrivals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mmsi INTEGER NOT NULL,
                    first_prediction_id INTEGER,
                    prediction_time_utc TEXT NOT NULL,
                    predicted_eta_utc TEXT NOT NULL,
                    arrival_time_utc TEXT NOT NULL,
                    actual_duration_h REAL NOT NULL,
                    absolute_error_h REAL NOT NULL,
                    fully_observed INTEGER NOT NULL,
                    method TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_mmsi ON predictions (mmsi, prediction_time_utc)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_arr_mmsi ON completed_arrivals (mmsi)")
            conn.commit()

    def log_prediction(self, eta: VesselLiveEta, obs_time: Optional[datetime] = None) -> Optional[int]:
        """Record an eligible prediction into the immutable log."""
        if not eta.is_eligible or eta.predicted_duration_h is None or eta.latitude is None or eta.longitude is None:
            return None

        t_pred = (obs_time or datetime.now(timezone.utc)).isoformat()
        with self._connection() as conn:
            cur = conn.execute("""
                INSERT INTO predictions (
                    mmsi, vessel_name, prediction_time_utc, latitude, longitude,
                    sog, cog, distance_to_berth_km, bucket_name, method,
                    predicted_duration_h, predicted_eta_utc, is_eligible_cohort_a
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                eta.mmsi, eta.name, t_pred, eta.latitude, eta.longitude,
                eta.sog, eta.cog, eta.distance_to_berth_km, eta.bucket_name,
                eta.method, eta.predicted_duration_h, eta.predicted_eta_utc or "",
            ))
            conn.commit()
            return cur.lastrowid

    def check_arrival_and_score(
        self,
        mmsi: int,
        lat: float,
        lon: float,
        sog: float,
        obs_time: Optional[datetime] = None,
    ) -> Optional[dict]:
        """
        Check if a vessel has reached the destination/berth zone.
        If yes, matches the first active prediction, computes actual arrival error,
        and records it in completed_arrivals.
        """
        if not in_box(lat, lon, DESTINATION_ZONE) or sog >= 1.5:
            return None

        t_arr = obs_time or datetime.now(timezone.utc)

        with self._connection() as conn:
            # Check if this vessel has already been recorded as arrived recently (within 6h)
            recent = conn.execute("""
                SELECT id FROM completed_arrivals
                WHERE mmsi = ? AND arrival_time_utc >= ?
            """, (mmsi, (t_arr - timedelta(hours=6)).isoformat())).fetchone()
            if recent:
                return None

            # Find the first prediction for this vessel in the last 24h
            first_pred = conn.execute("""
                SELECT * FROM predictions
                WHERE mmsi = ? AND prediction_time_utc >= ?
                ORDER BY prediction_time_utc ASC
                LIMIT 1
            """, (mmsi, (t_arr - timedelta(hours=24)).isoformat())).fetchone()

            if not first_pred:
                return None

            t_pred = datetime.fromisoformat(first_pred["prediction_time_utc"].replace("Z", "+00:00"))
            actual_dur_h = (t_arr - t_pred).total_seconds() / 3600.0
            if actual_dur_h <= 0.2:  # too short / already at berth when predicted
                return None

            abs_error_h = abs(actual_dur_h - first_pred["predicted_duration_h"])

            # Verify vessel was fully observed (e.g. at least 2 pings, no gap > 1h)
            pred_count = conn.execute("""
                SELECT COUNT(*) as cnt FROM predictions
                WHERE mmsi = ? AND prediction_time_utc >= ? AND prediction_time_utc <= ?
            """, (mmsi, first_pred["prediction_time_utc"], t_arr.isoformat())).fetchone()["cnt"]

            fully_observed = 1 if pred_count >= 2 else 0

            cur = conn.execute("""
                INSERT INTO completed_arrivals (
                    mmsi, first_prediction_id, prediction_time_utc,
                    predicted_eta_utc, arrival_time_utc, actual_duration_h,
                    absolute_error_h, fully_observed, method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mmsi, first_pred["id"], first_pred["prediction_time_utc"],
                first_pred["predicted_eta_utc"], t_arr.isoformat(),
                round(actual_dur_h, 3), round(abs_error_h, 3),
                fully_observed, first_pred["method"],
            ))
            conn.commit()

            return {
                "id": cur.lastrowid,
                "mmsi": mmsi,
                "predicted_duration_h": first_pred["predicted_duration_h"],
                "actual_duration_h": actual_dur_h,
                "absolute_error_h": abs_error_h,
                "fully_observed": bool(fully_observed),
            }

    def get_scoreboard_stats(self) -> dict[str, Any]:
        """
        Return scoreboard metrics across fully observed completed arrivals.
        If n < 30, reports 'collecting (n = X)' status with dates/hours covered.
        """
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT absolute_error_h, arrival_time_utc
                FROM completed_arrivals
                WHERE fully_observed = 1
                ORDER BY arrival_time_utc ASC
            """).fetchall()

            n = len(rows)
            if n == 0:
                return {
                    "status": "collecting (n = 0)",
                    "n": 0,
                    "mae_h": None,
                    "medae_h": None,
                    "ci_95": None,
                    "dates_covered": "None yet",
                    "utc_hours_covered": 0.0,
                }

            errors = [r["absolute_error_h"] for r in rows]
            t_first = datetime.fromisoformat(rows[0]["arrival_time_utc"].replace("Z", "+00:00"))
            t_last = datetime.fromisoformat(rows[-1]["arrival_time_utc"].replace("Z", "+00:00"))
            hours_cov = max(0.1, (t_last - t_first).total_seconds() / 3600.0)
            date_range = f"{t_first.strftime('%Y-%m-%d')} to {t_last.strftime('%Y-%m-%d')}"

            mae = float(sum(errors) / n)
            sorted_e = sorted(errors)
            medae = float(sorted_e[n // 2]) if n % 2 != 0 else float((sorted_e[n // 2 - 1] + sorted_e[n // 2]) / 2.0)

            status = f"collecting (n = {n})" if n < 30 else f"verified (n = {n})"

            ci = None
            if n >= 30:
                # 95% bootstrap CI
                import numpy as np
                boot_means = []
                arr = np.array(errors)
                rng = np.random.default_rng(42)
                for _ in range(1000):
                    boot_means.append(np.mean(rng.choice(arr, size=n, replace=True)))
                ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

            return {
                "status": status,
                "n": n,
                "mae_h": round(mae, 3),
                "medae_h": round(medae, 3),
                "ci_95": ci,
                "dates_covered": date_range,
                "utc_hours_covered": round(hours_cov, 1),
            }
