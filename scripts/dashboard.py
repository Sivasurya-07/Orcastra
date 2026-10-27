"""
ORCASTRA — Live ETA & Operations Dashboard.

Built with Streamlit and Plotly.
Supports dual operation modes:
  1. LIVE MODE: Monitors latest_state.json from the collector subprocess.
     Controls collector start/stop subprocess. Never auto-starts. Never displays API key.
  2. OFFLINE REPLAY MODE: Replays recorded AIS sessions from Parquet files on disk.
     Runs with zero network and no API key. Graceful offline map rendering.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add services/collector/src to path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
COLLECTOR_SRC = REPO_ROOT / "services" / "collector" / "src"
if str(COLLECTOR_SRC) not in sys.path:
    sys.path.insert(0, str(COLLECTOR_SRC))

from orcastra_collector.config import CollectorConfig, DEFAULT_CONFIG
from orcastra_collector.lock import get_running_collector_info
from orcastra_collector.state import read_latest_state
from orcastra_collector.live_eta import (
    compute_live_eta,
    PredictionLogger,
    haversine_km,
    channel_distance_to_berth,
    DISTANCE_BUCKETS,
    METHOD_NAME,
    P95_INTERVAL_NOTE,
    CENTERLINE_WP,
    CORRIDOR,
    ENTRANCE_ZONE,
    DESTINATION_ZONE,
)

# ── Paths & Constants ──
DATA_ROOT = Path(r"D:\ORCASTRA_DATA")
FIXTURE_SESSION = REPO_ROOT / "tests" / "fixtures" / "sample_session"
DEFAULT_STATE_PATH = DATA_ROOT / "latest_state.json"
DEFAULT_LOCK_PATH = DATA_ROOT / "collector.lock"
DEFAULT_PRED_LOG_PATH = DATA_ROOT / "prediction_log.sqlite"
BENCHMARK_RESULTS_V2_1 = DATA_ROOT / "historical" / "results" / "tier1_v2_1_short_horizon_results.json"
BENCHMARK_RESULTS_V2 = DATA_ROOT / "historical" / "results" / "tier1_v2_short_horizon_results.json"

st.set_page_config(
    page_title="ORCASTRA — Live Maritime ETA & AIS Corridor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for high-contrast, clean typography and status badges
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border: 1px solid #2d3748;
    }
    .badge-live {
        background-color: #166534;
        color: #4ade80;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-offline {
        background-color: #991b1b;
        color: #fca5a5;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-stopped {
        background-color: #374151;
        color: #9ca3af;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Collector Subprocess Management ──

def get_collector_subprocess_status() -> dict[str, Any]:
    """Check running collector process using the lock file."""
    lock_info = get_running_collector_info(DEFAULT_LOCK_PATH)
    if lock_info:
        return {
            "running": True,
            "pid": lock_info.get("pid"),
            "session_id": lock_info.get("session_id"),
            "start_time": lock_info.get("start_time_utc"),
            "mode": lock_info.get("mode"),
        }
    return {"running": False, "pid": None, "session_id": None, "start_time": None}


def start_collector_subprocess(duration_seconds: Optional[int] = None) -> tuple[bool, str]:
    """Launch the collector as an independent subprocess."""
    status = get_collector_subprocess_status()
    if status["running"]:
        return False, f"Collector is already running (PID: {status['pid']})"

    collector_py = REPO_ROOT / "services" / "collector" / "src" / "orcastra_collector" / "collector.py"
    cmd = [sys.executable, str(collector_py)]
    if duration_seconds:
        cmd.extend(["--duration", str(duration_seconds)])

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        # Give process a moment to write lock
        time.sleep(1.0)
        return True, f"Started collector process (PID: {proc.pid})"
    except Exception as e:
        return False, f"Failed to start collector: {e}"


def stop_collector_subprocess() -> tuple[bool, str]:
    """Stop the collector process gracefully."""
    status = get_collector_subprocess_status()
    if not status["running"]:
        return False, "Collector is not currently running"

    pid = status["pid"]
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except psutil.TimeoutExpired:
            proc.kill()
        return True, f"Stopped collector process (PID: {pid})"
    except Exception as e:
        return False, f"Error terminating process {pid}: {e}"


# ── Map Rendering with Graceful Offline Degradation ──

def render_corridor_map(vessels_df: pd.DataFrame, is_offline_mode: bool = False) -> go.Figure:
    """
    Renders map of corridor, entrance, berth boxes, and vessel positions.
    Degrades gracefully if tile servers/internet are unavailable by falling back
    to a 2D coordinate plot.
    """
    fig = go.Figure()

    # Centerline waypoints
    cw_lats = [wp[0] for wp in CENTERLINE_WP]
    cw_lons = [wp[1] for wp in CENTERLINE_WP]
    cw_texts = [f"Waypoint: {wp[2]} km to berth" for wp in CENTERLINE_WP]

    fig.add_trace(go.Scatter(
        x=cw_lons, y=cw_lats,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3, dash="dot"),
        marker=dict(size=8, color="#38bdf8", symbol="diamond"),
        name="Channel Centerline",
        hovertext=cw_texts,
    ))

    # Corridor outline polygon
    c_lons = [CORRIDOR["lon_min"], CORRIDOR["lon_max"], CORRIDOR["lon_max"], CORRIDOR["lon_min"], CORRIDOR["lon_min"]]
    c_lats = [CORRIDOR["lat_min"], CORRIDOR["lat_min"], CORRIDOR["lat_max"], CORRIDOR["lat_max"], CORRIDOR["lat_min"]]
    fig.add_trace(go.Scatter(
        x=c_lons, y=c_lats,
        mode="lines",
        line=dict(color="#64748b", width=1.5, dash="dash"),
        name="Corridor Zone",
        hoverinfo="name",
    ))

    # Entrance Zone polygon
    e_lons = [ENTRANCE_ZONE["lon_min"], ENTRANCE_ZONE["lon_max"], ENTRANCE_ZONE["lon_max"], ENTRANCE_ZONE["lon_min"], ENTRANCE_ZONE["lon_min"]]
    e_lats = [ENTRANCE_ZONE["lat_min"], ENTRANCE_ZONE["lat_min"], ENTRANCE_ZONE["lat_max"], ENTRANCE_ZONE["lat_max"], ENTRANCE_ZONE["lat_min"]]
    fig.add_trace(go.Scatter(
        x=e_lons, y=e_lats,
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(color="#3b82f6", width=2),
        name="Entrance Zone",
        hoverinfo="name",
    ))

    # Destination / Berth Zone polygon
    d_lons = [DESTINATION_ZONE["lon_min"], DESTINATION_ZONE["lon_max"], DESTINATION_ZONE["lon_max"], DESTINATION_ZONE["lon_min"], DESTINATION_ZONE["lon_min"]]
    d_lats = [DESTINATION_ZONE["lat_min"], DESTINATION_ZONE["lat_min"], DESTINATION_ZONE["lat_max"], DESTINATION_ZONE["lat_max"], DESTINATION_ZONE["lat_min"]]
    fig.add_trace(go.Scatter(
        x=d_lons, y=d_lats,
        fill="toself",
        fillcolor="rgba(34, 197, 94, 0.2)",
        line=dict(color="#22c55e", width=2),
        name="Destination Berth Box",
        hoverinfo="name",
    ))

    # Vessel markers
    if not vessels_df.empty:
        # Separate by data age category
        # <60s: Green, 1-3m: Yellow, 3-5m: Orange, >5m (Stale): Red
        color_map = {
            "Fresh (<60s)": "#22c55e",
            "Active (1-3m)": "#eab308",
            "Aging (3-5m)": "#f97316",
            "Stale (>5m)": "#ef4444",
        }

        for cat, color in color_map.items():
            sub = vessels_df[vessels_df["age_category"] == cat]
            if not sub.empty:
                hover_texts = [
                    f"<b>{row['name']}</b> (MMSI: {row['mmsi']})<br>"
                    f"SOG: {row['sog']} kn | COG: {row['cog']}°<br>"
                    f"Dist to Berth: {row['dist_to_berth']}<br>"
                    f"ETA: {row['eta']}<br>"
                    f"Status: {row['status']}<br>"
                    f"Data Age: {row['age_seconds']:.0f}s"
                    for _, row in sub.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=sub["longitude"], y=sub["latitude"],
                    mode="markers+text",
                    marker=dict(
                        size=12 if cat != "Stale (>5m)" else 9,
                        color=color,
                        symbol="circle" if cat != "Stale (>5m)" else "x",
                        line=dict(color="#000000", width=1),
                    ),
                    text=sub["label"],
                    textposition="top right",
                    textfont=dict(size=10, color="#ffffff"),
                    name=cat,
                    hovertext=hover_texts,
                    hoverinfo="text",
                ))

    # Clean styling with nautical bounds
    fig.update_layout(
        title="Houston Ship Channel AIS Vessel Radar",
        xaxis=dict(title="Longitude", range=[-95.25, -94.60], gridcolor="#1e293b", zeroline=False),
        yaxis=dict(title="Latitude", range=[29.25, 29.85], gridcolor="#1e293b", zeroline=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="#090d16",
        paper_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=620,
    )

    return fig


# ── Benchmark & Verification Comparison Charts ──

def render_benchmark_chart() -> go.Figure:
    """
    Renders bar chart comparing Protocol v2.1 verified MAEs across methods.
    Source: docs/EVAL_PROTOCOL_V2_1.md and tier1_v2_1_short_horizon_results.json.
    """
    methods = [
        "Kinematic Baseline (d / SOG)",
        "Segment Median",
        "Distance-Bucket Median Baseline",
        "Calibrated QR P50 (HistGradientBoosting)",
    ]
    maes = [3.113, 2.681, 2.708, 2.725]
    colors = ["#94a3b8", "#38bdf8", "#22c55e", "#a855f7"]

    fig = go.Figure(go.Bar(
        x=methods,
        y=maes,
        text=[f"{v:.3f} h ({v*60:.1f} m)" for v in maes],
        textposition="auto",
        marker=dict(color=colors),
    ))

    fig.update_layout(
        title="Protocol v2.1 Verified Overall MAE by Method (D: NOAA 2024 Test Set, N=5,067)",
        yaxis=dict(title="MAE (Hours)", range=[0, 4.0], gridcolor="#334155"),
        xaxis=dict(tickangle=-15),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#f8fafc"),
        margin=dict(l=40, r=20, t=50, b=60),
        height=360,
    )
    return fig


def render_selection_bias_chart() -> go.Figure:
    """
    Renders chart contrasting Protocol v2 (selection-biased) vs Protocol v2.1 (unconditioned).
    Source: docs/EVAL_PROTOCOL_V2.md vs docs/EVAL_PROTOCOL_V2_1.md.
    """
    labels = [
        "Protocol v2 (Selection-Biased)<br>Filtered: 0 < y_rem ≤ 3.5h",
        "Protocol v2.1 (Unconditioned)<br>Honest Distance Buckets",
    ]
    maes = [0.239, 2.708]
    colors = ["#ef4444", "#22c55e"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=maes,
        text=["0.239 h (~14.4 min)<br><b>ARTIFICIAL SELECTION BIAS</b>", "2.708 h (~162.5 min)<br><b>REALISTIC OPERATIONAL ERROR</b>"],
        textposition="auto",
        marker=dict(color=colors),
    ))

    fig.update_layout(
        title="Impact of Evaluation Bias: Protocol v2 vs Protocol v2.1",
        yaxis=dict(title="MAE (Hours)", range=[0, 3.5], gridcolor="#334155"),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#f8fafc"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=360,
    )
    return fig


# ── Offline Replay Data Loader ──

def load_offline_replay_events(session_path: Path) -> pd.DataFrame:
    """Load and format events from Parquet recording session."""
    events_dir = session_path / "events"
    if not events_dir.exists():
        return pd.DataFrame()

    parts = sorted(events_dir.glob("part_*.parquet"))
    if not parts:
        return pd.DataFrame()

    import polars as pl
    dfs = [pl.read_parquet(p) for p in parts]
    df = pl.concat(dfs, how="diagonal")
    if "is_duplicate" in df.columns:
        df = df.filter(pl.col("is_duplicate") == False)

    time_col = "aisstream_received_time" if "aisstream_received_time" in df.columns else "observation_time"
    df = df.sort(time_col)
    return df.to_pandas()


# ── Main Dashboard Application ──

def main():
    st.title("ORCASTRA — Live Maritime Corridor & Short-Horizon ETA")
    st.caption("Port of Houston Ship Channel | Baseline: Distance-Bucket Median | Real Data Only")

    # Mode Selection
    mode = st.sidebar.radio("Operating Mode", ["LIVE", "OFFLINE REPLAY"], index=0)

    # ─────────────────────────────────────────────────────────────
    # SIDEBAR: Mode Controls & Collector Subprocess Manager
    # ─────────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    collector_status = get_collector_subprocess_status()

    if mode == "OFFLINE REPLAY":
        st.sidebar.markdown("### 🔴 Offline Replay Settings")
        st.sidebar.info("Replaying recorded AIS data from disk. No network access. No API key used.")

        # Session selector
        recordings_dir = DATA_ROOT / "recordings"
        available_sessions = []
        if recordings_dir.exists():
            available_sessions = [d for d in recordings_dir.iterdir() if d.is_dir() and (d / "events").exists()]

        session_options = {s.name: s for s in available_sessions}
        if FIXTURE_SESSION.exists():
            session_options["Bundled Sample Fixture (tests/fixtures)"] = FIXTURE_SESSION

        selected_name = st.sidebar.selectbox("Choose Recording Session", list(session_options.keys()))
        selected_session = session_options[selected_name]

    else:
        st.sidebar.markdown("### 🟢 Live Collector Control")
        if collector_status["running"]:
            st.sidebar.markdown(f'<span class="badge-live">COLLECTOR RUNNING (PID {collector_status["pid"]})</span>', unsafe_allow_html=True)
            st.sidebar.caption(f"Started: {collector_status['start_time']}")
            if st.sidebar.button("🛑 Stop Live Collection"):
                ok, msg = stop_collector_subprocess()
                if ok:
                    st.sidebar.success(msg)
                    st.rerun()
                else:
                    st.sidebar.error(msg)
        else:
            st.sidebar.markdown('<span class="badge-stopped">COLLECTOR STOPPED</span>', unsafe_allow_html=True)
            st.sidebar.caption("Collector must be started explicitly. Does not auto-start.")

            duration = st.sidebar.number_input("Session Duration (seconds, 0=indefinite)", min_value=0, max_value=86400, value=300, step=60)
            dur_arg = duration if duration > 0 else None

            if st.sidebar.button("🚀 Start Live Collection"):
                ok, msg = start_collector_subprocess(duration_seconds=dur_arg)
                if ok:
                    st.sidebar.success(msg)
                    st.rerun()
                else:
                    st.sidebar.error(msg)

        auto_refresh = st.sidebar.checkbox("Auto-refresh (every 5s)", value=True)

    # ─────────────────────────────────────────────────────────────
    # MODE 1: OFFLINE REPLAY
    # ─────────────────────────────────────────────────────────────
    if mode == "OFFLINE REPLAY":
        st.error("### 🔴 OFFLINE REPLAY — RECORDED DATA, NOT LIVE")
        st.markdown(
            "This mode replays previously captured AIS position reports strictly from disk files. "
            "**No live WebSocket connection is open.** The map operates fully offline."
        )

        df_replay = load_offline_replay_events(selected_session)
        if df_replay.empty:
            st.warning("No Parquet event data found in selected session directory.")
            return

        time_col = "aisstream_received_time" if "aisstream_received_time" in df_replay.columns else "observation_time"
        st.caption(f"Loaded **{len(df_replay)}** events from `{selected_session.name}` ({df_replay[time_col].min()} to {df_replay[time_col].max()})")

        # Time slice slider
        unique_mmsis = df_replay["mmsi"].unique()
        vessels_state = {}
        for mmsi in unique_mmsis:
            v_recs = df_replay[df_replay["mmsi"] == mmsi].sort_values(time_col)
            last_rec = v_recs.iloc[-1]
            vessels_state[int(mmsi)] = {
                "mmsi": int(mmsi),
                "name": last_rec.get("vessel_name") or f"MMSI_{mmsi}",
                "latitude": last_rec.get("latitude"),
                "longitude": last_rec.get("longitude"),
                "sog": last_rec.get("sog"),
                "cog": last_rec.get("cog"),
                "vessel_type": last_rec.get("vessel_type"),
                "last_seen_utc": str(last_rec.get(time_col)),
            }

        _display_vessel_operations(vessels_state, is_live=False)

    # ─────────────────────────────────────────────────────────────
    # MODE 2: LIVE MODE
    # ─────────────────────────────────────────────────────────────
    else:
        state_data = read_latest_state(DEFAULT_STATE_PATH)

        # Header status row
        col_status, col_meta, col_conn = st.columns([1.5, 2, 1.5])
        with col_status:
            if collector_status["running"]:
                st.markdown(f"**Collector:** :green[● RUNNING] (PID: `{collector_status['pid']}`)")
            else:
                st.markdown("**Collector:** :red[● STOPPED]")

        with col_meta:
            if state_data:
                up_utc = state_data.get("updated_at_utc", "unknown")
                st.markdown(f"**Latest State Updated:** `{up_utc}`")
            else:
                st.markdown("**Latest State:** :orange[No state file found yet]")

        with col_conn:
            if state_data:
                conn_st = state_data.get("connection", {}).get("status", "disconnected")
                st.markdown(f"**Connection:** `{conn_st}`")

        # Summary Metrics
        if state_data:
            stats = state_data.get("stats", {})
            conn_info = state_data.get("connection", {})

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("WebSocket Frames", stats.get("socket_frames", 0))
            m2.metric("Total Events", stats.get("total_events", 0))
            m3.metric("Duplicates Flagged", stats.get("total_duplicates", 0))
            m4.metric("Active Vessels", stats.get("unique_vessels", 0))
            m5.metric("Reconnects / Drops", f"{conn_info.get('reconnects', 0)} / {conn_info.get('drops', 0)}")

            vessels_state = state_data.get("vessels", {})
            _display_vessel_operations(vessels_state, is_live=True)
        else:
            st.info("Start the live collector using the sidebar button to begin receiving AIS messages.")
            # Render empty map with corridor boundaries
            st.plotly_chart(render_corridor_map(pd.DataFrame()), use_container_width=True)

        if auto_refresh and collector_status["running"]:
            time.sleep(5)
            st.rerun()

    # ─────────────────────────────────────────────────────────────
    # BENCHMARKS & VERIFIED BASELINE COMPARISONS
    # ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Benchmark Comparisons & Protocol Evidence")

    tab1, tab2, tab3 = st.tabs(["Protocol v2.1 Verified MAEs", "Selection Bias Contrast", "Live Scoreboard"])

    with tab1:
        st.markdown(
            "**Verified Test MAEs on Unconditioned NOAA 2024 Test Set ($N = 5,067$ snapshots, 264 commercial vessels):**<br>"
            "Source: `docs/EVAL_PROTOCOL_V2_1.md` and `tier1_v2_1_short_horizon_results.json`",
            unsafe_allow_html=True,
        )
        st.plotly_chart(render_benchmark_chart(), use_container_width=True)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Kinematic d/SOG", "3.113 h", help="Simple distance / speed")
        b2.metric("Hist Segment Median", "2.681 h", help="Historical segment median")
        b3.metric("Bucket Median Baseline", "2.708 h", help="Our honest live baseline method")
        b4.metric("Calibrated QR P50", "2.725 h", help="Gradient boosting quantile regressor")

        st.caption(f"**Uncertainty Note:** {P95_INTERVAL_NOTE}. Live predictions display P50 point estimates.")

    with tab2:
        st.markdown(
            r"**Selection-Bias Audit (Protocol v2 vs Protocol v2.1):**<br>"
            r"Protocol v2 reported artificially low MAE (~14 min) by conditioning on true remaining duration $\le 3.5$h. "
            r"Protocol v2.1 removes all outcome conditioning and groups strictly by observable distance-to-berth buckets.",
            unsafe_allow_html=True,
        )
        st.plotly_chart(render_selection_bias_chart(), use_container_width=True)
        st.warning(
            "⚠️ **Evaluation Warning:** Benchmark results established on historical NOAA data do **not** establish "
            "live accuracy. Real-time stream density, loitering, and pilot delays require continuous live scoring."
        )

    with tab3:
        st.markdown("### 🏆 Live Arrival Scoreboard")
        st.caption("Scores only vessels that were fully observed from initial corridor prediction to berth arrival.")

        pred_logger = PredictionLogger(DEFAULT_PRED_LOG_PATH)
        sb_stats = pred_logger.get_scoreboard_stats()

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Scoreboard Status", sb_stats["status"])
        s2.metric("Observed Arrivals", sb_stats["n"])
        s3.metric("Live MAE", f"{sb_stats['mae_h']:.3f} h" if sb_stats["mae_h"] else "Collecting...")
        s4.metric("UTC Hours Covered", f"{sb_stats['utc_hours_covered']} h")

        if sb_stats["n"] < 30:
            st.info(f"Currently in collection phase: **{sb_stats['status']}**. A minimum of $n = 30$ fully observed arrivals is required before computing formal confidence intervals.")
        else:
            st.success(f"Verified live cohort size: $n = {sb_stats['n']}$. Bootstrap 95% CI: `{sb_stats['ci_95']}`")


def _display_vessel_operations(vessels_state: dict[str, Any], is_live: bool):
    """Processes vessel states, computes live ETAs, and renders map and table."""
    now_utc = datetime.now(timezone.utc)
    table_rows = []
    map_rows = []

    for mmsi_raw, v in vessels_state.items():
        eta_obj = compute_live_eta(v, now_utc=now_utc)
        mmsi = int(mmsi_raw)

        # Categorize data age
        age = eta_obj.data_age_seconds
        if age < 60:
            age_cat = "Fresh (<60s)"
        elif age < 180:
            age_cat = "Active (1-3m)"
        elif age <= 300:
            age_cat = "Aging (3-5m)"
        else:
            age_cat = "Stale (>5m)"

        # Display values
        eta_display = eta_obj.predicted_eta_utc if (eta_obj.is_eligible and not eta_obj.is_stale) else f"Unavailable ({eta_obj.status})"
        dist_display = f"{eta_obj.distance_to_berth_km:.1f} km" if eta_obj.distance_to_berth_km else "—"
        sog_display = f"{eta_obj.sog:.1f}" if eta_obj.sog is not None else "—"

        table_rows.append({
            "MMSI": mmsi,
            "Vessel Name": eta_obj.name,
            "Status": eta_obj.status.upper(),
            "Last Seen Age": f"{age:.0f}s" if age < 3600 else f"{age/60:.0f}m",
            "Speed (kn)": sog_display,
            "Dist to Berth": dist_display,
            "Predicted ETA": eta_display,
            "Bucket": eta_obj.bucket_name or "—",
            "Method": eta_obj.method,
            "Eligible Cohort A": "✓ Yes" if eta_obj.is_eligible else "✗ No",
            "Reason": eta_obj.reason,
        })

        if eta_obj.latitude is not None and eta_obj.longitude is not None:
            map_rows.append({
                "mmsi": mmsi,
                "name": eta_obj.name,
                "latitude": eta_obj.latitude,
                "longitude": eta_obj.longitude,
                "sog": eta_obj.sog,
                "cog": eta_obj.cog,
                "dist_to_berth": dist_display,
                "eta": eta_display,
                "status": eta_obj.status,
                "age_seconds": age,
                "age_category": age_cat,
                "label": f"{eta_obj.name} ({eta_display})",
            })

    # Render Corridor Map
    df_map = pd.DataFrame(map_rows)
    st.plotly_chart(render_corridor_map(df_map, is_offline_mode=not is_live), use_container_width=True)

    # Render Vessel Table
    st.subheader(f"Tracked Vessels ({len(table_rows)} total)")
    if table_rows:
        df_table = pd.DataFrame(table_rows)
        # Sort by distance to berth if available, else MMSI
        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No vessel position reports currently tracked.")


if __name__ == "__main__":
    main()
