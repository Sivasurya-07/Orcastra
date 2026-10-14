"""
ORCASTRA M0 — Live AISStream Feed Validation (15 minutes)
Houston Ship Channel bounding box.

Captures:
1. Start/end timestamps, hour of day
2. Total messages, msgs/sec, position-message count
3. Unique vessels, ship-type distribution, commercial count
4. Missing/invalid fields, duplicate rate
5. Reconnects, drops, errors, limits
6. Out-of-bbox messages
7. Disk/CPU/RAM (sampled)
8. Live-vs-archive comparison by ship type

SECURITY: API key is loaded from .env, never printed/logged/exposed.
"""

import asyncio
import json
import sys
import os
import time
import logging
import psutil
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

# ── Config ──
HOUSTON_BBOX = [[29.0, -95.3], [29.9, -94.4]]
TEST_DURATION = 900  # 15 minutes
RESULTS_DIR = Path(r"D:\ORCASTRA_DATA\historical\results")
RAW_DIR = Path(r"D:\ORCASTRA_DATA\live_test")

# AIS vessel type categories (same as validation script)
def categorize_vtype(vt):
    if vt is None: return "unknown"
    if 70 <= vt <= 79: return "cargo"
    if 80 <= vt <= 89: return "tanker"
    if 60 <= vt <= 69: return "passenger"
    if vt in (31, 32, 52): return "tug_towing"
    if vt == 30: return "fishing"
    if vt == 50: return "pilot"
    if vt in (36, 37): return "pleasure_sailing"
    if vt == 51: return "sar"
    return "other"

# Load API key from .env — never print it
def load_api_key():
    env_path = Path(r"D:\ORCASTRA\repo\.env")
    for line in env_path.read_text().splitlines():
        if line.startswith("AISSTREAM_API_KEY="):
            k = line.split("=", 1)[1].strip().strip('"').strip("'")
            if k and k != "your_api_key_here":
                return k
    return None


async def run_live_test():
    import websockets

    api_key = load_api_key()
    if not api_key:
        log.error("No valid API key in .env")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    subscription = {
        "APIKey": api_key,
        "BoundingBoxes": [HOUSTON_BBOX],
        "FilterMessageTypes": [
            "PositionReport",
            "StandardClassBPositionReport",
            "ExtendedClassBPositionReport",
            "ShipStaticData",
        ]
    }

    # ── Counters ──
    total_msgs = 0
    position_msgs = 0
    static_msgs = 0
    other_msgs = 0
    unique_mmsis = set()
    mmsi_vtype = {}          # mmsi -> vessel_type from static data
    mmsi_msg_count = Counter()
    msg_type_counts = Counter()
    minute_counts = defaultdict(int)
    minute_mmsis = defaultdict(set)

    # Field quality
    missing_fields = Counter()   # field_name -> count of nulls
    required_fields = ['MMSI', 'Latitude', 'Longitude']
    out_of_bbox = 0
    duplicate_check = set()     # (mmsi, timestamp) tuples
    duplicates = 0

    # Errors
    reconnects = 0
    errors = []
    drops = 0

    # Resource tracking
    process = psutil.Process()
    resource_samples = []
    disk_start = 0

    # Raw message samples (first 5 for schema inspection)
    raw_samples = []

    start_time = time.time()
    start_utc = datetime.now(timezone.utc)

    log.info(f"Live AISStream Feed Validation")
    log.info(f"BBox: {HOUSTON_BBOX}")
    log.info(f"Duration: {TEST_DURATION}s ({TEST_DURATION/60:.0f} min)")
    log.info(f"Start UTC: {start_utc.isoformat()}")
    log.info(f"Hour of day (UTC): {start_utc.hour}:{start_utc.minute:02d}")
    log.info(f"Houston local (CST=UTC-6): {start_utc.hour - 6 if start_utc.hour >= 6 else start_utc.hour + 18}:xx")

    url = "wss://stream.aisstream.io/v0/stream"
    first_msg_time = None

    while time.time() - start_time < TEST_DURATION:
        try:
            async with websockets.connect(url, compression="deflate",
                                          close_timeout=10, open_timeout=15) as ws:
                await ws.send(json.dumps(subscription))
                log.info(f"{'Re-' if reconnects > 0 else ''}Connected. Subscription sent.")
                if reconnects > 0:
                    reconnects += 1

                while time.time() - start_time < TEST_DURATION:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                        if isinstance(raw, bytes):
                            raw = raw.decode('utf-8')

                        msg = json.loads(raw)
                        total_msgs += 1

                        if first_msg_time is None:
                            first_msg_time = time.time()
                            latency = first_msg_time - start_time
                            log.info(f"First message after {latency:.1f}s")

                        # Save raw samples
                        if len(raw_samples) < 5:
                            raw_samples.append(msg)

                        # Track message type
                        msg_type = msg.get("MessageType", "Unknown")
                        msg_type_counts[msg_type] += 1

                        # Extract metadata
                        meta = msg.get("MetaData", {})
                        mmsi = meta.get("MMSI")
                        lat = meta.get("Latitude")
                        lon = meta.get("Longitude")
                        ship_name = meta.get("ShipName")
                        ts_str = meta.get("time_utc", "")

                        # Track MMSI
                        if mmsi:
                            unique_mmsis.add(mmsi)
                            mmsi_msg_count[mmsi] += 1

                        # Check required fields
                        if mmsi is None: missing_fields['MMSI'] += 1
                        if lat is None: missing_fields['Latitude'] += 1
                        if lon is None: missing_fields['Longitude'] += 1

                        # Bbox check
                        if lat is not None and lon is not None:
                            if not (HOUSTON_BBOX[0][0] <= lat <= HOUSTON_BBOX[1][0] and
                                    HOUSTON_BBOX[0][1] <= lon <= HOUSTON_BBOX[1][1]):
                                out_of_bbox += 1

                        # Duplicate check
                        if mmsi and ts_str:
                            key = (mmsi, ts_str)
                            if key in duplicate_check:
                                duplicates += 1
                            else:
                                duplicate_check.add(key)

                        # Categorize message
                        if msg_type in ("PositionReport", "StandardClassBPositionReport",
                                        "ExtendedClassBPositionReport"):
                            position_msgs += 1

                            # Extract SOG from position reports
                            msg_body = msg.get("Message", {})
                            pos = (msg_body.get("PositionReport") or
                                   msg_body.get("StandardClassBPositionReport") or
                                   msg_body.get("ExtendedClassBPositionReport") or {})
                            sog = pos.get("Sog")
                            cog = pos.get("Cog")
                            if sog is None: missing_fields['Sog'] += 1
                            if cog is None: missing_fields['Cog'] += 1

                        elif msg_type == "ShipStaticData":
                            static_msgs += 1
                            msg_body = msg.get("Message", {})
                            ssd = msg_body.get("ShipStaticData", {})
                            vtype = ssd.get("Type")
                            if mmsi and vtype is not None:
                                mmsi_vtype[mmsi] = vtype
                        else:
                            other_msgs += 1

                        # Per-minute tracking
                        elapsed_min = int((time.time() - start_time) / 60)
                        minute_counts[elapsed_min] += 1
                        if mmsi:
                            minute_mmsis[elapsed_min].add(mmsi)

                        # Progress logging
                        if total_msgs % 1000 == 0:
                            elapsed = time.time() - start_time
                            rate = total_msgs / elapsed
                            remaining = TEST_DURATION - elapsed
                            log.info(f"  {total_msgs:,} msgs | {len(unique_mmsis)} vessels | "
                                     f"{rate:.1f}/s | {remaining:.0f}s left")

                        # Resource sampling every 60s
                        if total_msgs % 2000 == 0:
                            mem = process.memory_info().rss / 1024 / 1024
                            cpu = process.cpu_percent(interval=0.1)
                            resource_samples.append({
                                'elapsed_s': round(time.time() - start_time, 0),
                                'rss_mb': round(mem, 1),
                                'cpu_pct': round(cpu, 1),
                                'msgs': total_msgs,
                            })

                    except asyncio.TimeoutError:
                        drops += 1
                        log.info(f"  No message for 15s (drop #{drops})")

                    except websockets.exceptions.ConnectionClosed as e:
                        log.warning(f"  Connection closed: {e}")
                        errors.append(f"ConnectionClosed at {time.time()-start_time:.0f}s: {e}")
                        break

        except Exception as e:
            reconnects += 1
            err_msg = f"Connection error #{reconnects} at {time.time()-start_time:.0f}s: {str(e)[:100]}"
            log.warning(err_msg)
            errors.append(err_msg)
            await asyncio.sleep(min(2 ** reconnects, 30))  # Exponential backoff

    end_time = time.time()
    end_utc = datetime.now(timezone.utc)
    elapsed = end_time - start_time

    # ── Results ──
    log.info(f"\n{'='*70}")
    log.info(f"LIVE TEST RESULTS")
    log.info(f"{'='*70}")

    # 1. Timestamps
    log.info(f"\n--- 1. Timestamps ---")
    log.info(f"Start: {start_utc.isoformat()} (UTC hour {start_utc.hour})")
    log.info(f"End:   {end_utc.isoformat()} (UTC hour {end_utc.hour})")
    log.info(f"Duration: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"Houston local: ~{start_utc.hour-6 if start_utc.hour>=6 else start_utc.hour+18}:xx CST")

    # 2. Message counts
    log.info(f"\n--- 2. Messages ---")
    log.info(f"Total messages: {total_msgs:,}")
    log.info(f"Messages/sec: {total_msgs/elapsed:.2f}" if elapsed > 0 else "N/A")
    log.info(f"Messages/min: {total_msgs/elapsed*60:.1f}" if elapsed > 0 else "N/A")
    log.info(f"Position messages: {position_msgs:,}")
    log.info(f"Static data messages: {static_msgs:,}")
    log.info(f"Other messages: {other_msgs:,}")
    log.info(f"Message type breakdown:")
    for mt, c in msg_type_counts.most_common():
        log.info(f"  {mt}: {c:,} ({c/total_msgs*100:.1f}%)" if total_msgs > 0 else f"  {mt}: {c}")

    # 3. Vessels
    log.info(f"\n--- 3. Unique vessels ---")
    log.info(f"Total unique MMSI: {len(unique_mmsis)}")
    # Vessel type distribution from static data
    vtype_dist = Counter()
    commercial_mmsis = set()
    for mmsi in unique_mmsis:
        vt = mmsi_vtype.get(mmsi)
        cat = categorize_vtype(vt)
        vtype_dist[cat] += 1
        if cat in ('cargo', 'tanker', 'passenger'):
            commercial_mmsis.add(mmsi)

    log.info(f"Vessels with type info (from ShipStaticData): {sum(1 for m in unique_mmsis if m in mmsi_vtype)}")
    log.info(f"Vessel type distribution:")
    for cat, c in vtype_dist.most_common():
        log.info(f"  {cat:20s}: {c:4d} ({c/len(unique_mmsis)*100:.1f}%)" if unique_mmsis else "")
    log.info(f"Commercial (cargo+tanker+passenger): {len(commercial_mmsis)}")

    # 4. Field quality
    log.info(f"\n--- 4. Field quality ---")
    for field, c in missing_fields.most_common():
        pct = c / total_msgs * 100 if total_msgs > 0 else 0
        log.info(f"  Missing {field}: {c:,} ({pct:.2f}%)")
    log.info(f"Duplicates (same MMSI+timestamp): {duplicates:,} ({duplicates/total_msgs*100:.2f}%)" if total_msgs > 0 else "N/A")

    # 5. Errors
    log.info(f"\n--- 5. Connectivity ---")
    log.info(f"Reconnects: {reconnects}")
    log.info(f"Drops (15s timeout): {drops}")
    log.info(f"Errors: {len(errors)}")
    for e in errors[:5]:
        log.info(f"  {e}")

    # 6. Bbox
    log.info(f"\n--- 6. Bbox compliance ---")
    log.info(f"Out-of-bbox messages: {out_of_bbox} ({out_of_bbox/total_msgs*100:.2f}%)" if total_msgs > 0 else "N/A")

    # 7. Resources
    log.info(f"\n--- 7. Resources ---")
    if resource_samples:
        last = resource_samples[-1]
        log.info(f"Final RSS: {last['rss_mb']:.1f} MB")
        log.info(f"Peak RSS: {max(s['rss_mb'] for s in resource_samples):.1f} MB")

    # 8. Live-vs-archive comparison
    log.info(f"\n--- 8. Live vs Archive comparison ---")
    # Archive stats (from NOAA 3-day sample, 72h total)
    archive_msgs_per_min = 351.8
    archive_vessels_per_hr = 916 / 72  # 12.7
    archive_commercial_per_hr = 110 / 72  # unique commercial vessels involved in approaches, not vessels/hr
    # But more useful: vessels per hour from hourly stats
    # Archive hourly median msgs: 21,762 → per minute: 362.7
    # Let's use per-minute for direct comparison

    live_msgs_per_min = total_msgs / elapsed * 60 if elapsed > 0 else 0
    live_vessels_per_hr = len(unique_mmsis) / elapsed * 3600 if elapsed > 0 else 0

    log.info(f"  Metric               | Archive (NOAA 2024)  | Live (AISStream)")
    log.info(f"  ---------------------|---------------------|------------------")
    log.info(f"  msgs/min             | {archive_msgs_per_min:.1f}               | {live_msgs_per_min:.1f}")
    log.info(f"  vessels/hr (extrap)  | {archive_vessels_per_hr:.1f}                | {live_vessels_per_hr:.1f}")
    log.info(f"  commercial vessels   | (110 over 72h)       | {len(commercial_mmsis)} in {elapsed/60:.0f}min")
    log.info(f"")
    log.info(f"  Note: Archive is 1-min downsampled. Live is broadcast-level.")
    log.info(f"  Note: Archive is Jan 2024 Mon-Wed. Live is {end_utc.strftime('%b %Y %A')}.")

    if live_msgs_per_min < archive_msgs_per_min * 0.5:
        log.info(f"  ⚠️ LIVE COVERAGE IS THINNER (live/archive ratio: {live_msgs_per_min/archive_msgs_per_min:.2f})")
    elif live_msgs_per_min > archive_msgs_per_min * 1.5:
        log.info(f"  Live is DENSER than archive (expected: live has higher temporal resolution)")
    else:
        log.info(f"  Live coverage is comparable to archive (ratio: {live_msgs_per_min/archive_msgs_per_min:.2f})")

    # Per-minute breakdown
    log.info(f"\n  Per-minute breakdown:")
    for m in sorted(minute_counts.keys()):
        log.info(f"    Min {m:2d}: {minute_counts[m]:5d} msgs, {len(minute_mmsis[m]):4d} vessels")

    # ── Save all results ──
    results = {
        'timestamps': {
            'start_utc': start_utc.isoformat(),
            'end_utc': end_utc.isoformat(),
            'start_hour_utc': start_utc.hour,
            'duration_seconds': round(elapsed, 1),
            'houston_local_approx': f"{start_utc.hour-6 if start_utc.hour>=6 else start_utc.hour+18}:xx CST",
        },
        'messages': {
            'total': total_msgs,
            'per_second': round(total_msgs / elapsed, 2) if elapsed > 0 else 0,
            'per_minute': round(total_msgs / elapsed * 60, 1) if elapsed > 0 else 0,
            'position_msgs': position_msgs,
            'static_msgs': static_msgs,
            'other_msgs': other_msgs,
            'type_breakdown': dict(msg_type_counts.most_common()),
        },
        'vessels': {
            'unique_mmsis': len(unique_mmsis),
            'with_type_info': sum(1 for m in unique_mmsis if m in mmsi_vtype),
            'type_distribution': dict(vtype_dist.most_common()),
            'commercial_count': len(commercial_mmsis),
        },
        'field_quality': {
            'missing': dict(missing_fields.most_common()),
            'duplicates': duplicates,
            'duplicate_pct': round(duplicates / total_msgs * 100, 3) if total_msgs > 0 else 0,
        },
        'connectivity': {
            'reconnects': reconnects,
            'drops': drops,
            'errors': errors[:10],
        },
        'bbox_compliance': {
            'out_of_bbox': out_of_bbox,
            'out_of_bbox_pct': round(out_of_bbox / total_msgs * 100, 3) if total_msgs > 0 else 0,
        },
        'resources': resource_samples[-3:] if resource_samples else [],
        'archive_comparison': {
            'archive_msgs_per_min': archive_msgs_per_min,
            'live_msgs_per_min': round(live_msgs_per_min, 1),
            'ratio': round(live_msgs_per_min / archive_msgs_per_min, 3) if archive_msgs_per_min > 0 else 0,
            'archive_note': 'NOAA 2024-01-15 to 2024-01-17, Mon-Wed, 1-min downsampled',
            'live_note': f'{end_utc.strftime("%Y-%m-%d %A")}, broadcast-level',
        },
        'per_minute': {str(m): {'msgs': minute_counts[m], 'vessels': len(minute_mmsis[m])}
                       for m in sorted(minute_counts.keys())},
    }

    # Save a raw sample (first 3 messages, with API key removed)
    safe_samples = []
    for s in raw_samples[:3]:
        sc = json.loads(json.dumps(s, default=str))
        # Remove API key if somehow present
        if 'APIKey' in sc:
            sc['APIKey'] = '[REDACTED]'
        safe_samples.append(sc)
    results['raw_message_samples'] = safe_samples

    results_path = RESULTS_DIR / "live_test_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    asyncio.run(run_live_test())
