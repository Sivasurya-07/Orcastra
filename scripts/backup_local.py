"""
ORCASTRA — Local Backup & Restore Verification.

Backs up critical SQLite ledgers, vessel cache, and Parquet data
to a timestamped local archive on D:, with SHA-256 verification and test restore.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path(r"D:\ORCASTRA_DATA")
BACKUP_ROOT = DATA_ROOT / "backups"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_backup(backup_name: str = None) -> Path:
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    if backup_name is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{ts}.tar.gz"
    
    archive_path = BACKUP_ROOT / backup_name
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "files": {}}

    # Files to backup: ledgers, cache, session ledgers
    targets = []
    
    # 1. Vessel cache
    vcache = DATA_ROOT / "vessel_cache.sqlite"
    if vcache.exists():
        targets.append(vcache)

    # 2. Collector sessions (ledger + events + raw)
    recordings_dir = DATA_ROOT / "recordings"
    if recordings_dir.exists():
        for sess in recordings_dir.iterdir():
            if sess.is_dir():
                for p in sess.rglob("*"):
                    if p.is_file() and not p.name.endswith(".tmp"):
                        targets.append(p)

    with tarfile.open(archive_path, "w:gz") as tar:
        for t in targets:
            rel = t.relative_to(DATA_ROOT)
            tar.add(t, arcname=str(rel))
            manifest["files"][str(rel)] = {
                "size": t.stat().st_size,
                "sha256": sha256_file(t),
            }

        # Add manifest to tar
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        import io
        ti = tarfile.TarInfo(name="_MANIFEST.json")
        ti.size = len(manifest_bytes)
        ti.mtime = int(datetime.now(timezone.utc).timestamp())
        tar.addfile(ti, io.BytesIO(manifest_bytes))

    print(f"Backup created: {archive_path} ({archive_path.stat().st_size} bytes, {len(targets)} files)")
    return archive_path


def test_restore(archive_path: Path) -> bool:
    """Restore to a temporary location and verify checksums against manifest."""
    with tempfile.TemporaryDirectory(dir=str(DATA_ROOT)) as tmp_dir:
        tmp_path = Path(tmp_dir) / "restore_test"
        tmp_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=tmp_path)

        manifest_file = tmp_path / "_MANIFEST.json"
        if not manifest_file.exists():
            print("ERROR: Manifest missing in backup archive")
            return False

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        all_ok = True
        for rel_path, meta in manifest["files"].items():
            restored = tmp_path / rel_path
            if not restored.exists():
                print(f"ERROR: Restored file missing: {rel_path}")
                all_ok = False
                continue
            if restored.stat().st_size != meta["size"]:
                print(f"ERROR: Size mismatch for {rel_path}")
                all_ok = False
                continue
            actual_hash = sha256_file(restored)
            if actual_hash != meta["sha256"]:
                print(f"ERROR: SHA256 mismatch for {rel_path}")
                all_ok = False
                continue

        # Test SQLite read integrity on restored vessel cache
        restored_vcache = tmp_path / "vessel_cache.sqlite"
        if restored_vcache.exists():
            conn = sqlite3.connect(str(restored_vcache))
            count = conn.execute("SELECT count(*) FROM vessel_cache").fetchone()[0]
            conn.close()
            print(f"Restored vessel cache SQLite verified: {count} rows")

        if all_ok:
            print("Restore verification SUCCESS: all files, sizes, and SHA-256 matched.")
        return all_ok


if __name__ == "__main__":
    archive = create_backup("test_m0_backup.tar.gz")
    success = test_restore(archive)
    # cleanup test backup
    if archive.exists():
        archive.unlink()
    sys.exit(0 if success else 1)
