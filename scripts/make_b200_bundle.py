"""Build b200_bundle.zip for the B200 Jupyter server.

Contents: src/, config/, data/processed/*.csv, requirements-train.txt,
notebooks/02_model_training.ipynb and a MANIFEST.json (git commit, file hashes).
Upload the zip and the notebook next to each other; the notebook unpacks the zip
into ./b200_bundle and runs from there.

    python scripts/make_b200_bundle.py [--out b200_bundle.zip]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["src", "config", "requirements-train.txt", "notebooks/02_model_training.ipynb"]
DATA_GLOB = "data/processed/*.csv"
SKIP = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*")) if f.is_file()
                    and not any(part in SKIP for part in f.parts) and f.suffix != ".pyc"]
        elif p.exists():
            out.append(p)
        else:
            sys.exit(f"missing {item}")
    data = sorted(ROOT.glob(DATA_GLOB))
    if not data:
        sys.exit("no data/processed/*.csv found")
    return out + data


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "b200_bundle.zip"))
    args = ap.parse_args()
    fs = files()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": commit,
                "files": {str(f.relative_to(ROOT)): sha256(f) for f in fs}}
    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in fs:
            z.write(f, f.relative_to(ROOT).as_posix())
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(fs)} files, commit {commit[:10]})")


if __name__ == "__main__":
    main()
