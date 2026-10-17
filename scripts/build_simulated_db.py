"""Build the SIMULATED reference database data/simulated/insurance.db (docs/simulated_data_plan.md).

    python scripts/build_simulated_db.py [--seed 42] [--check]

Deterministic for a given seed. Reads only allow-listed claim columns (never the fraud label)
from real train / validation / test. Writes data/simulated/manifest.json (committed) so a
rebuild can be verified; --check fails if the rebuilt tables differ from the manifest.
Rebuilding replaces the file, so the run-time tables (verification_results,
investigation_audit) start empty again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, load_config, path_for  # noqa: E402
from src.simulated.generate import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", action="store_true", help="fail if the tables differ from manifest.json")
    args = ap.parse_args()
    cfg = load_config()["simulated_data"]
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    out = PROJECT_ROOT / cfg["path"]
    if out.exists():
        print(f"replacing {out.relative_to(ROOT)} (its run-time tables start empty again)")
    inputs = [path_for(k) for k in ("real_train", "real_validation", "real_test")]
    m = build(out, inputs, PROJECT_ROOT / cfg["clauses"], seed=seed)
    manifest_path = out.parent / "manifest.json"
    if args.check:
        old = json.loads(manifest_path.read_text())
        if old["sha256"] != m["sha256"] or old["seed"] != m["seed"]:
            sys.exit(f"Rebuilt tables differ from {manifest_path.relative_to(ROOT)}")
        print("manifest check passed")
    else:
        manifest_path.write_text(json.dumps(m, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} (SIMULATED reference data, seed {seed}): "
          + ", ".join(f"{t} {n}" for t, n in m["rows"].items()))


if __name__ == "__main__":
    main()
