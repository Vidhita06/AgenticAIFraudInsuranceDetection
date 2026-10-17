"""Per-check PASS / FAIL / UNKNOWN counts of policy verification with SIMULATED data on, over
real validation claims. Writes docs/report/tables/03_simulated_clause_results.csv.

    python scripts/simulated_clause_report.py [--split validation]

The fraud label is read here only to report the fraud rate among failing claims (an
evaluation output); it is never used to generate the simulated data.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import path_for  # noqa: E402
from src.tools import verify_policy  # noqa: E402


def clause_table(df: pd.DataFrame) -> pd.DataFrame:
    stats = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "UNKNOWN": 0, "fail_frauds": 0})
    meta = {}
    for row in df.to_dict("records"):
        label = row.pop("FraudFound_P", None)
        out = verify_policy(row, simulated=True)
        if not out.get("ok"):
            continue
        for c in out["checks"]:
            st = stats[c["check_id"]]
            st[c["result"]] += 1
            st["fail_frauds"] += int(c["result"] == "FAIL" and label == 1)
            meta[c["check_id"]] = (c["level"], c["category"], c["rule"])
    rows = []
    for cid, st in sorted(stats.items()):
        n = st["PASS"] + st["FAIL"] + st["UNKNOWN"]
        rows.append({"check_id": cid, "level": meta[cid][0], "category": meta[cid][1], "rule": meta[cid][2],
                     "claims": n, "PASS": st["PASS"], "FAIL": st["FAIL"], "UNKNOWN": st["UNKNOWN"],
                     "FAIL %": round(100 * st["FAIL"] / n, 2) if n else None,
                     "fraud % among FAIL": round(100 * st["fail_frauds"] / st["FAIL"], 1) if st["FAIL"] else None})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation", "train"])
    args = ap.parse_args()
    df = pd.read_csv(path_for(f"real_{args.split}"))
    t = clause_table(df)
    out = ROOT / "docs" / "report" / "tables" / "03_simulated_clause_results.csv"
    t.to_csv(out, index=False)
    print(t.to_string(index=False))
    print(f"\n{len(df)} {args.split} claims -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
