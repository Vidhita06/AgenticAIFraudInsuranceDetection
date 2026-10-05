"""Column profiling and integrity checks of the provided splits (Phase 1 audit)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.load import CATEGORIES, ID_COL, RAW_COLUMNS, TARGET

NON_ID_COLUMNS = [c for c in RAW_COLUMNS if c != ID_COL]


def profile_columns(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per column: dtype, cardinality, nulls and value set (pooled over splits)."""
    pooled = pd.concat(splits.values(), ignore_index=True)
    rows = []
    for col in RAW_COLUMNS:
        s = pooled[col]
        values = sorted(s.dropna().unique().tolist(), key=str)
        allowed = CATEGORIES.get(col)
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "cardinality": s.nunique(),
            "nulls": int(s.isna().sum()),
            "min": s.min() if s.dtype.kind in "iuf" else "",
            "max": s.max() if s.dtype.kind in "iuf" else "",
            "out_of_domain": ", ".join(str(v) for v in values if v not in allowed) if allowed else "",
            "values": ", ".join(map(str, values)) if len(values) <= 20 else f"{len(values)} distinct",
        })
    return pd.DataFrame(rows)


def frequency_table(splits: dict[str, pd.DataFrame], col: str,
                    fraud_splits: list[str] | None = None) -> pd.DataFrame:
    """Per-split share (%) of each value of `col`, plus fraud rate (%) for `fraud_splits`
    (default: all). Pass fraud_splits without real_test to keep test labels unseen."""
    share = pd.DataFrame({
        name: df[col].value_counts(normalize=True, dropna=False) * 100 for name, df in splits.items()
    })
    fraud_splits = list(splits) if fraud_splits is None else fraud_splits
    fraud = pd.DataFrame({
        f"fraud%_{name}": splits[name].groupby(col, dropna=False)[TARGET].mean() * 100
        for name in fraud_splits
    })
    out = share.join(fraud)
    order = CATEGORIES.get(col)
    if order:
        out = out.reindex([v for v in order if v in out.index] + [v for v in out.index if v not in order])
    else:
        out = out.sort_index()
    out.index.name = col
    return out


def _row_keys(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[columns].astype(str), index=False)


def verify_split_facts(raw: pd.DataFrame, splits: dict[str, pd.DataFrame],
                       real_max_policy: int = 15420) -> list[dict[str, Any]]:
    """Re-check the facts stated in the brief. Returns [{check, expected, observed, ok}]."""
    rt, syn = splits["real_train"], splits["synthetic"]
    va, te = splits["real_validation"], splits["real_test"]
    checks: list[dict[str, Any]] = []

    def add(name, expected, observed):
        checks.append({"check": name, "expected": expected, "observed": observed,
                       "ok": expected == observed})

    def add_approx(name, expected, observed, tol):
        checks.append({"check": name, "expected": f"~{expected:.2%}",
                       "observed": f"{observed:.2%}", "ok": abs(expected - observed) <= tol})

    add("raw shape", (15420, 33), raw.shape)
    add("augmented rows (real train + synthetic)", 300000, len(rt) + len(syn))
    add("real train rows / fraud", (10793, 646), (len(rt), int(rt[TARGET].sum())))
    add("synthetic rows / fraud", (289207, 17310), (len(syn), int(syn[TARGET].sum())))
    add("validation rows / fraud", (2313, 139), (len(va), int(va[TARGET].sum())))
    add("test rows / fraud", (2313, 138), (len(te), int(te[TARGET].sum())))

    pn = {k: set(v[ID_COL]) for k, v in splits.items()}
    real_sets = [pn["real_train"], pn["real_validation"], pn["real_test"]]
    overlaps = sum(len(a & b) for i, a in enumerate(real_sets) for b in real_sets[i + 1:])
    add("real splits disjoint by PolicyNumber (overlapping ids)", 0, overlaps)
    missing = sorted(set(raw[ID_COL]) - set().union(*real_sets))
    add("original rows in no real split", [1517], missing)
    dropped = raw[raw[ID_COL].isin(missing)]
    add("dropped row has DayOfWeekClaimed='0', MonthClaimed='0', Age=0", True,
        bool(len(dropped) == 1 and (dropped["DayOfWeekClaimed"] == "0").all()
             and (dropped["MonthClaimed"] == "0").all() and (dropped["Age"] == 0).all()))

    # Real split rows are identical to the original rows with the same PolicyNumber.
    raw_idx = raw.set_index(ID_COL)
    mism = 0
    for name in ("real_train", "real_validation", "real_test"):
        df = splits[name].set_index(ID_COL)
        mism += int((raw_idx.loc[df.index, df.columns].astype(str) != df.astype(str)).any(axis=1).sum())
    add("real split rows identical to original rows (mismatching rows)", 0, mism)

    add("synthetic PolicyNumber min", real_max_policy + 1, int(syn[ID_COL].min()))
    add("synthetic PolicyNumber sequential", True,
        bool((syn[ID_COL].sort_values().diff().dropna() == 1).all()))
    add("real PolicyNumber range within 1..15420", True,
        bool(all(splits[k][ID_COL].between(1, real_max_policy).all()
                 for k in ("real_train", "real_validation", "real_test"))))

    cols = NON_ID_COLUMNS
    syn_keys = set(_row_keys(syn, cols))
    for name in ("real_train", "real_validation", "real_test"):
        hits = int(_row_keys(splits[name], cols).isin(syn_keys).sum())
        add(f"synthetic exact copies of {name} rows (all cols but PolicyNumber)", 0, hits)
    pooled = pd.concat([rt, syn, va, te], ignore_index=True)
    add("duplicate rows across all splits (all cols but PolicyNumber)", 0,
        int(pooled.duplicated(subset=cols).sum()))
    add("duplicate synthetic rows (all cols but PolicyNumber)", 0,
        int(syn.duplicated(subset=cols).sum()))

    add("Age = 0 in real train", 232, int((rt["Age"] == 0).sum()))
    add("Age = 0 in synthetic", 5242, int((syn["Age"] == 0).sum()))
    for name, df in (("real train", rt), ("synthetic", syn)):
        mismatch = (df["PolicyType"] != df["VehicleCategory"] + " - " + df["BasePolicy"]).mean()
        add_approx(f"PolicyType mismatch share in {name}", 0.32, mismatch, 0.01)
    for name, df in splits.items():
        vc = df["Year"].value_counts()
        add(f"Year order 1994 > 1995 > 1996 in {name}", True,
            bool(vc.get(1994, 0) > vc.get(1995, 0) > vc.get(1996, 0)))
    for name, df in (("real train", rt), ("synthetic", syn)):
        add_approx(f"Deductible = 400 share in {name}", 0.93, (df["Deductible"] == 400).mean(), 0.01)
    return checks


def univariate_signal_check(df: pd.DataFrame) -> pd.DataFrame:
    """Fraud rate (%) for the brief's stated strong signals, to compare with stated values."""
    stated = {
        ("Fault", "Policy Holder"): 7.82, ("Fault", "Third Party"): 1.17,
        ("BasePolicy", "All Perils"): 10.28, ("BasePolicy", "Collision"): 7.11,
        ("BasePolicy", "Liability"): 0.77,
        ("VehicleCategory", "Utility"): 11.04, ("VehicleCategory", "Sedan"): 8.28,
        ("VehicleCategory", "Sport"): 1.42,
        ("VehiclePrice", "less than 20000"): 8.48, ("VehiclePrice", "more than 69000"): 7.94,
    }
    rows = []
    for (col, val), pct in stated.items():
        sub = df[df[col] == val]
        rows.append({"column": col, "value": val, "stated_pct": pct, "n": len(sub),
                     "observed_pct": round(100 * sub[TARGET].mean(), 2)})
    return pd.DataFrame(rows)
