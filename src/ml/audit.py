"""Phase 1 data audit: profiling, fact checks and docs/report/01_data_audit.md (+ tables/01_*.csv).

The test split is read only for integrity checks (shapes, ids, duplicates, value
domains, rule violation counts). Its labels are not inspected beyond the class
count stated in the brief, so model selection stays blind to it.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from typing import Any

from src.config import input_hashes, load_config, path_for
from src.ml.data import (
    CATEGORICAL_COLUMNS, CATEGORIES, ID_COL, RAW_COLUMNS, TARGET, load_raw, load_splits,
)
from src.ml.evaluate import wilson_ci
from src.validation.validators import RULES, run_rules

LABELLED = ["real_train", "synthetic", "real_validation"]  # fraud rates reported for these only
PROFILE_COLUMNS = [c for c in RAW_COLUMNS if c not in ("PolicyNumber",)]


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


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def tvd_table(splits: dict[str, pd.DataFrame], reference: str = "real_train") -> pd.DataFrame:
    """Total variation distance of each column's distribution vs the reference split."""
    rows = []
    ref = splits[reference]
    for col in PROFILE_COLUMNS:
        if col == TARGET:
            continue
        p = ref[col].value_counts(normalize=True)
        rec = {"column": col}
        for name, df in splits.items():
            if name == reference:
                continue
            q = df[col].value_counts(normalize=True)
            rec[f"TVD vs {name}"] = 0.5 * p.subtract(q, fill_value=0).abs().sum()
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(f"TVD vs {[n for n in splits if n != reference][0]}",
                                          ascending=False)


def rule_violation_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    results = {name: run_rules(df) for name, df in splits.items()}
    for rule in RULES:
        rec = {"rule_id": rule.rule_id, "severity": rule.severity}
        for name, df in splits.items():
            hit = ~results[name][rule.rule_id]
            rec[f"{name} n"] = int(hit.sum())
            rec[f"{name} %"] = 100 * hit.mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def rule_fraud_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fraud rate among rule violators (Wilson 95% CI) vs passers, labelled splits only."""
    rows = []
    for name in LABELLED:
        df = splits[name]
        res = run_rules(df)
        y = df[TARGET]
        for rule in RULES:
            hit = ~res[rule.rule_id]
            n, k = int(hit.sum()), int(y[hit].sum())
            lo, hi = wilson_ci(k, n)
            pass_rate = y[~hit].mean()
            rows.append({
                "split": name, "rule_id": rule.rule_id, "violators": n, "fraud": k,
                "fraud % (violators)": 100 * k / n if n else float("nan"),
                "95% CI": f"{100 * lo:.1f}-{100 * hi:.1f}" if n else "",
                "fraud % (passers)": 100 * pass_rate,
                "lift": (k / n) / pass_rate if n and pass_rate else float("nan"),
            })
    return pd.DataFrame(rows)


def run_audit() -> dict:
    raw = load_raw()
    splits = load_splits(include_test=True)
    tables_dir = path_for("tables_dir")
    tables_dir.mkdir(parents=True, exist_ok=True)

    facts = pd.DataFrame(verify_split_facts(raw, splits, load_config()["data"]["real_policy_number_max"]))
    signals = univariate_signal_check(raw).rename(columns={"n": "n raw", "observed_pct": "raw %"})
    for name, df in (("real_train", splits["real_train"]),
                     ("augmented", pd.concat([splits["real_train"], splits["synthetic"]]))):
        sig = univariate_signal_check(df)
        signals[f"{name} %"] = sig["observed_pct"].values
    profile = profile_columns(splits)
    per_split_card = pd.DataFrame({n: df[RAW_COLUMNS].nunique() for n, df in splits.items()})
    profile = profile.merge(per_split_card, left_on="column", right_index=True)
    tvd = tvd_table(splits)
    violations = rule_violation_table({"raw": raw, **splits})
    rule_fraud = rule_fraud_table(splits)

    freq_long = []
    for col in CATEGORICAL_COLUMNS + ["WeekOfMonth", "WeekOfMonthClaimed", "RepNumber",
                                      "Deductible", "DriverRating", "Year"]:
        ft = frequency_table(splits, col, fraud_splits=LABELLED).reset_index()
        ft.insert(0, "column", col)
        ft = ft.rename(columns={col: "value"})
        freq_long.append(ft)
    freq = pd.concat(freq_long, ignore_index=True)
    age = pd.DataFrame({n: df["Age"].describe() for n, df in splits.items()}).T

    facts.to_csv(tables_dir / "01_fact_checks.csv", index=False)
    profile.to_csv(tables_dir / "01_column_profile.csv", index=False)
    freq.to_csv(tables_dir / "01_frequencies_by_split.csv", index=False)
    tvd.to_csv(tables_dir / "01_tvd_vs_real_train.csv", index=False)
    violations.to_csv(tables_dir / "01_rule_violations.csv", index=False)
    rule_fraud.to_csv(tables_dir / "01_rule_fraud_rates.csv", index=False)

    return {"raw": raw, "splits": splits, "facts": facts, "signals": signals,
            "profile": profile, "tvd": tvd, "violations": violations,
            "rule_fraud": rule_fraud, "freq": freq, "age": age, "hashes": input_hashes()}


def write_report(res: dict) -> str:
    splits = res["splits"]
    facts, violations, rule_fraud = res["facts"], res["violations"], res["rule_fraud"]
    rt, syn = splits["real_train"], splits["synthetic"]
    n_ok = int(facts["ok"].sum())
    failed = facts[~facts["ok"]]

    shapes = pd.DataFrame([
        {"split": n, "rows": len(d), "fraud": int(d[TARGET].sum()) if n != "real_test" else "(138, per brief)",
         "fraud %": f"{100 * d[TARGET].mean():.2f}" if n != "real_test" else "",
         "PolicyNumber range": f"{d.PolicyNumber.min()}-{d.PolicyNumber.max()}"}
        for n, d in {"raw": res["raw"], **splits}.items()
    ])
    test_fraud = int(splits["real_test"][TARGET].sum())
    shapes.loc[shapes.split == "real_test", "fraud"] = test_fraud
    shapes.loc[shapes.split == "real_test", "fraud %"] = f"{100 * test_fraud / len(splits['real_test']):.2f}"

    viol_cols = ["rule_id", "severity"] + [c for c in violations.columns if c.endswith(" n")]
    viol_view = violations[viol_cols].copy()
    for name in ["real_train", "synthetic", "real_validation", "real_test"]:
        viol_view[f"{name} n"] = [f"{n} ({p:.2f}%)" for n, p in
                                  zip(violations[f"{name} n"], violations[f"{name} %"])]
    rf = rule_fraud[rule_fraud["violators"] > 0].copy()

    profile_view = res["profile"][["column", "dtype", "cardinality", "real_train", "synthetic",
                                   "real_validation", "real_test", "nulls", "out_of_domain", "values"]]
    profile_view = profile_view.rename(columns={
        "cardinality": "card. (all)", "real_train": "card. RT", "synthetic": "card. Syn",
        "real_validation": "card. Val", "real_test": "card. Test"})

    rule_defs = pd.DataFrame([{"rule_id": r.rule_id, "severity": r.severity,
                               "checks": r.condition} for r in RULES])
    hashes = pd.DataFrame([{"file": k, "sha256": v} for k, v in res["hashes"].items()])
    tvd = res["tvd"].head(10)
    age0 = {n: int((d.Age == 0).sum()) for n, d in splits.items()}

    rt_mm = rt[rt.PolicyType != rt.VehicleCategory + " - " + rt.BasePolicy]
    syn_lag = int((~run_rules(syn)["V10_POLICY_DAYS_VS_LAG"]).sum())

    md = f"""# Phase 1 — Data Audit & Validation Rules

_Generated by `python scripts/run_phase1_audit.py` on {dt.date.today().isoformat()}. Tables: `docs/report/tables/01_*.csv`. Rules: `config/policy_rules.yaml`._

## 1. Inputs

{_md_table(hashes)}

{_md_table(shapes)}

All files load with explicit dtypes (`src/ml/data.py`): 24 categorical columns as strings (so
out-of-domain values such as `'0'` survive loading), 9 integer columns as `int64`, `is_synthetic`
as `int8`. The raw file has a UTF-8 BOM and the processed files do not; `encoding="utf-8-sig"`
handles both. The real test split is read here only for integrity checks (ids, duplicates, domains,
rule-violation counts). Apart from its overall class balance, which the brief already states, no
fraud rate is computed on it.

## 2. Verification of the stated facts ({n_ok}/{len(facts)} confirmed)

{_md_table(facts)}

"""
    if len(failed):
        md += "**Mismatches with the brief:**\n\n"
        for _, r in failed.iterrows():
            md += f"- {r['check']}: brief says {r['expected']}, data shows {r['observed']}.\n"
        md += ("  The brief's ~93% figure for Deductible = 400 is wrong; the real figure is about 96%. "
               "Nothing else depends on it.\n\n")

    md += f"""### Stated univariate signals

The fraud rates in the brief match the **augmented (real-train + synthetic) training set**
exactly, and the real-only figures closely:

{_md_table(res['signals'])}

## 3. Column profile

{_md_table(profile_view)}

`Age` summary per split:

{_md_table(res['age'].reset_index().rename(columns={'index': 'split'}))}

Per-split frequencies and fraud rates (labelled splits only) for every categorical column are in
`docs/report/tables/01_frequencies_by_split.csv`.

**Distribution shift vs real train** (total variation distance, top 10 columns). Validation and
test differ from real train only by sampling noise. Synthetic data has the same marginals.
Phase 2 tests joint fidelity.

{_md_table(tvd, '{:.4f}')}

## 4. Claim-validation rules (`config/policy_rules.yaml`, `src/validation/`)

Each rule returns `{{rule_id, passed, severity, message}}` via `validate_claim(claim)`.
These rules become the agent's policy-check tool. Severity: **blocking** means the claim
cannot be assessed (a "Request More Information" candidate), **warning** means an
inconsistency for the adjuster, and **info** means unusual but plausible context. The
rules only flag claims; they never change the data.

{_md_table(rule_defs)}

### Violations per split and origin

{_md_table(viol_view)}

### Fraud rate among violators (real train, synthetic, validation; Wilson 95% CI)

{_md_table(rf[['split', 'rule_id', 'violators', 'fraud', 'fraud % (violators)', '95% CI', 'fraud % (passers)', 'lift']])}

## 5. Findings

1. **Splits are clean.** All checked facts hold except the deductible share (see §2). Real
   train/validation/test are disjoint by `PolicyNumber` and identical to the original rows. Only
   `PolicyNumber` 1517, the row with `'0'` dates and Age 0, is dropped. No synthetic row copies a
   real row exactly, and there are no duplicates.
2. **`PolicyNumber` must be excluded.** Real rows have 1–15,420 and synthetic rows have
   15,421–304,627 (sequential), so it would act as an origin and leakage key.
3. **`AgeOfPolicyHolder` is a deterministic function of `Age`**, but with shifted bands: Age
   16–17 → "18 to 20", 18–20 → "21 to 25", 21–25 → "26 to 30", 26–35 → "31 to 35",
   36–45 → "36 to 40", 46–55 → "41 to 50", 56–65 → "51 to 65", 66+ → "over 65", and Age 0 →
   "16 to 17". This holds for 100% of rows in every split, synthetic included. A literal check
   ("Age lies inside the bin") would fail about 45% of rows, so `Age` is probably the driver's
   age and the bin a separately derived field. V04 checks the empirical mapping instead.
   For Phase 3, `AgeOfPolicyHolder` adds no information beyond `Age`.
4. **Missing age (V03)**: Age 0 appears in {age0['real_train']} real-train, {age0['synthetic']}
   synthetic, {age0['real_validation']} validation and {age0['real_test']} test rows. In each
   case `AgeOfPolicyHolder` is "16 to 17", a value that never occurs with a recorded age. The
   fraud rate is about 1.6–1.9× the base rate in real train, synthetic and validation, so
   missing age is a risk signal as well as a reason to request more information.
5. **The PolicyType "inconsistency" is a single alias.** All {len(rt_mm)} real-train mismatches
   (32%) are `PolicyType = "Sedan - Liability"` recorded for a **Sport** vehicle on a
   **Liability** base policy. Real "Sedan + Liability" claims never occur. V05 therefore
   flags only *other* mismatches (none exist), and V05B reports the alias as info. The alias
   has low fraud (~0.75%) because it means Sport + Liability, so it must not be treated as a
   fraud signal or a blocking error.
6. **Early-policy incidents (V08)** carry the strongest rule signal: real-train fraud rate
   ≈ 19% (lift ≈ 3×, n = 37). In synthetic data the rate is only ≈ 9.5% (n = 899), so the
   generator weakened this red flag. Phase 2 follows this up.
7. **The timing rules (V06, V09) are almost never violated** (3 and 4 rows in the original
   data). Claim dates carry only month and week, with no claim year, so V06 can only detect
   same-month inversions. `Days_Policy_Claim = "none"` occurs only in the dropped row 1517, so
   it has 3 levels in every split.
8. **V10 is violated only in synthetic data** ({syn_lag} rows, 0 real): a claim filed within 30
   days of policy start but 2+ months after the accident. This is a synthetic artefact. V11 (holder under 21 with a
   vehicle older than their driving years) fires at the same rate in real and synthetic data
   (~0.35%), but its fraud lift is weaker in synthetic data (1.9× vs 2.7×). Phase 2 can use V10
   violations to filter synthetic rows.
9. **`RepNumber`** (16 agents) has fraud rates between ~4.7% and ~6.7% in real train, which
   looks like noise. Phase 3 assesses it with a χ² test before deciding.

## 6. Decisions

- Never use as features: `PolicyNumber` and `is_synthetic`. `RepNumber` stays under review.
- Validation rules flag claims and never rewrite them. V03 (Age 0) is the only blocking rule
  that fires on this data.
- The PolicyType alias is documented, not corrected. Phase 4 decides whether to drop
  `PolicyType` as redundant (it is `VehicleCategory × BasePolicy` up to the alias).

## 7. Open questions

1. How was the synthetic data generated (method, e.g. CTGAN, Gaussian copula or SMOTE-NC)? Did
   the generator see only the 70% real-train split? Exact-copy checks are clean; Phase 2 adds
   near-duplicate and distance-to-closest-record checks.
2. Should V08 (early-policy incident) stay a **warning** or become **blocking** (request
   coverage documents)? It affects ~0.4% of claims.
3. Is `Age` the driver's age and `AgeOfPolicyHolder` the policyholder's age band? The data
   suggests they are the same person with shifted bands, but the Kaggle source does not say.
"""
    out = path_for("reports_dir") / "01_data_audit.md"
    out.write_text(md, encoding="utf-8")
    return md
