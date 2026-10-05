"""Field validation (V01-V03) and the rule engine (synopsis Phase 2: business-rule checks).

Rule ids, severities, conditions and explanation text are read from
config/policy_rules.yaml; the Python checks are registered under the same ids here
(missing or invalid fields) and in consistency_rules.py (timeline and logic checks).

Every check is vectorised over a DataFrame of claims and returns a boolean Series
(True = passed). `validate_claim` applies the same rules to one claim and returns
`[{rule_id, passed, severity, message}]`, the output of the agent's policy-check tool.
Rules flag problems; they never modify or "fix" the data.

Severity levels:
    blocking: the claim cannot be assessed reliably (REQUEST_MORE_INFO).
    warning:  inconsistent or suspicious fields an adjuster should look at.
    info:     unusual but plausible context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.config import path_for
from src.ml.data import CATEGORIES, CLAIM_FIELDS, INTEGER_DOMAINS
from src.validation import consistency_rules
from src.validation.consistency_rules import expected_holder_bin

SEVERITIES = ("blocking", "warning", "info")
RULE_TYPES = ("validation", "consistency")


def required_fields(df, params):
    return df[CLAIM_FIELDS].notna().all(axis=1)


def domain_values(df, params):
    ok = pd.Series(True, index=df.index)
    for col, allowed in CATEGORIES.items():
        ok &= df[col].isna() | df[col].isin(allowed)
    for col, allowed in INTEGER_DOMAINS.items():
        if col not in df or allowed is None:
            continue
        ok &= df[col].isna() | df[col].isin(allowed)
    age = pd.to_numeric(df["Age"], errors="coerce")
    ok &= age.isna() | (age == 0) | age.between(params["age_min"], params["age_max"])
    return ok


def age_present(df, params):
    return df["Age"].isna() | (df["Age"] != params["missing_sentinel"])


CHECKS = {
    "V01_REQUIRED_FIELDS": required_fields,
    "V02_DOMAIN_VALUES": domain_values,
    "V03_AGE_MISSING": age_present,
    **consistency_rules.CHECKS,
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    rule_type: str
    severity: str
    condition: str
    explanation: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def check(self, df: pd.DataFrame) -> pd.Series:
        return CHECKS[self.rule_id](df, self.params).astype(bool)


def load_policy_rules(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or path_for("policy_rules"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> tuple[Rule, ...]:
    """Validation and consistency rules from policy_rules.yaml, checked against the code."""
    spec = load_policy_rules(path)
    rules = []
    for r in spec["rules"]:
        if r["type"] not in RULE_TYPES:
            continue
        if r["severity"] not in SEVERITIES:
            raise ValueError(f"{r['id']}: unknown severity {r['severity']!r}")
        if r["id"] not in CHECKS:
            raise ValueError(f"{r['id']} is in policy_rules.yaml but has no Python check")
        rules.append(Rule(r["id"], r["type"], r["severity"], r["condition"].strip(),
                          " ".join(r["explanation"].split()), r.get("params") or {}))
    missing = set(CHECKS) - {r.rule_id for r in rules}
    if missing:
        raise ValueError(f"Checks without a rule in policy_rules.yaml: {sorted(missing)}")
    return tuple(rules)


RULES: tuple[Rule, ...] = load_rules()
RULES_BY_ID = {r.rule_id: r for r in RULES}


def _invalid_values(claim: Mapping[str, Any], age_min: int, age_max: int) -> str:
    bad = []
    for col, allowed in CATEGORIES.items():
        v = claim.get(col)
        if v is not None and not pd.isna(v) and v not in allowed:
            bad.append(f"{col}={v!r}")
    for col, allowed in INTEGER_DOMAINS.items():
        v = claim.get(col)
        if allowed is not None and v is not None and not pd.isna(v) and v not in allowed:
            bad.append(f"{col}={v!r}")
    age = claim.get("Age")
    if age is not None and not pd.isna(age) and age != 0 and not age_min <= age <= age_max:
        bad.append(f"Age={age!r}")
    return ", ".join(bad)


class _Template(dict):
    def __missing__(self, key):
        return "?"


def _message_context(claim: Mapping[str, Any]) -> dict[str, Any]:
    v02 = RULES_BY_ID["V02_DOMAIN_VALUES"].params
    ctx = {f: claim.get(f) for f in CLAIM_FIELDS}
    ctx["missing_fields"] = ", ".join(
        f for f in CLAIM_FIELDS if claim.get(f) is None or pd.isna(claim.get(f)))
    ctx["invalid_values"] = _invalid_values(claim, v02["age_min"], v02["age_max"])
    age = pd.to_numeric(pd.Series([claim.get("Age")]), errors="coerce")
    ctx["expected_holder_bin"] = expected_holder_bin(age).iloc[0]
    return _Template(ctx)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CLAIM_FIELDS if c not in df.columns]
    if missing:
        df = df.assign(**{c: pd.NA for c in missing})
    return df


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean frame (rows x rule_id), True = passed."""
    df = _prepare(df)
    return pd.DataFrame({r.rule_id: r.check(df) for r in RULES}, index=df.index)


def validate_claim(claim: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply every rule to a single claim (a mapping of raw field -> value)."""
    row = pd.DataFrame([{f: claim.get(f) for f in CLAIM_FIELDS}])
    for col in INTEGER_DOMAINS:
        if col in row:
            row[col] = pd.to_numeric(row[col], errors="coerce")
    results = run_rules(row).iloc[0]
    ctx = _message_context(claim)
    out = []
    for rule in RULES:
        passed = bool(results[rule.rule_id])
        out.append({
            "rule_id": rule.rule_id,
            "passed": passed,
            "severity": rule.severity,
            "message": rule.condition if passed else rule.explanation.format_map(ctx),
        })
    return out


def has_blocking_failure(results: list[dict[str, Any]]) -> bool:
    return any(not r["passed"] and r["severity"] == "blocking" for r in results)


def rule_hit_summary(df: pd.DataFrame, target: str = "FraudFound_P") -> pd.DataFrame:
    """Per-rule violation counts and fraud rate among violators vs passers."""
    passed = run_rules(df)
    rows = []
    for rule in RULES:
        hit = ~passed[rule.rule_id]
        rec = {
            "rule_id": rule.rule_id,
            "severity": rule.severity,
            "n": len(df),
            "violations": int(hit.sum()),
            "violation_pct": 100 * hit.mean() if len(df) else np.nan,
        }
        if target in df:
            y = df[target]
            rec["fraud_rate_violators_pct"] = 100 * y[hit].mean() if hit.any() else np.nan
            rec["fraud_rate_passers_pct"] = 100 * y[~hit].mean() if (~hit).any() else np.nan
            rec["fraud_violators"] = int(y[hit].sum())
        rows.append(rec)
    return pd.DataFrame(rows)
