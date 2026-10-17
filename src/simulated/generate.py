"""Generate the SIMULATED reference database (docs/simulated_data_plan.md).

Hard rule: the fraud label is never read. Inputs are loaded with an explicit column
allow-list (`INPUT_COLUMNS`), so the label column is never even parsed;
tests/test_simulated_db.py proves the output does not depend on it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import yaml

from src.simulated import DATA_ORIGIN
from src.simulated.dates import accident_date, reporting_lag_days

GENERATOR_VERSION = 1
# Every column the generator may read. Never add the label or the origin flag here.
INPUT_COLUMNS = ["PolicyNumber", "BasePolicy", "VehicleCategory", "Deductible", "AgentType", "VehiclePrice",
                 "Year", "Month", "WeekOfMonth", "DayOfWeek", "MonthClaimed", "WeekOfMonthClaimed",
                 "DayOfWeekClaimed", "Days_Policy_Accident", "Days_Policy_Claim", "PastNumberOfClaims"]
DAY_BINS = {"none": (0, 0), "1 to 7": (1, 7), "8 to 15": (8, 15), "15 to 30": (15, 30), "more than 30": (31, 1095)}
CLAIM_BIN_MAX = {"more than 30": 10 ** 6}
HISTORY_COUNTS = {"none": (0, 0), "1": (1, 1), "2 to 4": (2, 4), "more than 4": (5, 7)}
HISTORY_TYPES = ["collision", "single-vehicle collision", "theft", "vandalism", "glass", "weather",
                 "third-party liability"]
HISTORY_STATUSES = ["closed - paid", "closed - no payment", "withdrawn by policyholder"]
LIABILITY_TIERS = [(15000, 30000, 5000), (25000, 50000, 10000), (50000, 100000, 25000), (100000, 300000, 50000)]
PRICE_CAP = {"less than 20000": 20000, "20000 to 29000": 29000, "30000 to 39000": 39000,
             "40000 to 59000": 59000, "60000 to 69000": 69000}

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE policies (
  policy_number INTEGER PRIMARY KEY, policy_id TEXT, customer_id TEXT,
  base_policy TEXT, vehicle_category TEXT, deductible INTEGER, agent_type TEXT, vehicle_price_band TEXT,
  inception_date TEXT, term_start TEXT, term_end TEXT, accident_date_derived TEXT,
  days_policy_accident_bin TEXT, days_policy_claim_bin TEXT, dates_consistent INTEGER,
  limit_bodily_injury_per_person INTEGER, limit_bodily_injury_per_accident INTEGER,
  limit_property_damage INTEGER, limit_vehicle_damage INTEGER, limit_comprehensive INTEGER,
  data_origin TEXT NOT NULL);
CREATE TABLE policy_rules (
  clause_id TEXT, base_policy TEXT, category TEXT, title TEXT, wording TEXT, test_json TEXT,
  wording_note TEXT NOT NULL, data_origin TEXT NOT NULL, PRIMARY KEY (clause_id, base_policy));
CREATE TABLE claim_history (
  prior_claim_id TEXT PRIMARY KEY, customer_id TEXT, policy_number INTEGER, claim_date TEXT,
  incident_type TEXT, status TEXT, data_origin TEXT NOT NULL);
CREATE INDEX claim_history_customer ON claim_history (customer_id);
CREATE TABLE verification_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT, investigation_id TEXT, claim_id TEXT,
  check_id TEXT, level TEXT, category TEXT, result TEXT, reason TEXT, clause_reference TEXT, source TEXT,
  data_origin TEXT NOT NULL DEFAULT 'agent run-time output');
CREATE TABLE investigation_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT, investigation_id TEXT, claim_id TEXT,
  decision TEXT, flag_category TEXT, decided_by TEXT, reason_codes TEXT, tool_statuses TEXT,
  simulated_data INTEGER, model_version TEXT, data_origin TEXT NOT NULL DEFAULT 'agent run-time output');
"""
GENERATED_TABLES = ("policies", "policy_rules", "claim_history")


def read_inputs(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Only the allow-listed columns are parsed; the label column is never loaded."""
    frames = [pd.read_csv(p, usecols=lambda c: c in INPUT_COLUMNS) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    missing = set(INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"input files lack columns: {sorted(missing)}")
    return df.drop_duplicates("PolicyNumber").sort_values("PolicyNumber").reset_index(drop=True)


def _anniversary(start: date, year: int) -> date:
    try:
        return start.replace(year=year)
    except ValueError:                       # 29 Feb in a non-leap year
        return start.replace(year=year, day=28)


def _rng(seed: int, pn: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(pn)])


def policy_row(r: dict, seed: int) -> tuple[dict, list[dict]]:
    pn = int(r["PolicyNumber"])
    rng = _rng(seed, pn)
    acc = accident_date(r)
    lag = reporting_lag_days(r)
    a_lo, a_hi = DAY_BINS.get(r["Days_Policy_Accident"], (31, 1095))
    c_lo, c_hi = DAY_BINS.get(r["Days_Policy_Claim"], (31, 1095))
    c_hi = CLAIM_BIN_MAX.get(r["Days_Policy_Claim"], c_hi)
    lo, hi = max(a_lo, c_lo - (lag or 0)), min(a_hi, c_hi - (lag or 0))
    consistent = acc is not None and lag is not None and lo <= hi
    if not consistent:
        lo, hi = a_lo, a_hi
    d = int(rng.integers(lo, hi + 1))
    inception = acc - timedelta(days=d) if acc else None
    term_start = term_end = None
    if acc:
        term_start = _anniversary(inception, acc.year)
        if term_start > acc:
            term_start = _anniversary(inception, acc.year - 1)
        term_end = _anniversary(term_start, term_start.year + 1) - timedelta(days=1)
    bi_person, bi_acc, pd_lim = LIABILITY_TIERS[int(rng.integers(len(LIABILITY_TIERS)))]
    vehicle = None
    if r["BasePolicy"] in ("Collision", "All Perils"):
        vehicle = PRICE_CAP.get(r["VehiclePrice"]) or int(rng.choice([80000, 100000, 120000]))
    row = {"policy_number": pn, "policy_id": f"PN-{pn}", "customer_id": f"SIM-CUST-{pn}",
           "base_policy": r["BasePolicy"], "vehicle_category": r["VehicleCategory"],
           "deductible": int(r["Deductible"]), "agent_type": r["AgentType"], "vehicle_price_band": r["VehiclePrice"],
           "inception_date": inception.isoformat() if inception else None,
           "term_start": term_start.isoformat() if term_start else None,
           "term_end": term_end.isoformat() if term_end else None,
           "accident_date_derived": acc.isoformat() if acc else None,
           "days_policy_accident_bin": r["Days_Policy_Accident"], "days_policy_claim_bin": r["Days_Policy_Claim"],
           "dates_consistent": int(consistent),
           "limit_bodily_injury_per_person": bi_person, "limit_bodily_injury_per_accident": bi_acc,
           "limit_property_damage": pd_lim, "limit_vehicle_damage": vehicle,
           "limit_comprehensive": vehicle if r["BasePolicy"] == "All Perils" else None,
           "data_origin": DATA_ORIGIN}
    n_lo, n_hi = HISTORY_COUNTS.get(r["PastNumberOfClaims"], (0, 0))
    n = int(rng.integers(n_lo, n_hi + 1))
    history = []
    if acc:
        days_back = sorted(int(x) for x in rng.integers(30, 5 * 365, size=n))
        for i, back in enumerate(reversed(days_back)):
            history.append({"prior_claim_id": f"SIM-H-{pn}-{i + 1}", "customer_id": row["customer_id"],
                            "policy_number": pn, "claim_date": (acc - timedelta(days=back)).isoformat(),
                            "incident_type": str(rng.choice(HISTORY_TYPES)),
                            "status": str(rng.choice(HISTORY_STATUSES)), "data_origin": DATA_ORIGIN})
    return row, history


def clause_rows(clauses_path: str | Path) -> list[dict]:
    spec = yaml.safe_load(Path(clauses_path).read_text(encoding="utf-8"))
    rows = []
    for c in spec["clauses"]:
        for bp in c["coverage"]:
            rows.append({"clause_id": c["id"], "base_policy": bp, "category": c["category"], "title": c["title"],
                         "wording": c["wording"], "test_json": json.dumps(c.get("test")) if c.get("test") else None,
                         "wording_note": spec["wording_note"], "data_origin": DATA_ORIGIN})
    return rows


def table_hash(con: sqlite3.Connection, table: str) -> str:
    cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
    h = hashlib.sha256()
    for row in con.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY {', '.join(cols[:2])}"):
        h.update(json.dumps(row, default=str).encode())
    return h.hexdigest()


def build(out_path: str | Path, input_paths: Iterable[str | Path], clauses_path: str | Path,
          seed: int = 42, built_at: Optional[str] = None) -> dict:
    """Write the database (replacing any existing file) and return its manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    df = read_inputs(input_paths)
    policies, history = [], []
    for r in df.to_dict("records"):
        p, h = policy_row(r, seed)
        policies.append(p)
        history += h
    con = sqlite3.connect(out_path)
    try:
        con.executescript(SCHEMA)
        for table, rows in (("policies", policies), ("policy_rules", clause_rows(clauses_path)),
                            ("claim_history", history)):
            if rows:
                cols = list(rows[0])
                con.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                [tuple(r[c] for c in cols) for r in rows])
        manifest = {"data_origin": DATA_ORIGIN, "seed": int(seed), "generator_version": GENERATOR_VERSION,
                    "input_columns": INPUT_COLUMNS,
                    "rows": {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in GENERATED_TABLES},
                    "sha256": {t: table_hash(con, t) for t in GENERATED_TABLES}}
        meta = {**{k: json.dumps(v) for k, v in manifest.items()}, "built_at": built_at or date.today().isoformat(),
                "note": "SIMULATED reference data for the project prototype. Not real policies or claim history."}
        con.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items()))
        con.commit()
    finally:
        con.close()
    return manifest
