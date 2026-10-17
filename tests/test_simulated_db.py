"""The SIMULATED reference database: never reads the fraud label, deterministic, internally
consistent, and labelled as simulated everywhere."""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config, path_for
from src.simulated import generate
from src.simulated.dates import accident_date, reporting_lag_days

CLAUSES = PROJECT_ROOT / load_config()["simulated_data"]["clauses"]
SPLITS = ("real_train", "real_validation", "real_test")


def small_inputs(tmp_path, n=400, transform=None):
    paths = []
    for k in SPLITS:
        df = pd.read_csv(path_for(k)).head(n)
        if transform:
            df = transform(df)
        p = tmp_path / f"{k}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


def test_reads_never_request_the_label(tmp_path, monkeypatch):
    inputs = small_inputs(tmp_path, 50)
    calls = []
    real = pd.read_csv

    def spy(path, *a, **kw):
        calls.append(kw.get("usecols"))
        return real(path, *a, **kw)
    monkeypatch.setattr(generate.pd, "read_csv", spy)
    generate.build(tmp_path / "db.sqlite", inputs, CLAUSES)
    assert len(calls) == 3
    for usecols in calls:
        assert callable(usecols)
        assert not usecols("FraudFound_P") and not usecols("is_synthetic")
        assert usecols("PastNumberOfClaims") and usecols("PolicyNumber")


def test_output_does_not_depend_on_the_label(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "a").mkdir()
    base = generate.build(tmp_path / "a.db", small_inputs(tmp_path / "a"), CLAUSES)
    (tmp_path / "b").mkdir()
    permuted = generate.build(tmp_path / "b.db", small_inputs(
        tmp_path / "b", transform=lambda d: d.assign(FraudFound_P=rng.permutation(d.FraudFound_P.to_numpy()))), CLAUSES)
    (tmp_path / "c").mkdir()
    dropped = generate.build(tmp_path / "c.db", small_inputs(
        tmp_path / "c", transform=lambda d: d.drop(columns=["FraudFound_P"])), CLAUSES)
    (tmp_path / "d").mkdir()
    flipped = generate.build(tmp_path / "d.db", small_inputs(
        tmp_path / "d", transform=lambda d: d.assign(FraudFound_P=1 - d.FraudFound_P)), CLAUSES)
    assert base["sha256"] == permuted["sha256"] == dropped["sha256"] == flipped["sha256"]


def test_generator_code_never_names_the_label():
    for f in ("src/simulated/generate.py", "src/simulated/dates.py", "src/simulated/__init__.py",
              "scripts/build_simulated_db.py"):
        text = (PROJECT_ROOT / f).read_text()
        assert "FraudFound_P" not in text and "TARGET" not in text and "is_synthetic" not in text, f
    assert "FraudFound_P" not in generate.INPUT_COLUMNS


def test_deterministic_for_a_seed(tmp_path):
    inputs = small_inputs(tmp_path, 100)
    a = generate.build(tmp_path / "a.db", inputs, CLAUSES, seed=42)
    b = generate.build(tmp_path / "b.db", inputs, CLAUSES, seed=42)
    c = generate.build(tmp_path / "c.db", inputs, CLAUSES, seed=7)
    assert a["sha256"] == b["sha256"] and a["sha256"]["policies"] != c["sha256"]["policies"]


def test_full_build_matches_the_committed_manifest(simulated_db):
    _, manifest = simulated_db
    committed = json.loads((PROJECT_ROOT / "data" / "simulated" / "manifest.json").read_text())
    assert manifest["sha256"] == committed["sha256"] and manifest["rows"] == committed["rows"]


def test_tables_are_labelled_and_have_no_outcome_columns(simulated_db):
    con = sqlite3.connect(simulated_db[0])
    for t in ("policies", "policy_rules", "claim_history"):
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
        assert not [c for c in cols if any(w in c.lower() for w in ("fraud", "label", "outcome", "found"))], t
        assert con.execute(f"SELECT COUNT(*) FROM {t} WHERE data_origin != 'simulated reference data'").fetchone()[0] == 0
    notes = {r[0] for r in con.execute("SELECT DISTINCT wording_note FROM policy_rules")}
    assert notes == {"illustrative wording written by the project team"}
    assert "SIMULATED" in dict(con.execute("SELECT key, value FROM meta").fetchall())["note"]


def test_dates_match_the_real_bins_and_the_term(simulated_db):
    con = sqlite3.connect(simulated_db[0])
    pol = pd.read_sql("SELECT * FROM policies", con)
    real = pd.concat([pd.read_csv(path_for(k)) for k in SPLITS]).drop_duplicates("PolicyNumber")
    m = pol.merge(real, left_on="policy_number", right_on="PolicyNumber")
    assert len(m) == len(pol)
    assert (m.accident_date_derived >= m.term_start).all() and (m.accident_date_derived <= m.term_end).all()
    assert (m.base_policy == m.BasePolicy).all() and (m.deductible == m.Deductible).all()
    ok = m[m.dates_consistent == 1]
    assert len(ok) / len(m) > 0.99
    d_acc = (pd.to_datetime(ok.accident_date_derived) - pd.to_datetime(ok.inception_date)).dt.days
    lag = ok.apply(lambda r: reporting_lag_days(r), axis=1)
    for bin_, (lo, hi) in generate.DAY_BINS.items():
        sel = ok.Days_Policy_Accident == bin_
        assert d_acc[sel].between(lo, hi).all(), bin_
        sel = ok.Days_Policy_Claim == bin_
        hi = generate.CLAIM_BIN_MAX.get(bin_, hi)
        assert (d_acc[sel] + lag[sel]).between(lo, hi).all(), bin_


def test_history_counts_follow_past_number_of_claims(simulated_db):
    con = sqlite3.connect(simulated_db[0])
    counts = dict(con.execute("SELECT policy_number, COUNT(*) FROM claim_history GROUP BY policy_number").fetchall())
    real = pd.concat([pd.read_csv(path_for(k), usecols=["PolicyNumber", "PastNumberOfClaims"]) for k in SPLITS])
    for pn, past in real.drop_duplicates("PolicyNumber").itertuples(index=False):
        lo, hi = generate.HISTORY_COUNTS[past]
        assert lo <= counts.get(pn, 0) <= hi, (pn, past)
    statuses = {r[0] for r in con.execute("SELECT DISTINCT status FROM claim_history")}
    assert statuses <= set(generate.HISTORY_STATUSES)


def test_reporting_lag_is_never_negative():
    claim = {"Year": 1994, "Month": "Mar", "WeekOfMonth": 3, "DayOfWeek": "Friday",
             "MonthClaimed": "Mar", "WeekOfMonthClaimed": 3, "DayOfWeekClaimed": "Monday"}
    assert reporting_lag_days(claim) == 0                 # same week: earlier weekday is not "before"
    later = {**claim, "MonthClaimed": "May", "WeekOfMonthClaimed": 1}
    assert reporting_lag_days(later) > 30
    assert accident_date(claim).weekday() == 4
