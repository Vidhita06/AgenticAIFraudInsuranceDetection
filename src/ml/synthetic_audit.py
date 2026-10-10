"""Synthetic-data audit: fidelity, adversarial validation, utility (TRTR/TSTR) and
memorisation. The generator is a black box, so every check compares synthetic rows
with real-train rows and uses held-out real rows as the reference behaviour.

PolicyNumber and is_synthetic are never used as inputs: PolicyNumber alone
separates the two origins perfectly.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Iterable, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.ml.data import CATEGORIES, CLAIM_FIELDS, ID_COL, ORDINAL_COLUMNS, TARGET
from src.ml.eda import cramers_matrix
from src.ml.evaluate import bootstrap_ci, paired_bootstrap_diff

warnings.filterwarnings("ignore", message=".*eval_set.*")

# Columns compared between origins: every field except the identifier.
COMPARE_COLUMNS = CLAIM_FIELDS + [TARGET]
NUMERIC_COLUMNS = ["Age", "WeekOfMonth", "WeekOfMonthClaimed", "Deductible", "DriverRating", "Year"]
# Inputs for the audit models: claim fields minus display-only columns.
AUDIT_FEATURES = [c for c in CLAIM_FIELDS if c not in ("PolicyType", "AgeOfPolicyHolder")]


# ---------------------------------------------------------------------------
# Fidelity
# ---------------------------------------------------------------------------

def tvd_and_chi2(real: pd.DataFrame, syn: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Per-column total variation distance and a chi-square homogeneity test."""
    from scipy.stats import chi2_contingency

    rows = []
    for col in cols:
        p = real[col].value_counts(normalize=True)
        q = syn[col].value_counts(normalize=True)
        table = pd.concat([real[col].value_counts(), syn[col].value_counts()], axis=1).fillna(0)
        chi2, pval, dof, _ = chi2_contingency(table.to_numpy(dtype=float), correction=False)
        rows.append({"column": col, "tvd": 0.5 * p.subtract(q, fill_value=0).abs().sum(),
                     "chi2": chi2, "dof": dof, "p_value": pval})
    return pd.DataFrame(rows).sort_values("tvd", ascending=False).reset_index(drop=True)


def fraud_rate_agreement(real: pd.DataFrame, syn: pd.DataFrame, cols: Iterable[str],
                         min_n: int = 30) -> pd.DataFrame:
    """Fraud rate per category in each origin, for categories with >= min_n real rows."""
    rows = []
    for col in cols:
        r = real.groupby(col, observed=True)[TARGET].agg(["size", "mean"])
        s = syn.groupby(col, observed=True)[TARGET].agg(["size", "mean"])
        for v in r.index:
            if r.loc[v, "size"] < min_n:
                continue
            rows.append({"column": col, "value": v, "n_real": int(r.loc[v, "size"]),
                         "rate_real": r.loc[v, "mean"],
                         "n_syn": int(s.loc[v, "size"]) if v in s.index else 0,
                         "rate_syn": s.loc[v, "mean"] if v in s.index else np.nan})
    out = pd.DataFrame(rows)
    base_r, base_s = real[TARGET].mean(), syn[TARGET].mean()
    out["lift_real"] = out["rate_real"] / base_r
    out["lift_syn"] = out["rate_syn"] / base_s
    return out


def cramers_difference(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str]):
    m_real = cramers_matrix(real, cols)
    m_syn = cramers_matrix(syn, cols)
    return m_real, m_syn, m_syn - m_real


def _mi_miller_madow(x: pd.Series, y: pd.Series) -> float:
    """Plug-in MI minus the Miller-Madow bias term (R-1)(C-1)/(2N): without it, columns
    with many levels look informative in the 10.8k-row real sample purely by chance."""
    mi = mutual_info_score(x.astype(str), y)
    return mi - (x.nunique() - 1) * (y.nunique() - 1) / (2 * len(x))


def mutual_information(real: pd.DataFrame, syn: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Mutual information (nats) with the target per origin, bias-corrected."""
    rows = [{"column": c, "mi_real": _mi_miller_madow(real[c], real[TARGET]),
             "mi_syn": _mi_miller_madow(syn[c], syn[TARGET])} for c in cols]
    out = pd.DataFrame(rows)
    out["rank_real"] = out["mi_real"].rank(ascending=False).astype(int)
    out["rank_syn"] = out["mi_syn"].rank(ascending=False).astype(int)
    return out.sort_values("mi_real", ascending=False).reset_index(drop=True)


def category_coverage(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> dict:
    """Single-value and pairwise-combination coverage between origins."""
    single_missing = [(c, v, int(n)) for c in cols for v, n in real[c].value_counts().items()
                      if v not in set(syn[c].unique())]
    single_novel = [(c, v, int(n)) for c in cols for v, n in syn[c].value_counts().items()
                    if v not in set(real[c].unique())]
    real_pairs_total = real_pairs_missing = real_rows_missing_weight = 0
    novel_row_flag = np.zeros(len(syn), dtype=bool)
    syn_str = syn[cols].astype(str)
    real_str = real[cols].astype(str)
    for a, b in itertools.combinations(cols, 2):
        rp = real_str[a] + "\x1f" + real_str[b]
        sp = syn_str[a] + "\x1f" + syn_str[b]
        rset, sset = set(rp.unique()), set(sp.unique())
        real_pairs_total += len(rset)
        missing = rset - sset
        real_pairs_missing += len(missing)
        real_rows_missing_weight += int(rp.isin(missing).sum())
        novel_row_flag |= ~sp.isin(rset).to_numpy()
    return {
        "real_values_absent_in_synthetic": single_missing,
        "synthetic_values_absent_in_real": single_novel,
        "real_pair_combinations": real_pairs_total,
        "real_pair_combinations_absent_in_synthetic": real_pairs_missing,
        "synthetic_rows_with_pair_unseen_in_real_pct": 100 * novel_row_flag.mean(),
        "novel_pair_row_mask": novel_row_flag,
    }


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def to_model_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Native-categorical frame for LightGBM with fixed category sets (stable codes)."""
    out = pd.DataFrame(index=df.index)
    for col in features:
        if col in CATEGORIES:
            out[col] = pd.Categorical(df[col].astype(object), categories=CATEGORIES[col])
        elif col == "RepNumber":
            out[col] = pd.Categorical(df[col], categories=list(range(1, 17)))
        else:
            out[col] = pd.to_numeric(df[col]).astype(float)
    return out


LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40,
                  colsample_bytree=0.8, reg_lambda=1.0, n_estimators=2000, verbose=-1,
                  random_state=RANDOM_STATE, n_jobs=-1)


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, weight: np.ndarray | None = None,
             seed: int = RANDOM_STATE, balance: bool = True) -> lgb.LGBMClassifier:
    """LightGBM with early stopping on a 10% stratified holdout of the TRAINING data."""
    weight = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    X_tr, X_es, y_tr, y_es, w_tr, w_es = train_test_split(
        X, y, weight, test_size=0.1, stratify=y, random_state=seed)
    params = dict(LGB_PARAMS, random_state=seed)
    if balance:
        params["scale_pos_weight"] = float((w_tr * (y_tr == 0)).sum() / (w_tr * (y_tr == 1)).sum())
    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_es, y_es)], eval_sample_weight=[w_es],
              eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    return model


def fit_logreg(train: pd.DataFrame, y: np.ndarray, features: list[str],
               weight: np.ndarray | None = None):
    cats = [c for c in features if c not in NUMERIC_COLUMNS]
    nums = [c for c in features if c in NUMERIC_COLUMNS]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                             ("num", StandardScaler(), nums)])
    model = make_pipeline(pre, LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5))
    X = train[features].copy()
    X[cats] = X[cats].astype(str)
    model.fit(X, y, logisticregression__sample_weight=weight)
    return model


def _logreg_predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].copy()
    cats = [c for c in features if c not in NUMERIC_COLUMNS]
    X[cats] = X[cats].astype(str)
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Adversarial validation
# ---------------------------------------------------------------------------

def adversarial_validation(real: pd.DataFrame, syn: pd.DataFrame,
                           features: list[str] = COMPARE_COLUMNS, n_splits: int = 5,
                           seed: int = RANDOM_STATE) -> dict:
    """Real vs synthetic classifier on a balanced sample (no PolicyNumber).
    Returns CV ROC-AUC, gain importances, and P(synthetic) for every synthetic row
    (out-of-fold for sampled rows, from the full-sample model for the rest)."""
    assert ID_COL not in features
    rng = np.random.default_rng(seed)
    syn_idx = rng.choice(len(syn), size=len(real), replace=False)
    sample = pd.concat([real[features], syn.iloc[syn_idx][features]], ignore_index=True)
    y = np.r_[np.zeros(len(real)), np.ones(len(real))].astype(int)
    X = to_model_frame(sample, features)
    params = dict(LGB_PARAMS, n_estimators=400)
    oof = np.zeros(len(y))
    importances = []
    aucs = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(**params).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
        importances.append(pd.Series(m.booster_.feature_importance("gain"), index=features))
    full = lgb.LGBMClassifier(**params).fit(X, y)
    p_syn = full.predict_proba(to_model_frame(syn[features], features))[:, 1]
    p_syn[syn_idx] = oof[len(real):]
    imp = pd.concat(importances, axis=1).mean(axis=1)
    imp = (imp / imp.sum()).sort_values(ascending=False).rename("gain_share")
    return {"auc_folds": aucs, "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "oof_auc": float(roc_auc_score(y, oof)), "importance": imp,
            "p_synthetic": p_syn, "p_real_oof": oof[:len(real)]}


def adversarial_drop_top(real: pd.DataFrame, syn: pd.DataFrame, ranked: list[str], k: int = 3,
                         seed: int = RANDOM_STATE) -> list[dict]:
    """Re-run adversarial validation after removing the top-1..k features in turn."""
    out = []
    feats = list(COMPARE_COLUMNS)
    for f in ranked[:k]:
        feats = [c for c in feats if c != f]
        res = adversarial_validation(real, syn, feats, n_splits=3, seed=seed)
        out.append({"dropped": f, "auc_mean": res["auc_mean"]})
    return out


# ---------------------------------------------------------------------------
# Utility: TRTR vs TSTR vs mixes, scored on real validation
# ---------------------------------------------------------------------------

def utility_experiment(real: pd.DataFrame, syn: pd.DataFrame, val: pd.DataFrame,
                       configs: Mapping[str, dict], features: list[str] = AUDIT_FEATURES,
                       seeds: Iterable[int] = (42, 43, 44), n_boot: int = 1000) -> dict:
    """Each config: {"real": bool, "syn_mask": bool array or None, "syn_frac": float,
    "syn_weight": float, "syn_n": int|None, "model": "lgbm"|"logreg"}.
    Returns per-config metrics (mean over seeds) and validation scores."""
    seeds = list(seeds)
    y_val = val[TARGET].to_numpy()
    X_val = to_model_frame(val, features)
    rows, scores = [], {}
    for name, cfg in configs.items():
        seed_scores = []
        run_seeds = seeds if cfg.get("model", "lgbm") == "lgbm" else seeds[:1]  # LR is deterministic
        for seed in run_seeds:
            rng = np.random.default_rng(seed)
            parts, weights = [], []
            if cfg.get("real", False):
                parts.append(real)
                weights.append(np.ones(len(real)))
            s = syn
            if cfg.get("syn_mask") is not None:
                s = s[cfg["syn_mask"]]
            n_syn = cfg.get("syn_n") or int(round(cfg.get("syn_frac", 0.0) * len(s)))
            if n_syn:
                idx = rng.choice(len(s), size=min(n_syn, len(s)), replace=False)
                parts.append(s.iloc[idx])
                weights.append(np.full(len(idx), cfg.get("syn_weight", 1.0)))
            train = pd.concat(parts, ignore_index=True)
            w = np.concatenate(weights)
            y = train[TARGET].to_numpy()
            if cfg.get("model", "lgbm") == "lgbm":
                m = fit_lgbm(to_model_frame(train, features), y, w, seed=seed)
                p = m.predict_proba(X_val)[:, 1]
            else:
                m = fit_logreg(train, y, features, w)
                p = _logreg_predict(m, val, features)
            seed_scores.append(p)
        p_mean = np.mean(seed_scores, axis=0)
        scores[name] = p_mean
        pr = bootstrap_ci(y_val, p_mean, average_precision_score, n_boot=n_boot)
        roc = bootstrap_ci(y_val, p_mean, roc_auc_score, n_boot=n_boot)
        per_seed = [average_precision_score(y_val, s) for s in seed_scores]
        rows.append({"config": name, "model": cfg.get("model", "lgbm"), "train_rows": len(train),
                     "pr_auc": pr[0], "pr_lo": pr[1], "pr_hi": pr[2],
                     "pr_auc_seed_sd": float(np.std(per_seed)),
                     "roc_auc": roc[0], "roc_lo": roc[1], "roc_hi": roc[2]})
    return {"table": pd.DataFrame(rows), "scores": scores, "y_val": y_val}


def utility_differences(result: dict, reference: str, n_boot: int = 1000) -> pd.DataFrame:
    y = result["y_val"]
    rows = []
    for name, s in result["scores"].items():
        if name == reference:
            continue
        d = paired_bootstrap_diff(y, s, result["scores"][reference], average_precision_score,
                                  n_boot=n_boot)
        rows.append({"config": name, "delta_pr_auc_vs_" + reference: d[0], "lo": d[1], "hi": d[2]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Memorisation: Gower distance to closest record and near-duplicate matching
# ---------------------------------------------------------------------------

class GowerEncoder:
    """Encodes rows for Gower distance: nominal columns as integer codes (0/1 mismatch),
    ordinal and numeric columns scaled to [0, 1] (absolute difference)."""

    def __init__(self, columns: list[str] = COMPARE_COLUMNS):
        self.columns = columns
        self.nominal = [c for c in columns if c in CATEGORIES and c not in ORDINAL_COLUMNS] + \
                       [c for c in columns if c == "RepNumber" or c == TARGET]
        self.scaled = [c for c in columns if c not in self.nominal]

    def fit(self, frames: Iterable[pd.DataFrame]) -> "GowerEncoder":
        pooled = pd.concat([f[self.columns] for f in frames], ignore_index=True)
        self.codes = {c: {v: i for i, v in enumerate(sorted(pooled[c].astype(str).unique()))}
                      for c in self.nominal}
        self.ranges = {}
        for c in self.scaled:
            num = self._numeric(pooled[c], c)
            self.ranges[c] = (float(num.min()), float(num.max()) - float(num.min()) or 1.0)
        return self

    @staticmethod
    def _numeric(s: pd.Series, col: str) -> pd.Series:
        if col in ORDINAL_COLUMNS:
            return s.map({v: i for i, v in enumerate(CATEGORIES[col])}).astype(float)
        return pd.to_numeric(s).astype(float)

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        nom = np.stack([df[c].astype(str).map(self.codes[c]).fillna(-1).to_numpy()
                        for c in self.nominal], axis=1).astype(np.int16)
        num = np.stack([((self._numeric(df[c], c) - self.ranges[c][0]) / self.ranges[c][1]).to_numpy()
                        for c in self.scaled], axis=1).astype(np.float32)
        return nom, num


def gower_dcr(query: tuple, ref: tuple, chunk: int = 128) -> np.ndarray:
    """Distance from each query row to its closest reference row (Gower, mean over columns)."""
    qn, qx = query
    rn, rx = ref
    p = qn.shape[1] + qx.shape[1]
    out = np.empty(len(qn), dtype=np.float32)
    for s in range(0, len(qn), chunk):
        d = (qn[s:s + chunk, None, :] != rn[None, :, :]).sum(axis=2).astype(np.float32)
        d += np.abs(qx[s:s + chunk, None, :] - rx[None, :, :]).sum(axis=2)
        out[s:s + chunk] = d.min(axis=1) / p
    return out


def max_matching_columns(query: np.ndarray, ref: np.ndarray, chunk: int = 256) -> np.ndarray:
    """For each query row, the largest number of exactly matching columns with any ref row.
    Inputs are integer-coded arrays (rows x columns)."""
    out = np.empty(len(query), dtype=np.int16)
    for s in range(0, len(query), chunk):
        out[s:s + chunk] = (query[s:s + chunk, None, :] == ref[None, :, :]).sum(axis=2).max(axis=1)
    return out


def exact_codes(frames: Mapping[str, pd.DataFrame], columns: list[str] = COMPARE_COLUMNS
                ) -> dict[str, np.ndarray]:
    pooled = pd.concat([f[columns] for f in frames.values()], ignore_index=True).astype(str)
    maps = {c: {v: i for i, v in enumerate(sorted(pooled[c].unique()))} for c in columns}
    return {name: np.stack([f[c].astype(str).map(maps[c]).to_numpy() for c in columns], axis=1)
            .astype(np.int16) for name, f in frames.items()}
