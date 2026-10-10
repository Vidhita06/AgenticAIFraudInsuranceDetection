"""Evaluation and statistics helpers (Wilson and bootstrap CIs; model metrics follow in Step B)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(0.5 + level / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(y_true, y_score, metric, n_boot: int = 1000, level: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap CI of metric(y_true, y_score).
    Resamples are stratified by class so every replicate contains both classes."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    stats_ = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats_[b] = metric(y_true[idx], y_score[idx])
    alpha = (1 - level) / 2
    return (float(metric(y_true, y_score)), float(np.quantile(stats_, alpha)),
            float(np.quantile(stats_, 1 - alpha)))


def paired_bootstrap_diff(y_true, score_a, score_b, metric, n_boot: int = 1000,
                          level: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """metric(a) - metric(b) on the same resamples, with a percentile CI."""
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[b] = metric(y_true[idx], score_a[idx]) - metric(y_true[idx], score_b[idx])
    alpha = (1 - level) / 2
    point = metric(y_true, score_a) - metric(y_true, score_b)
    return (float(point), float(np.quantile(diffs, alpha)), float(np.quantile(diffs, 1 - alpha)))


# ---------------------------------------------------------------------------
# Model metrics (Step B)
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score,
)


def threshold_metrics(y, p, threshold: float) -> dict[str, float]:
    """Confusion-matrix metrics for the rule `flag if p >= threshold`."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= threshold
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0
    return {"threshold": float(threshold), "accuracy": (tp + tn) / len(y), "precision": prec,
            "recall": rec, "f1": f1, "f2": f2, "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "flag_rate": float(pred.mean()), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def recall_at_precision(y, p, min_precision: float) -> float:
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def precision_at_top(y, p, frac: float) -> float:
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(frac * len(y))))
    return float(y[np.argsort(-p, kind="stable")[:k]].mean())


def ranking_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, np.clip(p, 0, 1))),
            "recall_at_precision_0.3": recall_at_precision(y, p, 0.3),
            "precision_at_top_5pct": precision_at_top(y, p, 0.05),
            "precision_at_top_10pct": precision_at_top(y, p, 0.10)}


def all_metrics(y, p, threshold: float) -> dict[str, float]:
    return {**ranking_metrics(y, p), **threshold_metrics(y, p, threshold)}


def metrics_with_ci(y, p, threshold: float, n_boot: int = 1000, level: float = 0.95,
                    seed: int = 42) -> pd.DataFrame:
    """Every metric with a stratified percentile-bootstrap CI (the threshold is held fixed)."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    point = all_metrics(y, p, threshold)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keys = [k for k in point if k not in ("threshold", "tp", "fp", "fn", "tn")]
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = all_metrics(y[idx], p[idx], threshold)
        for k in keys:
            boots[k].append(m[k])
    a = (1 - level) / 2
    rows = [{"metric": k, "value": point[k], "lo": float(np.quantile(boots[k], a)),
             "hi": float(np.quantile(boots[k], 1 - a))} for k in keys]
    rows += [{"metric": k, "value": point[k], "lo": np.nan, "hi": np.nan}
             for k in ("threshold", "tp", "fp", "fn", "tn")]
    return pd.DataFrame(rows)


def cost_threshold(y, p, cost_ratio: float) -> dict[str, float]:
    """Threshold minimising cost_ratio * FN + FP on (y, p); ties go to the higher threshold.
    Also reports the Bayes threshold 1 / (1 + cost_ratio), optimal for calibrated scores."""
    y, p = np.asarray(y).astype(int), np.asarray(p)
    cands = np.unique(np.concatenate([np.quantile(p, np.linspace(0, 1, 401)), [1.0 + 1e-9]]))
    best = None
    for t in cands:
        pred = p >= t
        cost = cost_ratio * int((~pred & (y == 1)).sum()) + int((pred & (y == 0)).sum())
        if best is None or cost < best[1] or (cost == best[1] and t > best[0]):
            best = (float(t), cost)
    m = threshold_metrics(y, p, best[0])
    return {"cost_ratio": cost_ratio, "threshold": best[0], "bayes_threshold": 1 / (1 + cost_ratio),
            "cost_per_claim": best[1] / len(y), "flag_rate": m["flag_rate"],
            "precision": m["precision"], "recall": m["recall"]}


def capacity_threshold(p, top_frac: float) -> float:
    """Score above which the top `top_frac` of claims fall."""
    return float(np.quantile(np.asarray(p), 1 - top_frac))


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Calibration by score quantile bins: mean predicted vs observed fraud rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    ci = [wilson_ci(int(o * n), int(n)) for o, n in zip(g["observed"], g["n"])]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g.reset_index()


def lift_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift: fraud rate and cumulative recall by score decile (decile 1 = highest)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    df["decile"] = np.arange(len(df)) * n_bins // len(df) + 1
    base = df["y"].mean()
    g = df.groupby("decile").agg(n=("y", "size"), fraud=("y", "sum"), min_score=("p", "min"))
    g["rate"] = g["fraud"] / g["n"]
    g["lift"] = g["rate"] / base
    g["cumulative_recall"] = g["fraud"].cumsum() / df["y"].sum()
    return g.reset_index()


def group_metrics(y, p, groups: pd.Series, threshold: float) -> pd.DataFrame:
    """Recall, FPR, precision and flag rate per group (fairness audit)."""
    rows = []
    y, p = np.asarray(y).astype(int), np.asarray(p)
    groups = pd.Series(groups).reset_index(drop=True)
    for g in groups.dropna().unique():
        m = (groups == g).to_numpy()
        t = threshold_metrics(y[m], p[m], threshold)
        rows.append({"group": g, "n": int(m.sum()), "fraud": int(y[m].sum()),
                     "fraud_rate": float(y[m].mean()), "flag_rate": t["flag_rate"],
                     "recall": t["recall"] if y[m].sum() else np.nan, "fpr": t["fpr"],
                     "precision": t["precision"]})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
