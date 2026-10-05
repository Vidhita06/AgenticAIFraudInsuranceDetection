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
