"""Fraud-focused EDA: rates with Wilson CIs, association ranking, interactions,
red-flag mining and the plotting helpers used by notebooks/01_eda.ipynb.

All statistics are computed on training data (real first, then confirmed on
synthetic). Real validation is used only to report how red flags hold up on
fresh real claims; the test split is never touched here.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy import stats  # noqa: E402

from src.ml.data import CATEGORIES, TARGET  # noqa: E402
from src.ml.evaluate import wilson_ci  # noqa: E402
from src.config import path_for  # noqa: E402
from src.validation.feature_engineering import AGE_BAND_ORDER  # noqa: E402
from src.validation.validators import condition_mask, load_policy_rules  # noqa: E402

# ---------------------------------------------------------------------------
# Palette (reference data-viz palette; validated for CVD separation)
# ---------------------------------------------------------------------------
ORIGIN_COLORS = {"real_train": "#2a78d6", "synthetic": "#eb6834", "real_validation": "#1baf7a"}
ORIGIN_LABELS = {"real_train": "Real train", "synthetic": "Synthetic",
                 "real_validation": "Real validation"}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                          "#c3c2b7", "#fcfcfb")
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div_blue_red", ["#256abf", "#f0efec", "#e34948"])

# Feature groups used for univariate profiles.
FEATURE_GROUPS: dict[str, list[str]] = {
    "Accident & claim timing": ["Month", "WeekOfMonth", "DayOfWeek", "MonthClaimed",
                                "WeekOfMonthClaimed", "DayOfWeekClaimed", "claim_lag_bucket",
                                "Year"],
    "Policy": ["BasePolicy", "PolicyType", "Deductible", "Days_Policy_Accident",
               "Days_Policy_Claim", "AgentType", "RepNumber"],
    "Vehicle": ["Make", "VehicleCategory", "VehiclePrice", "AgeOfVehicle", "NumberOfCars"],
    "Policyholder": ["age_band", "AgeOfPolicyHolder", "Sex", "MaritalStatus", "DriverRating",
                     "Fault", "AccidentArea"],
    "Claim process": ["PoliceReportFiled", "WitnessPresent", "NumberOfSuppliments",
                      "AddressChange_Claim", "PastNumberOfClaims"],
}
SENSITIVE = ["Sex", "MaritalStatus", "age_band"]
LAG_ORDER = ["same month", "1 month", "2+ months"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def value_order(col: str, values: Iterable) -> list:
    values = list(pd.unique(pd.Series(list(values)).dropna()))
    if col in CATEGORIES:
        order = CATEGORIES[col]
        return [v for v in order if v in values] + [v for v in values if v not in order]
    if col == "age_band":
        return [v for v in AGE_BAND_ORDER if v in values]
    if col == "claim_lag_bucket":
        return [v for v in LAG_ORDER if v in values]
    return sorted(values)


# ---------------------------------------------------------------------------
# Rates and association statistics
# ---------------------------------------------------------------------------

def rate_table(df: pd.DataFrame, col: str, target: str = TARGET) -> pd.DataFrame:
    """Per-category n, fraud count, fraud rate with Wilson 95% CI, and lift vs base rate."""
    g = df.groupby(col, observed=True, dropna=False)[target].agg(n="size", fraud="sum")
    base = df[target].mean()
    ci = [wilson_ci(int(k), int(n)) for k, n in zip(g["fraud"], g["n"])]
    g["rate"] = g["fraud"] / g["n"]
    g["lo"] = [c[0] for c in ci]
    g["hi"] = [c[1] for c in ci]
    g["lift"] = g["rate"] / base
    g = g.reindex(value_order(col, g.index))
    g.index.name = "value"
    return g


def rates_by_origin(splits: Mapping[str, pd.DataFrame], col: str) -> pd.DataFrame:
    parts = []
    for name, df in splits.items():
        t = rate_table(df, col).reset_index()
        t.insert(0, "origin", name)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "column", col)
    return out


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def information_value(x: pd.Series, y: pd.Series, eps: float = 0.5) -> float:
    """IV with additive smoothing so empty cells do not produce infinities."""
    t = pd.crosstab(x, y)
    bad = (t.get(1, 0) + eps) / (t.get(1, 0).sum() + eps * len(t))
    good = (t.get(0, 0) + eps) / (t.get(0, 0).sum() + eps * len(t))
    return float(((good - bad) * np.log(good / bad)).sum())


def association_ranking(df: pd.DataFrame, cols: list[str], target: str = TARGET) -> pd.DataFrame:
    rows = []
    for col in cols:
        table = pd.crosstab(df[col], df[target])
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        rows.append({"feature": col, "levels": int(df[col].nunique()), "chi2": chi2,
                     "dof": dof, "p_value": p, "cramers_v": cramers_v(df[col], df[target]),
                     "iv": information_value(df[col], df[target])})
    out = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    out["iv_strength"] = pd.cut(out["iv"], [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
                                labels=["useless", "weak", "medium", "strong", "suspicious"])
    return out


def cramers_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for a, b in itertools.combinations(cols, 2):
        m.loc[a, b] = m.loc[b, a] = cramers_v(df[a], df[b])
    return m


def interaction_table(df: pd.DataFrame, row: str, col: str, target: str = TARGET) -> pd.DataFrame:
    g = df.groupby([row, col], observed=True)[target].agg(n="size", fraud="sum").reset_index()
    g["rate"] = g["fraud"] / g["n"]
    ci = [wilson_ci(int(k), int(n)) for k, n in zip(g["fraud"], g["n"])]
    g["lo"], g["hi"] = [c[0] for c in ci], [c[1] for c in ci]
    return g


# ---------------------------------------------------------------------------
# Red-flag mining
# ---------------------------------------------------------------------------

def mine_conditions(df: pd.DataFrame, cols: list[str], min_support: int = 30,
                    min_lift: float = 1.5, max_order: int = 2,
                    target: str = TARGET) -> pd.DataFrame:
    """Single and pairwise `col == value` conditions whose fraud rate has a Wilson lower
    bound above the base rate and lift >= min_lift. A screening aid for curating rules."""
    base = df[target].mean()
    y = df[target].to_numpy()
    masks = {(c, v): (df[c] == v).to_numpy() for c in cols for v in df[c].dropna().unique()}
    rows = []
    items = list(masks.items())
    combos: list[tuple] = [(k,) for k, _ in items]
    if max_order >= 2:
        combos += [(a, b) for (a, _), (b, _) in itertools.combinations(items, 2) if a[0] != b[0]]
    for combo in combos:
        m = masks[combo[0]] if len(combo) == 1 else masks[combo[0]] & masks[combo[1]]
        n = int(m.sum())
        if n < min_support:
            continue
        k = int(y[m].sum())
        lo, hi = wilson_ci(k, n)
        rate = k / n
        if lo > base and rate / base >= min_lift:
            rows.append({"condition": " & ".join(f"{c}={v}" for c, v in combo), "order": len(combo),
                         "n": n, "fraud": k, "rate": rate, "lo": lo, "hi": hi, "lift": rate / base})
    return pd.DataFrame(rows).sort_values(["lo", "n"], ascending=False).reset_index(drop=True)


def red_flag_stats(splits: Mapping[str, pd.DataFrame], rules: list[dict[str, Any]],
                   target: str = TARGET) -> pd.DataFrame:
    rows = []
    for rule in rules:
        for name, df in splits.items():
            m = condition_mask(df, rule["when"])
            n, k = int(m.sum()), int(df.loc[m, target].sum())
            lo, hi = wilson_ci(k, n)
            base = df[target].mean()
            rows.append({"id": rule["id"], "origin": name, "n": n, "support_pct": 100 * n / len(df),
                         "fraud": k, "rate": k / n if n else np.nan, "lo": lo, "hi": hi,
                         "lift": (k / n) / base if n else np.nan,
                         "recall": k / df[target].sum()})
    return pd.DataFrame(rows)


def write_red_flag_stats(stats_df: pd.DataFrame, path: Path | None = None) -> None:
    """Rewrite the `stats` of each red flag in policy_rules.yaml (the red_flags block is
    the last key in the file, so everything above it, comments included, is kept)."""
    import yaml

    path = Path(path or path_for("policy_rules"))
    text = path.read_text(encoding="utf-8")
    flags = load_policy_rules(path)["red_flags"]
    for flag in flags:
        sub = stats_df[stats_df["id"] == flag["id"]].set_index("origin")
        flag["stats"] = {
            origin: {"n": int(r["n"]), "support_pct": round(float(r["support_pct"]), 2),
                     "fraud_rate": round(float(r["rate"]), 4) if pd.notna(r["rate"]) else None,
                     "lift": round(float(r["lift"]), 2) if pd.notna(r["lift"]) else None,
                     "recall": round(float(r["recall"]), 4)}
            for origin, r in sub.iterrows()}
    head = text[: text.index("\nred_flags:") + 1]
    body = yaml.safe_dump({"red_flags": flags}, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(head + body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_rate_panel(ax, splits: Mapping[str, pd.DataFrame], col: str, base_rate: float) -> None:
    """Fraud rate per category with Wilson CIs, one dot per origin; n for the first origin."""
    names = list(splits)
    tables = {n: rate_table(splits[n], col) for n in names}
    order = value_order(col, set().union(*[set(t.index) for t in tables.values()]))
    y = np.arange(len(order))
    offsets = np.linspace(-0.18, 0.18, len(names)) if len(names) > 1 else [0.0]
    for off, name in zip(offsets, names):
        t = tables[name].reindex(order)
        ax.errorbar(100 * t["rate"], y + off,
                    xerr=np.clip([100 * (t["rate"] - t["lo"]), 100 * (t["hi"] - t["rate"])], 0, None),
                    fmt="o", ms=4, lw=1.2, color=ORIGIN_COLORS.get(name, MUTED),
                    label=ORIGIN_LABELS.get(name, name), capsize=0)
    first = tables[names[0]].reindex(order)
    labels = [f"{v}  (n={int(n):,})" if pd.notna(n) else str(v) for v, n in zip(order, first["n"])]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(100 * base_rate, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Fraud rate (%)")
    ax.set_title(col, fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    style_axes(ax)


def plot_rate_grid(splits, cols: list[str], base_rate: float, title: str, path: Path,
                   ncols: int = 2) -> Path:
    nrows = int(np.ceil(len(cols) / ncols))
    heights = []
    for i in range(nrows):
        row_cols = cols[i * ncols:(i + 1) * ncols]
        heights.append(max(splits[next(iter(splits))][c].nunique() for c in row_cols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, sum(0.32 * h + 1.0 for h in heights)),
                             gridspec_kw={"height_ratios": [0.32 * h + 1.0 for h in heights]},
                             squeeze=False)
    for ax, col in zip(axes.flat, cols):
        plot_rate_panel(ax, splits, col, base_rate)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=len(labels), fontsize=9)
    fig.suptitle(f"{title}: fraud rate by category (95% Wilson CI; dashed = real-train base rate)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save(fig, path)


def plot_heatmap(ax, matrix: pd.DataFrame, cmap, norm=None, annot: pd.DataFrame | None = None,
                 fmt: str = "{:.2f}", vmin=None, vmax=None, annot_size: int = 7):
    im = ax.imshow(matrix.values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=7)
    if annot is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                v = annot.iloc[i, j]
                if isinstance(v, str) or pd.notna(v):
                    val = matrix.iloc[i, j]
                    lo, hi = (vmin, vmax) if norm is None else (norm.vmin, norm.vmax)
                    dark = pd.notna(val) and hi is not None and (val - lo) / (hi - lo + 1e-12) > 0.6
                    ax.text(j, i, v if isinstance(v, str) else fmt.format(v), ha="center",
                            va="center", fontsize=annot_size, color="white" if dark else INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=INK_2)
    return im


def plot_interaction(df: pd.DataFrame, row: str, col: str, path: Path, title: str,
                     min_n: int = 30) -> Path:
    """Fraud-rate heatmap with n per cell. Cells with n < min_n are greyed out (rate and n
    still printed) so a handful of claims cannot dominate the colour scale."""
    t = interaction_table(df, row, col)
    rate = t.pivot(index=row, columns=col, values="rate")
    n = t.pivot(index=row, columns=col, values="n")
    rate = rate.reindex(index=value_order(row, rate.index), columns=value_order(col, rate.columns))
    n = n.reindex_like(rate)
    shown = rate.where(n >= min_n)
    annot = pd.DataFrame(
        [[f"{100 * r:.1f}%\nn={int(k):,}" if pd.notna(r) else "" for r, k in zip(rr, nn)]
         for rr, nn in zip(rate.values, n.values)], index=rate.index, columns=rate.columns)
    fig, ax = plt.subplots(figsize=(1.25 * rate.shape[1] + 2.5, 0.55 * rate.shape[0] + 1.6))
    vmax = max(0.15, float(np.nanmax(shown.values)) if shown.notna().any().any() else 0.15)
    small = (n < min_n) & n.notna()
    ax.imshow(np.where(small, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("g", [GRID, GRID]),
              vmin=0, vmax=1, aspect="auto")
    im = plot_heatmap(ax, shown, SEQ_BLUE, annot=annot.where(~small, ""), vmin=0, vmax=vmax)
    for i, j in zip(*np.nonzero(small.values)):
        ax.text(j, i, annot.iloc[i, j], ha="center", va="center", fontsize=7, color=MUTED)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Fraud rate", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    ax.set_title(title + f" — grey: n < {min_n}", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    return save(fig, path)


def plot_dots(ax, labels: list[str], series: Mapping[str, tuple], xlabel: str,
              ref: float | None = None, colors: Mapping[str, str] | None = None) -> None:
    """Horizontal dot plot: series name -> (values, lo, hi); lo/hi may be None."""
    colors = colors or {}
    palette = list(ORIGIN_COLORS.values())
    y = np.arange(len(labels))
    offsets = np.linspace(-0.18, 0.18, len(series)) if len(series) > 1 else [0.0]
    for i, (off, (name, (vals, lo, hi))) in enumerate(zip(offsets, series.items())):
        vals = np.asarray(vals, dtype=float)
        # Percentile bootstrap CIs can sit slightly off the point estimate; clip at 0.
        err = None if lo is None else np.clip([vals - np.asarray(lo, float),
                                               np.asarray(hi, float) - vals], 0, None)
        ax.errorbar(vals, y + off, xerr=err, fmt="o", ms=5, lw=1.2, capsize=0,
                    color=colors.get(name, palette[i % len(palette)]), label=name)
    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls="--")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=GRID, lw=0.6)
    style_axes(ax)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_ecdf(ax, samples: Mapping[str, np.ndarray], xlabel: str,
              colors: Mapping[str, str] | None = None) -> None:
    colors = colors or {}
    palette = list(ORIGIN_COLORS.values())
    for i, (name, x) in enumerate(samples.items()):
        x = np.sort(np.asarray(x))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), lw=2,
                color=colors.get(name, palette[i % len(palette)]), label=f"{name} (n={len(x):,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative share")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
