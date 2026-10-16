"""Regenerate the Step B report assets from the frozen artifacts in models/.

    python scripts/regenerate_model_report.py

Writes docs/report/tables/02_*.csv and docs/report/figures/02_*.png:
thresholds, test metrics with CIs vs baselines, lift and reliability, SHAP global /
dependence / local examples, error analysis and fairness.

This reproduces the B200 run's single test evaluation with the same frozen model. It makes
no modelling choices, and it stops if the recomputed test metrics differ from those
recorded in models/model_card.json.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ml import eda  # noqa: E402
from src.ml.predict import load_artifacts  # noqa: E402
from src.ml.train import TrainingRun  # noqa: E402

TABLES = ROOT / "docs" / "report" / "tables"
FIGS = ROOT / "docs" / "report" / "figures"


def main() -> None:
    art = load_artifacts()
    card = art.card
    if card.get("fast_mode"):
        sys.exit("models/ holds FAST_MODE placeholder artifacts; nothing to report.")
    work = ROOT / "outputs" / "report_regen"          # git-ignored scratch for TrainingRun
    if work.exists():
        shutil.rmtree(work)
    run = TrainingRun(work, fast=False, device="cpu", seeds=[42, 43, 44], n_boot=2000, log=lambda m: None)
    run.final = {"family": card["family"], "params": {}, "preprocessor": art.preprocessor, "model": art.model,
                 "thresholds": card["thresholds"], "risk_bands": card["risk_bands"]}

    ev = run.evaluate("real_test")
    rec = {m["metric"]: m["value"] for m in card["evaluation"]["metrics"]}
    now = ev["metrics"].set_index("metric")["value"].to_dict()
    for k in ("roc_auc", "pr_auc", "precision", "recall", "fpr", "tp", "fp", "fn", "tn"):
        if abs(float(now[k]) - float(rec[k])) > 1e-9:
            sys.exit(f"Recomputed test {k} = {now[k]} differs from the model card ({rec[k]}); stopping.")
    print("test metrics reproduce the model card exactly")
    sh = run.explain()
    err = run.error_analysis()
    fair = run.fairness()

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = {"02_thresholds.csv": pd.DataFrame(card["thresholds"]).T,
           "02_test_metrics.csv": ev["metrics"], "02_test_metrics_capacity_top10.csv": ev["metrics_capacity"],
           "02_test_comparison.csv": ev["comparison"], "02_test_lift.csv": ev["lift"],
           "02_test_reliability.csv": ev["reliability"], "02_shap_global_importance.csv": sh["importance"].rename("mean_abs_shap"),
           "02_shap_local_examples.csv": sh["local"], "02_error_analysis.csv": err, "02_fairness.csv": fair}
    for name, df in out.items():
        df.to_csv(TABLES / name, index=not isinstance(df.index, pd.RangeIndex))

    # Lift + cumulative gains
    lt = ev["lift"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(lt.decile, lt.lift, color=eda.ORIGIN_COLORS["real_train"], width=0.7)
    for x, v in zip(lt.decile, lt.lift):
        axes[0].text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color=eda.INK_2)
    axes[0].axhline(1, color=eda.MUTED, ls="--", lw=1)
    axes[0].set_xlabel("Score decile (1 = highest)"); axes[0].set_ylabel("Lift over base rate")
    axes[0].set_title("Lift by decile (real test)", loc="left", fontsize=10); eda.style_axes(axes[0])
    axes[1].plot(np.r_[0, lt.decile / 10], np.r_[0, lt.cumulative_recall], "o-", lw=2,
                 color=eda.ORIGIN_COLORS["real_train"], label="model")
    axes[1].plot([0, 1], [0, 1], color=eda.MUTED, ls="--", lw=1, label="random")
    axes[1].set_xlabel("Share of claims investigated"); axes[1].set_ylabel("Share of frauds caught")
    axes[1].legend(frameon=False, fontsize=8); axes[1].grid(color=eda.GRID, lw=0.6)
    axes[1].set_title("Cumulative gains (real test)", loc="left", fontsize=10); eda.style_axes(axes[1])
    eda.save(fig, FIGS / "02_lift.png")

    # Reliability on test
    rel = ev["reliability"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    lim = max(rel.mean_pred.max(), rel.observed.max()) * 1.1
    ax.plot([0, lim], [0, lim], color=eda.MUTED, ls="--", lw=1)
    ax.errorbar(rel.mean_pred, rel.observed, yerr=np.clip([rel.observed - rel.lo, rel.hi - rel.observed], 0, None),
                fmt="o-", color=eda.ORIGIN_COLORS["real_train"], lw=2, ms=5, capsize=0)
    ax.set_xlabel("Mean predicted fraud probability"); ax.set_ylabel("Observed fraud rate (95% CI)")
    ax.set_title(f"Reliability on real test ({card['calibration']['method']} calibration)", loc="left", fontsize=10)
    ax.grid(color=eda.GRID, lw=0.6); eda.style_axes(ax)
    eda.save(fig, FIGS / "02_reliability.png")

    # SHAP global
    imp = sh["importance"].head(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    eda.plot_dots(ax, imp.index.tolist(), {"mean |SHAP|": (imp.values, None, None)}, "Mean |SHAP| (log-odds)")
    ax.set_title("Global feature importance (SHAP, real test)", loc="left", fontsize=10)
    eda.save(fig, FIGS / "02_shap_global.png")

    # SHAP dependence for the top 4 model columns
    X = np.asarray(sh["X"], dtype=float)
    names = sh["names"]
    mean_abs = np.abs(sh["values"]).mean(axis=0)
    top = [names[i] for i in np.argsort(-mean_abs)[:4]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    rng = np.random.default_rng(0)
    for ax, fname in zip(axes, top):
        j = names.index(fname)
        ax.scatter(X[:, j] + rng.normal(0, 0.03, len(X)), sh["values"][:, j], s=6, alpha=0.35,
                   color=eda.ORIGIN_COLORS["real_train"])
        ax.axhline(0, color=eda.MUTED, lw=1); ax.set_xlabel(fname); ax.set_ylabel("SHAP (log-odds)")
        eda.style_axes(ax)
    fig.suptitle("SHAP dependence: top 4 model columns (real test)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); eda.save(fig, FIGS / "02_shap_dependence.png")

    # Fairness
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t = fair[fair.n >= 30]
    labels = (t.attribute + ": " + t.group.astype(str) + "  (n=" + t.n.astype(str) + ")").tolist()
    for ax, metric in zip(axes, ["recall", "fpr", "flag_rate"]):
        eda.plot_dots(ax, labels, {metric: (t[metric], None, None)}, metric)
        ax.set_title(f"{metric} by group (groups with n ≥ 30)", loc="left", fontsize=10)
    fig.tight_layout(); eda.save(fig, FIGS / "02_fairness.png")
    shutil.rmtree(work, ignore_errors=True)
    print("wrote", ", ".join(sorted(out)), "and figures 02_*.png")


if __name__ == "__main__":
    main()
