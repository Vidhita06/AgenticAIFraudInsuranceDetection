"""CPU MLP vs the B200 LightGBM on REAL VALIDATION only (no test data).

Usage: python scripts/compare_mlp_cpu.py [out_dir]   (results: docs/report/02_model_report.md section 5)
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score as ap, roc_auc_score as roc
from src.ml.train import make_training_data, load_data, fit_model, tune, stable_choice, build_preprocessor
from src.ml.predict import load_artifacts
from src.ml.evaluate import bootstrap_ci, paired_bootstrap_diff
OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/mlp_cpu"
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(4)
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
aug, val = load_data()
data = make_training_data(aug, val, 1.0, 0.1)          # same mix as the B200 final model
y = data.y_val
# LightGBM (B200 artifacts): per-member raw scores and the calibrated ensemble
art = load_artifacts()
Xv = art.preprocessor.transform(val)
members = [m.predict_proba(Xv)[:, 1] for m in art.model.base.models]
lgb_ens = art.model.base.predict_proba(Xv)[:, 1]
res = {"lightgbm_member_pr_auc": [ap(y, s) for s in members], "lightgbm_ensemble_pr_auc": ap(y, lgb_ens),
       "lightgbm_ensemble_roc_auc": roc(y, lgb_ens)}
log(f"LightGBM members {np.round(res['lightgbm_member_pr_auc'], 4)}, ensemble {res['lightgbm_ensemble_pr_auc']:.4f}")
pre = build_preprocessor("embedding").fit(data.train)
default_scores = []
for s in (42, 43, 44):
    fm = fit_model("mlp", data, device="cpu", seed=s, preprocessor=pre)
    default_scores.append(fm.val_scores)
    log(f"MLP default seed {s}: PR-AUC {ap(y, fm.val_scores):.4f}  best epoch {fm.extra['best_epoch']}  {fm.fit_seconds:.0f}s")
res["mlp_default_member_pr_auc"] = [ap(y, s) for s in default_scores]
res["mlp_default_ensemble_pr_auc"] = ap(y, np.mean(default_scores, axis=0))
study, _ = tune("mlp", data, n_trials=8, device="cpu", preprocessor=pre, log=log,
                storage=f"sqlite:///{OUT}/optuna_mlp.db", study_name="mlp_cpu")
study.trials_dataframe().to_csv(f"{OUT}/mlp_trials.csv", index=False)
choice = stable_choice("mlp", study, data, "cpu", top_k=2, seeds=(42, 43, 44), preprocessor=pre)
choice["table"].to_csv(f"{OUT}/mlp_top_trials.csv", index=False)
best = choice["params"]
tuned_scores = [fit_model("mlp", data, best, device="cpu", seed=s, preprocessor=pre).val_scores for s in (42, 43, 44)]
res["mlp_tuned_params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()}
res["mlp_tuned_member_pr_auc"] = [ap(y, s) for s in tuned_scores]
res["mlp_tuned_ensemble_pr_auc"] = ap(y, np.mean(tuned_scores, axis=0))
best_mlp = max([("default", np.mean(default_scores, axis=0)), ("tuned", np.mean(tuned_scores, axis=0))],
               key=lambda t: ap(y, t[1]))
res["best_mlp"] = best_mlp[0]
res["mlp_best_ensemble_ci"] = bootstrap_ci(y, best_mlp[1], ap, n_boot=2000)
res["lightgbm_ensemble_ci"] = bootstrap_ci(y, lgb_ens, ap, n_boot=2000)
res["diff_mlp_minus_lgbm"] = paired_bootstrap_diff(y, best_mlp[1], lgb_ens, ap, n_boot=2000)
res["mlp_best_roc_auc"] = roc(y, best_mlp[1])
np.save(f"{OUT}/mlp_best_val_scores.npy", best_mlp[1])
json.dump(res, open(f"{OUT}/mlp_vs_lgbm.json", "w"), indent=2, default=float)
log("DONE " + json.dumps({k: res[k] for k in ("lightgbm_ensemble_pr_auc", "mlp_default_ensemble_pr_auc",
                                                "mlp_tuned_ensemble_pr_auc", "diff_mlp_minus_lgbm")}, default=float))
