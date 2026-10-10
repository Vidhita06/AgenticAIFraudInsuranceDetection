"""Model training: data mixes, the model zoo (LogReg, RF, XGBoost, LightGBM, CatBoost,
PyTorch MLP), Optuna tuning, calibration and artifact export.

Used by notebooks/02_model_training.ipynb (B200) and scripts/train_model.py (headless).
Model selection always uses real validation; synthetic rows only enter training.
GPU: XGBoost device="cuda", CatBoost task_type="GPU", MLP bf16 autocast on CUDA.
LightGBM, Random Forest and Logistic Regression run on CPU.
"""
from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE
from src.ml.data import ORIGIN_COL, TARGET, load_augmented_train, load_real_validation
from src.ml.preprocess import build_preprocessor, catboost_cat_features, embedding_layout

warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")

PREPROCESSOR_KIND = {"logreg": "linear", "rf": "tree", "xgboost": "tree", "lightgbm": "tree",
                     "catboost": "catboost", "mlp": "embedding"}
GPU_MODELS = {"xgboost", "catboost", "mlp"}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def environment_report() -> dict[str, Any]:
    """Library versions and GPU facts recorded in the model card."""
    import platform
    import sklearn
    rep: dict[str, Any] = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("xgboost", "lightgbm", "catboost", "shap", "optuna", "torch", "joblib"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            rep[mod] = None
    try:
        import torch
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rep["gpu"] = torch.cuda.get_device_name(0)
            rep["cuda"] = torch.version.cuda
            rep["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            rep["arch_list"] = torch.cuda.get_arch_list()
            rep["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        rep["cuda_available"] = False
    return rep


def check_blackwell(rep: Mapping[str, Any], require_gpu: bool) -> None:
    """Fail early if a GPU run cannot use the B200 (sm_100)."""
    if not rep.get("cuda_available"):
        if require_gpu:
            raise RuntimeError("No CUDA GPU visible. Set FAST_MODE=True for a CPU smoke test, "
                               "or check the Jupyter kernel is the GPU one.")
        return
    major = int(str(rep["compute_capability"]).split(".")[0])
    archs = rep.get("arch_list") or []
    if major >= 10 and not any(a.startswith(("sm_100", "sm_10", "sm_12", "compute_10")) for a in archs):
        raise RuntimeError(
            f"GPU {rep['gpu']} is compute capability {rep['compute_capability']} (Blackwell) but "
            f"this torch build only supports {archs}. Use the server's preinstalled CUDA 12.8+ "
            "PyTorch (do not pip-install torch).")
    if not rep.get("bf16_supported"):
        raise RuntimeError("bf16 is not supported on this GPU/torch build.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class TrainingData:
    train: pd.DataFrame                  # raw claim columns + target + is_synthetic
    weight: np.ndarray                   # sample weight (1 for real, w for synthetic)
    val: pd.DataFrame
    description: str = ""

    @property
    def y(self) -> np.ndarray:
        return self.train[TARGET].to_numpy().astype(int)

    @property
    def y_val(self) -> np.ndarray:
        return self.val[TARGET].to_numpy().astype(int)


def make_training_data(aug: pd.DataFrame, val: pd.DataFrame, synthetic_fraction: float = 1.0,
                       synthetic_weight: float = 1.0, include_real: bool = True,
                       max_rows: int | None = None, seed: int = RANDOM_STATE) -> TrainingData:
    """Real train (+ a fraction of synthetic rows at a sample weight). `max_rows` subsamples
    the result stratified by origin and label (FAST_MODE)."""
    rng = np.random.default_rng(seed)
    real = aug[aug[ORIGIN_COL] == 0]
    syn = aug[aug[ORIGIN_COL] == 1]
    parts = [real] if include_real else []
    if synthetic_fraction > 0:
        n = int(round(synthetic_fraction * len(syn)))
        parts.append(syn.iloc[np.sort(rng.choice(len(syn), n, replace=False))])
    train = pd.concat(parts, ignore_index=True)
    if max_rows and len(train) > max_rows:
        strata = train[ORIGIN_COL].astype(str) + "_" + train[TARGET].astype(str)
        frac = max_rows / len(train)
        train = (train.groupby(strata, group_keys=False)
                 .apply(lambda g: g.sample(max(1, int(round(frac * len(g)))), random_state=seed))
                 .reset_index(drop=True))
    w = np.where(train[ORIGIN_COL].to_numpy() == 1, synthetic_weight, 1.0)
    desc = (f"{'real' if include_real else 'no real'} + {synthetic_fraction:.0%} synthetic "
            f"(w={synthetic_weight})")
    return TrainingData(train, w, val.reset_index(drop=True), desc)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_augmented_train(), load_real_validation()


# ---------------------------------------------------------------------------
# PyTorch MLP with entity embeddings
# ---------------------------------------------------------------------------

def _build_mlp_net(n_cat: int, cards: list[int], n_num: int, hidden: tuple[int, ...],
                   dropout: float, emb_dim_max: int):
    import torch
    from torch import nn

    class MLPNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_cat = n_cat
            self.embs = nn.ModuleList(
                [nn.Embedding(c + 1, min(emb_dim_max, (c + 1) // 2 + 1)) for c in cards])
            d_in = sum(e.embedding_dim for e in self.embs) + n_num
            layers: list[nn.Module] = []
            for h in hidden:
                layers += [nn.Linear(d_in, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
                d_in = h
            layers.append(nn.Linear(d_in, 1))
            self.mlp = nn.Sequential(*layers)

        def forward(self, x):
            codes = (x[:, : self.n_cat].long() + 1).clamp(min=0)
            parts = [emb(codes[:, i].clamp(max=emb.num_embeddings - 1))
                     for i, emb in enumerate(self.embs)]
            parts.append(x[:, self.n_cat:].float())
            return self.mlp(torch.cat(parts, dim=1)).squeeze(1)

    return MLPNet()


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Entity-embedding MLP. Input = output of the "embedding" preprocessor (first n_cat
    columns are integer codes). Weighted BCE or focal loss, bf16 autocast on CUDA, early
    stopping on validation PR-AUC. Pickles as a CPU state_dict (torch needed to load)."""

    def __init__(self, n_cat: int = 0, cards: tuple = (), hidden: tuple = (256, 128),
                 dropout: float = 0.2, emb_dim_max: int = 16, lr: float = 2e-3,
                 weight_decay: float = 1e-5, batch_size: int = 4096, max_epochs: int = 60,
                 patience: int = 8, loss: str = "bce", focal_gamma: float = 2.0,
                 pos_weight: float | None = None, device: str = "cpu", use_bf16: bool = True,
                 use_compile: bool = False, seed: int = RANDOM_STATE, verbose: bool = False):
        self.n_cat = n_cat
        self.cards = cards
        self.hidden = hidden
        self.dropout = dropout
        self.emb_dim_max = emb_dim_max
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.device = device
        self.use_bf16 = use_bf16
        self.use_compile = use_compile
        self.seed = seed
        self.verbose = verbose

    def _net(self):
        return _build_mlp_net(self.n_cat, list(self.cards), self.n_features_in_ - self.n_cat,
                              tuple(self.hidden), self.dropout, self.emb_dim_max)

    def fit(self, X, y, sample_weight=None, eval_set=None):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        w = np.ones(len(y), np.float32) if sample_weight is None else np.asarray(sample_weight, np.float32)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        dev = torch.device(self.device)
        net = self._net().to(dev)
        model = net
        if self.use_compile and self.device == "cuda":
            try:
                model = torch.compile(net)
            except Exception:  # noqa: BLE001
                model = net
        pos_w = self.pos_weight or float((w * (1 - y)).sum() / max((w * y).sum(), 1e-9))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        Xt, yt, wt = (torch.from_numpy(a).to(dev) for a in (X, y, w))
        amp = self.use_bf16 and self.device == "cuda"
        best, best_state, bad = -1.0, None, 0
        self.history_ = []
        n = len(y)
        for epoch in range(self.max_epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, self.batch_size):
                idx = perm[s:s + self.batch_size]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(Xt[idx])
                logits = logits.float()
                target = yt[idx]
                bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="none",
                    pos_weight=torch.tensor(pos_w, device=dev))
                if self.loss == "focal":
                    pt = torch.exp(-torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, target, reduction="none"))
                    bce = bce * (1 - pt) ** self.focal_gamma
                loss = (bce * wt[idx]).sum() / wt[idx].sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            if eval_set is not None:
                self.net_ = net
                pv = self.predict_proba(eval_set[0])[:, 1]
                score = average_precision_score(eval_set[1], pv)
                self.history_.append(score)
                if score > best + 1e-5:
                    best, bad = score, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_ = net
        self.best_score_ = best
        self.best_epoch_ = int(np.argmax(self.history_)) + 1 if self.history_ else self.max_epochs
        return self

    def predict_proba(self, X):
        import torch
        X = np.asarray(X, dtype=np.float32)
        dev = next(self.net_.parameters()).device
        self.net_.eval()
        out = []
        amp = self.use_bf16 and dev.type == "cuda"
        with torch.no_grad():
            for s in range(0, len(X), 65536):
                xb = torch.from_numpy(X[s:s + 65536]).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                    logit = self.net_(xb)
                out.append(torch.sigmoid(logit.float()).cpu().numpy())
        p = np.concatenate(out)
        return np.column_stack([1 - p, p])

    def to_cpu(self):
        self.net_ = self.net_.to("cpu")
        self.device = "cpu"
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            state["net_state_"] = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
            del state["net_"]
        return state

    def __setstate__(self, state):
        net_state = state.pop("net_state_", None)
        self.__dict__.update(state)
        if net_state is not None:
            self.net_ = self._net()
            self.net_.load_state_dict(net_state)
            self.net_.eval()
            self.device = "cpu"


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 0.5, "max_iter": 3000},
    "rf": {"n_estimators": 500, "min_samples_leaf": 20, "max_features": "sqrt"},
    "xgboost": {"n_estimators": 3000, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0},
    "lightgbm": {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 31,
                 "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1,
                 "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "catboost": {"iterations": 3000, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
    "mlp": {"hidden": (256, 128), "dropout": 0.2, "lr": 2e-3, "batch_size": 4096,
            "max_epochs": 60, "patience": 8, "loss": "bce"},
}
FAST_OVERRIDES = {"rf": {"n_estimators": 150}, "xgboost": {"n_estimators": 600},
                  "lightgbm": {"n_estimators": 600}, "catboost": {"iterations": 600},
                  "mlp": {"max_epochs": 8, "patience": 3, "batch_size": 1024}}


@dataclass
class FittedModel:
    name: str
    preprocessor: Any
    model: Any
    params: dict[str, Any]
    data_description: str
    val_scores: np.ndarray
    fit_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)

    def predict_proba_raw(self, raw: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.preprocessor.transform(raw))[:, 1]


def _pos_weight(y, w) -> float:
    return float((w * (y == 0)).sum() / max((w * (y == 1)).sum(), 1e-9))


def fit_model(name: str, data: TrainingData, params: Mapping[str, Any] | None = None,
              device: str = "cpu", seed: int = RANDOM_STATE, preprocessor=None,
              fast: bool = False) -> FittedModel:
    """Fit one model family on `data` and score real validation. The preprocessor is fitted
    on the training rows only (pass a fitted one to reuse it)."""
    params = {**DEFAULT_PARAMS[name], **(FAST_OVERRIDES.get(name, {}) if fast else {}),
              **(params or {})}
    t0 = time.time()
    pre = preprocessor or build_preprocessor(PREPROCESSOR_KIND[name]).fit(data.train)
    X, Xv = pre.transform(data.train), pre.transform(data.val)
    y, w, yv = data.y, data.weight, data.y_val
    spw = _pos_weight(y, w)
    extra: dict[str, Any] = {}

    if name == "logreg":
        model = LogisticRegression(class_weight="balanced", random_state=seed, **params)
        model.fit(X, y, sample_weight=w)
    elif name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                       random_state=seed, **params)
        model.fit(X, y, sample_weight=w)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(tree_method="hist", device=device, scale_pos_weight=spw,
                              eval_metric="aucpr", early_stopping_rounds=150,
                              random_state=seed, n_jobs=-1, **params)
        model.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
        extra["best_iteration"] = int(model.best_iteration)
    elif name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=-1,
                                   verbose=-1, **params)
        model.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)])
        extra["best_iteration"] = int(model.best_iteration_ or params["n_estimators"])
    elif name == "catboost":
        from catboost import CatBoostClassifier, Pool
        cats = catboost_cat_features(pre)
        gpu = device == "cuda"
        train_pool = Pool(X, y, cat_features=cats, weight=w)
        val_pool = Pool(Xv, yv, cat_features=cats)
        # PRAUC is evaluated on the CPU during GPU training; fall back to AUC if refused.
        for metric in ("PRAUC", "AUC"):
            model = CatBoostClassifier(
                task_type="GPU" if gpu else "CPU", eval_metric=metric, scale_pos_weight=spw,
                od_type="Iter", od_wait=150, use_best_model=True, random_seed=seed,
                allow_writing_files=False,
                verbose=False, **({"thread_count": -1} if not gpu else {}), **params)
            try:
                model.fit(train_pool, eval_set=val_pool)
                extra["eval_metric"] = metric
                break
            except Exception:  # noqa: BLE001
                if metric == "AUC":
                    raise
        extra["best_iteration"] = int(model.get_best_iteration() or 0)
    elif name == "mlp":
        n_cat, cards = embedding_layout(pre)
        model = TorchMLPClassifier(n_cat=n_cat, cards=tuple(cards), device=device, seed=seed,
                                   **params)
        model.fit(X, y, sample_weight=w, eval_set=(Xv, yv))
        extra["best_epoch"] = model.best_epoch_
    else:
        raise ValueError(name)
    pv = model.predict_proba(Xv)[:, 1]
    return FittedModel(name, pre, model, dict(params), data.description, pv, time.time() - t0, extra)


# ---------------------------------------------------------------------------
# Optuna search spaces (objective: real-validation PR-AUC)
# ---------------------------------------------------------------------------

def suggest_params(name: str, trial) -> dict[str, Any]:
    if name == "xgboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 5, log=True)}
    if name == "lightgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True)}
    if name == "catboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 3)}
    if name == "mlp":
        width = trial.suggest_categorical("width", [128, 256, 512, 1024])
        depth = trial.suggest_int("depth", 1, 3)
        return {"hidden": tuple(max(32, width // (2 ** i)) for i in range(depth)),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "lr": trial.suggest_float("lr", 3e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [1024, 4096, 16384]),
                "loss": trial.suggest_categorical("loss", ["bce", "focal"])}
    raise ValueError(f"no search space for {name}")


def tune(name: str, data: TrainingData, n_trials: int, device: str, fast: bool = False,
         timeout: float | None = None, preprocessor=None, log: Callable[[str], None] = print,
         storage: str | None = None, study_name: str | None = None):
    """TPE search maximising real-validation PR-AUC (median pruner on the trial history).
    With `storage` (e.g. sqlite:///outputs/optuna.db) a disconnected run resumes."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pre = preprocessor or build_preprocessor(PREPROCESSOR_KIND[name]).fit(data.train)

    def objective(trial):
        fm = fit_model(name, data, suggest_params(name, trial), device=device, preprocessor=pre,
                       fast=fast)
        score = average_precision_score(data.y_val, fm.val_scores)
        trial.set_user_attr("best_iteration", fm.extra.get("best_iteration") or fm.extra.get("best_epoch"))
        trial.set_user_attr("params", {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in fm.params.items()})
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        storage=storage, study_name=study_name or f"tune_{name}", load_if_exists=storage is not None)
    remaining = n_trials - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=timeout,
                       callbacks=[lambda s, t: log(f"  {name} trial {t.number}: PR-AUC "
                                                   f"{t.value if t.value is not None else float('nan'):.4f} "
                                                   f"(best {s.best_value:.4f})")])
    return study, pre


def stable_choice(name: str, study, data: TrainingData, device: str, top_k: int = 5,
                  seeds=(42, 43, 44), fast: bool = False, preprocessor=None) -> dict[str, Any]:
    """Re-fit the top-k trials with several seeds and pick the best MEAN validation PR-AUC
    (guards against a lucky single trial on only ~139 validation frauds)."""
    done = [t for t in study.trials if t.value is not None]
    top = sorted(done, key=lambda t: t.value, reverse=True)[:top_k]
    rows = []
    for t in top:
        params = {k: (tuple(v) if k == "hidden" else v) for k, v in t.user_attrs["params"].items()}
        scores = []
        for s in seeds:
            fm = fit_model(name, data, params, device=device, seed=s, preprocessor=preprocessor,
                           fast=fast)
            scores.append(average_precision_score(data.y_val, fm.val_scores))
        rows.append({"trial": t.number, "single_run": t.value, "mean": float(np.mean(scores)),
                     "sd": float(np.std(scores)), "params": params})
    table = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return {"table": table, "params": table.loc[0, "params"]}


# ---------------------------------------------------------------------------
# Seed ensemble + calibration
# ---------------------------------------------------------------------------

class SeedEnsemble(BaseEstimator, ClassifierMixin):
    """Average of the same model fitted with different seeds (reduces variance)."""

    def __init__(self, models: list | None = None):
        self.models = models or []
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - p, p])


class CalibratedModel(BaseEstimator, ClassifierMixin):
    """Base model + monotone calibrator (isotonic, or Platt on the logit).
    `base` is kept so SHAP can explain the uncalibrated model."""

    def __init__(self, base=None, method: str = "sigmoid", calibrator=None):
        self.base = base
        self.method = method
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit_calibrator(self, raw_scores, y):
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_scores, y)
        else:
            self.calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw_scores), y)
        return self

    def calibrate(self, raw_scores):
        if self.method == "isotonic":
            return self.calibrator.predict(raw_scores)
        return self.calibrator.predict_proba(self._logit(raw_scores))[:, 1]

    def predict_proba(self, X):
        p = self.calibrate(self.base.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def crossfit_calibration(raw_scores: np.ndarray, y: np.ndarray, n_splits: int = 5,
                         seed: int = RANDOM_STATE) -> dict[str, Any]:
    """Out-of-fold calibrated validation scores for isotonic and Platt; chooses the method
    with the lower out-of-fold Brier score. Thresholds are then chosen on the OOF scores,
    so neither calibration nor thresholds are fitted on the claims they are judged on."""
    out = {}
    for method in ("sigmoid", "isotonic"):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(raw_scores, y):
            cm = CalibratedModel(method=method).fit_calibrator(raw_scores[tr], y[tr])
            oof[te] = cm.calibrate(raw_scores[te])
        out[method] = {"oof": oof, "brier": float(brier_score_loss(y, oof)),
                       "pr_auc": float(average_precision_score(y, oof))}
    out["uncalibrated_brier"] = float(brier_score_loss(y, raw_scores))
    out["method"] = min(("sigmoid", "isotonic"), key=lambda m: out[m]["brier"])
    return out


# ---------------------------------------------------------------------------
# Checkpointing and export
# ---------------------------------------------------------------------------

def to_cpu_inference(model):
    """Switch GPU-trained models to CPU inference before saving."""
    if isinstance(model, (CalibratedModel,)):
        model.base = to_cpu_inference(model.base)
        return model
    if isinstance(model, SeedEnsemble):
        model.models = [to_cpu_inference(m) for m in model.models]
        return model
    cls = type(model).__name__
    if cls == "XGBClassifier":
        model.set_params(device="cpu")
    elif cls == "TorchMLPClassifier":
        model.to_cpu()
    # CatBoost GPU-trained models predict on CPU as-is; sklearn/LightGBM are CPU already.
    return model


def save_checkpoint(fm: FittedModel, directory: Path, metrics: Mapping[str, Any]) -> Path:
    import joblib
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fm.name}.joblib"
    joblib.dump({"fitted": fm}, path)
    (directory / f"{fm.name}.json").write_text(json.dumps(
        {"name": fm.name, "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in fm.params.items()},
         "data": fm.data_description, "fit_seconds": fm.fit_seconds, "extra": fm.extra,
         "metrics": metrics}, indent=2, default=str))
    return path


def load_checkpoint(name: str, directory: Path) -> FittedModel | None:
    import joblib
    path = directory / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)["fitted"]
    except Exception:  # noqa: BLE001
        return None


def clone_fitted(fm: FittedModel) -> FittedModel:
    return copy.deepcopy(fm)


# ---------------------------------------------------------------------------
# End-to-end run (shared by the B200 notebook and scripts/train_model.py)
# ---------------------------------------------------------------------------

TUNABLE = ("xgboost", "lightgbm", "catboost", "mlp")


class TrainingRun:
    """One training run writing everything under `out_dir`:
    checkpoints/ (one file per finished model), figures/, tables/, models/ (final CPU
    artifacts), run_log.csv and optuna.db. Each step is a method so the notebook can
    rerun sections; finished models are reloaded from checkpoints when `resume` is set."""

    def __init__(self, out_dir: str | Path, fast: bool = False, device: str | None = None,
                 seeds=(42, 43, 44), n_trials: int = 60, tune_top: int = 2, resume: bool = True,
                 n_boot: int = 1000, log: Callable[[str], None] = print):
        from src.config import input_hashes, load_config
        self.out = Path(out_dir)
        for sub in ("checkpoints", "figures", "tables", "models"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.fast = fast
        self.device = device or detect_device()
        self.seeds = list(seeds)[:1] if fast else list(seeds)
        self.n_trials = n_trials
        self.tune_top = tune_top
        self.resume = resume
        self.n_boot = 200 if fast else n_boot
        self.log = log
        self.cfg = load_config()
        self.max_rows = self.cfg["training"]["fast_mode_rows"] if fast else None
        self.hashes = input_hashes()
        self.aug, self.val = load_data()
        self.y_val = self.val[TARGET].to_numpy().astype(int)
        w = self.cfg["training"]["default_synthetic_weight"]
        self.mix = {"synthetic_fraction": 1.0, "synthetic_weight": w}
        self.results: dict[str, dict[str, Any]] = {}
        self.fitted: dict[str, FittedModel] = {}
        self.baseline_scores: dict[str, np.ndarray] = {}

    # -- helpers -----------------------------------------------------------
    def data(self, synthetic_fraction=None, synthetic_weight=None, include_real=True, seed=RANDOM_STATE):
        return make_training_data(
            self.aug, self.val,
            self.mix["synthetic_fraction"] if synthetic_fraction is None else synthetic_fraction,
            self.mix["synthetic_weight"] if synthetic_weight is None else synthetic_weight,
            include_real=include_real, max_rows=self.max_rows, seed=seed)

    def _record(self, key: str, family: str, scores: np.ndarray, data_desc: str,
                params: Mapping | None = None, seconds: float = 0.0, extra: Mapping | None = None):
        from src.ml.evaluate import bootstrap_ci, ranking_metrics
        from sklearn.metrics import average_precision_score as ap
        m = ranking_metrics(self.y_val, scores)
        pr = bootstrap_ci(self.y_val, scores, ap, n_boot=self.n_boot)
        row = {"model": key, "family": family, "data": data_desc, "pr_auc": m["pr_auc"],
               "pr_lo": pr[1], "pr_hi": pr[2], "roc_auc": m["roc_auc"],
               "precision_at_top_10pct": m["precision_at_top_10pct"],
               "recall_at_precision_0.3": m["recall_at_precision_0.3"], "fit_seconds": seconds}
        self.results[key] = row
        log_row = {**row, "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
                   "device": self.device, "fast_mode": self.fast,
                   "params": json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                         for k, v in (params or {}).items()}, default=str),
                   "extra": json.dumps(extra or {}, default=str),
                   "data_sha256": json.dumps({k: v[:12] for k, v in self.hashes.items()})}
        path = self.out / "run_log.csv"
        pd.DataFrame([log_row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        self.log(f"{key:<32} val PR-AUC {row['pr_auc']:.4f} [{pr[1]:.3f}, {pr[2]:.3f}]  "
                 f"ROC-AUC {row['roc_auc']:.4f}  ({seconds:.0f}s)")
        return row

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.results.values()).sort_values("pr_auc", ascending=False)

    # -- 1. baselines ------------------------------------------------------
    def baselines(self) -> pd.DataFrame:
        from src.ml.preprocess import ClaimFeatureBuilder
        self.baseline_scores["all_legit"] = np.zeros(len(self.val))
        self._record("baseline: all-legit", "constant", self.baseline_scores["all_legit"], "-")
        b = ClaimFeatureBuilder().fit(self.val)
        self.baseline_scores["rules_only"] = b.transform(self.val)["red_flag_score"].to_numpy()
        self._record("baseline: rules-only (red flags)", "rules",
                     self.baseline_scores["rules_only"], "config/policy_rules.yaml")
        fm = self._fit_or_load("logreg", self.data())
        self.baseline_scores["logreg"] = fm.val_scores
        return self.table()

    def _fit_or_load(self, name: str, data: TrainingData, key: str | None = None,
                     params=None, seed=RANDOM_STATE) -> FittedModel:
        key = key or name
        ck = self.out / "checkpoints"
        fm = load_checkpoint(key, ck) if self.resume else None
        if fm is None:
            fm = fit_model(name, data, params, device=self.device if name in GPU_MODELS else "cpu",
                           seed=seed, fast=self.fast)
            fm.name = key
            row = self._record(key, name, fm.val_scores, fm.data_description, fm.params,
                               fm.fit_seconds, fm.extra)
            save_checkpoint(fm, ck, row)
        else:
            self.log(f"{key:<32} loaded from checkpoint")
            self._record(key, name, fm.val_scores, fm.data_description, fm.params, fm.fit_seconds, fm.extra)
        self.fitted[key] = fm
        return fm

    # -- 2. data-mix experiment --------------------------------------------
    def data_mix(self, family: str = "xgboost") -> pd.DataFrame:
        tr = self.cfg["training"]
        fracs = [0.0, 0.25, 1.0] if self.fast else tr["synthetic_fractions"]
        weights = [1.0, 0.3] if self.fast else tr["synthetic_weights"]
        configs = [(f, 1.0, True) for f in fracs] + [(1.0, w, True) for w in weights if w != 1.0]
        configs.append((1.0, 1.0, False))   # synthetic only
        rows = []
        for frac, w, real in configs:
            scores = []
            for s in self.seeds[:2]:
                d = self.data(frac, w, include_real=real, seed=s)
                key = f"mix_{family}_f{frac}_w{w}_{'real' if real else 'synonly'}_s{s}"
                fm = self._fit_or_load(family, d, key=key, seed=s)
                scores.append(fm.val_scores)
            from sklearn.metrics import average_precision_score as ap
            per_seed = [ap(self.y_val, sc) for sc in scores]
            rows.append({"synthetic_fraction": frac, "synthetic_weight": w, "include_real": real,
                         "train_rows": len(d.train), "synthetic_rows": int((d.train[ORIGIN_COL] == 1).sum()),
                         "pr_auc_mean": float(np.mean(per_seed)), "pr_auc_sd": float(np.std(per_seed)),
                         "pr_auc_seed_avg_scores": float(ap(self.y_val, np.mean(scores, axis=0)))})
        tab = pd.DataFrame(rows)
        best = tab[tab.include_real].sort_values("pr_auc_mean", ascending=False).iloc[0]
        self.mix = {"synthetic_fraction": float(best.synthetic_fraction),
                    "synthetic_weight": float(best.synthetic_weight)}
        self.log(f"chosen data mix: {self.mix}")
        tab.to_csv(self.out / "tables" / "data_mix.csv", index=False)
        return tab

    # -- 3. model zoo --------------------------------------------------------
    def zoo(self, families=("rf", "xgboost", "lightgbm", "catboost", "mlp")) -> pd.DataFrame:
        for name in families:
            try:
                self._fit_or_load(name, self.data())
            except Exception as e:  # noqa: BLE001  (one family failing must not kill the run)
                self.log(f"!! {name} failed: {type(e).__name__}: {e}")
        t = self.table()
        t.to_csv(self.out / "tables" / "model_comparison.csv", index=False)
        return t

    # -- 4. tuning -----------------------------------------------------------
    def tune_best(self) -> pd.DataFrame:
        cands = [r for r in self.table().to_dict("records") if r["model"] in TUNABLE]
        cands = sorted(cands, key=lambda r: r["pr_auc"], reverse=True)[: self.tune_top]
        rows = []
        self.tuned: dict[str, dict[str, Any]] = {}
        for r in cands:
            name = r["model"]
            data = self.data()
            self.log(f"tuning {name}: {self.n_trials} trials")
            study, pre = tune(name, data, self.n_trials, self.device if name in GPU_MODELS else "cpu",
                              fast=self.fast, log=self.log,
                              storage=f"sqlite:///{(self.out / 'optuna.db').resolve()}",
                              study_name=f"{name}_{'fast' if self.fast else 'full'}")
            choice = stable_choice(name, study, data, self.device if name in GPU_MODELS else "cpu",
                                   top_k=2 if self.fast else 5, seeds=self.seeds, fast=self.fast,
                                   preprocessor=pre)
            choice["table"].to_csv(self.out / "tables" / f"tuning_{name}_top_trials.csv", index=False)
            study.trials_dataframe().to_csv(self.out / "tables" / f"tuning_{name}_trials.csv", index=False)
            self.tuned[name] = {"params": choice["params"], "mean_pr_auc": float(choice["table"]["mean"].iloc[0]),
                                "default_pr_auc": r["pr_auc"], "preprocessor": pre}
            rows.append({"family": name, "default_pr_auc": r["pr_auc"],
                         "best_single_trial": float(study.best_value),
                         "chosen_mean_pr_auc_over_seeds": self.tuned[name]["mean_pr_auc"],
                         "chosen_sd": float(choice["table"]["sd"].iloc[0]), "trials": len(study.trials)})
        t = pd.DataFrame(rows)
        t.to_csv(self.out / "tables" / "tuning_summary.csv", index=False)
        return t

    # -- 5. final model: seed ensemble + calibration -------------------------
    def finalize(self) -> dict[str, Any]:
        from src.ml.evaluate import cost_threshold, capacity_threshold, threshold_metrics
        options = []
        for name, t in getattr(self, "tuned", {}).items():
            use_tuned = t["mean_pr_auc"] >= t["default_pr_auc"]
            options.append((t["mean_pr_auc"] if use_tuned else t["default_pr_auc"], name,
                            t["params"] if use_tuned else {}, t["preprocessor"]))
        if not options:   # no tuning ran: best family from the zoo
            best = [r for r in self.table().to_dict("records") if r["model"] in PREPROCESSOR_KIND][0]
            options = [(best["pr_auc"], best["model"], {}, None)]
        _, name, params, pre = max(options, key=lambda o: o[0])
        data = self.data()
        members = []
        for s in self.seeds:
            fm = fit_model(name, data, params, device=self.device if name in GPU_MODELS else "cpu",
                           seed=s, preprocessor=pre, fast=self.fast)
            pre = fm.preprocessor
            members.append(fm.model)
            self.log(f"final {name} seed {s}: val PR-AUC {average_precision_score(self.y_val, fm.val_scores):.4f}")
        ensemble = SeedEnsemble(members)
        raw_val = ensemble.predict_proba(pre.transform(self.val))[:, 1]
        cal = crossfit_calibration(raw_val, self.y_val)
        model = CalibratedModel(ensemble, cal["method"]).fit_calibrator(raw_val, self.y_val)
        oof = cal[cal["method"]]["oof"]
        thresholds = {}
        for c in self.cfg["decision"]["cost_ratios"]:
            thresholds[f"cost_{c}to1"] = cost_threshold(self.y_val, oof, c)
        cap = self.cfg["decision"]["capacity_top_fraction"]
        t_cap = capacity_threshold(oof, cap)
        thresholds[f"capacity_top_{int(cap * 100)}pct"] = {
            "threshold": t_cap, **{k: v for k, v in threshold_metrics(self.y_val, oof, t_cap).items()
                                   if k in ("flag_rate", "precision", "recall")}}
        high = thresholds["cost_10to1"]["threshold"]
        medium = min(thresholds["cost_20to1"]["threshold"], high)
        if medium >= high:
            medium = high / 2
        self.final = {"family": name, "params": params, "preprocessor": pre, "model": model,
                      "raw_val": raw_val, "oof_cal": oof, "calibration": {k: v for k, v in cal.items() if k not in ("sigmoid", "isotonic")}
                      | {m: {"brier": cal[m]["brier"], "pr_auc": cal[m]["pr_auc"]} for m in ("sigmoid", "isotonic")},
                      "thresholds": thresholds, "risk_bands": {"medium": float(medium), "high": float(high)},
                      "data": data.description, "mix": dict(self.mix), "train_rows": len(data.train)}
        self.log(f"final model: {name} x{len(members)} seeds, calibration={cal['method']}, "
                 f"bands={self.final['risk_bands']}")
        pd.DataFrame(thresholds).T.to_csv(self.out / "tables" / "thresholds.csv")
        return self.final

    # -- 6. final evaluation (test opened once) ------------------------------
    def evaluate(self, split: str = "real_test") -> dict[str, Any]:
        from src.ml.data import load_real_test
        from src.ml.evaluate import lift_table, metrics_with_ci, reliability_table
        from src.ml.preprocess import ClaimFeatureBuilder
        df = load_real_test() if split == "real_test" else self.val
        y = df[TARGET].to_numpy().astype(int)
        f = self.final
        p = f["model"].predict_proba(f["preprocessor"].transform(df))[:, 1]
        t_default = f["risk_bands"]["high"]
        res = {"split": split, "n": len(y), "fraud": int(y.sum()), "scores": p, "y": y, "frame": df}
        res["metrics"] = metrics_with_ci(y, p, t_default, n_boot=self.n_boot)
        cap_key = [k for k in f["thresholds"] if k.startswith("capacity")][0]
        res["metrics_capacity"] = metrics_with_ci(y, p, f["thresholds"][cap_key]["threshold"], n_boot=self.n_boot)
        # Baselines on the same split
        comp = {"final model (calibrated)": p,
                "rules-only (red flags)": ClaimFeatureBuilder().fit(df).transform(df)["red_flag_score"].to_numpy(),
                "all-legit": np.zeros(len(y))}
        if "logreg" in self.fitted:
            lr = self.fitted["logreg"]
            comp["logistic regression"] = lr.model.predict_proba(lr.preprocessor.transform(df))[:, 1]
        rows = []
        for k, s in comp.items():
            thr = t_default if k == "final model (calibrated)" else np.quantile(s, 1 - (p >= t_default).mean())
            m = metrics_with_ci(y, s, thr, n_boot=self.n_boot).set_index("metric")
            rows.append({"model": k, **{f"{c}": m.loc[c, "value"] for c in
                                        ("pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "fpr", "accuracy")},
                         "pr_auc_ci": f"[{m.loc['pr_auc', 'lo']:.3f}, {m.loc['pr_auc', 'hi']:.3f}]"})
        res["comparison"] = pd.DataFrame(rows)
        m = res["metrics"].set_index("metric")
        prevalence, flag_rate = y.mean(), m.loc["flag_rate", "value"]
        res["success"] = {
            "precision_lo": float(m.loc["precision", "lo"]), "random_precision": float(prevalence),
            "recall_lo": float(m.loc["recall", "lo"]), "random_recall_at_same_flag_rate": float(flag_rate),
            "met": bool(m.loc["precision", "lo"] > 2 * prevalence and m.loc["recall", "lo"] > 2 * flag_rate)}
        res["lift"] = lift_table(y, p)
        res["reliability"] = reliability_table(y, p)
        tag = "test" if split == "real_test" else "valstandin"
        res["metrics"].to_csv(self.out / "tables" / f"final_metrics_{tag}.csv", index=False)
        res["comparison"].to_csv(self.out / "tables" / f"final_comparison_{tag}.csv", index=False)
        res["lift"].to_csv(self.out / "tables" / f"lift_{tag}.csv", index=False)
        self.evaluation = res
        return res

    # -- 7. explainability, errors, fairness ---------------------------------
    def explain(self, n_rows: int | None = None) -> dict[str, Any]:
        from src.ml.explain import build_explainer, describe_feature, top_contributions
        f, ev = self.final, self.evaluation
        ex = build_explainer(f["preprocessor"], f["model"], self.aug[self.aug[ORIGIN_COL] == 0])
        df = ev["frame"]
        if n_rows is None:
            n_rows = 300 if f["family"] == "mlp" else len(df)
        idx = np.arange(len(df)) if n_rows >= len(df) else np.random.default_rng(0).choice(len(df), n_rows, replace=False)
        X = f["preprocessor"].transform(df.iloc[idx])
        sv = ex.shap_values(X)
        names = ex.feature_names
        labels = [describe_feature(n)[0].split(" = ")[0] for n in names]
        imp = pd.DataFrame(np.abs(sv), columns=labels).T.groupby(level=0).sum().T.mean().sort_values(ascending=False)
        imp.to_csv(self.out / "tables" / "shap_global_importance.csv", header=["mean_abs_shap"])
        y, p = ev["y"][idx], ev["scores"][idx]
        thr = f["risk_bands"]["high"]
        cases = {"true_positive": np.flatnonzero((y == 1) & (p >= thr)),
                 "false_positive": np.flatnonzero((y == 0) & (p >= thr)),
                 "false_negative": np.flatnonzero((y == 1) & (p < thr))}
        local = []
        for kind, ids in cases.items():
            for i in ids[np.argsort(-p[ids])][:2] if kind != "false_negative" else ids[np.argsort(p[ids])][:2]:
                raw = df.iloc[idx[i]].to_dict()
                for c in top_contributions(sv[i], np.asarray(X)[i], names, raw, top_k=5):
                    local.append({"case": kind, "row": int(idx[i]), "score": float(p[i]), **c})
        local = pd.DataFrame(local)
        local.to_csv(self.out / "tables" / "shap_local_examples.csv", index=False)
        self.shap = {"values": sv, "X": X, "names": names, "importance": imp, "local": local,
                     "explainer": ex, "rows": idx}
        return self.shap

    def error_analysis(self, columns=("BasePolicy", "Fault", "VehicleCategory", "age_band",
                                      "AddressChange_Claim", "Deductible", "VehiclePrice")) -> pd.DataFrame:
        from src.ml.evaluate import group_metrics
        from src.validation.feature_engineering import add_engineered_features
        ev = self.evaluation
        df = add_engineered_features(ev["frame"])
        parts = []
        for c in columns:
            g = group_metrics(ev["y"], ev["scores"], df[c].astype(str), self.final["risk_bands"]["high"])
            g.insert(0, "segment", c)
            parts.append(g)
        t = pd.concat(parts, ignore_index=True)
        t.to_csv(self.out / "tables" / "error_analysis.csv", index=False)
        return t

    def fairness(self) -> pd.DataFrame:
        from src.ml.evaluate import group_metrics
        from src.validation.feature_engineering import add_engineered_features
        ev = self.evaluation
        df = add_engineered_features(ev["frame"])
        parts = []
        for c in self.cfg["features"]["fairness_groups"]:
            g = group_metrics(ev["y"], ev["scores"], df[c].astype(str), self.final["risk_bands"]["high"])
            g.insert(0, "attribute", c)
            parts.append(g)
        t = pd.concat(parts, ignore_index=True)
        t.to_csv(self.out / "tables" / "fairness.csv", index=False)
        return t

    # -- 8. artifacts ----------------------------------------------------------
    def save_artifacts(self) -> Path:
        import joblib
        from src.ml.evaluate import ranking_metrics
        from src.ml.preprocess import feature_names
        d = self.out / "models"
        f = self.final
        model = to_cpu_inference(f["model"])
        joblib.dump(f["preprocessor"], d / "preprocessor.joblib", compress=3)
        joblib.dump(model, d / "fraud_model.joblib", compress=3)
        ex = getattr(self, "shap", {}).get("explainer")
        if ex is not None:
            joblib.dump(ex, d / "shap_explainer.pkl", compress=3)
        ev = getattr(self, "evaluation", None)
        env = environment_report()
        date = pd.Timestamp.now().strftime("%Y%m%d-%H%M")
        card = {
            "model_version": f"{f['family']}-{'fast' if self.fast else 'b200'}-{date}",
            "placeholder": bool(self.fast),
            "family": f["family"], "n_seed_members": len(self.seeds),
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in f["params"].items()},
            "preprocessor_kind": PREPROCESSOR_KIND[f["family"]],
            "features": feature_names(f["preprocessor"]),
            "excluded_features": ["PolicyNumber", "is_synthetic", "PolicyType", "AgeOfPolicyHolder",
                                  "RepNumber", "Sex", "MaritalStatus", "Age", "MonthClaimed",
                                  "DayOfWeekClaimed", "WeekOfMonthClaimed"],
            "training_data": {"description": f["data"], "mix": f["mix"], "rows": f["train_rows"]},
            "calibration": f["calibration"],
            "thresholds": f["thresholds"], "risk_bands": f["risk_bands"],
            "decision_policy": {"high": "FLAG_FOR_INVESTIGATION candidate (score >= high band)",
                                "blocking_validation_failure": "REQUEST_MORE_INFO",
                                "otherwise": "APPROVE (always subject to adjuster review)"},
            "validation_metrics": ranking_metrics(self.y_val, f["oof_cal"]),
            "evaluation": None if ev is None else {
                "split": ev["split"], "n": ev["n"], "fraud": ev["fraud"],
                "metrics": ev["metrics"].to_dict("records"),
                "success_criterion": ev["success"]},
            "data_sha256": self.hashes, "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "device": self.device, "environment": env, "fast_mode": self.fast,
        }
        (d / "model_card.json").write_text(json.dumps(card, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        self.card = card
        self.log(f"saved artifacts to {d}: " + ", ".join(sorted(p.name for p in d.iterdir())))
        return d

    def zip_outputs(self) -> Path:
        import shutil
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        base = self.out.parent / f"b200_outputs_{stamp}"
        path = Path(shutil.make_archive(str(base), "zip", root_dir=self.out.parent, base_dir=self.out.name))
        self.log(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
        return path
