"""Headless training run: the same pipeline as notebooks/02_model_training.ipynb.

    python scripts/train_model.py --fast                 # CPU smoke test (20k rows, few trials)
    python scripts/train_model.py --trials 80 --out outputs
    python scripts/train_model.py --fast --install-models   # also copy artifacts to models/

Real test is evaluated only without --fast (with --fast, validation stands in).
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ml.train import TrainingRun, check_blackwell, environment_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--tune-top", type=int, default=2)
    ap.add_argument("--models", default="rf,xgboost,lightgbm,catboost,mlp")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--install-models", action="store_true",
                    help="copy the final artifacts into models/ (placeholder or real)")
    args = ap.parse_args()

    env = environment_report()
    check_blackwell(env, require_gpu=not args.fast)
    run = TrainingRun(args.out, fast=args.fast, n_trials=3 if args.fast else args.trials,
                      tune_top=args.tune_top, resume=not args.no_resume)
    run.baselines()
    run.data_mix("xgboost")
    run.zoo(args.models.split(","))
    run.tune_best()
    run.finalize()
    run.save_artifacts()
    run.evaluate("real_validation" if args.fast else "real_test")
    run.explain()
    run.error_analysis()
    run.fairness()
    models_dir = run.save_artifacts()
    run.zip_outputs()
    if args.install_models:
        dest = ROOT / "models"
        dest.mkdir(exist_ok=True)
        for f in models_dir.iterdir():
            shutil.copy2(f, dest / f.name)
        print(f"installed artifacts into {dest}")


if __name__ == "__main__":
    main()
