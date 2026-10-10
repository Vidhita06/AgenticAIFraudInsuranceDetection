"""Install the artifacts from a B200 run, rebuild the similar-claims index and run the tests.

    python scripts/install_b200_outputs.py path/to/b200_outputs_<timestamp>.zip

1. extracts outputs/models/* from the zip into models/ (preprocessor, calibrated model,
   SHAP explainer, model card) and outputs/{figures,tables,run_log.*} into docs/report/b200_run/;
2. checks the model card is a full (non-FAST_MODE) run and that the artifacts load and score;
3. rebuilds data/vector_store (its staleness check is tied to models/preprocessor.joblib);
4. runs pytest.
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_FILES = ("preprocessor.joblib", "fraud_model.joblib", "shap_explainer.pkl", "model_card.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("--allow-fast", action="store_true", help="accept a FAST_MODE (placeholder) run")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.zip_path)
    names = z.namelist()
    found = {}
    for f in MODEL_FILES:
        hits = [n for n in names if n.endswith(f"models/{f}")]
        if not hits and f != "shap_explainer.pkl":
            sys.exit(f"{f} not found in {args.zip_path}")
        if hits:
            found[f] = hits[0]
    card = json.loads(z.read(found["model_card.json"]))
    print(f"model {card['model_version']} | family {card['family']} | fast_mode={card['fast_mode']} "
          f"| trained {card['trained_at']} on {card.get('environment', {}).get('gpu', card['device'])}")
    if card.get("fast_mode") and not args.allow_fast:
        sys.exit("This is a FAST_MODE run (placeholder). Pass --allow-fast to install it anyway.")

    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    for f, member in found.items():
        (models / f).write_bytes(z.read(member))
        print(f"installed models/{f}")
    report = ROOT / "docs" / "report" / "b200_run"
    for n in names:
        rel = n.split("outputs/", 1)[-1] if "outputs/" in n else None
        if rel and (rel.startswith(("figures/", "tables/")) or rel.startswith("run_log")) and not n.endswith("/"):
            dest = report / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))
    print(f"copied figures/tables/run logs to {report.relative_to(ROOT)}")

    from src.ml.data import load_real_validation
    from src.ml.predict import load_artifacts, score_claims
    load_artifacts.cache_clear()
    print("scoring check:", score_claims(load_real_validation().head(2), load_artifacts())[0])

    from src.retrieval.build_index import build_index
    info = build_index()
    print(f"rebuilt data/vector_store: {info['n_claims']} claims, dim {info['dim']}")

    if not args.skip_tests:
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))


if __name__ == "__main__":
    main()
