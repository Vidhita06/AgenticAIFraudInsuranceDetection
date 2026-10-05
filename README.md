# agentic-claims-triage

**Agentic AI: Autonomous Insurance Claims Triage & Fraud Assistant** (IDEA Lab project,
Ramdeobaba University, Session 2026-27)

A LangGraph agent takes a vehicle insurance claim and works through it in five steps:

1. validates it;
2. scores it with a trained fraud model;
3. retrieves similar past claims;
4. runs policy checks;
5. recommends **APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFO** with a grounded
   rationale.

A human adjuster has the final say.

Current scope is **EDA, model training (on a remote NVIDIA B200 Jupyter server) and the
agent**. The full work plan is in `prompts/claude_code_eda_and_training_prompt.md`.

## Repository structure (built so far)

```
config/
  settings.yaml             paths and parameters (RANDOM_STATE = 42, cost ratios, top_k, max_steps)
  policy_rules.yaml         validation / consistency rules (+ red flags from the EDA)
data/
  raw/fraud_oracle.csv      original Kaggle data (read-only)
  processed/                augmented_train_300k, real_train, real_validation, real_test (read-only)
details/                    synopsis, proposal, presentation template
docs/
  synopsis_requirements.md  checklist of synopsis objectives, deliverables, metrics
  report/                   01_data_audit.md, tables/, figures/
notebooks/01_eda.ipynb      EDA and synthetic-data audit (thin wrapper over src/)
notebooks/02_model_training.ipynb   B200 training notebook (Run All; see below)
models/                     preprocessor, calibrated model, SHAP explainer, model card
b200_bundle.zip             upload bundle for the B200 (built by scripts/make_b200_bundle.py)
prompts/                    the work-plan prompt
src/
  config.py                 settings loader, input-file SHA-256 hashes
  ml/data.py                explicit-dtype loaders, category sets and natural orders
  ml/audit.py               Phase 1 profiling, fact checks, report builder
  ml/eda.py, ml/synthetic_audit.py   Step A analysis code
  ml/preprocess.py          feature builder + preprocessors (single source of truth)
  ml/train.py               model zoo, data mix, Optuna, calibration, TrainingRun
  ml/evaluate.py            metrics with bootstrap CIs, thresholds, lift, fairness
  ml/predict.py, ml/explain.py   CPU scoring and SHAP explanations from models/
  validation/feature_engineering.py  engineered features
  validation/validators.py  missing/invalid-field rules and the rule engine
  validation/consistency_rules.py  timeline and logic rules
scripts/run_phase1_audit.py rebuilds docs/report/01_data_audit.md
scripts/make_b200_bundle.py, scripts/train_model.py   B200 bundle; headless training
tests/test_validation.py    rules, YAML-vs-code consistency, data-loading checks
```

```
config/prompts/            orchestrator_system.md, synthesis.md
src/schemas/               ClaimSchema (33 fields), TriageDecision, ClaimState
src/ingestion/             parser (CSV/JSON/dict) + normalizer (lists missing/invalid fields)
src/retrieval/             encoder, FAISS index build (real train only), search
src/tools/                 score_claim, find_similar_claims, check_policy_rules (+ LangChain TOOLS)
src/agent/                 llm.py, nodes.py, guardrails.py, graph.py, runner.py
scripts/build_vector_index.py
tests/                     test_validation, test_tools, test_guardrails, test_graph (+ fixtures/)
```

Step D adds `evaluation/` and the agent-evaluation notebook.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.ml.data --write-real-train   # regenerates data/processed/real_train.csv
python scripts/run_phase1_audit.py         # regenerates docs/report/01_data_audit.md
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # Step A (~10 min CPU)
```

## Training on the B200 (Step B)

`notebooks/02_model_training.ipynb` runs top to bottom with *Run All*. It reads everything it
needs from `b200_bundle.zip`:

- `src/`, `config/`, `data/processed/*.csv` and `requirements-train.txt`;
- `MANIFEST.json`, which records the git commit and the SHA-256 of every file.

**1. Get the files.** Download two files from the branch (or PR): `b200_bundle.zip` (repo
root) and `notebooks/02_model_training.ipynb`. To rebuild the bundle from a checkout, run
`python scripts/make_b200_bundle.py`.

**2. Upload.** In the B200 Jupyter file browser, create an empty folder (e.g. `fraud_b200/`)
and upload both files into it, side by side. Do not unzip the bundle: the notebook does that
itself.

**3. Pick the kernel.** Use the server's default Python 3 kernel, the one with the
preinstalled CUDA PyTorch. Do not `pip install torch`. The B200 (Blackwell, sm_100) needs a
CUDA 12.8+ build, and the notebook checks this for you.

**4. Configure (optional).** The first code cell is the only one to edit:

| Setting | Default | Effect |
|---|---|---|
| `FAST_MODE` | `False` | Leave off on the server. |
| `N_TRIALS` | `80` | Optuna trials per finalist. Fewer is faster: about 30 for a quick pass. |
| `TUNE_TOP` | `2` | Number of model families to tune. |
| `MODELS` | all five | Drop `"mlp"` or `"catboost"` to save time. |
| `RESUME` | `True` | Keep on. A rerun after a disconnect reuses finished models and Optuna trials. |

**5. Run All.** The notebook:

1. unpacks the bundle;
2. installs `requirements-train.txt`, with every preinstalled torch/CUDA/numpy package pinned
   so pip cannot change them;
3. prints `nvidia-smi` and an environment report, and stops early with a clear message if
   the GPU, sm_100 or bf16 is unusable;
4. runs baselines → data-mix experiment → model zoo → Optuna → final seed ensemble →
   calibration → thresholds → **one** test evaluation → SHAP, error analysis and fairness →
   CPU export.

Each finished model is checkpointed to `outputs/checkpoints/`. If the kernel dies, choose
*Run All* again. Expect roughly 1–3 h with the defaults; most of it is Optuna.

**6. Download.** The last cell prints `DOWNLOAD THIS FILE: …/b200_outputs_<timestamp>.zip`
(next to the notebook). Download it before the server is reclaimed. It contains:

- `outputs/models/`: the artifacts (`preprocessor.joblib`, `fraud_model.joblib`,
  `shap_explainer.pkl`, `model_card.json`);
- `figures/`, `tables/`, `run_log.csv` / `run_log.txt` and `optuna.db`.

**7. Bring it back.** Unzip it in the repo root and copy `outputs/models/*` into `models/`.
The artifacts are saved for CPU inference and load with `requirements.txt` (same sklearn,
xgboost, catboost, lightgbm and shap pins). If the MLP wins, CPU torch is also needed.

**Verify locally:**

```bash
python -c "from src.ml.predict import load_artifacts, score_claims; from src.ml.data import load_real_validation; print(score_claims(load_real_validation().head(3), load_artifacts()))"
```

The `models/` files currently in the repo are **FAST_MODE placeholders**: a 20k-row CPU run
with `"placeholder": true` in `model_card.json`. They exist so Step C can be built and tested;
the B200 artifacts replace them. `docs/report/02_model_training_fast_mode_run.ipynb` is the
executed FAST_MODE smoke test of the same notebook. In FAST_MODE the evaluation section uses
real validation as a stand-in, so `real_test.csv` stays unopened until the B200 run.

**Headless alternative** (no Jupyter): `python scripts/train_model.py --trials 80`, or
`--fast` for a CPU smoke test.

## Triage agent (Step C)

```bash
cp .env.example .env                      # set LLM_API_KEY (Anthropic by default)
python scripts/build_vector_index.py      # FAISS index over real training claims -> data/vector_store/
```

```python
from src.agent.runner import TriageRunner
runner = TriageRunner(audit_log="evaluation/results/audit.jsonl")
thread_id, decision, trace = runner.run_claim(claim_dict)      # pauses at human review
print(decision.decision, decision.rationale, decision.evidence)
runner.resume(thread_id, {"decision": "APPROVE", "adjuster_id": "a1", "override_reason": None})
```

How a claim flows through the graph:

1. `ingest → validate → score` always run.
2. A ReAct loop of at most `agent.max_steps` turns (`config/settings.yaml`) lets the LLM call
   `find_similar_claims` and `check_policy_rules`.
3. `synthesize` writes a JSON decision.
4. `guardrails` enforces, in code:
   - only APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFO, never a denial;
   - a blocking validation failure forces REQUEST_MORE_INFO;
   - a `high` risk band is never approved;
   - evidence that is not traceable to a tool output is removed;
   - after two LLM failures, a rule-based decision is used.
5. `human_review` pauses the run (LangGraph `interrupt`) for the adjuster.

Without an API key the agent still runs and returns the rule-based decision.
`pytest -q` covers every graph path with a scripted fake chat model, so it needs no key and
no network.

## Data

| File | Rows | Fraud | Use |
|---|---|---|---|
| `data/raw/fraud_oracle.csv` | 15,420 | 923 (5.99%) | original Kaggle *Vehicle Claim Fraud Detection* |
| `data/processed/augmented_train_300k.csv` | 300,000 | 17,956 | 10,793 real train + 289,207 synthetic (`is_synthetic`) |
| `data/processed/real_train.csv` | 10,793 | 646 | the `is_synthetic == 0` rows |
| `data/processed/real_validation.csv` | 2,313 | 139 | model selection |
| `data/processed/real_test.csv` | 2,313 | 138 | opened once for the final model evaluation and once for the final agent evaluation |

Rules that hold throughout:

- Real data is the only ground truth; synthetic rows are used for training only.
- `PolicyNumber` and `is_synthetic` are never features. `PolicyNumber` alone identifies a row
  as real or synthetic: real rows are ≤ 15,420, synthetic rows are above.
- `AgeOfPolicyHolder` and `PolicyType` are for display only, because `Age` and
  `VehicleCategory` + `BasePolicy` fully determine them.

## Results so far

**Data audit** (`docs/report/01_data_audit.md`):

- 26 of the 28 stated data facts were confirmed. The exception: Deductible = 400 covers about
  96% of rows, not 93%.
- 13 validation and consistency rules are defined in `config/policy_rules.yaml`.
- Missing age (Age = 0) is the only blocking rule that fires on this data.
- Early-policy incidents carry about 3× the base fraud rate in real data.

See [`docs/synopsis_requirements.md`](docs/synopsis_requirements.md) for the synopsis checklist,
including the items marked deferred.
