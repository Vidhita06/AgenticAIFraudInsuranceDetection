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
prompts/                    the work-plan prompt
src/
  config.py                 settings loader, input-file SHA-256 hashes
  ml/data.py                explicit-dtype loaders, category sets and natural orders
  ml/audit.py               Phase 1 profiling, fact checks, report builder
  ml/evaluate.py            statistics helpers (Wilson CI; metrics follow in Step B)
  validation/validators.py  missing/invalid-field rules and the rule engine
  validation/consistency_rules.py  timeline and logic rules
scripts/run_phase1_audit.py rebuilds docs/report/01_data_audit.md
tests/test_validation.py    rules, YAML-vs-code consistency, data-loading checks
```

Steps A–D add the rest of the target layout: the training notebook and B200 bundle, `models/`,
`src/schemas`, `src/ingestion`, `src/retrieval`, `src/tools`, `src/agent` and `evaluation/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m src.ml.data --write-real-train   # regenerates data/processed/real_train.csv
python scripts/run_phase1_audit.py         # regenerates docs/report/01_data_audit.md
```

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
