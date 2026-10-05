# AgenticAIFraudInsuranceDetection

**Agentic AI: Autonomous Insurance Claims Triage & Fraud Assistant** (IDEA Lab project,
Ramdeobaba University, Session 2026-27)

The goal is an LLM orchestrator (LangGraph) that takes a vehicle insurance claim, validates it,
scores its fraud risk with a trained model, retrieves similar historical claims, runs policy
checks, and recommends **Approve / Flag for Investigation / Request More Information** with a
written rationale. A human adjuster has the final say in a Streamlit dashboard, and their
decisions feed back into retraining.

This repository currently covers the ML foundation: data audit, synthetic-data audit, EDA,
leakage-safe modelling, and agent-ready tools.

## Repository structure

```
config.yaml                 paths and parameters (RANDOM_STATE = 42)
requirements.txt            pinned dependencies
data/
  raw/                      original Kaggle data (read-only)
  processed/                augmented train (real + synthetic), real validation, real test (read-only)
  interim/, features/       derived data (git-ignored)
details/                    synopsis, proposal, presentation template
docs/
  synopsis_requirements.md  checklist of every synopsis objective, deliverable and metric
src/
  config.py                 config loading, input-file SHA-256 hashes
  stats.py                  Wilson CI and other stats helpers
  data/load.py              explicit-dtype loaders, category sets and natural orders
  data/profile.py           column profiling, verification of split facts
  data/validation.py        claim-validation rules (the agent's policy-check tool)
  data/audit.py             Phase 1 report builder
scripts/run_phase1_audit.py Phase 1 entry point
notebooks/01_data_audit.ipynb
reports/                    phase reports (*.md), tables/, figures/
tests/                      pytest suite
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Reproducing each phase

| Phase | Command | Output |
|---|---|---|
| 1. Data audit & validation rules | `python scripts/run_phase1_audit.py` | `reports/01_data_audit.md`, `reports/tables/01_*.csv` |
| 2. Synthetic data audit | pending | `reports/02_synthetic_audit.md` |
| 3. EDA | pending | `reports/03_eda_report.md`, `reports/red_flag_rules.yaml` |
| 4. Features | pending | `src/features/`, `src/schema.py` |
| 5. Modelling | pending | `reports/05_model_comparison.md` |
| 6. Final evaluation | pending | `reports/06_final_evaluation.md`, `reports/model_card.md` |
| 7. Agent-ready tools | pending | `src/tools/`, `src/feedback/`, `docs/agent_design.md` |

## Data

| File | Rows | Fraud | Use |
|---|---|---|---|
| `data/raw/vehicle_fraud_oracle.csv` | 15,420 | 923 (5.99%) | original Kaggle *Vehicle Claim Fraud Detection* |
| `data/processed/augmented_train_300k.csv` | 300,000 | 17,956 | 10,793 real train + 289,207 synthetic (`is_synthetic`) |
| `data/processed/real_validation.csv` | 2,313 | 139 | model selection |
| `data/processed/real_test.csv` | 2,313 | 138 | opened once, for the final evaluation |

Rules that hold in every phase: real data is the only ground truth, synthetic rows are used for
training only, and `PolicyNumber` and `is_synthetic` are never features. `PolicyNumber`
identifies the origin perfectly (real ≤ 15,420 < synthetic).

## Results so far

**Phase 1** (`reports/01_data_audit.md`):

- 26 of the 28 stated data facts were confirmed. The exception: Deductible = 400 covers about
  96% of rows, not 93%.
- `AgeOfPolicyHolder` is a deterministic, band-shifted function of `Age`.
- All PolicyType mismatches are a single alias, "Sedan - Liability" for Sport/Liability claims.
- Early-policy incidents have about 3× the fraud rate in real data but only about 1.6× in
  synthetic data.

## Synopsis checklist

See [`docs/synopsis_requirements.md`](docs/synopsis_requirements.md) for the full checklist
with status. Summary:

- Done: data audit, claim-validation rules (policy-check groundwork).
- Pending:
  - EDA report
  - baseline models (LR, RF) and advanced models (XGBoost, PyTorch NN)
  - imbalance handling
  - accuracy/precision/recall/F1/AUC-ROC evaluation and held-out FPR
  - tools for scoring, retrieval and policy checks
  - agent, dashboard, feedback loop
  - final report and demo
