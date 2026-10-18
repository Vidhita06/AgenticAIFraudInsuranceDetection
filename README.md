# agentic-claims-triage

**Agentic AI: Autonomous Insurance Claims Triage & Fraud Assistant** (IDEA Lab project,
Ramdeobaba University, Session 2026-27)

A LangGraph agent investigates a vehicle insurance claim. One orchestrator works with six
specialised tools (the team plan, mapped in `docs/agent_restructure_plan.md`):

1. it validates the claim;
2. it calls the tools: fraud score, SHAP explanation, policy retrieval, policy verification,
   similar claims and claim history;
3. it aggregates the evidence, including missing information and conflicts;
4. it recommends **APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFORMATION**, with a
   grounded rationale and reason codes.

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
src/tools/                 the six tools: score_claim, explain_claim, retrieve_policy, verify_policy,
                           find_similar_claims, get_claim_history (+ LangChain TOOLS)
src/agent/                 llm.py, nodes.py, evidence.py (aggregation), guardrails.py, graph.py, runner.py
scripts/build_vector_index.py
tests/                     test_validation, test_tools, test_six_tools, test_guardrails, test_graph (+ fixtures/)
```

```
evaluation/agent_eval.py   sampling, cost estimates, batch runs, metrics, LLM-as-judge, report
evaluation/results/        JSONL runs (git-ignored), judge scores, manual-review sheets
scripts/run_batch_triage.py   offline | estimate | pilot | final | all
scripts/install_b200_outputs.py   install B200 artifacts, rebuild the index, run tests
notebooks/03_agent_evaluation.ipynb   analysis notebook (never spends money)
docs/architecture.png      graph diagram (docs/make_architecture_diagram.py)
```

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

`models/` holds the **B200 artifacts** (`lightgbm-b200-20261016-0915`). They load on CPU with
`requirements.txt`. After replacing them, rebuild the index (`python scripts/build_vector_index.py`)
and regenerate the report assets (`python scripts/regenerate_model_report.py`).
`docs/report/02_model_training_fast_mode_run.ipynb` is the executed FAST_MODE smoke test of the
notebook.

**Headless alternative** (no Jupyter): `python scripts/train_model.py --trials 80`, or
`--fast` for a CPU smoke test.

## Triage agent (Step C)

```bash
cp .env.example .env                      # Ollama by default (free); see .env.example
python scripts/build_vector_index.py      # FAISS index over real training claims -> data/vector_store/
```

```python
from src.agent.runner import TriageRunner
runner = TriageRunner(audit_log="evaluation/results/audit.jsonl")
thread_id, decision, trace = runner.run_claim(claim_dict)      # pauses at human review
print(decision.decision, decision.rationale, decision.evidence)
runner.resume(thread_id, {"decision": "APPROVE", "adjuster_id": "a1", "override_reason": None})
```

How a claim flows through the graph (`docs/architecture.png`):

1. `ingest → validate → score` always run. Scoring is Tool 1: probability, predicted class,
   risk band and the red flags that fired.
2. The **orchestrator** is a ReAct loop of at most `agent.max_steps` turns. In it, the LLM
   chooses among Tools 2–6:
   - `explain_fraud_score`, `retrieve_policy`, `verify_policy`, `find_similar_claims` and
     `get_claim_history`;
   - the tools are bound to the claim, so the LLM cannot change it;
   - prerequisites run first (verification needs retrieval).
3. `complete_investigation` runs any required tool the LLM skipped
   (`agent.required_tools`), logged as `auto`. Without an LLM it runs all six.
4. `aggregate_evidence` builds the evidence record in code:
   - each tool's status (completed / partial / unavailable / failed / not_run) and origin;
   - the findings, missing information, conflicts and investigation status.
5. `synthesize` (the reasoning step) writes a JSON decision, rationale, cited evidence and
   unresolved questions.
6. `guardrails` enforces, in code:
   - only APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFORMATION, never a denial;
   - a blocking validation failure forces REQUEST_MORE_INFORMATION;
   - a `high` risk band is never approved;
   - APPROVE needs a fraud score and no policy-verification FAIL;
   - evidence that cannot be traced to the output of the tool it cites is removed;
   - after two LLM failures, a rule-based decision is used.

   It also adds the reason codes and the plan's final-output fields from the evidence record.
7. `human_review` pauses the run (LangGraph `interrupt`) for the adjuster.

**Data limits, reported rather than invented:**

- The Kaggle data has no policy documents. Policy retrieval reads `data/policies/<PolicyNumber>.json`
  when one exists; otherwise it reports `partial` and lists what is missing. Verification
  checks that need policy terms return `UNKNOWN`.
- `PolicyNumber` is unique per claim, so there is no linked claim history. Tool 6 reports
  `unavailable` unless `claim_history.source` names an authorised file.

**Simulated reference data (optional, clearly labelled).** `python scripts/build_simulated_db.py`
builds `data/simulated/insurance.db` (see `docs/simulated_data_plan.md`). It has four parts:

- simulated policies, from the claims' real policy fields plus simulated dates and limits;
- illustrative clauses written by the team (`config/simulated_policy_clauses.yaml`);
- simulated prior claims, with counts that follow `PastNumberOfClaims` and no outcomes;
- run-time tables, where the agent records its verification results and an audit row.

The generator never reads the fraud label, and tests prove it. With `simulated_data.enabled: true`
(or `NodeDeps(simulated=True)`), Tools 3, 4 and 6 use the database:

- every output from it says `source: "simulated reference data"`;
- evidence cited from it is labelled the same way in the recommendation;
- Tool 4 checks the clauses for real PASS/FAIL results.

With `enabled: false` (the default) the tools behave exactly as without it.

**Decisions and flag categories:**

- REQUEST_MORE_INFORMATION is only for claim-level problems: blocking validation, or a claim
  record that contradicts itself (PV02, PV04).
- Every FLAG_FOR_INVESTIGATION carries a category:
  - `FRAUD_RISK`: fraud risk;
  - `POLICY_CONDITION`: a coverage problem, such as a missed reporting deadline;
  - `SYSTEM_FAILURE`: the fraud score or a required tool failed.
- The evaluation counts only FRAUD_RISK flags as fraud caught.
- **Systemic gaps never justify a flag.** These hold for every claim: no policy document, UNKNOWN
  policy checks, no linked history and no claim amount.
  - If a FRAUD_RISK flag's reason codes are all low-risk or systemic ones, the guardrail
    replaces it with the rule-based decision (`unsupported_fraud_flag`).
  - The synthesis prompt gives a worked example of such a claim being approved.
- **In the medium band, only information the fraud model does not see can justify a flag**
  (`agent.medium_band_concrete_factors`):
  - a policy-condition FAIL, tagged POLICY_CONDITION, not FRAUD_RISK;
  - a claim-record contradiction (→ REQUEST_MORE_INFORMATION);
  - a claim-history contradiction: the history lists more prior claims than the claim reports.

  Red flags, SHAP drivers and similar-claim fraud are model features, so they appear only as
  context in the rationale. On validation they don't separate fraud in the medium band: 4 frauds
  among the 71 claims with one. See `docs/report/03b_medium_band_factors.md`.

Without an API key the agent still runs and returns the rule-based decision.
`pytest -q` covers every graph path with a scripted fake chat model, so it needs no key and
no network.

## Agent evaluation (Step D)

Every stage takes `--simulated-data off|on|both` (default `both`). `both` runs the same claims
without and with the simulated reference database, and `docs/report/03_agent_evaluation.md`
shows the two side by side. Build the database first with `python scripts/build_simulated_db.py`.

The live evaluation is one command, `scripts/run_batch_triage.py all`:

1. **Pilot:** 5 validation claims. It prints the measured tokens, latency and cost per claim,
   and how often the model fell back to the rule-based decision.
2. **Projection:** it projects the time and cost of the final run.
3. **Final run:** it runs the test set once per model:
   - all 138 test frauds plus 138 legitimate claims, or `--sample-size N` frauds plus N legit;
   - 20 claims run 3 times, to check run-to-run consistency;
   - up to 40 LLM-as-judge rubric checks.
4. **Report:** it writes `docs/report/03_agent_evaluation.md`, plus a per-model copy that
   names the model and sample size. It also writes
   `evaluation/results/manual_review_<model>_*.csv` with 20 claims for hand review.

**Free models** (Ollama, free-tier APIs) show **$0**, need no API key and run without a
confirmation prompt. **Paid models** (Anthropic) show a cost estimate and ask first; that's
about $0.08 per claim, or about $25 for the full run on Sonnet 5.5.

### Option A: Ollama on the B200 with `qwen2.5:72b` (recommended)

In a terminal on the B200 server (in Jupyter: *File → New → Terminal*):

```bash
# 1. Ollama: install if `ollama --version` fails, start it, pull the model (~47 GB)
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > ollama.log 2>&1 &
ollama pull qwen2.5:72b

# 2. The repository, in its own virtualenv (keeps the server's torch/numpy untouched)
git clone -b claude/brave-hawking-ive43h https://github.com/Vidhita06/AgenticAIFraudInsuranceDetection.git
cd AgenticAIFraudInsuranceDetection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure, build the similar-claims index, check the tests
cp .env.example .env
sed -i 's/^MODEL_NAME=.*/MODEL_NAME=qwen2.5:72b/; s/^BATCH_MODEL_NAME=.*/BATCH_MODEL_NAME=qwen2.5:72b/' .env
python scripts/build_vector_index.py
pytest -q

# 4. Run: estimate, then pilot -> full final run -> report
python scripts/run_batch_triage.py estimate
python scripts/run_batch_triage.py all
```

I couldn't measure the 72B model's speed here. At typical single-stream speeds (tens of
tokens per second) and 3–4 calls per claim, the full run (276 test claims plus consistency and
judge calls) may take several hours. The pilot prints the measured seconds per claim and the
projected total before the final run starts. If it's too long, add `--sample-size 70`.

**Bring the results back:**

```bash
git add docs/report/03_agent_evaluation*.md evaluation/results/*.csv evaluation/results/*.json evaluation/results/.test_evaluation_done_*
git commit -m "Agent evaluation: qwen2.5:72b" && git push
```

The raw `*.jsonl` runs are git-ignored; download them from the file browser if you want them.

### Option B: Ollama on a laptop with `qwen2.5:7b`

1. Install Ollama from https://ollama.com/download (macOS or Windows installer; on Linux,
   `curl -fsSL https://ollama.com/install.sh | sh`). The app starts the server; otherwise run
   `ollama serve`.
2. Run:

   ```bash
   ollama pull qwen2.5:7b                      # ~4.7 GB; needs 8+ GB RAM

   git clone -b claude/brave-hawking-ive43h https://github.com/Vidhita06/AgenticAIFraudInsuranceDetection.git
   cd AgenticAIFraudInsuranceDetection
   python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env                         # already set to ollama + qwen2.5:7b
   python scripts/build_vector_index.py
   pytest -q

   python scripts/run_batch_triage.py estimate --sample-size 40
   python scripts/run_batch_triage.py all --sample-size 40 --consistency-claims 10 --judge-claims 20
   ```

- **Time:** a 7B model on a laptop CPU takes tens of seconds per call, so use the smaller
  sample (40 frauds + 40 legit). The pilot prints the measured time per claim and projects
  the total; with a GPU or Apple Silicon it's much faster.
- **Quality:** a 7B model calls tools less reliably and makes more JSON mistakes than a 72B
  one. The graph recovers tool calls the model writes as text, and runs the policy check
  automatically if the model skips it (logged as `auto_tools`). It retries malformed JSON
  once and then uses the rule-based decision. The report shows the fallback rate, so you can
  compare models.
- **Re-runs:** the test set is evaluated once *per model*, so a later 72B run on the B200 is
  allowed.

### Option C: free OpenAI-compatible APIs

Groq, OpenRouter `:free` models and Google Gemini have free tiers; see the blocks in
`.env.example`. Use `--sample-size 40 --pause 2` to stay under their rate limits. If a model
is billed, add its price to `llm_pricing_usd_per_mtok` in `config/settings.yaml`; models not
listed there are assumed free.

### Without any LLM

`python scripts/run_batch_triage.py offline` runs the rule-based path on validation claims.
`notebooks/03_agent_evaluation.ipynb` analyses whichever results exist and never spends money.

### Updating the model

When new B200 outputs arrive, run:

```bash
python scripts/install_b200_outputs.py b200_outputs_<timestamp>.zip
```

It installs `models/`, rebuilds `data/vector_store` and runs the tests. Artifacts uploaded
directly to `models/` need `python scripts/build_vector_index.py` instead. The index records
the SHA-256 of `models/preprocessor.joblib`, `policy_rules.yaml` and the feature config, and
the similar-claims tool and the tests refuse a stale index.

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

**EDA and synthetic data** (`docs/report/01_eda_report.md`):

- The synthetic rows are indistinguishable from real ones: adversarial AUC 0.511.
- There is no memorisation.
- Adding them raises real-validation PR-AUC by about 0.05.

**Fraud model** (`docs/report/02_model_report.md`, `docs/report/model_card.md`): LightGBM, 3
seeds, Platt-calibrated, trained on real + synthetic data (w = 0.1). On the held-out real test
set (evaluated once):

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.874 | |
| PR-AUC (random = 0.06) | 0.252 | |
| Recall at the 10:1 cost threshold | 0.79 | 0.72–0.86 |
| Precision at the 10:1 cost threshold | 0.21 | 0.19–0.23 |
| False-positive rate | 0.19 | |

- The synopsis success criterion is met: precision and recall are both well above a random
  flagger at the same flag rate.
- A CPU-trained PyTorch MLP scores 0.262 validation PR-AUC (tuned, 3 seeds) vs LightGBM's 0.291,
  so LightGBM stays and the MLP was not run on test (`02_model_report.md` §5).

**Agent:** built and tested with a scripted model. The live LLM evaluation is pending (Step D).

See [`docs/synopsis_requirements.md`](docs/synopsis_requirements.md) for the synopsis checklist,
including the items marked deferred.
