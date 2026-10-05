# Claude Code Prompt — EDA, B200 Training Notebooks and the Triage Agent

> How to use: open Claude Code in the repo root and paste everything below the
> line. It works for a fresh session and for the session that already did the
> Phase 1 audit on `claude/brave-hawking-ive43h` (step 0 migrates that work).
> Claude Code stops at each checkpoint for your go-ahead.

---

You are working in the repo `AgenticAIFraudInsuranceDetection`, an IDEA Lab
project: **"Agentic AI: Autonomous Insurance Claims Triage & Fraud Assistant"**.
A LangGraph agent takes a vehicle insurance claim, validates it, scores it with a
trained fraud model, retrieves similar past claims, runs policy checks, and
recommends **APPROVE / FLAG_FOR_INVESTIGATION / REQUEST_MORE_INFO** with a
grounded rationale. A human adjuster has the final say.

**Scope of this work: EDA, model training, and the agent. Nothing else.**
Model training runs on a remote **NVIDIA B200 Jupyter server** that I upload
notebooks and data to. Everything else (validation rules, tools, retrieval, the
agent, tests) runs locally.

## 0. Context and migration

1. Read the synopsis `details/RBU_Project_Synopsis_Agentic_AI_Insurance_Claims (3).pdf`
   and the proposal `details/Agentic_AI_Insurance_Claims_Project_Proposal (1).docx`.
   If `docs/synopsis_requirements.md` exists, keep it; otherwise create it as a
   checklist of objectives, deliverables and metrics. Mark items owned by the
   deferred parts (dashboard, API, feedback loop) as "deferred".
2. If Phase 1 work exists on branch `claude/brave-hawking-ive43h` (`src/data/*`,
   `reports/01_data_audit.md`, 13 validation rules, 25 tests), merge it in and
   **move it into the layout below**. Keep the logic and tests, change the
   locations, and use `git mv` where possible so history is kept:
   - `src/data/load.py` → `src/ml/data.py` (loaders, category sets, natural orders)
   - `src/data/validation.py` → split into `src/validation/validators.py`
     (missing or invalid fields) and `src/validation/consistency_rules.py`
     (timeline and logic checks). Rule ids, conditions, severities and
     explanation text move to `config/policy_rules.yaml`, and the code reads them
     from there.
   - `config.yaml` → `config/settings.yaml`; `reports/` → `docs/report/`
     (tables in `docs/report/tables/`, figures in `docs/report/figures/`).
   - Remove `src/data/` once it's empty and fix all imports and tests.
3. Data files, renamed to match the layout (`git mv`, never change contents):
   - `data/raw/vehicle_fraud_oracle.csv` → `data/raw/fraud_oracle.csv`
     (15,420 × 33, UTF-8 with BOM; read with `encoding="utf-8-sig"`).
   - `data/processed/augmented_train_300k.csv`: 300,000 × 34 = 10,793 real rows
     (`is_synthetic = 0`, 646 fraud) + 289,207 synthetic rows (17,310 fraud).
   - `data/processed/real_validation.csv`: 2,313 real rows, 139 fraud.
   - `data/processed/real_test.csv`: 2,313 real rows, 138 fraud.
   - Create `data/processed/real_train.csv` = the `is_synthetic == 0` rows of the
     augmented file (drop `is_synthetic`), using a small script.
   - `data/raw/insurance_claims.csv` (the supplementary Kaggle "Auto Insurance
     Claims Data") is not in the repo. Leave it out; don't block on it.
4. Verified facts and decisions already made (don't re-litigate):
   - The real train, validation and test splits don't overlap. No synthetic row
     copies a real claim. Every synthetic row has `PolicyNumber` ≥ 15,421, so
     `PolicyNumber` (and `is_synthetic`) must **never** be a feature.
   - `Age = 0` means missing age. It is a blocking validation failure that leads
     to REQUEST_MORE_INFO.
   - `AgeOfPolicyHolder` is fully determined by `Age` (shifted bands), and
     `PolicyType` is fully determined by `VehicleCategory` + `BasePolicy`
     (Sport + Liability is always recorded as "Sedan - Liability"). Drop both as
     model inputs but keep them for display.
   - Early-policy incidents (V08) are a warning and a red-flag feature, not a
     blocking rule.
   - Deductible is 400 for ~96% of rows.
   - Cost ratio: missed fraud vs unnecessary investigation defaults to 10:1. Also
     report 5:1 and 20:1, plus a fixed-capacity option (investigate the top 10%).
   - How the synthetic data was generated is unknown. Treat the generator as a
     black box.

## 1. Target layout (repo root = `agentic-claims-triage/`)

Build **only** what is marked ✅. Don't create files or empty placeholders for
⏸ items; they come later and will follow this same layout.

```
README.md ✅   requirements.txt ✅   requirements-train.txt ✅ (B200 extras)   .env.example ✅
config/settings.yaml ✅  config/policy_rules.yaml ✅  config/prompts/{orchestrator_system,synthesis}.md ✅
data/raw/{fraud_oracle.csv ✅, insurance_claims.csv ⏸}
data/processed/{augmented_train_300k,real_train,real_validation,real_test}.csv ✅
data/vector_store/{claims.faiss, claims_meta.parquet} ✅ (generated, git-ignored)
data/feedback/ ⏸
models/{fraud_model.joblib, preprocessor.joblib, shap_explainer.pkl, model_card.json} ✅ (from the B200)
notebooks/01_eda.ipynb ✅  02_model_training.ipynb ✅  03_agent_evaluation.ipynb ✅
src/schemas/{claim.py, decision.py, state.py} ✅
src/ingestion/{parser.py, normalizer.py} ✅ (minimal)   text_extractor.py ⏸
src/validation/{validators.py, consistency_rules.py, feature_engineering.py} ✅
src/ml/{data.py, preprocess.py, train.py, evaluate.py, predict.py, explain.py} ✅
src/retrieval/{encoder.py, build_index.py, search.py} ✅
src/tools/{__init__.py, fraud_score_tool.py, similar_claims_tool.py, policy_check_tool.py} ✅   claimant_history_tool.py ⏸
src/agent/{llm.py, nodes.py, guardrails.py, graph.py, runner.py} ✅
src/feedback/ ⏸   src/api/ ⏸   dashboard/ ⏸
scripts/{make_b200_bundle.py, build_vector_index.py, run_batch_triage.py} ✅   train_model.py ✅   retrain_from_feedback.py ⏸
evaluation/agent_eval.py ✅  evaluation/results/ ✅
tests/{fixtures/sample_claims.json, test_validation.py, test_tools.py, test_guardrails.py, test_graph.py} ✅
docs/report/ ✅ (EDA report, model report, model card, agent eval)   docs/architecture.png ✅   docs/presentation/ ⏸
```

## 2. Rules that apply everywhere

- **Only real data counts as ground truth.** Select models on
  `real_validation.csv`. Open `real_test.csv` once for the final model evaluation
  and once for the final agent evaluation. Synthetic rows only ever go into
  training.
- **No leakage.** Fit every encoder, scaler and resampler on training data only.
  Don't SMOTE on top of data that is already 96% synthetic unless real-validation
  results justify it.
- **Metrics:** accuracy, precision, recall, F1 and ROC-AUC (required by the
  synopsis), plus PR-AUC (the selection metric at 6% prevalence), F2,
  false-positive rate and recall at fixed precision. Compare against naive
  baselines and give 95% bootstrap CIs (there are only ~139 frauds per split).
- **Single source of truth.** Feature engineering and preprocessing live **only**
  in `src/validation/feature_engineering.py` and `src/ml/preprocess.py`. The
  notebooks import them; they never copy the code. The pickled pipeline then
  references `src.*` classes, which exist both on the B200 and locally.
- `RANDOM_STATE = 42`; SHA-256 of every input file goes in reports and
  `model_card.json`.
- `.gitignore`: `.env`, `data/vector_store/`, `evaluation/results/*.jsonl`, the
  B200 bundle zip, notebook checkpoints. Commit `models/` only if every file is
  < 50 MB; otherwise tell me.
- **Stop at every CHECKPOINT:** summarise in ≤ 15 bullets, list open questions,
  commit, and wait for my approval.

## 3. Step A — EDA and synthetic-data audit → `notebooks/01_eda.ipynb` (CHECKPOINT)

It needs no GPU but must also run on the B200 server, so make it portable as in
§4.1. Write the narrative to `docs/report/01_eda_report.md` and figures to
`docs/report/figures/`.

1. **Data audit summary:** reuse the Phase 1 findings; don't redo them.
2. **Synthetic audit:** per-column TVD; fraud rate per category per origin
   (real vs synthetic); Cramér's V matrix difference; adversarial validation
   (LightGBM, real vs synthetic, *without* `PolicyNumber`) with the features
   driving it; TRTR vs TSTR vs train-on-real+synthetic, all scored on real
   validation; Gower distance-to-closest-record. End with a verdict: use all
   synthetic rows, a filtered subset, down-weighted rows, or none.
3. **Fraud EDA on training data** (real first, then confirmed on synthetic):
   fraud rate per category with Wilson CIs and n; chi-square, Cramér's V and
   information value ranking; key interactions (`Fault × BasePolicy`,
   `VehicleCategory × VehiclePrice`, `AgeOfPolicyHolder × Fault`,
   `PastNumberOfClaims × AddressChange_Claim`, claim lag × `PoliceReportFiled`);
   sensitive attributes (`Sex`, `MaritalStatus`, `Age`) and whether to exclude
   them.
4. **Outputs used later:** a feature-engineering plan, and red-flag rules with
   support and lift added to `config/policy_rules.yaml` (type `red_flag`, next to
   the validation rules).
5. Charts: consistent palette, labelled axes, CIs and n on rates, no pie charts.

## 4. Step B — B200 training → `notebooks/02_model_training.ipynb` (CHECKPOINT)

### 4.1 Portability: how I'll run it

`scripts/make_b200_bundle.py` builds `b200_bundle.zip` containing `src/`,
`config/`, `data/processed/*.csv` and `requirements-train.txt`. I upload the zip
and the notebook to the B200 Jupyter server. The notebook's first cells must:

- unzip the bundle if it sits next to the notebook (or use the repo if it is
  already there), add it to `sys.path`, and set `PROJECT_ROOT`;
- `pip install -r requirements-train.txt` **without reinstalling or upgrading
  torch/CUDA** (use the server's preinstalled PyTorch);
- print an environment report: `nvidia-smi`, GPU name, torch and CUDA versions,
  `torch.cuda.is_bf16_supported()`, versions of xgboost, catboost, lightgbm,
  sklearn and optuna. The B200 is Blackwell (sm_100): check that torch was
  built with sm_100 support, and fail early with a clear message if not;
- have a `FAST_MODE` flag (a 20k-row sample, a few trials) so I can smoke-test
  "Run All" on a CPU laptop first; and `RUN_SECTIONS` toggles so I can rerun one
  part;
- run top to bottom with "Run All" and no manual edits beyond the config cell;
- write everything to `outputs/` (models, metrics, figures, run log) and finish by
  zipping `outputs/` into `b200_outputs_<timestamp>.zip` for download, because
  the server is ephemeral. Checkpoint each finished model to disk as it completes
  so a disconnect doesn't lose the whole run.

### 4.2 GPU usage (the data is only 300k rows, so be pragmatic)

- XGBoost: `device="cuda"`, `tree_method="hist"`. CatBoost: `task_type="GPU"`.
  LightGBM on CPU (its GPU build is unreliable; the node has plenty of cores).
  Random Forest and Logistic Regression on CPU with `n_jobs=-1` (cuML is
  optional).
- PyTorch MLP: entity embeddings for categoricals, bf16 autocast, large batches,
  weighted BCE or focal loss, early stopping on real-validation PR-AUC,
  `torch.compile` only if it works on the installed version.
- Optuna (TPE, 50–100 trials per finalist, pruning) with an objective of
  real-validation PR-AUC. Prefer stable configurations over a lucky top trial.

### 4.3 Experiments, in order

1. Baselines: all-legit, rules-only (red-flag rules), class-weighted Logistic
   Regression.
2. Random Forest, XGBoost, LightGBM, CatBoost, MLP, all class-weighted.
3. **Data-mix experiment:** real-only vs real + 10/25/50/100% synthetic vs
   synthetic-only, plus synthetic sample weights of 0.1, 0.3 and 1.0. Plot
   real-validation PR-AUC against synthetic volume.
4. Tune the best 1–2. Calibrate (isotonic or Platt, on real validation) and
   report Brier score and a reliability curve.
5. Thresholds for 5:1, 10:1 and 20:1 costs and for top-10% capacity, saved as
   risk bands (`low`/`medium`/`high`) in `model_card.json`.
6. **Final test evaluation, once:** all metrics with CIs vs baselines; SHAP
   global and local explanations for TP, FP and FN examples; error analysis;
   fairness (recall, FPR and flag rate by `Sex`, age band and `MaritalStatus`).

### 4.4 Artifacts (must load on my CPU laptop)

- `models/preprocessor.joblib` (fitted `ColumnTransformer` from
  `src/ml/preprocess.py`), `models/fraud_model.joblib` (calibrated final model,
  **switched to CPU inference** before saving, e.g. XGBoost `device="cpu"`;
  if the MLP wins, save TorchScript or `state_dict` + config and give
  `predict.py` a CPU loader), `models/shap_explainer.pkl`, and
  `models/model_card.json` (metrics with CIs, thresholds and risk bands,
  features, data hashes, training date, GPU and **exact package versions**).
- `requirements.txt` pins the **same** sklearn, xgboost, catboost, lightgbm and
  shap versions that the B200 run used, so the joblib files load locally. Add a
  test that loads the artifacts and scores `tests/fixtures/sample_claims.json`.
- `src/ml/explain.py` falls back to building a fresh `TreeExplainer` from the
  model if unpickling `shap_explainer.pkl` fails.
- `scripts/train_model.py` runs the same pipeline headless (for reruns
  without Jupyter); the notebook calls `src/ml/train.py` as well.
- `docs/report/02_model_report.md` and `docs/report/model_card.md`.

At this checkpoint, give me exact upload/run/download instructions. Until I bring
the real artifacts back, train a `FAST_MODE` model locally and save it as the
placeholder artifacts, so Step C can be built and tested.

## 5. Step C — Tools, retrieval and agent (CHECKPOINT)

1. **Schemas.** `src/schemas/claim.py`: pydantic `ClaimSchema` with all 33 raw
   fields and allowed values (`PolicyNumber` is an optional id).
   `decision.py`: `TriageDecision {claim_id, decision, fraud_probability,
   risk_band, rationale, evidence: [{source_tool, fact}], validation_issues,
   similar_claim_ids, requires_human_review: true}`. `state.py`: LangGraph
   `ClaimState` (TypedDict with message history, claim, tool results, step
   count, decision).
2. **Ingestion (minimal):** `parser.py` reads CSV rows, JSON or a form dict;
   `normalizer.py` turns them into `ClaimSchema` and lists missing or invalid
   fields instead of raising.
3. **Retrieval:** `encoder.py` encodes a claim with the fitted preprocessor
   (L2-normalised). `build_index.py` / `scripts/build_vector_index.py` builds a
   FAISS `IndexFlatIP` over **real training claims only** (no synthetic, no
   validation/test rows), with `claims_meta.parquet` holding ids, key fields and
   fraud labels. `search.py` returns the top k with similarity scores. Check
   that neighbours look sensible.
4. **Tools** (`src/tools/`): plain typed functions wrapped as LangChain tools,
   with `TOOLS = [...]` exported from `__init__.py`; JSON-serialisable output;
   each catches its own errors and returns `{ok: false, error}`.
   - `fraud_score_tool`: probability, risk band, top-5 SHAP factors in plain
     English, model version.
   - `similar_claims_tool`: top-k similar past claims, their fraud rate, and key
     differences.
   - `policy_check_tool`: validation and consistency rules plus red flags from
     `config/policy_rules.yaml`.
5. **Agent** (`src/agent/`, LangGraph):
   - `llm.py`: provider switch from `.env` (`LLM_PROVIDER` = anthropic, openai or
     ollama; `MODEL_NAME`; `LLM_API_KEY`). Default to Anthropic `claude-sonnet-5-5`;
     use `claude-haiku-4-5-20251001` for cheap batch runs.
   - Graph (`graph.py`): `ingest → validate → score → agent ⇄ tools` (a ReAct
     loop bounded by `MAX_STEPS` from settings) `→ synthesize → guardrails →
     human_review` (LangGraph `interrupt`, with a SQLite or memory checkpointer)
     `→ END`. Validate and score always run; the agent decides which further
     tools to call (similar claims and policy checks), as the synopsis requires.
   - `synthesize`: structured output into `TriageDecision` using
     `config/prompts/synthesis.md`. Every rationale point must cite a tool
     result.
   - `guardrails.py`, enforced in code (not only in the prompt): only the three
     allowed decisions, and never a denial; any blocking validation failure
     forces REQUEST_MORE_INFO; a `high` risk band can't be APPROVE; any
     evidence fact not traceable to a tool output is removed and logged; if the
     LLM fails or returns an invalid output twice, fall back to a deterministic
     rule-based decision. Everything is marked for human review.
   - `runner.py`: `run_claim(claim) -> (thread_id, TriageDecision, trace)` and
     `resume(thread_id, adjuster_decision)`.
6. **Tests:** use a fake chat model (LangChain `GenericFakeChatModel` with
   scripted tool calls) so the full suite runs with no API key and no network.
   Cover validation rules, each tool, each guardrail and the graph paths
   (approve, flag, request info, LLM failure fallback, max-steps cutoff).

## 6. Step D — Agent evaluation → `notebooks/03_agent_evaluation.ipynb` (CHECKPOINT)

Runs locally (needs an LLM key, not a GPU). Develop the prompts on validation
claims; run the final evaluation once on test claims. Use a stratified sample:
all test frauds plus a matched number of legit claims, with the size
configurable to control cost. `scripts/run_batch_triage.py` runs it end to end
and writes JSONL to `evaluation/results/`. `evaluation/agent_eval.py` reports:

- decision quality vs labels (fraud → FLAG recall, legit → APPROVE rate,
  REQUEST_MORE_INFO rate), compared with the model-only threshold policy;
- tool-use rate per tool, average steps, MAX_STEPS hits;
- guardrail interventions, fallback rate, evidence-grounding failures;
- rationale quality on a sample: an LLM-as-judge rubric (cites evidence,
  consistent with the score, no invented facts) plus 20 claims for me to review
  by hand;
- latency and token cost per claim; run-to-run consistency on 20 claims run 3
  times.

Write `docs/report/03_agent_evaluation.md`, a `docs/architecture.png` diagram of
the graph, and update `README.md` (setup, B200 workflow, how to run each step,
headline results, synopsis checklist with deferred items marked).

Start with step 0. Before moving files, show me the migration plan (old path →
new path).
