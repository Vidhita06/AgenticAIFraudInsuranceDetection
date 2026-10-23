# Tech Stack and Architectural Rationale

_Agentic AI: Autonomous Insurance Claims Triage & Fraud Assistant (IDEA Lab)._

This document lists every technology the project uses and explains, for each one:

- what problem it solves here;
- why it was chosen over the alternatives;
- what it costs us (its known limitation).

All numbers come from the project's own reports:

- `docs/report/01_eda_report.md`: EDA and synthetic-data audit;
- `docs/report/02_model_report.md` and `model_card.md`: the fraud model;
- `docs/report/03_agent_evaluation.md`: the agent evaluation;
- `docs/report/03b_medium_band_factors.md`: the medium-band analysis.

Where something is not yet measured, the document says so.

---

## 1. System at a glance

```
Claim (CSV row / dashboard form / JSON)
  │
  ▼
Layer 1  ingest → validate (V01–V12 rules)                  [deterministic code]
  │
  ▼
Tool 1   score: calibrated LightGBM + red flags             [deterministic ML]
  │
  ▼
Layer 2  agent ⇄ tools (LLM ReAct loop, ≤ 8 steps)          [LLM plans; code executes]
           Tool 2 SHAP explanation      Tool 5 similar claims (FAISS)
           Tool 3 policy retrieval      Tool 6 claim history
           Tool 4 policy verification
  │
  ▼
         complete_investigation (runs any required tool the agent skipped, logged "auto")
  │
  ▼
Layer 4  aggregate_evidence (statuses, conflicts, reason codes, missing info)  [code]
  │
  ▼
Layer 5  synthesize (LLM writes decision + rationale + cited evidence)
         → guardrails (code can override; rule-based fallback)
  │
  ▼
Layer 6  human_review (LangGraph interrupt) → adjuster decides → SQLite audit store
```

The diagram is `docs/architecture.png`; `docs/make_architecture_diagram.py` regenerates it.

**Core design principle: the LLM orchestrates and explains; it never computes, and it never
decides alone.**

- Scores, thresholds, policy checks, reason codes, conflicts and tool statuses are all
  computed in Python.
- The LLM chooses which tools to call and writes the decision and rationale.
- Guardrails check that output against the evidence, and a human adjuster makes the final
  call.

Only three decisions exist: `APPROVE`, `FLAG_FOR_INVESTIGATION` and
`REQUEST_MORE_INFORMATION`. The system can **never deny** a claim.

---

## 2. Language and runtime

| Technology | Version | Why | Alternatives considered |
|---|---|---|---|
| **Python** | 3.11 (training); 3.11–3.13 supported, 3.11/3.12 recommended | The ML (LightGBM, SHAP, scikit-learn), agent (LangGraph/LangChain) and UI (Streamlit) ecosystems are all Python-first, so a single language covers the whole stack. 3.11 was the B200 server's Python. Every pin has wheels for 3.11–3.13 on Windows and Linux (§11). | **R:** no agent frameworks. **Java/JS:** weak tabular ML and SHAP support. Python 3.13 works, but 3.11/3.12 is recommended because those are the versions the training ran on and the wheels have been out longest. |
| **NVIDIA B200 GPU** (CUDA 12.8+) | — | The university's training server. Used for the model zoo and Optuna searches: XGBoost `device=cuda`, CatBoost GPU, MLP in bf16. It also served `qwen2.5:72b` through Ollama for the agent pilot. | A laptop CPU would take hours for 300k-row Optuna searches. **Not needed at inference time:** the final model runs on CPU. |
| **CPU laptop / small cloud VM** | — | The deployment target. The LightGBM artifact is 1.9 MB, inference is CPU-only, and the dashboard peaks at about **445 MB RAM**. | A GPU at inference: unnecessary for a gradient-boosted tree model. |

---

## 3. Data layer and preprocessing

| Technology | Version | Role in this project | Why chosen |
|---|---|---|---|
| **pandas** | 3.0.6 | Loading the CSVs (real train/validation/test, 300k augmented), EDA, feature engineering, evaluation tables. | The standard tabular tool. The data fits in memory easily (300k × 33). **Polars** would be faster, but nothing in the pipeline is pandas-bound, and every downstream library (scikit-learn, SHAP, Streamlit) takes pandas directly. |
| **NumPy** | 2.4.6 | Arrays under every model, SHAP values, FAISS vectors. | Required by everything else. NumPy 2.x is needed by the pandas 3 / scikit-learn 1.9 pins. |
| **SciPy** | 1.17.1 | χ² tests, Wilson intervals and other statistics in the EDA and synthetic audit. | Standard. Already a scikit-learn dependency, so it adds nothing. |
| **scikit-learn** | 1.9.1 | `ColumnTransformer` preprocessing (`src/ml/preprocess.py`), the logistic-regression and Random Forest baselines, Platt/isotonic calibration, metrics (PR-AUC, ROC-AUC, Brier), stratified splits and CV. | It gives one fitted, picklable preprocessing object, so training and inference use *exactly* the same transform (no train/serve skew). Pinned to equal `requirements-train.txt`, because a pickled scikit-learn object should be loaded with the version that saved it. |
| **PyArrow** | 25.0.1 | Fast parquet/CSV I/O backend for pandas 3. | pandas 3 uses Arrow-backed strings; pinning it avoids version drift. |
| **Pydantic** | 2.13.4 | Schemas for the claim (`src/schemas/claim.py`), agent state, `TriageDecision`, `EvidenceItem`, `ToolStatusEntry`. | Validates the LLM's JSON output against a strict schema: an invalid decision label or a missing field is caught, not passed on. **dataclasses** have no validation; **marshmallow** is older and not what LangChain uses. |
| **Custom validation rules** (`src/validation/`) | — | V01–V12: schema and missing values (e.g. Age = 0 → blocking), plus consistency rules (PolicyType = VehicleCategory + BasePolicy, claim timeline, early-policy incident...). | The Phase 1 data audit found real data quirks that need explicit, testable rules. A generic validator such as Great Expectations cannot express "Sport + Liability is recorded as Sedan - Liability". The rules also feed a model feature (the consistency-failure count). |

**Preprocessing decisions, each made by the EDA (`01_eda_report.md` §5):**

- **`PolicyNumber` and `is_synthetic` are never features.** PolicyNumber perfectly separates
  real from synthetic rows, so using it would be leakage.
- **`PolicyType` and `AgeOfPolicyHolder` are display-only.** Each is fully determined by other
  columns.
- **`Sex` and `MaritalStatus` are excluded for fairness.** The ablation showed no predictive
  gain from them.
- **Raw `Age` is replaced** by `age_band` + `age_missing`.
- **Claim-date fields are replaced** by lag and weekend features.
- **`RepNumber` is excluded:** no signal.
- **Encodings:** nominal → one-hot (rare levels with fewer than 20 rows grouped); ordinal
  bins → ordered integers; `Month`/`DayOfWeek` → cyclic sin/cos.
- **Red flags:** 12 rule-based red-flag indicators plus a weighted red-flag score (RF01–RF13
  from the EDA), and the consistency-failure count.
- **Result:** 89 model columns.

---

## 4. Synthetic data

| Item | Choice | Why |
|---|---|---|
| **Source** | 289,207 synthetic claims supplied with the project (`augmented_train_300k.csv`). The generator was a black box, fitted on the 70% train split only. | Only 646 real training frauds exist, so more examples of the minority class help the trees. |
| **Audit** (`src/ml/synthetic_audit.py`) | Marginal fidelity (TVD at most 0.008); fraud-rate agreement per category (r = 0.89); **adversarial validation** (a classifier separating real from synthetic gets ROC-AUC **0.511**, i.e. chance); **DCR** (distance to closest record) for memorisation and privacy; **TRTR vs TSTR** utility. | We never trust synthetic data blindly. These checks prove it is realistic, not a copy of real rows, and actually useful. |
| **How it is used** | All synthetic rows are added to training at **sample weight w = 0.1**. Validation and test are **real only**. | The data-mix experiment showed +0.04–0.05 PR-AUC over real-only, and a down-weighted mix slightly beat w = 1.0. Real data stays the only ground truth for model selection and testing. |
| **Imbalance handling** | `scale_pos_weight` (class weights, about 15.7); **no SMOTE / oversampling**. | 96% of rows are already synthetic. SMOTE would add a second layer of artificial points and can create unrealistic categorical mixes. Class weights change the loss without inventing data. |

---

## 5. Fraud-scoring model (Tool 1)

### 5.1 LightGBM: why it is the final model

| Model | Real-validation PR-AUC | Notes |
|---|---|---|
| **LightGBM** (defaults, 3 seeds, calibrated) | **0.291** [0.238, 0.363] | Final model |
| XGBoost | 0.283 | Within noise of LightGBM |
| MLP (PyTorch, tuned, 3 seeds) | 0.262 | Every MLP member is below every LightGBM member |
| Random Forest | 0.246 | |
| Logistic regression (class-weighted) | 0.193 | Linear baseline |
| Rules-only red-flag score | 0.142 | Domain-rules baseline |
| CatBoost | not compared | Not installed in the partial run; the final run's zoo table was not uploaded (stated honestly in the model report §4) |

**Why LightGBM over XGBoost** (the honest answer: they are within noise, ±0.06 bootstrap CI):

- **Best validation PR-AUC.** It was highest on the primary metric, chosen on real
  validation only.
- **Speed.** Histogram-based, leaf-wise growth trains fastest on 300k rows. This matters for
  the 3-seed ensembles and the Optuna trials.
- **Small artifacts.** `fraud_model.joblib` is 1.9 MB and loads
  in milliseconds on CPU.
- **Exact TreeSHAP.** `shap.TreeExplainer` gives exact, fast SHAP values for tree ensembles.
  This is essential for Tool 2: an adjuster must see *why* a claim was scored.
- **CPU inference.** No GPU or CUDA runtime is needed in deployment.

**Why not CatBoost:** it handles categoricals natively and is often strongest on such data.
But it was not available in the comparison runs. It is a listed follow-up, not a rejected
option. It stays pinned (1.2.10) so `train.py` can include it.

**Why not the MLP (PyTorch):**

- It scored lower: −0.029 PR-AUC (paired-bootstrap CI [−0.074, +0.012]).
- It is slower.
- It needs PyTorch at inference.
- It has no exact TreeSHAP.

Per the protocol, it never touched the test set.

**Why not plain scikit-learn models:**

- **Logistic regression** is the interpretable baseline (0.193) but misses non-linear
  interactions (e.g. Fault × BasePolicy).
- **Random Forest** (0.246) is weaker than boosting and produces larger artifacts.

### 5.2 Ensembling, early stopping, tuning

| Technique | Why |
|---|---|
| **3 seeds (42/43/44), averaged** | Seed-to-seed PR-AUC variation (SD about 0.003–0.004) is comparable to the gaps between options. Averaging gives a stabler score and a fair basis for comparisons. |
| **Early stopping on real-validation PR-AUC** | Picks the tree count (488/490/381) on the metric we care about, using real data. |
| **Optuna 5.0.0** (TPE sampler + median pruner, SQLite storage) | A Bayesian search is more sample-efficient than grid search. The pruner stops bad trials early. SQLite storage makes a long B200 search resumable. **Outcome:** tuned parameters did not beat library defaults over the same seeds, so the defaults were kept. The adoption rule compares means over the same seeds, not one lucky trial. |

### 5.3 Calibration and thresholds

| Item | Choice | Why |
|---|---|---|
| **Calibration** | **Platt scaling**, 5-fold cross-fitted on real validation | Class weights push raw scores far above true probabilities (uncalibrated Brier 0.159). Platt: 0.0485; isotonic: 0.0500. Platt has 2 parameters, so it is more stable on only 139 validation frauds; isotonic can overfit with so few positives. **Test Brier: 0.0488** (base-rate predictor: 0.056). |
| **Decision thresholds** | **Cost-based.** Missed fraud : unnecessary investigation = 10:1 (default) → **0.108**. Also reported: 5:1 → 0.179, 20:1 → 0.037, top-10% capacity → 0.189. | A threshold of 0.5 is meaningless with 6% prevalence. The cost ratio is a business assumption, so we report several and make the ratio configurable (`decision.cost_ratios`). The thresholds were chosen on out-of-fold validation scores and held up on test (recall 0.791 → 0.790). |
| **Risk bands** | high ≥ 0.108; medium 0.037–0.108; low < 0.037 | Bands map onto the cost thresholds, so the agent gets one simple, auditable signal. |

**Final test result** (evaluated once): ROC-AUC 0.874, PR-AUC 0.252, recall 0.790, precision
0.208, flag rate 22.7%. A random flagger at the same flag rate gets precision 0.060 and
recall 0.227.

### 5.4 Explainability: SHAP 0.51.0 (Tool 2)

- **TreeSHAP** on the uncalibrated ensemble, in log-odds.
- **One-hot levels are summed back per original feature.** An adjuster sees "BasePolicy",
  not "BasePolicy_All Perils".
- **Why SHAP over alternatives:**
  - **LIME** samples randomly, so the same claim can get different explanations.
  - **Permutation importance** is global only.
  - **Built-in gain importance** is global and biased toward high-cardinality features.
  - **TreeSHAP** is exact, deterministic, additive (base value + contributions = score) and
    fast.
- **Known anomaly (reported honestly):** accident Month ranks 3rd in SHAP, more than its
  univariate EDA signal explains. A Month ablation is a listed follow-up.

### 5.5 Artifacts: joblib 1.6.0

- Persists the preprocessor, the 3 LightGBM boosters, the calibrator and the thresholds.
- **Why joblib:** it handles NumPy-heavy scikit-learn objects efficiently, and it is the
  scikit-learn-recommended format.
- **Why not ONNX:** ONNX would need a converter for the calibrator + preprocessor pipeline,
  and would lose SHAP compatibility.
- **Constraint:** load with the same library versions that saved the artifact. That is why
  `requirements.txt` and `requirements-train.txt` share the same ML pins.
- `model_card.json` records the effective parameters, the version id
  (`lightgbm-b200-20261016-0915`) and the metrics.

---

## 6. Similar-claims retrieval (Tool 5): FAISS

| Item | Choice | Why |
|---|---|---|
| **Library** | **faiss-cpu 1.15.1** | Mature, fast and runs in-process. The CPU wheel is available on Windows/Linux for 3.11–3.13. |
| **Index type** | **`IndexFlatIP`** (exact inner product) on **L2-normalised vectors** = exact cosine similarity | 10,793 vectors is tiny. Exact search takes about a millisecond, so an approximate index (IVF, HNSW) would add tuning and recall loss for no benefit. |
| **Vectors** | The model's own "linear" preprocessor output (`src/retrieval/encoder.py`), not text embeddings | Claims are structured rows, not text. A sentence-embedding model would add a dependency (and possibly a GPU) and would measure *wording* similarity. Encoding with the model's features means "similar" uses the same view of a claim as the fraud model. |
| **Indexed claims** | **Real training claims only** (10,793), never synthetic, never validation/test | Precedent shown to an adjuster must be a genuine historical claim. Excluding validation/test prevents label leakage into evaluation. Every match carries provenance. |
| **Staleness fingerprint** | The index stores a hash of the encoder/training data | If the preprocessor changes, the tool refuses a stale index instead of silently returning wrong neighbours. |
| **Why not a vector DB** (Chroma, Pinecone, Weaviate, pgvector) | — | A server or cloud service is unnecessary for 10.8k static vectors. Pinecone would also send claim vectors to a third party (privacy). FAISS is a single file loaded in memory. |
| **Optional filter** | `same_base_policy` | Lets the agent compare like with like (e.g. only All Perils claims). |

---

## 7. Agent and orchestration layer

### 7.1 LangGraph 1.2.12

**What it does here:**

- Defines the explicit state graph of §1.
- Runs the agent ⇄ tools ReAct loop with `max_steps = 8`.
- Pauses for the adjuster with an **interrupt** and checkpoints the state (`InMemorySaver`),
  so `resume()` continues the same investigation.

**Why LangGraph:**

- **Deterministic and LLM steps mix in one graph.** Validation, scoring, evidence aggregation
  and guardrails are ordinary Python nodes, and only two nodes call an LLM.
  - A free-running agent (a plain LangChain `AgentExecutor`, AutoGPT-style loops) could skip
    the fraud score or the guardrails.
  - Here the graph structure makes them unskippable.
- **Human-in-the-loop is built in.** `interrupt` + checkpointer is exactly the "adjuster
  reviews before anything is final" requirement.
- **Step limits and an auditable trace.** Every node and tool call is recorded with its
  origin: `agent`, `dependency` or `auto`.

**Alternatives considered:**

- **CrewAI / AutoGen:** multi-agent conversation frameworks. They are harder to make
  deterministic, and multiple agents are unnecessary: one orchestrator plus tools is enough.
- **Hand-written loop:** possible, but it would re-implement checkpointing, interrupts and
  tool-call plumbing.
- **LlamaIndex:** focused on document RAG, which this project is not.

### 7.2 LangChain core 1.6.11 + provider packages

| Package | Version | Why |
|---|---|---|
| `langchain-core` | 1.6.11 | Common message/tool interfaces, so the same graph runs on any provider. Pinned ≥ 1.6.10 because `langchain-openai` 1.7.2 requires it; the earlier 1.6.6 pin broke the B200 install. |
| `langchain-anthropic` + `anthropic` | 1.7.5 / 1.11.0 | Claude (default `claude-sonnet-5-5`; Haiku for batch). Strong tool calling and JSON adherence. Paid, with prices in `settings.yaml` for cost estimates. |
| `langchain-openai` | 1.7.2 | **Any OpenAI-compatible API** through `OPENAI_BASE_URL`: Groq, OpenRouter, Google Gemini's OpenAI endpoint, vLLM, LM Studio. One package covers the free-tier hosted options. |
| `langchain-ollama` | 1.1.0 | **Fully local/offline LLMs** (default `qwen2.5:7b`; `qwen2.5:72b` on the B200). `num_ctx` is raised to 16384 (the Ollama default of 2048 truncates our prompts); `format=json` for synthesis; temperature 0. |
| `python-dotenv` | 1.2.2 | Loads `LLM_PROVIDER`, `MODEL_NAME`, `LLM_API_KEY`, `OPENAI_BASE_URL` and `OLLAMA_BASE_URL` from `.env`. Keys never live in code or git: `.env` is git-ignored. |

**Why a provider switch (`src/agent/llm.py`) instead of committing to one LLM:**

- **Privacy:** Ollama keeps claims on our own hardware.
- **Cost:** a free tier (Groq/Gemini) for demos, Claude for best quality.
- **Availability:** no single vendor outage or deprecation stops the project.

The LLM is a replaceable component; the safety comes from code (§8).

**Robustness code around the LLM:**

- lenient JSON parsing (strips fences, finds the JSON object);
- recovery of tool calls written as text by open models;
- claim-bound tools: the LLM chooses tools and arguments but can never retype claim fields;
- dependencies enforced in code (`verify_policy` runs `retrieve_policy` first);
- `complete_investigation` runs the required tools if the LLM skipped them.

### 7.3 The six tools (`src/tools/`)

| Tool | Implementation | Why it is a separate tool |
|---|---|---|
| 1 Fraud Score | calibrated LightGBM + red flags; always runs first | The initial model-based risk assessment; never optional |
| 2 SHAP Explanation | TreeSHAP, signed factors | Explanations are expensive to read and needed mainly in medium/high bands; required by default |
| 3 Policy Retrieval | the policy store `data/policies/*.json`, or the simulated DB, or claim-recorded attributes (`partial`) | The data has **no policy documents**; the tool says so instead of inventing terms |
| 4 Policy Verification | PV01–PV09 + simulated clauses → PASS/FAIL/UNKNOWN, with `level` claim_record or policy_condition | Separates "the claim contradicts itself" (→ more info) from "the claim breaks a policy condition" (→ coverage flag) |
| 5 Similar Claims | FAISS (§6) | Precedent for the adjuster; required |
| 6 Claim History | authorised CSV/JSONL, or the simulated DB, else `unavailable` | `PolicyNumber` is unique per claim, so the real data has no linked history; the tool reports this instead of claiming "clean history" |

**Statuses** are `completed`, `partial`, `unavailable`, `failed` and `not_run`. They keep "we
checked and found nothing" apart from "we could not check".

### 7.4 Evidence aggregation (`src/agent/evidence.py`)

This is deterministic code that builds the evidence record the LLM reasons over:

- per-tool status and origin;
- missing information;
- **conflicts** (e.g. high band but similar claims are at the base fraud rate);
- **reason codes**: `POLICY_CONDITION_FAILED`, `CLAIM_RECORD_CONTRADICTION`,
  `HISTORY_CONTRADICTION`, `POLICY_TERMS_UNVERIFIED`, ...

**Why in code:** these fields go into the final output. If the LLM wrote them, it could
hallucinate a conflict or a missing check.

---

## 8. Guardrails and fallback (`src/agent/guardrails.py`)

Rules applied after the LLM's synthesis; any violation is overridden and noted in
`guardrail_notes`:

1. Only the three decisions exist; **deny is impossible**.
2. **High band** → never APPROVE.
3. **APPROVE** requires all of: a fraud score, a consistent claim record, no
   policy-condition FAIL, and no failed required tool.
4. **REQUEST_MORE_INFORMATION** only with claim-level grounds: blocking validation, or a
   PV02/PV04 contradiction. A tool failure is a SYSTEM_FAILURE flag, not "ask the claimant".
5. **`unsupported_fraud_flag`:**
   - A low-band flag whose reason codes are only systemic gaps (missing policy text, missing
     history) is replaced.
   - A medium-band flag without an out-of-model factor (`medium_band_concrete_factors`) is
     replaced.
6. **Grounded evidence:** every cited number or ID must appear in the cited tool's output. An
   `unavailable` source can be cited only for that limitation.
7. **Flag categories:** each flag is tagged FRAUD_RISK, POLICY_CONDITION or SYSTEM_FAILURE.
   Fraud metrics count FRAUD_RISK only.
8. **Rule-based fallback:** after 2 failed LLM attempts (timeout, invalid JSON, provider
   down), `rule_based_decision()` produces a decision from the same evidence. The system
   degrades gracefully instead of failing.
9. **Simulated evidence must be labelled** "simulated" in the rationale.

**Why rules 5 and the medium-band list exist** (evidence-driven, from the pilots):

- The first pilot flagged low-band claims because the policy document was missing, which is
  true for every claim.
- The second flagged every medium-band claim.
- The offline analysis (`03b_medium_band_factors.md`) showed that red flags, SHAP drivers and
  similar-claim fraud do **not** separate fraud inside the medium band, because the model has
  already used them.
- So only information the model does not see can trigger a medium-band flag.
- After the fix, the pilot agreed with the model policy on 10/10 claims in both modes.

---

## 9. Simulated reference database (`src/simulated/`)

| Item | Choice | Why |
|---|---|---|
| **Purpose** | Gives Tools 3, 4 and 6 realistic policy terms, clauses and prior claims, so the agent's reasoning over them can be demonstrated | The Kaggle data has no policy documents or history |
| **Storage** | **SQLite** (`data/simulated/insurance.db`) | Zero-install (Python stdlib `sqlite3`), a single file, SQL queries, a reproducible manifest of SHA-256 table hashes |
| **Generation** | Deterministic per-policy RNG (seed 42); dates derived from the week/weekday fields; conservative reporting lag | Reproducible builds; plausible dates |
| **Label independence** | The generator reads an explicit **`INPUT_COLUMNS` allow-list**; a test proves `FraudFound_P` is never read | Otherwise simulated "evidence" could leak the answer and inflate results |
| **Labelling** | Every output carries `source: "simulated reference data"` | Nobody can mistake it for a real record |
| **Clauses** | `config/simulated_policy_clauses.yaml` (SIM-ALL-01…05, SIM-LIA, SIM-COL, SIM-AP); the reporting deadline is set to 60 days after a fail-rate check | A clause failing on more than about 5% of claims would flag claims for an artefact of the simulation |
| **Toggle** | `simulated_data.enabled: false` by default; evaluation runs `off`, `on` or `both` | With it off, the tools behave exactly as before |

**Measured effect (offline, 278 validation claims):** off vs on gave the same FRAUD_RISK recall
(0.748). It added one POLICY_CONDITION flag and changed one approval. This shows the
simulated data affects coverage questions only, not fraud detection, which is what we want.

---

## 10. Adjuster dashboard and audit

| Technology | Version | Why |
|---|---|---|
| **Streamlit** | 1.62.0 | A pure-Python UI that calls the same `TriageRunner` as the batch scripts. No second codebase, no REST API, no JavaScript build. A team of ML students can maintain it. **Flask/FastAPI + React** would need an API layer, a frontend build and auth plumbing for a prototype UI. **Gradio** is aimed at model demos, not multi-page review workflows with forms. |
| **Altair** | 6.3.0 | Declarative charts (SHAP bars, score position against thresholds); Streamlit's native chart library. **matplotlib** images are static; **Plotly** is a heavier dependency. |
| **SQLite audit store** (`data/audit/audit.db`, `src/agent/audit.py`) | stdlib | Tables `recommendations` and `adjuster_decisions`. An override needs a written reason, and a claim can be reviewed only once. This gives a durable, queryable audit trail with zero setup. Git-ignored, because it holds review data. **PostgreSQL** would be the production upgrade (multi-user, concurrent writes). |
| **CPU-only** | — | The dashboard runs on a laptop without a GPU. Peak about 445 MB RSS; the LLM is remote (API) or local (Ollama). |

---

## 11. Dependency management and packaging

| Item | Choice | Why |
|---|---|---|
| **Exact pins (`==`)** in `requirements.txt` | — | Pickled artifacts must load with the versions that saved them, and exact pins make the B200 and laptop installs identical. |
| **`requirements-train.txt`** | the ML subset, same pins | The B200 already has a CUDA build of torch (Blackwell sm_100 needs CUDA 12.8+), so torch is deliberately **not** listed: reinstalling it could break the GPU build. CPU torch is optional (only to load an MLP artifact). |
| **PyYAML 6.0.2** | — | 6.0.1 has **no cp313 Windows wheel**, so pip tried to build from source and failed. 6.0.2 ships wheels for 3.8–3.13. |
| **langchain-core 1.6.11** | — | Fixed a resolver conflict with `langchain-openai` 1.7.2 (needs ≥ 1.6.10). |
| **Wheel verification** | `uv pip compile --python-platform x86_64-pc-windows-msvc --only-binary :all:` + PyPI JSON / `packaging.tags` checks | Resolved 144 packages, every one with a binary wheel for Windows on 3.11, 3.12 and 3.13, so no compiler is needed. pip's own cross-platform mode mis-evaluated a Linux-only marker in xgboost, so **uv** was used for the resolution. Clean Linux installs on 3.11/3.12/3.13 pass all tests. |
| **uv** | (tool) | A fast resolver that can resolve *for another platform*, which plain pip cannot do reliably. Only a development tool; users still run `pip install -r requirements.txt`. |
| **YAML config** (`config/settings.yaml`) | PyYAML | Every threshold, path, required tool and factor list lives in one human-readable file. Behaviour changes without code changes, and the settings are reviewable in a viva. |
| **Markdown prompts** (`config/prompts/`) | — | Prompts are versioned text files, not code strings, so they are easy to diff and review. |

---

## 12. Testing, evaluation and reproducibility

| Technology | Version | Role / why |
|---|---|---|
| **pytest** | 9.1.1 | About 118 test functions (123 test cases with parametrisation): validation rules, every tool, guardrails, the graph (with a stub LLM, so no API needed), the simulated DB including the label-independence test, and the dashboard. The guardrail tests replay real pilot failures (e.g. the PN-5736 low-band case). |
| **Streamlit `AppTest`** | (in streamlit) | Headless UI tests of the dashboard in pytest, no browser. |
| **Playwright + Chromium** | (pre-installed) | Only for dashboard screenshots during development; not a dependency. |
| **Jupyter: ipykernel, nbconvert, nbclient** | 7.4.0 / 7.17.1 / 0.11.0 | Notebooks 01 (EDA), 02 (training) and 03 (agent evaluation) executed headless with `jupyter nbconvert --execute`, so the reports regenerate from code. |
| **matplotlib** | 3.11.2 | Static report figures (lift, reliability, SHAP, fairness, TVD, DCR, architecture diagram). |
| **Evaluation harness** (`evaluation/agent_eval.py`, `scripts/run_batch_triage.py`) | — | Stratified/band-stratified sampling; FRAUD_RISK-only recall and precision; prevalence re-weighting; **Wilson CIs**; **bootstrap CIs** (2000 iterations); consistency re-runs; LLM-as-judge; off/on comparison; `--resume`; **one test evaluation per model and mode**, enforced with marker files. |
| **Hashes and manifests** | SHA-256 | Input data hashes (data audit), simulated-DB manifest, FAISS staleness fingerprint. |
| **Git / GitHub** | — | Version control; PR-based review; the B200 pulls the branch. |

---

## 13. Security and privacy

- **Secrets:** API keys live only in `.env` or the environment settings, never in git or chat.
- **Data sent to an LLM:** the compacted evidence record (claim fields, scores, tool outputs).
  It contains no names or addresses; the dataset has none. `PolicyNumber` appears as an ID.
  - **Ollama: nothing leaves the machine.** This is the recommended setup for real claim
    data.
  - **Hosted APIs (Claude, Groq, Gemini):** data goes to a third party under that provider's
    terms. Free tiers may log or train on inputs, which is acceptable for the public Kaggle
    data only.
- **Fairness:** `Sex` and `MaritalStatus` are not features. Flag-rate and FPR gaps are under
  1.5 points between groups.
- **Human authority:** every decision has `requires_human_review = true`; the adjuster's
  decision and override reason are audited.

---

## 14. Deployment options

| Option | Fraud model | LLM | Notes |
|---|---|---|---|
| Laptop demo | CPU | Ollama `qwen2.5:7b`, or an API | Fully offline possible |
| Free cloud (Streamlit Community Cloud / Render free) | CPU | Groq / Gemini via `LLM_PROVIDER=openai` + `OPENAI_BASE_URL` | Peak about 445 MB fits a 512 MB tier tightly; rate limits; no SLA; needs a re-pilot with that model |
| B200 / GPU server | CPU (fine) | Ollama `qwen2.5:72b` | Best local quality; about 83 s per investigation in the pilot |
| Production (future) | CPU container | Contracted API or self-hosted vLLM | Add PostgreSQL audit, auth, monitoring and drift checks |

**If the LLM is unavailable,** the rule-based fallback still returns a guardrailed decision.
The system never blocks on the LLM.

---

## 15. Known limitations (stated, not hidden)

- **LightGBM vs XGBoost is within noise.** CatBoost was not compared, and the final run's
  zoo/tuning tables were not uploaded.
- **Liability claims:** recall is 0 (5 frauds, 0.6% rate). **All Perils:** FPR is 0.38.
- **Accident Month's SHAP rank is unexplained.** An ablation is pending.
- **No real policy documents or claim history exist.** Tools 3/4/6 rely on attributes, the
  simulated DB, or `unavailable`.
- **The live LLM test-set evaluation has not been run yet.** The pilot (10 claims, 10/10
  agreement) and the offline rule-based baseline are done. Free-API models need their own
  pilot before use.
- **The 10:1 cost ratio is an assumption,** not insurer data.
