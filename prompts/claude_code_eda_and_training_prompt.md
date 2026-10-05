# Claude Code Prompt — EDA → Model Training → Agent-Ready Fraud Scoring

> How to use: open Claude Code in the repo root and paste everything below the
> line as your first message. Claude Code works phase by phase and stops at each
> checkpoint for your go-ahead. It covers synopsis weeks 1–4 (dataset, EDA,
> ML/DL model) and prepares the tool interfaces for weeks 5–7 (agent,
> retrieval, dashboard).

---

You are working in the repo `AgenticAIFraudInsuranceDetection` — an IDEA Lab
project, **"Agentic AI: Autonomous Insurance Claims Triage & Fraud Assistant"**.
The final system: an LLM orchestrator (LangGraph) that takes a vehicle insurance
claim, validates it, calls a fraud-scoring model, retrieves similar historical
claims (FAISS/ChromaDB), runs policy checks, and produces a recommendation —
**Approve / Flag for Investigation / Request More Information** — with a written
rationale, reviewed by a human adjuster in a Streamlit dashboard whose decisions
feed back into retraining.

This session covers everything *before* the agent: rigorous EDA, a synthetic-data
audit, leakage-safe modelling, and packaging the model plus supporting functions
as clean, explainable, callable tools.

## 0. Context you must load first

1. Read `details/Project_Synopsis_Agentic_AI_Insurance_Claims (3).pdf` and
   `details/Agentic_AI_Insurance_Claims_Project_Proposal (1).docx` (use
   `pdftotext` and `unzip -p … word/document.xml` if no Python readers are
   installed). Write `docs/synopsis_requirements.md`: a checklist of every
   objective, deliverable, metric and success criterion, each tagged with the phase
   below that delivers it. If the synopsis asks for something this prompt doesn't
   cover, add it. Required items include: an EDA report; baseline (Logistic
   Regression, Random Forest) **and** advanced (XGBoost **and** a neural network
   in PyTorch or TensorFlow) models; evaluation with accuracy, precision, recall,
   F1 and AUC-ROC; class-imbalance handling; tools for fraud scoring,
   similar-claims retrieval and policy/history checks; false-positive rate on
   held-out claims; and "precision and recall both meaningfully above the naive
   baseline".
2. Data (already in the repo; never modify these files, write derived data to
   `data/interim/` or `data/features/`):
   - `data/raw/vehicle_fraud_oracle.csv` — original Kaggle data, 15,420 × 33,
     UTF-8 with BOM (read with `encoding="utf-8-sig"`). Target `FraudFound_P`.
   - `data/processed/augmented_train_300k.csv` — 300,000 × 34 (the 33 columns plus
     `is_synthetic`): 10,793 real training rows (646 fraud) + 289,207 synthetic
     rows (17,310 fraud), both at a 5.99% fraud rate.
   - `data/processed/real_validation.csv` — 2,313 real rows, 139 fraud (6.01%).
   - `data/processed/real_test.csv` — 2,313 real rows, 138 fraud (5.97%).
3. Facts already verified (re-check them in Phase 1, report any mismatch):
   - Real train / validation / test are a stratified 70/15/15 split of the
     original, disjoint by `PolicyNumber`. Exactly one original row is in none of
     them: `PolicyNumber` 1517, the row with `DayOfWeekClaimed = '0'`,
     `MonthClaimed = '0'` and `Age = 0` (dropped as invalid).
   - No synthetic row is an exact copy (all columns except `PolicyNumber`) of any
     real train, validation or test row, and there are no duplicate rows.
   - **Every synthetic row has `PolicyNumber` ≥ 15,421** (sequential), real rows
     have 1–15,420. `PolicyNumber` is therefore a perfect proxy for
     `is_synthetic` and must never be a feature.
   - `Age = 0` (missing-age sentinel, higher fraud risk): 232 real-train rows,
     5,242 synthetic.
   - `PolicyType` ≠ `VehicleCategory + " - " + BasePolicy` in ~32% of rows in both
     real and synthetic data — a known inconsistency in the original data.
   - Year distribution is similar across splits (1994 > 1995 > 1996).
   - Strong univariate signals: `Fault` (Policy Holder 7.82% vs Third Party 1.17%),
     `BasePolicy` (All Perils 10.28%, Collision 7.11%, Liability 0.77%),
     `VehicleCategory` (Utility 11.04%, Sedan 8.28%, Sport 1.42%), `VehiclePrice`
     extremes (<20k 8.48%, >69k 7.94%), `PoliceReportFiled`, `AgentType`.
   - `Deductible` is 400 for ~93% of rows; `DriverRating` is uniform over 1–4.

## 1. Non-negotiable rules (apply in every phase)

- **Real data is the only ground truth.** Model selection uses
  `real_validation.csv`; `real_test.csv` is opened **once**, for the final
  evaluation in Phase 6. Synthetic rows only ever go into training.
- **No leakage.** Every encoder, scaler, feature selector and resampler is fit
  inside a scikit-learn `Pipeline` on training data only. Drop `PolicyNumber` and
  `is_synthetic` from features; assess `RepNumber` (16 agent IDs) before keeping it.
  Don't apply SMOTE/oversampling on top of data that is already 96% synthetic
  unless an experiment shows it helps on real validation data.
- **Metrics.** Always report the synopsis metrics (accuracy, precision, recall, F1,
  ROC-AUC) **and** PR-AUC (average precision), which is the primary selection
  metric at 6% prevalence. Also report recall at a fixed precision, F2, the
  false-positive rate, and the naive baselines (all-legit predictor: 94% accuracy,
  0 recall; random scorer: PR-AUC ≈ 0.06). With only ~139 fraud cases per real
  split, give every metric a 95% bootstrap CI.
- Reproducibility: `RANDOM_STATE = 42`, pinned `requirements.txt`, a `config.yaml`
  for paths and parameters, and SHA-256 hashes of the input files logged in each
  report.
- Code lives in `src/` as importable modules; numbered notebooks in `notebooks/`
  only call `src/` functions. Figures go to `reports/figures/`, findings to
  `reports/*.md`. Add `pytest` tests for the validation rules, the feature
  pipeline and the tool functions. Large outputs (model binaries > 50 MB, run
  logs) go in `.gitignore`.
- **Stop at every CHECKPOINT:** summarise findings and decisions in ≤ 15 bullets,
  list open questions, commit, and wait for my approval.

## 2. Phase 1 — Data audit & validation rules (CHECKPOINT)

1. Load the four files with explicit dtypes, re-verify §0.3, and profile every
   column: dtype, cardinality, value set, and per-split frequency.
2. Write reusable claim-validation rules in `src/data/validation.py`. These
   become the agent's **policy-check tool** (synopsis Phase 2 "business-rule
   checks"). Each rule returns `{rule_id, passed, severity, message}`; report
   violation counts per split and origin:
   - Required fields present, values inside the known category sets.
   - `Age = 0` → missing age (→ "Request More Information" candidate).
   - `Age` inside the `AgeOfPolicyHolder` bin.
   - `PolicyType` consistent with `VehicleCategory` + `BasePolicy`.
   - Claim not before accident: `MonthClaimed`/`WeekOfMonthClaimed` vs
     `Month`/`WeekOfMonth`, allowing year wrap-around.
   - Coverage period: `Days_Policy_Accident` / `Days_Policy_Claim` = `"none"` or
     `"1 to 7"` (incident at the very start of the policy) and their ordering.
   - Vehicle age vs policyholder age, and other plausibility checks you find.
   Quantify rule hits by fraud rate — some violations may themselves be fraud
   signals. Don't silently "fix" them.
3. Deliverables: `src/data/load.py`, `src/data/validation.py`,
   `reports/01_data_audit.md`.

## 3. Phase 2 — Synthetic data audit (CHECKPOINT)

The model's value depends on whether 289k synthetic rows behave like real claims.
Compare real-train vs synthetic:

1. **Fidelity:** per-column distributions (total variation distance, chi-square),
   fraud rate *per category per origin* (does each red flag hold in both?),
   Cramér's V association matrices and their difference heatmap, mutual
   information with the target per origin, and rare-category coverage.
2. **Adversarial validation:** a LightGBM classifier for real vs synthetic
   (without `PolicyNumber`!). ROC-AUC ≈ 0.5 means indistinguishable; > 0.7 means
   detectable artefacts. Report the features driving it.
3. **Utility:** TRTR (train real) vs TSTR (train synthetic) vs train
   real+synthetic, all scored on `real_validation.csv` with the same model.
4. **Memorisation:** distribution of Gower distance to the closest record,
   synthetic→real-train vs real-validation→real-train. Also check near-duplicates
   of validation/test rows (e.g. ≥ 31/32 matching columns), which would mean the
   generator saw held-out data. Ask me how the synthetic data was generated
   (method, and whether only the 70% train split was used) if it's unclear.
5. Verdict: use all synthetic data, a filtered subset (drop rule violators or
   rows the adversarial model finds trivially synthetic), down-weighted samples,
   or none. Deliverable: `reports/02_synthetic_audit.md`.

## 4. Phase 3 — Fraud-focused EDA (the synopsis EDA report) (CHECKPOINT)

Run it on the training data (real-train first, confirm on synthetic); never on test.

1. Target distribution and imbalance by origin and split.
2. Univariate profiles of all features, grouped: accident & claim timing, policy,
   vehicle, policyholder, claim process (police report, witness, agent type,
   supplements, address change, past claims).
3. Fraud rate per category with Wilson CIs and support counts (sorted bars),
   chi-square + Cramér's V ranking, and information value / weight of evidence.
4. Interactions: `Fault × BasePolicy`, `VehicleCategory × VehiclePrice`,
   `AgeOfPolicyHolder × Fault`, `PastNumberOfClaims × AddressChange_Claim`,
   claim lag × `PoliceReportFiled`, `Year` trend.
5. Redundancy: `PolicyType` vs `VehicleCategory` + `BasePolicy`; `Age` vs
   `AgeOfPolicyHolder`; `Month` vs `MonthClaimed`. Decide what to keep.
6. Sensitive attributes (`Sex`, `MaritalStatus`, `Age`): document fraud-rate
   differences and recommend whether to exclude them from the model. Keep them
   available for the fairness audit either way.
7. Outputs that feed later phases: (a) a **feature-engineering plan**, (b)
   **human-readable red-flag rules** with support and lift in
   `reports/red_flag_rules.yaml`, used by the agent's policy-check tool and its
   rationale text.
8. Charts: consistent palette, labelled axes, fraud rates with CIs and n,
   no pie charts. Deliverables: `notebooks/03_eda.ipynb`, `reports/03_eda_report.md`
   (written so it can go straight into the final project report).

## 5. Phase 4 — Preprocessing & feature engineering (CHECKPOINT)

1. Ordinal-encode the range columns in their natural order (`VehiclePrice`,
   `Days_Policy_Accident`, `Days_Policy_Claim`, `PastNumberOfClaims`,
   `AgeOfVehicle`, `AgeOfPolicyHolder`, `NumberOfSuppliments`,
   `AddressChange_Claim`, `NumberOfCars`); ordinal or cyclic encoding for months
   and weekdays; one-hot (or in-fold target encoding) for nominal columns.
   Use native categoricals for CatBoost/LightGBM; scale for LR and the NN.
2. Engineered features, each justified from EDA: `age_missing`, claim lag in weeks
   (accident → claim), weekend accident/claim flags, `policytype_mismatch`,
   validation-rule hit count, high-risk combination flags, price-extreme flag.
3. `build_preprocessor()` in `src/features/` returns the `ColumnTransformer`.
   `src/schema.py` defines a **pydantic `Claim` model** with every raw field, its
   type and allowed values. This is the common claim schema from synopsis Phase 1
   and the input contract for every agent tool.

## 6. Phase 5 — Modelling & experiments (CHECKPOINT)

1. Baselines: all-legit, rules-only scorer from `red_flag_rules.yaml`, and
   class-weighted Logistic Regression.
2. Models required by the synopsis: Random Forest, XGBoost, and a PyTorch MLP
   (entity embeddings or one-hot input, weighted BCE or focal loss, early stopping
   on real-validation PR-AUC). Add LightGBM and CatBoost for comparison. Use class
   weights / `scale_pos_weight` throughout.
3. **Data-mix experiment (the core research result):** real-only vs real +
   synthetic at 10/25/50/100% vs synthetic-only, plus synthetic sample weights
   (0.1, 0.3, 1.0). Plot real-validation PR-AUC against synthetic volume to show
   whether augmentation actually helps.
4. Tune the best 1–2 configurations with Optuna (objective: PR-AUC on
   `real_validation.csv`, or stratified CV with real-only validation folds; about
   50–100 trials). Watch for overfitting to the 139 validation frauds: prefer
   stable configurations over a lucky top trial.
5. Calibrate probabilities (isotonic or Platt) and report the Brier score and a
   reliability curve. The agent reasons over the probability, so it must be
   meaningful.
6. Thresholds: ask me for the cost ratio of a missed fraud vs an unnecessary
   investigation (default 10:1). Derive the decision policy:
   - score ≥ high threshold → **Flag for Investigation**
   - blocking validation failures (e.g. missing age, inconsistent fields) →
     **Request More Information**
   - otherwise → **Approve** (always subject to adjuster review)
7. Track every run (MLflow locally, or a CSV run log): params, data mix, metrics,
   data hashes. Deliverable: `reports/05_model_comparison.md` with a comparison
   table of all models, synopsis metrics and PR-AUC side by side.

## 7. Phase 6 — Final held-out evaluation, explainability & fairness (CHECKPOINT)

1. Evaluate the chosen model **once** on `real_test.csv`: accuracy, precision,
   recall, F1, F2, ROC-AUC, PR-AUC, false-positive rate, confusion matrix,
   precision at the top 5%/10%, and a lift chart, all with bootstrap CIs and
   compared against the baselines. State plainly whether the synopsis success
   criterion (precision and recall well above the naive baseline) is met.
2. SHAP: global importance, dependence plots, and local explanations for sample
   true positives, false positives and false negatives. Check that SHAP agrees with
   the EDA red flags and explain any disagreement.
3. Error analysis: which claim segments are missed or over-flagged?
4. Fairness: recall, FPR and flag rate by `Sex`, age band and `MaritalStatus`.
5. Deliverables: `reports/06_final_evaluation.md` and `reports/model_card.md`
   (intended use, data and synthetic-data caveats, metrics, limitations,
   fairness, thresholds).

## 8. Phase 7 — Agent-ready tools (CHECKPOINT)

Build the tools; don't build the agent yet.

1. Persist the full pipeline (preprocessor + calibrated model) with `joblib` in
   `models/`, plus `metadata.json` (version, features, thresholds, metrics, data
   hashes, training date).
2. `src/tools/` with pure, typed functions that validate input against `Claim`
   and return JSON-serialisable output, ready to wrap as LangGraph tools:
   - `score_claim(claim) -> {fraud_probability, risk_tier, thresholds, model_version}`
   - `explain_claim(claim, top_k=5) -> [{feature, value, shap_contribution, reason}]`
   - `check_policy_rules(claim) -> [{rule_id, passed, severity, message}]`
     (validation rules + red-flag rules)
   - `find_similar_claims(claim, k=5) -> [{policy_number, similarity, fraud_label,
     key_fields}]` — a FAISS or ChromaDB index built over **real training claims
     only** (not synthetic, not validation/test), using the encoded features or
     embeddings; check that the neighbours look sensible.
3. `src/feedback/`: the schema and a storage helper (SQLite or CSV) for adjuster
   decisions (claim, model score, agent recommendation, adjuster decision,
   override reason, timestamp). Add a `retrain.py` stub that shows how logged
   decisions would be merged into training data and the model re-evaluated
   (closes the synopsis feedback loop).
4. An optional thin FastAPI wrapper over the tools, with tests.
5. `docs/agent_design.md`: the LangGraph architecture mapped to synopsis phases
   1–6 (intake → validate → score → reason with tools → decide → human review).
   Cover state schema, tool-calling policy, the decision-synthesis prompt, guardrails
   (the agent never auto-denies; the human has the final say), audit logging, the
   Streamlit dashboard screens, and how to evaluate the agent on held-out claims.
   Also note how the supplementary Kaggle "Auto Insurance Claims Data" could
   enrich the retrieval knowledge base.

## 9. Final wrap-up

Update `README.md` with the project overview, repo structure, setup, how to
reproduce each phase, headline results, and the synopsis checklist with every item
marked done or pending.

Start with Phase 0 and Phase 1 now. Before writing code, show me a short plan of the
files you will create in Phase 1.
