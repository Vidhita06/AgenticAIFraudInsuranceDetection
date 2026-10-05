# Claude Code Prompt — EDA → Model Training → Agent-Ready Fraud Scoring

> How to use: put the augmented dataset at `data/raw/` (and the original
> `fraud_oracle.csv` too, if you still have it), put your synopsis PDF/DOCX/MD
> in `docs/`, open Claude Code in the repo root, and paste everything below the
> line as your first message. Claude Code will work phase by phase and stop at
> each checkpoint for your go-ahead.

---

You are working in the repo `AgenticAIFraudInsuranceDetection`. The end goal is an
**agentic AI system for vehicle insurance claim fraud detection**: an LLM agent
that takes a claim, calls a fraud-scoring model as a tool, explains the risk with
reason codes, checks business red-flag rules, and routes the claim (auto-approve /
review / investigate) with a human in the loop. This session covers everything
*before* the agent: rigorous EDA, a synthetic-data audit, leakage-safe modelling,
and packaging the model as a clean, explainable, callable tool.

## 0. Context you must load first

1. Read the project synopsis in `docs/` (any PDF/DOCX/MD there). Extract its stated
   objectives, workflow stages, required deliverables, and evaluation criteria, and
   write them to `docs/synopsis_requirements.md` as a checklist. Every phase below
   must map back to that checklist; if the synopsis asks for something this prompt
   doesn't cover, add it. If no synopsis is found, tell me and continue with this
   prompt as the spec.
2. Locate the dataset in `data/raw/`. Never commit data: add `data/` (except a
   `.gitkeep`/README), model binaries, and large outputs to `.gitignore`.
3. Known facts about the dataset (verify each one, report any mismatch):
   - 300,000 rows × 34 columns, 0 missing values. It is the Kaggle
     "Vehicle Insurance Claim Fraud Detection" (`fraud_oracle.csv`) schema plus an
     `is_synthetic` flag. Target: `FraudFound_P`.
   - Real rows (`is_synthetic=0`): 10,793 (3.6%), 646 fraud (5.99%).
     Synthetic rows (`is_synthetic=1`): 289,207, 17,310 fraud (5.99%).
     Overall imbalance ≈ 15.7 : 1.
   - 10,793 ≈ 70% of the original 15,420 rows, so the augmentation was **probably
     generated from a 70% train split**. Where are the other ~4,627 real rows?
     This is the single most important question for a valid evaluation (see §2).
   - `Age = 0` is a missing-age sentinel with elevated fraud risk.
   - `Deductible`: 300–700, ~93% at 400. `DriverRating`: 1–4 uniform.
     `Year`: 1994–1996. `WeekOfMonth`/`WeekOfMonthClaimed`: 1–5.
   - Strong univariate signals: `Fault` (Policy Holder 7.82% vs Third Party 1.17%),
     `BasePolicy` (All Perils 10.28%, Collision 7.11%, Liability 0.77%),
     `VehicleCategory` (Utility 11.04%, Sedan 8.28%, Sport 1.42%), `VehiclePrice`
     extremes (<20k 8.48%, >69k 7.94%), `PoliceReportFiled`, `AgentType`.

## 1. Non-negotiable rules (apply in every phase)

- **Real data is the only ground truth.** Validation and test sets contain only
  real rows (`is_synthetic=0`). Synthetic rows may only ever enter *training* folds.
- **Split before you look too hard.** Create the real holdout test set at the start
  of Phase 2, save its row IDs, and do not touch it again until final evaluation.
  All EDA conclusions that drive modelling decisions come from the training portion.
- **No leakage.** Every encoder, scaler, imputer, feature selector, and resampler is
  fit inside a scikit-learn `Pipeline` on training folds only. Drop identifiers
  (`PolicyNumber`; assess `RepNumber`) from features. Don't apply SMOTE/oversampling
  on top of a dataset that is already 96% synthetic unless an experiment shows it
  helps on real validation data.
- **Primary metric is PR-AUC (average precision) on real data**, plus recall at a
  fixed precision / precision at top-k%, F2, and a cost-based threshold. ROC-AUC
  and accuracy are secondary (accuracy is meaningless at 6% prevalence). Report
  every metric with a 95% bootstrap CI — the real test set has only a few hundred
  fraud cases at most, so point estimates alone are misleading.
- Reproducibility: fixed `RANDOM_STATE=42`, pinned `requirements.txt`, a single
  `config.yaml` for paths/params, and a SHA-256 hash of the input data logged in
  every report.
- Code lives in `src/` as importable modules; notebooks in `notebooks/` only call
  `src/` functions and are numbered (`01_data_audit.ipynb`, …). Figures go to
  `reports/figures/`, written findings to `reports/*.md`. Add small `pytest` tests
  for the data-validation rules and the feature pipeline.
- **Stop at every CHECKPOINT**, summarise findings and decisions in ≤15 bullet
  points, list open questions, and wait for my approval before continuing.

## 2. Phase 1 — Data audit & split integrity (CHECKPOINT)

1. Load with explicit dtypes; confirm shape, columns, dtypes, cardinalities,
   value sets for every categorical, duplicates (full-row and excluding
   `PolicyNumber`), and the stats listed in §0.3.
2. Encoding anomalies from the original dataset to check for in both real and
   synthetic rows: `DayOfWeekClaimed`/`MonthClaimed` = `'0'`, `Age = 0`
   vs `AgeOfPolicyHolder` bin (`16 to 17`), unknown/rare categories.
3. **Logical-consistency rules** (write them as reusable validators in
   `src/data/validation.py`; report violation counts for real vs synthetic):
   - `PolicyType` must equal `VehicleCategory - BasePolicy` (known to be
     inconsistent in some original rows — quantify, don't silently fix).
   - `Age` must fall inside the `AgeOfPolicyHolder` bin.
   - `MonthClaimed`/`WeekOfMonthClaimed` should not precede the accident
     `Month`/`WeekOfMonth` (allowing year wrap-around).
   - `Days_Policy_Claim` vs `Days_Policy_Accident` ordering.
   - `Year` vs `AgeOfVehicle` and other plausibility checks you find.
4. **Locate the held-out real rows.** If the original `fraud_oracle.csv` is
   present, match it against the real rows (on all columns except
   `is_synthetic`; `PolicyNumber` is unique in the original) and identify the
   ~4,627 rows absent from the augmented file. Then check whether any synthetic
   row is an exact or near-duplicate of those held-out rows (would indicate the
   generator saw them → leakage). Decide the test strategy:
   - **Best:** held-out real rows not seen by the generator = final test set;
     the 10,793 real rows split into train/validation.
   - **Otherwise:** stratified 70/15/15 split of the 10,793 real rows into
     real-train / real-val / real-test, and flag in the report that the
     generator may have seen real-val/test rows (so synthetic-data gains are
     optimistic). Also run a **temporal check**: train on 1994–1995, test on
     1996 real rows.
   Ask me which original file / generator settings were used if unclear.
5. Deliverables: `reports/01_data_audit.md`, `src/data/load.py`,
   `src/data/validation.py`, `src/data/split.py`, saved split indices in
   `data/splits/`.

## 3. Phase 2 — Synthetic data fidelity, utility & privacy audit (CHECKPOINT)

The model's value depends on whether 289k synthetic rows behave like real claims.
Compare real-train vs synthetic:

1. **Fidelity:** per-column distributions (TVD / Jensen–Shannon distance, chi-square;
   KS for numeric), fraud rate *per category per origin* (does each red flag hold
   in both?), pairwise association matrices (Cramér's V) and their difference
   heatmap, mutual information with the target per origin.
2. **Adversarial validation:** train a LightGBM to classify real vs synthetic.
   ROC-AUC ≈ 0.5 means indistinguishable; > 0.7 means detectable differences —
   report the top features driving it.
3. **Utility:** TRTR (train real, test real) vs TSTR (train synthetic, test real)
   vs TRSTR (train real+synthetic, test real), same model, same real validation set.
4. **Privacy / memorisation:** exact-copy rate of real rows inside synthetic, and
   distance-to-closest-record distribution (Gower distance) vs real-to-real baseline.
5. **Rule violations** from §2.3 in synthetic vs real.
6. Output a clear verdict: use all synthetic, a filtered subset (e.g. drop rule
   violators or rows the adversarial model is very confident are synthetic), a
   down-weighted sample, or none. Deliverable: `reports/02_synthetic_audit.md`.

## 4. Phase 3 — Fraud-focused EDA on the training split (CHECKPOINT)

Follow the synopsis's EDA requirements; at minimum:

1. Target distribution and imbalance (real vs synthetic vs combined).
2. Univariate profiles for all 33 features, grouped: claim timing, policy,
   vehicle, policyholder, claim-process (police report, witness, agent,
   supplements, address change).
3. Bivariate fraud-rate analysis: fraud rate per category with Wilson CIs and
   support counts (sorted bar charts), chi-square + Cramér's V ranking, and
   information value / weight of evidence for every feature. Do this on real-train
   first, then confirm on synthetic.
4. Interactions that likely matter: `Fault × BasePolicy`,
   `VehicleCategory × VehiclePrice`, `AgeOfPolicyHolder × Fault`,
   `PastNumberOfClaims × AddressChange_Claim`, claim-lag × `PoliceReportFiled`.
5. Redundancy: `PolicyType` vs `VehicleCategory`+`BasePolicy`; `Age` vs
   `AgeOfPolicyHolder`; `Month` vs `MonthClaimed`. Decide what to keep.
6. Fairness-sensitive attributes (`Sex`, `MaritalStatus`, `Age`): document their
   fraud-rate differences and recommend whether to exclude them from the model
   (they must at least be available for a fairness audit later).
7. Turn findings into (a) a **feature-engineering plan** and (b) a list of
   **human-readable red-flag rules** with their support and fraud lift — the agent
   will use these later as a rules tool.
8. Visual quality: consistent palette, labelled axes, fraud rate shown with CIs
   and sample sizes, no pie charts. Deliverables: `notebooks/03_eda.ipynb`,
   `reports/03_eda_findings.md`, `reports/red_flag_rules.yaml`.

## 5. Phase 4 — Preprocessing & feature engineering (CHECKPOINT)

1. Ordinal-encode range/bin columns in their natural order (`VehiclePrice`,
   `Days_Policy_Accident`, `Days_Policy_Claim`, `PastNumberOfClaims`,
   `AgeOfVehicle`, `AgeOfPolicyHolder`, `NumberOfSuppliments`,
   `AddressChange_Claim`, `NumberOfCars`); cyclic or ordinal encoding for
   months/days; one-hot or target encoding (in-fold only) for nominal columns
   (`Make`, `RepNumber` if kept); native categoricals for CatBoost/LightGBM.
2. Engineered features (justify each from EDA): `age_missing` (Age = 0),
   claim-lag in weeks between accident and claim, weekend accident/claim flags,
   `policytype_mismatch`, high-risk-combo flags (e.g. policy-holder fault ×
   All Perils), price-extreme flag.
3. One `build_preprocessor()` in `src/features/` returning a `ColumnTransformer`,
   plus a JSON/pydantic **input schema** (`src/schema.py`) describing every raw
   field, its allowed values, and its type — this schema becomes the agent's tool
   input contract.

## 6. Phase 5 — Modelling & experiments (CHECKPOINT)

1. Baselines: majority class, a rules-only scorer from `red_flag_rules.yaml`,
   logistic regression (class-weighted).
2. Main models: Random Forest, XGBoost, LightGBM, CatBoost — each with
   class weighting / `scale_pos_weight`. Use stratified K-fold CV in which
   **validation folds are real rows only** (synthetic rows are added to training
   folds only).
3. **Data-mix experiment (the core research result):** real-only vs
   real + synthetic at 25/50/100% vs synthetic-only, and synthetic sample weights
   (e.g. 0.1, 0.3, 1.0). Plot real-validation PR-AUC vs synthetic volume. This tells
   us whether the augmentation actually helps.
4. Tune the best 1–2 configs with Optuna (objective: CV PR-AUC on real folds,
   ~50–100 trials, early stopping).
5. Calibrate probabilities (isotonic or Platt, fit on real validation data) and
   report Brier score + reliability curve. Calibrated probabilities are needed
   because the agent will reason over them.
6. Threshold selection: define a cost matrix (missed fraud vs unnecessary
   investigation — ask me for the ratio, default 10:1) and pick thresholds for
   three routing tiers: low / medium (manual review) / high (investigate).
7. Track every run (MLflow locally, or a CSV run log) with params, data mix,
   metrics, and data hash.

## 7. Phase 6 — Final evaluation, explainability & fairness (CHECKPOINT)

1. Evaluate the chosen model **once** on the untouched real test set (plus the
   temporal 1996 check): PR-AUC, ROC-AUC, recall/precision/F1/F2 at chosen
   thresholds, confusion matrix, precision@top-5%/10%, lift/gain chart, all with
   bootstrap CIs. Compare against the baselines.
2. SHAP: global importance, dependence plots for the top features, and local
   explanations for sample true positives / false positives / false negatives.
   Check SHAP findings agree with the EDA red flags; investigate disagreements.
3. Error analysis: which claim segments are missed?
4. Fairness: recall, FPR and flag rate by `Sex`, age band, `MaritalStatus`.
5. Deliverables: `reports/06_model_evaluation.md` and a **model card**
   (`reports/model_card.md`: intended use, data, synthetic-data caveats, metrics,
   limitations, fairness, thresholds).

## 8. Phase 7 — Package the model as an agent-ready tool (CHECKPOINT)

Prepare — don't yet build — the agentic layer:

1. Persist the full pipeline (preprocessor + calibrated model) with `joblib` in
   `models/` along with `metadata.json` (version, features, thresholds, metrics,
   data hash, train date).
2. `src/inference/predict.py` exposing pure functions with typed, validated I/O:
   - `score_claim(claim: dict) -> {fraud_probability, risk_tier, threshold_used}`
   - `explain_claim(claim: dict, top_k=5) -> [{feature, value, shap_contribution,
     plain_english_reason}]`
   - `check_red_flags(claim: dict) -> [{rule_id, description, lift}]`
   - `find_similar_claims(claim: dict, k=5)` over real training claims (nearest
     neighbours on the encoded space) with their outcomes.
   All validate input against `src/schema.py` and return JSON-serialisable output.
3. A thin FastAPI app (`src/api/`) wrapping these functions, with tests.
4. Write `docs/agent_design.md`: the proposed agent architecture (LLM with tool
   use calling the four tools above; routing policy by risk tier; human-in-the-loop
   for medium/high; investigator-report generation; guardrails — the agent never
   auto-denies a claim; logging/audit trail; how the agent will be evaluated), and
   map it to the synopsis's agentic-AI stage.

## 9. Final wrap-up

Update `README.md` with project overview, structure, setup, how to reproduce each
phase, headline results, and the synopsis-requirements checklist with each item
marked done / pending. Commit after each phase with a descriptive message.

Start with Phase 0 and Phase 1 now. Before writing code, show me a short plan of
the files you will create in Phase 1.
