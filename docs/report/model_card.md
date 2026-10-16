# Model Card: Vehicle-Claim Fraud Scoring Model

_Machine-readable version: `models/model_card.json`. Training and evaluation details are in
`docs/report/02_model_report.md`._

| | |
|---|---|
| **Model version** | `lightgbm-b200-20261016-0915` |
| **Type** | LightGBM gradient-boosted trees, an ensemble of 3 seeds (42, 43, 44), averaged, then Platt-calibrated |
| **Trained** | 2026-10-16 on an NVIDIA B200 server (compute capability 10.0, CUDA 12.9, Python 3.12.3). LightGBM itself runs on CPU by design. |
| **Artifacts** | `models/preprocessor.joblib`, `models/fraud_model.joblib`, `models/shap_explainer.pkl`, `models/model_card.json`. They load on CPU with the pinned `requirements.txt` (scikit-learn 1.9.1, LightGBM 4.7.0, SHAP 0.51.0, joblib 1.6.0). |
| **Inference** | `src.ml.predict.score_claims(claims)` → `{fraud_probability, risk_band, thresholds, model_version}`; explanations via `src.ml.explain` |

## Intended use

- **Use:** triage support for **vehicle** insurance claims. The calibrated fraud probability
  and its risk band are one input to the LangGraph triage agent. The agent recommends
  *Approve*, *Flag for investigation* or *Request more information*, and **a human adjuster
  makes every final decision**.
- **Not for:**
  - automatic denial of claims;
  - pricing or underwriting;
  - judging individuals;
  - other insurance lines (health, life, property);
  - claims whose fields fall outside the schema in `src/schemas/claim.py`.
- **Users:** claims adjusters and fraud investigators, through the agent and its rationale.

## Training data

- **Source:** the Kaggle *Vehicle Claim Fraud Detection* data (15,420 claims, 1994–1996, 6%
  fraud).
- **Splits:** real claims were split 70/15/15 into train, validation and test, disjoint by
  `PolicyNumber`.
- **Training set:** 10,793 real training claims plus 289,207 synthetic claims at sample weight
  0.1 (300,000 rows). The weight was chosen by the data-mix experiment on real validation.
- **Synthetic data:**
  - The generator is unknown.
  - It was audited in `docs/report/01_eda_report.md` §2: real and synthetic rows are
    indistinguishable (adversarial AUC 0.51), there is no memorisation or held-out leakage,
    and adding the synthetic rows raised real-validation PR-AUC by about 0.05.
  - It **flattens weak, small-category signals** (Make, Month, early-policy incidents).
    Engineered features and red-flag indicators give the model these signals directly.
- **Ground truth:** real data only. Model selection, calibration and thresholds use
  `real_validation.csv` (2,313 claims, 139 frauds). `real_test.csv` (2,313 claims, 138
  frauds) was used once, for the final evaluation.
- **Input hashes (SHA-256):**

  | File | Hash |
  |---|---|
  | augmented_train | `8dea8f3b…0368` |
  | real_validation | `9e106895…c18c` |
  | real_test | `9c10316d…371b` |

## Features

There are 89 model columns, built by `src/ml/preprocess.py` (`ClaimFeatureBuilder` + "tree"
preprocessor):

- one-hot nominal fields: Fault, BasePolicy, VehicleCategory, AgentType, AccidentArea, Make,
  PoliceReportFiled, WitnessPresent, Month, DayOfWeek;
- ordinal range fields: price, policy-day bins, past claims, vehicle age, supplements,
  address change, number of cars, age band;
- numeric fields: deductible, driver rating, year, week, claim lag;
- engineered flags: missing age, early-policy incident, price extreme, policyholder at fault,
  liability-only, weekend accident/claim;
- 12 red-flag indicators plus a weighted red-flag score;
- a count of failed consistency rules.

**Excluded from the model:**

- `PolicyNumber` and `is_synthetic` (leakage);
- `PolicyType` and `AgeOfPolicyHolder` (determined by other fields);
- `RepNumber` (no signal);
- `Sex` and `MaritalStatus` (fairness; removing them cost no accuracy in the EDA ablation);
- raw `Age` (replaced by age band and missing-age flag);
- the claim-date fields (replaced by lag and weekend features).

## Hyperparameters

- **Library defaults** (`DEFAULT_PARAMS` in `src/ml/train.py`): learning rate 0.03, 31 leaves,
  `min_child_samples` 50, subsample 0.8, `colsample_bytree` 0.8, L2 regularisation 1.0, and
  `scale_pos_weight` 15.7 (class balance, computed from the weighted training set).
- **Early stopping:** on real-validation PR-AUC, with up to 3,000 trees. It stopped at 488,
  490 and 381 trees for the three seeds.
- **Tuning:**
  - The effective parameters are the defaults. They were recovered from the pickled
    estimators, because the uploaded `model_card.json` recorded `"params": {}` (the
    overrides-on-defaults dict).
  - Two things can cause that: the Optuna-tuned configuration lost to the defaults, or the
    tuning section did not run. The B200 run's `tuning_summary.csv` or `run_log.txt` would
    show which.
  - The old code also compared the tuned mean over 3 seeds with the defaults' *single* run.
    It now compares means over the same seeds and records the outcome in the card (see
    `02_model_report.md` §4).

## Performance on the held-out real test set (evaluated once)

Threshold: 0.108, the cost-optimal threshold for a 10:1 cost of a missed fraud vs an
unnecessary investigation. CIs are 95% stratified bootstrap (2,000 resamples).

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | 0.874 | 0.852 – 0.895 |
| PR-AUC (random scorer = 0.060) | 0.252 | 0.213 – 0.316 |
| Recall | 0.790 (109 of 138 frauds) | 0.717 – 0.855 |
| Precision | 0.208 | 0.188 – 0.229 |
| F1 / F2 | 0.329 / 0.506 | 0.299 – 0.358 / 0.461 – 0.546 |
| False-positive rate | 0.191 (416 of 2,175 legit) | 0.174 – 0.208 |
| Flag rate | 22.7% of claims | 21.1% – 24.4% |
| Accuracy (all-legit baseline = 0.940) | 0.808 | 0.792 – 0.824 |
| Precision in the top 5% / 10% of scores | 0.293 / 0.247 | — |
| Brier score (base rate alone = 0.056) | 0.0488 | 0.0468 – 0.0505 |

**Success criterion: met.** The synopsis asks for precision and recall both meaningfully above
the naive baseline. A random flagger with the same flag rate would score:

| | This model (lower CI) | Random flagger |
|---|---|---|
| Precision | 0.188 | 0.060 |
| Recall | 0.717 | 0.227 |

Both lower CIs are more than twice the random flagger's values. Accuracy is lower than the
all-legit baseline's 94% by design: at 6% prevalence, accuracy rewards never flagging.

## Decision thresholds and risk bands

Thresholds were chosen on out-of-fold calibrated real-validation scores:

| Policy | Threshold | Flag rate | Precision | Recall (validation) |
|---|---|---|---|---|
| Cost 5:1 | 0.179 | 11.5% | 0.256 | 0.489 |
| **Cost 10:1 (default)** | **0.108** | 24.0% | 0.198 | 0.791 |
| Cost 20:1 | 0.037 | 35.8% | 0.161 | 0.957 |
| Capacity: investigate the top 10% | 0.189 | 10.0% | 0.254 | 0.425 |

**Risk bands used by the agent:**

- `high`: score ≥ 0.108 (the 10:1 threshold); never approved by the agent.
- `medium`: 0.037 ≤ score < 0.108 (the 20:1 threshold).
- `low`: score < 0.037.

## Explainability

- `src.ml.explain` returns TreeSHAP values for the uncalibrated ensemble, in log-odds.
  Calibration is monotone, so the ranking of contributions is unchanged.
- The fraud-score tool reports the top 5 factors in plain English.
- Global importance, dependence plots and local examples: `02_model_report.md` §6.

## Fairness

- Sex and marital status are not model inputs.
- Recall, false-positive rate and flag rate by Sex, MaritalStatus and age band on the test set
  are in `02_model_report.md` §7, with the caveat that small groups have wide intervals.

## Limitations and risks

- **Data:**
  - The data is 1994–1996 claims from one insurer, with an unknown labelling process
    (`FraudFound_P`).
  - 96% of the training rows are synthetic. Their effect on real performance was measured,
    but they cannot add patterns the real data lacks.
- **Precision is low:**
  - At the default threshold about 4 of 5 flagged claims are legitimate. This is acceptable
    only because a flag means *investigate* and a human decides.
  - Use the 5:1 or capacity thresholds if investigation capacity is tight.
- **Calibration:** Platt calibration was fitted on 139 validation frauds. Probabilities are
  well-calibrated on average (Brier 0.049) but coarse at the extremes.
- **Dataset quirks:**
  - The model leans on known quirks such as the third-party-fault cluster with a 500
    deductible.
  - The red-flag rules make these visible; they may not transfer to another insurer's data.
- **Retraining:** the model will drift. The (deferred) feedback loop is meant to collect
  adjuster decisions for retraining.
