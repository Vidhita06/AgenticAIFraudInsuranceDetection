# Written Test – Agentic AI Insurance Claims Fraud Detection

---

## 1. Title of the Project

**Agentic AI for Autonomous Insurance Claims Fraud Detection**

An intelligent, multi-stage fraud detection system that combines a trained LightGBM machine learning model with a LangGraph-based LLM agent to triage vehicle insurance claims into Approve, Flag for Investigation, or Request More Information — with a human adjuster making every final decision.

Developed at RBU IDEA Lab on an NVIDIA DGX B200 server using open-source tools (Python, LightGBM, LangGraph, Ollama, Streamlit).

---

## 2. Objectives

1. Build a machine learning model that scores insurance claims for fraud probability with precision and recall both meaningfully above the naive baseline.
2. Design an agentic AI pipeline using LangGraph's ReAct architecture that reasons over evidence from multiple specialised tools before making a triage recommendation.
3. Implement deterministic code-level guardrails that ensure the system never denies a claim, never auto-approves high-risk claims, and always requires human review.
4. Ensure fairness by excluding sensitive attributes (Sex, MaritalStatus) from model features without sacrificing accuracy.
5. Provide explainability through SHAP values so that every flagged claim comes with human-readable top contributing factors.
6. Create a Streamlit dashboard for real-time claim processing with both rule-based and LLM-powered modes.

---

## 3. Algorithm / Flowchart / Block Diagram

### Pipeline Flowchart

```
                    ┌─────────────────────┐
                    │   Raw Claim JSON    │
                    │   (33 columns)      │
                    └─────────┬───────────┘
                              ▼
                ┌─────────────────────────────┐
                │  Stage 1: Claim Ingestion   │
                │  Parse & validate fields    │
                │  (Pydantic v2 schema)       │
                └─────────────┬───────────────┘
                              ▼
                ┌─────────────────────────────┐
                │  Stage 2: Feature Engg.     │
                │  33 raw → 89 features       │
                │  (one-hot, red flags, etc.) │
                └─────────────┬───────────────┘
                              ▼
                ┌─────────────────────────────┐
                │  Stage 3: Fraud Scoring     │
                │  LightGBM 3-seed ensemble   │
                │  + Platt calibration        │
                │  → probability + risk band  │
                └─────────────┬───────────────┘
                              ▼
                ┌─────────────────────────────┐
                │  Stage 4: Evidence          │
                │  Aggregation (ReAct Loop)   │◄────┐
                │  LLM reasons + calls tools  │     │
                └─────────────┬───────────────┘     │
                              │                     │
                    ┌─────────▼──────────┐          │
                    │  6 Specialised     │          │
                    │  Tools             │──────────┘
                    │  FS, PL, CH,       │  (iterative
                    │  EA, RA, CC        │   loop)
                    └─────────┬──────────┘
                              ▼
                ┌─────────────────────────────┐
                │  Stage 5: Guardrails        │
                │  8 deterministic rules      │
                │  (pure Python, no LLM)      │
                └─────────────┬───────────────┘
                              ▼
                ┌─────────────────────────────┐
                │  Stage 6: Triage Output     │
                └───┬─────────┼───────────┬───┘
                    ▼         ▼           ▼
              ┌─────────┐ ┌────────┐ ┌──────────┐
              │ APPROVE │ │ FLAG   │ │ REQUEST  │
              │ < 0.037 │ │ >=0.108│ │ 0.037-   │
              │         │ │        │ │ 0.108    │
              └─────────┘ └────────┘ └──────────┘
                              │
                              ▼
                    Human Adjuster Decision
```

### Algorithm (LightGBM Training)

1. Load raw dataset (15,420 claims, 33 columns, 6% fraud rate)
2. Split into train (70%), validation (15%), test (15%) — disjoint by PolicyNumber
3. Augment training set: 10,793 real + 289,207 synthetic claims at sample weight 0.1
4. Feature engineering: transform 33 raw columns into 89 model features
5. Train 3 LightGBM models with seeds 42, 43, 44 (early stopping on validation PR-AUC)
6. Average predictions across 3 models
7. Apply Platt calibration on validation set to get calibrated probabilities
8. Set decision thresholds using cost-sensitive analysis (10:1 cost ratio)
9. Evaluate once on held-out test set

### Algorithm (ReAct Agent Loop)

```
Input: claim_data, fraud_score, risk_band
Repeat:
    1. LLM THINKS about current evidence
    2. LLM DECIDES which tool to call next
    3. Tool EXECUTES and returns results
    4. LLM OBSERVES the results
    Until: LLM has enough evidence to decide
Output: evidence_record + triage_recommendation
```

---

## 4. Methodology

### 4.1 Data Preparation

- **Dataset:** Kaggle Vehicle Claim Fraud Detection (15,420 claims, 1994–1996, 6% fraud)
- **33 raw columns:** PolicyNumber, Age, Fault, VehicleCategory, Make, Deductible, DriverRating, PastNumberOfClaims, AgeOfVehicle, PoliceReportFiled, WitnessPresent, BasePolicy, etc.
- **Synthetic augmentation:** 289,207 synthetic claims added at sample weight 0.1 to address class imbalance. Verified via adversarial AUC (0.51 — indistinguishable from real data).
- **Fairness:** Sex and MaritalStatus excluded from features (ablation confirmed no accuracy loss).

### 4.2 Model Training

- **Algorithm:** LightGBM (gradient-boosted decision trees)
- **Ensemble:** 3 models trained with different seeds (42, 43, 44), predictions averaged
- **Calibration:** Platt scaling on validation set for calibrated probabilities
- **Key hyperparameters:** learning rate 0.03, 31 leaves, min_child_samples 50, scale_pos_weight 15.7
- **Early stopping:** On real-validation PR-AUC (stopped at 381–490 trees)

### 4.3 Feature Engineering

33 raw columns transformed into 89 model features:
- One-hot encoding of categorical fields (Fault, Make, BasePolicy, etc.)
- Ordinal binning (price ranges, age bands, past claims count)
- 12 red-flag indicators (e.g., third-party fault + 500 deductible, new policy + major claim)
- Engineered flags (missing age, early-policy incident, weekend claim, policyholder at fault)
- Weighted red-flag score and consistency rule failure count

### 4.4 Agentic AI Pipeline

- **Framework:** LangGraph with ReAct (Reason + Act) architecture
- **LLM:** Ollama + qwen2.5:72b (72B parameter open-source model, runs locally on B200)
- **6 Tools:** Fraud Score, Policy Lookup, Claim History, Evidence Analyser, Risk Assessment, Consistency Checker
- **Guardrails:** 8 deterministic Python rules enforced after LLM reasoning
- **Fallback:** If LLM fails, a rule-based path processes the claim without any LLM

### 4.5 Technology Stack

| Component | Technology |
|-----------|------------|
| ML Model | LightGBM 4.7.0 + scikit-learn 1.9.1 |
| Agent Framework | LangGraph (ReAct architecture) |
| LLM | Ollama + qwen2.5:72b (open-source, free) |
| Explainability | SHAP 0.51.0 (TreeSHAP) |
| Vector Store | FAISS (10,793 training claims) |
| Dashboard | Streamlit |
| Hardware | NVIDIA DGX B200 (192 GB HBM3e, CUDA 12.9) |
| Language | Python 3.12.3 |

---

## 5. Results

### 5.1 Model Performance (Held-Out Test Set)

Test set: 2,313 claims, 138 frauds. Evaluated once. Threshold: 0.108 (10:1 cost ratio).

| Metric | Value | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.874 | 0.852 – 0.895 |
| PR-AUC | 0.252 | 0.213 – 0.316 |
| Recall | 0.790 (109/138 frauds caught) | 0.717 – 0.855 |
| Precision | 0.208 | 0.188 – 0.229 |
| F1 Score | 0.329 | 0.299 – 0.358 |
| F2 Score | 0.506 | 0.461 – 0.546 |
| False Positive Rate | 0.191 | 0.174 – 0.208 |
| Brier Score | 0.0488 | 0.0468 – 0.0505 |

Confusion Matrix: TP 109, FP 416, FN 29, TN 1,759

### 5.2 Success Criterion — MET

| Metric | Our Model (lower CI) | Random Flagger |
|--------|---------------------|----------------|
| Precision | 0.188 | 0.060 |
| Recall | 0.717 | 0.227 |

Both metrics are more than 3x the random baseline.

### 5.3 Lift Analysis

- Top decile: 4.1x the base fraud rate (24.6% fraud)
- Top 3 deciles (30% of claims) contain 90% of all frauds
- Deciles 5–10 contain only 5 frauds in 1,387 claims

### 5.4 Agent Evaluation (Offline Baseline)

556 claims processed through the complete pipeline:

| Metric | Result |
|--------|--------|
| Fraud → FLAG recall | 74.8% |
| Legit → APPROVE rate | 84.2% |
| False-flag rate | 15.8% |
| FRAUD_RISK precision | 82.5% |
| Consistency across runs | 100% |
| Processing errors | 0 |

### 5.5 Risk Bands and Thresholds

| Risk Band | Threshold | Flag Rate | Action |
|-----------|-----------|-----------|--------|
| LOW | Score < 0.037 | — | Approve |
| MEDIUM | 0.037 – 0.108 | — | Request More Info |
| HIGH | Score >= 0.108 | 22.7% | Flag for Investigation |

---

## 6. Conclusion

1. The system successfully combines traditional ML (LightGBM) with agentic AI (LangGraph + LLM) to create an intelligent fraud triage pipeline.

2. The LightGBM model achieves 0.874 ROC-AUC and catches 79% of frauds (109 out of 138) — significantly outperforming both the random baseline and a rules-only approach.

3. The agentic architecture adds reasoning capability — the LLM agent doesn't just flag claims, it explains WHY by citing specific evidence from 6 specialised tools.

4. 8 code-enforced guardrails ensure the system never denies a claim, never auto-approves high-risk claims, and always requires human review — making it safe for production.

5. Fairness is maintained by excluding Sex and MaritalStatus from model features without any loss in accuracy.

6. The entire system was developed using free, open-source tools (LightGBM, Ollama, LangGraph) on the NVIDIA B200, demonstrating that enterprise-grade fraud detection doesn't require expensive proprietary APIs.

7. The system is designed as a decision-support tool — it assists human adjusters, never replaces them.

---

## 7. Future Scope

1. **Feedback loop:** Collect adjuster accept/reject decisions to retrain the model on real outcomes, improving precision over time.

2. **Live LLM evaluation:** The current offline evaluation uses the rule-based fallback. A live evaluation with the full LLM pipeline on production claims would measure the added value of LLM reasoning.

3. **Multi-line insurance:** Extend beyond vehicle claims to health, property, and life insurance with line-specific feature engineering and recalibrated thresholds.

4. **Real-time streaming:** Integrate with Kafka or similar event streams for real-time claim processing as they are filed, rather than batch processing.

5. **Graph-based fraud detection:** Add social network analysis to detect organised fraud rings (e.g., claims from the same repair shop, linked policyholders).

6. **Advanced explainability:** Add counterfactual explanations ("this claim would not have been flagged if the deductible were above 700") for better adjuster understanding.

7. **Model monitoring and drift detection:** Deploy automated monitoring to detect when the model's predictions degrade due to changing fraud patterns, triggering automatic retraining.

8. **Multi-modal evidence:** Incorporate claim photographs (vehicle damage images) using vision models to cross-validate reported damage against visual evidence.

9. **Deployment to cloud:** Deploy the Streamlit dashboard on Streamlit Community Cloud or Hugging Face Spaces with a free LLM API (Groq/Google AI Studio) replacing the local Ollama setup.

---

*Project developed at RBU IDEA Lab on NVIDIA DGX B200 (192 GB HBM3e, CUDA 12.9, Python 3.12.3)*
