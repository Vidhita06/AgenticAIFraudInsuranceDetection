# Synopsis Requirements Checklist

Sources: `details/RBU_Project_Synopsis_Agentic_AI_Insurance_Claims (3).pdf` (RBU synopsis, Session
2026-27, Sem V, IDEA Lab) and `details/Agentic_AI_Insurance_Claims_Project_Proposal (1).docx`
(IDEA Lab proposal). Each item is tagged with the phase of this work plan that delivers it:

- **P0–P7** are the work-plan phases: P0 context, P1 data audit, P2 synthetic audit, P3 EDA,
  P4 features, P5 modelling, P6 final evaluation, P7 agent-ready tools.
- **Later** means the agent, dashboard and integration phase (synopsis weeks 5–9), outside this session.

Status values: ✅ done · 🟡 partly done · ⬜ pending.

## 1. Objectives (synopsis §3, proposal §3)

| # | Objective | Phase | Status |
|---|---|---|---|
| O1 | ML/DL model that predicts the probability that a claim is fraudulent | P5, P6 | ⬜ |
| O2 | Autonomous LLM orchestrator that plans and runs a multi-step triage workflow with tools | P7 (design + tools), Later (agent) | ⬜ |
| O3 | Similar-claims retrieval to compare a new claim with historical patterns | P7 | ⬜ |
| O4 | Structured, human-readable decision with rationale (Approve / Flag for Investigation / Request More Information) | P5 (decision policy), P7 (explain + rules tools), Later | ⬜ |
| O5 | Lightweight human-in-the-loop dashboard (review, override, feedback) | P7 (design), Later (Streamlit) | ⬜ |
| O6 | Close the feedback loop: adjuster decisions used to retrain the model | P7 (`src/feedback/`, `retrain.py` stub), Later | ⬜ |

## 2. Methodology phases (synopsis §5, proposal §6)

| # | Synopsis phase | What it requires | Delivered by | Status |
|---|---|---|---|---|
| M1 | Phase 1: Ingestion & extraction | Parse claims into a **standardized common schema** (structured fields, optional free text) | P4 (`src/schema.py` pydantic `Claim`), Later (free-text extraction) | ⬜ |
| M2 | Phase 2: Validation & enrichment | Missing/incomplete/inconsistent field checks; business rules (incident within the policy coverage period, logical consistency); feature engineering | P1 (`src/data/validation.py`), P4 (features) | 🟡 rules done (P1); features pending |
| M3 | Phase 3: ML/DL scoring | Trained classifier outputs a fraud **probability** (not just a label) | P5 (calibrated model), P7 (`score_claim`) | ⬜ |
| M4 | Phase 4: Agentic reasoning core | LLM picks tools dynamically: vector-DB similar-claim retrieval, policy verification, claimant history checks | P7 (tools + `docs/agent_design.md`), Later | ⬜ |
| M5 | Phase 5: Decision synthesis | Combine score, validation results, similar claims, policy and history into a recommendation with rationale | P5 (thresholds), P7 (design), Later | ⬜ |
| M6 | Phase 6: Human review & feedback | Adjuster approves, modifies or overrides; decisions are logged securely and used for evaluation and retraining | P7 (feedback store), Later (dashboard) | ⬜ |
| M7 | Modularity | ML model and agent can be developed, tested and validated independently | All (importable `src/` modules, tests) | 🟡 |

## 3. Deliverables (synopsis §7, proposal §4)

| # | Deliverable | Phase | Status |
|---|---|---|---|
| D1 | Cleaned and documented dataset | P1, P2 | 🟡 audit done (`reports/01_data_audit.md`) |
| D2 | **EDA report** | P3 (`reports/03_eda_report.md`, `notebooks/03_eda.ipynb`) | ⬜ |
| D3 | **Baseline models**: Logistic Regression, Random Forest | P5 | ⬜ |
| D4 | **Advanced models**: XGBoost and a neural network (PyTorch or TensorFlow) | P5 | ⬜ |
| D5 | **Class-imbalance handling** | P5 (class weights, `scale_pos_weight`, focal loss; synthetic augmentation experiment) | ⬜ |
| D6 | Evaluation with **accuracy, precision, recall, F1, AUC-ROC** | P5, P6 | ⬜ |
| D7 | Callable **tool: fraud scoring** | P7 (`score_claim`, `explain_claim`) | ⬜ |
| D8 | Callable **tool: similar-claims retrieval** (FAISS/ChromaDB) | P7 (`find_similar_claims`) | ⬜ |
| D9 | Callable **tool: policy/history checks** | P1 (rules), P3 (red flags), P7 (`check_policy_rules`) | 🟡 rules done |
| D10 | Functional agentic pipeline (orchestrator + tools) | Later | ⬜ |
| D11 | Dashboard: submit a claim, view decision and rationale, approve or override | Later | ⬜ |
| D12 | End-to-end pipeline runs on **held-out claims without manual intervention** | Later (P7 designs the evaluation) | ⬜ |
| D13 | Final project report (problem, methodology, architecture, results, future scope) | P3/P6 reports feed it; Later | ⬜ |
| D14 | Presentation deck / demo video (`details/PROJECT PRESENTATION TEMPLATE.pptx`) | Later | ⬜ |
| D15 | Evaluation report with trained baseline and advanced models (timeline, weeks 3–4) | P5 (`reports/05_model_comparison.md`), P6 | ⬜ |
| D16 | Finalised dataset, fraud-labelling scheme and feature list (timeline, weeks 1–2) | P1, P3, P4 | 🟡 |

## 4. Metrics and success criteria (proposal §13, timeline)

| # | Criterion | Phase | Status |
|---|---|---|---|
| S1 | **Precision and recall both meaningfully above the naive baseline** (all-legit: 94% accuracy, 0 recall; random: PR-AUC ≈ 0.06) | P5, P6 | ⬜ |
| S2 | **False-positive rate on held-out claims** (proposal, testing phase) | P6 | ⬜ |
| S3 | Accuracy measured on held-out claims | P6 | ⬜ |
| S4 | Agent retrieves relevant similar claims and writes a coherent, policy-consistent rationale for most test claims | P7 (retrieval sanity check), Later | ⬜ |
| S5 | End-to-end pipeline produces a reviewable recommendation for each sampled held-out claim | Later | ⬜ |
| S6 | Dashboard lets an adjuster view details, score and rationale, and record approve/override | Later | ⬜ |

## 5. Additions from this work plan (not in the synopsis, needed for rigour)

| # | Item | Phase | Status |
|---|---|---|---|
| A1 | Synthetic-data audit (fidelity, adversarial validation, TSTR utility, memorisation) | P2 | ⬜ |
| A2 | PR-AUC as primary selection metric; F2, recall at fixed precision; 95% bootstrap CIs | P5, P6 | ⬜ |
| A3 | Data-mix experiment: real-only vs real + synthetic (10–100%) vs synthetic-only, sample weights | P5 | ⬜ |
| A4 | Probability calibration (Brier score, reliability curve) | P5 | ⬜ |
| A5 | Cost-based thresholds and decision policy (default 10:1 missed fraud vs investigation) | P5 | ⬜ |
| A6 | Explainability with SHAP (global and local), agreement with EDA red flags (synopsis cites Lundberg & Lee [7]) | P6 | ⬜ |
| A7 | Fairness audit by Sex, age band and MaritalStatus | P3, P6 | ⬜ |
| A8 | Model card | P6 | ⬜ |
| A9 | Red-flag rules with support and lift (`reports/red_flag_rules.yaml`) for policy checks and rationale text | P3 | ⬜ |
| A10 | Reproducibility: `RANDOM_STATE = 42`, pinned requirements, `config.yaml`, input SHA-256 hashes, run log | P0–P7 | 🟡 set up in P1 |
| A11 | pytest tests for validation rules, feature pipeline and tools | P1, P4, P7 | 🟡 validation tests done |
| A12 | Optional FastAPI wrapper over the tools | P7 | ⬜ |
| A13 | Agent design document (LangGraph state, tool policy, prompt, guardrails, audit log, dashboard screens, agent evaluation) | P7 (`docs/agent_design.md`) | ⬜ |
| A14 | Supplementary Kaggle "Auto Insurance Claims Data" (~1,000 rows) as an extra retrieval knowledge base (proposal §10.2) | P7 (documented in agent design), Later | ⬜ |

## 6. Scope constraints to respect

- One insurance line (auto/vehicle); structured claim fields first, optional free text.
- Agent on an open-source or free-tier LLM API with 2–3 callable tools (fraud model, retrieval,
  basic policy checks).
- Human adjuster has final authority. The system recommends and never auto-denies.
- Out of scope: live insurer integration, regulatory certification, other insurance lines,
  image-based damage assessment (stretch goal only).
- Tech stack named in the synopsis: Python, scikit-learn, XGBoost, TensorFlow/PyTorch,
  LangChain/LangGraph or CrewAI, FAISS/ChromaDB, an OpenAI/Anthropic/open-source LLM,
  Streamlit or Flask.
