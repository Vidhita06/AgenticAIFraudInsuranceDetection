# Viva Preparation Guide

_Agentic AI: Autonomous Insurance Claims Triage & Fraud Assistant._

Each question has two answers:

- **Technical answer:** the precise points, with our real numbers. Use it when an evaluator
  digs deeper.
- **Say it simply:** a short, conversational version to open with.

Background detail is in `TECH_AND_ARCHITECTURE.md` and `docs/report/`.

**Numbers worth memorising:**

| Fact | Value |
|---|---|
| Real claims | 15,420 (train 10,793 / validation 2,313 / test 2,313), about 6% fraud |
| Synthetic claims | 289,207, used at weight 0.1 |
| Model features | 89 |
| Test ROC-AUC / PR-AUC | 0.874 / 0.252 (random = 0.060) |
| Test recall / precision at 0.108 | 0.790 / 0.208, flag rate 22.7% |
| Thresholds | 0.108 (10:1, default), 0.179 (5:1), 0.037 (20:1), 0.189 (top 10%) |
| Bands | high ≥ 0.108; medium 0.037–0.108; low < 0.037 |
| Validation PR-AUC | LightGBM 0.291, XGBoost 0.283, MLP 0.262, RF 0.246, LogReg 0.193, rules 0.142 |
| Calibration (Brier) | Platt 0.0485, isotonic 0.0500, uncalibrated 0.159; test 0.0488 |
| Synthetic adversarial AUC | 0.511 (indistinguishable from real) |
| Agent | 6 tools, ≤ 8 steps, 3 decisions, never deny |
| Pilot | 10/10 agreement with the model policy, both modes |
| Memory | dashboard about 445 MB peak; model file 1.9 MB |

---

## A. Project overview

### A1. What does your system do, in one minute?

**Technical answer:**

- A claim enters, is validated (V01–V12) and scored by a calibrated LightGBM model (Tool 1).
- An LLM orchestrator then decides which of five more tools to call: SHAP, policy retrieval,
  policy verification, similar claims, claim history.
- Code aggregates the evidence. The LLM writes a decision with cited evidence, guardrails
  check it, and the graph pauses for a human adjuster, whose decision is audited in SQLite.

**Say it simply:**

> "Think of it as a junior investigator. A machine-learning model gives each claim a fraud
> score, then an AI agent gathers supporting evidence (why the score is high, whether the
> policy conditions hold, what similar past claims looked like) and writes a recommendation.
> A human adjuster always makes the final decision."

### A2. Why is this "agentic" and not just a classifier?

**Technical answer:**

- The LLM *plans*: it chooses tools and arguments based on the band and the evidence so far,
  in a ReAct loop.
- A classifier only outputs a probability. The agent produces an investigation: evidence,
  conflicts, missing information, unresolved questions and a rationale.

**Say it simply:**

> "The model answers 'how risky is this claim?'. The agent answers 'what should the adjuster
> look at, and why?' — and it decides by itself which checks to run."

### A3. What is the success criterion, and did you meet it?

**Technical answer:**

- The synopsis asks for precision and recall "meaningfully above the naive baseline".
- At the same flag rate, our lower CI bounds (precision 0.188, recall 0.717) are far above a
  random flagger (0.060, 0.227).
- We also beat the rules-only red-flag score on every metric (PR-AUC 0.252 vs 0.139).

**Say it simply:**

> "Yes. If you flagged the same number of claims at random, you'd catch about 23% of fraud.
> We catch 79%, and even the pessimistic end of our confidence interval is 72%."

### A4. Why not report accuracy?

**Technical answer:** With 6% fraud, predicting "legit" for everything is 94% accurate and
catches no fraud. We use PR-AUC (sensitive to the minority class), recall, precision, FPR and
Brier.

**Say it simply:**

> "A model that never flags anything is 94% accurate and useless. So we measure how much fraud
> we catch and how many of our flags are right."

---

## B. Data, EDA and synthetic data

### B1. How did you split the data, and why does the test set matter?

**Technical answer:**

- A stratified 70/15/15 split of the real claims.
- Synthetic rows are only ever in training.
- Model selection, early stopping, calibration and thresholds all used **real validation**.
- **Real test was opened once**, by the final run. The MLP never touched it, because it did not
  beat LightGBM on validation.

**Say it simply:**

> "We locked the test set away and used it once at the end. Every choice was made on
> validation data, so the test number is an honest estimate."

### B2. Why trust synthetic data? Isn't that risky?

**Technical answer:** We audited it in four ways:

1. **Marginals:** TVD at most 0.008.
2. **Fraud-rate agreement per category:** r = 0.89.
3. **Adversarial validation:** a classifier trying to tell real from synthetic gets AUC
   **0.511**, i.e. chance.
4. **Distance to closest record:** no memorised copies of real rows.

Utility (TSTR vs TRTR) showed it helps: +0.04–0.05 PR-AUC. We still down-weight it (w = 0.1)
and never evaluate on it.

**Say it simply:**

> "We tested it like a forgery: even a classifier trained to spot fakes couldn't tell the
> difference. And we never let it into validation or test, so it can help learning but can't
> fool our results."

### B3. Why weight 0.1 for synthetic rows?

**Technical answer:**

- The data-mix experiment: down-weighted mixes (0.1, 0.3) beat w = 1.0.
- 0.1 and 0.3 are within seed noise (gap 0.002, SD about 0.004).
- The final run picked 0.1. Down-weighting keeps the real distribution dominant while
  synthetic rows add minority-class coverage.

**Say it simply:**

> "Synthetic data helps, but real data should have the last word. Giving each synthetic row a
> tenth of the weight worked best."

### B4. Why exclude PolicyNumber, Sex and MaritalStatus?

**Technical answer:**

- **PolicyNumber** perfectly separates real from synthetic rows. That is leakage, and an ID
  carries no causal meaning.
- **Sex and MaritalStatus:** the ablation showed no predictive gain, so including them adds
  fairness risk for nothing. The fairness audit shows flag-rate/FPR gaps under 1.5 points.

**Say it simply:**

> "An ID number shouldn't predict fraud, and if gender doesn't improve the model, it has no
> business being in it."

### B5. How did you handle class imbalance? Why not SMOTE?

**Technical answer:**

- `scale_pos_weight` of about 15.7 (class weights) plus threshold tuning.
- SMOTE would create interpolated points on top of data that is already 96% synthetic.
  Interpolating one-hot categoricals also produces unrealistic claims.

**Say it simply:**

> "We tell the model a missed fraud matters about 16 times more, instead of inventing more fake
> fraud cases."

---

## C. Model choice

### C1. Why LightGBM over XGBoost, CatBoost or scikit-learn models?

**Technical answer:**

- **Validation PR-AUC:** LightGBM 0.291, XGBoost 0.283, Random Forest 0.246, logistic
  regression 0.193, MLP 0.262.
- **Honestly, LightGBM vs XGBoost is within noise** (CIs of about ±0.06).
- The tie-breakers:
  - fastest training on 300k rows (histogram-based, leaf-wise);
  - a 1.9 MB artifact with CPU inference;
  - exact, fast TreeSHAP for Tool 2.
- **CatBoost** was not in the final comparison (not installed in the run whose tables we
  have). It is a follow-up, not a rejection.
- **Random Forest and logistic regression** are clearly weaker. Logistic regression misses
  interactions such as Fault × BasePolicy.

**Say it simply:**

> "XGBoost and LightGBM were basically tied. We picked LightGBM because it scored highest on
> validation, it's fast, it runs on any laptop without a GPU, and we can explain every score
> exactly with SHAP."

### C2. Why not a neural network?

**Technical answer:**

- The tuned MLP (3 seeds, entity embeddings) scored 0.262 vs 0.291. Every MLP member was
  below every LightGBM member.
- Paired bootstrap difference: −0.029 [−0.074, +0.012].
- It is also slower, needs PyTorch, and has no exact SHAP.
- Trees are the known state of the art on small tabular data.

**Say it simply:**

> "We tried it fairly. It was worse on every seed and harder to explain, so there was no
> reason to switch."

### C3. Why did you not tune hyperparameters?

**Technical answer:**

- We *did* tune: Optuna TPE with a median pruner, on the B200.
- The adoption rule compares the tuned and default configurations as **means over the same 3
  seeds**. Tuned did not beat the defaults, so the defaults were kept.
- This avoids adopting a lucky single trial; for the MLP, the best trial (0.268) didn't
  survive 3 seeds.

**Say it simply:**

> "We searched, but the 'improvements' disappeared when we re-ran with different random seeds.
> So we kept the simpler defaults rather than fool ourselves."

### C4. Why a 3-seed ensemble?

**Technical answer:** Seed variance (SD about 0.003–0.004 PR-AUC) is the same size as the
differences we compare. Averaging 3 seeds stabilises the scores and makes comparisons fair.

**Say it simply:**

> "Same model, different random seed, slightly different result. Averaging three removes the
> luck."

---

## D. Thresholds and trade-offs (deep dive)

### D1. Why is your threshold 0.108 and not 0.5?

**Technical answer:**

- The threshold minimises expected cost with **missed fraud : unnecessary investigation =
  10:1**.
- With calibrated probabilities, the cost-optimal rule is roughly "flag if p > 1/(1+10)",
  about 0.09. We pick the empirical optimum on out-of-fold validation scores: **0.108**.
- At 6% prevalence, 0.5 would flag almost nothing.

**Say it simply:**

> "Missing a fraud costs far more than checking an honest claim. If it's ten times worse, you
> should investigate anything with more than about a 10% chance of fraud, and that's where
> 0.108 comes from."

### D2. What do the other thresholds give you?

**Technical answer** (all on test):

| Threshold | Recall | Precision | Flag rate |
|---|---|---|---|
| 0.108 (10:1) | 0.79 | 0.21 | 22.7% |
| Top-10% capacity, 0.189 | 0.41 | 0.26 | 9.5% |

- 5:1 (0.179) and 20:1 (0.037) are the stricter and looser alternatives.
- The cost ratio and capacity fraction are in `settings.yaml`, so the business owner chooses.

**Say it simply:**

> "It's a dial. If your team can only check 10% of claims, use the capacity threshold and catch
> 41% of fraud with better precision. If you can check more, catch 79%."

### D3. Why three bands, and why is the medium band special?

**Technical answer:**

- high ≥ 0.108; medium 0.037–0.108; low < 0.037.
- In the medium band, fraud is about **9%**, barely above the 6% base rate.
- Our analysis (`03b_medium_band_factors.md`) showed that red flags, SHAP drivers and
  similar-claim fraud **do not separate fraud within the medium band**, because the model has
  already used them.
- So a medium-band flag needs **out-of-model information**: a policy-condition FAIL, a
  claim-record contradiction, or a history contradiction.

**Say it simply:**

> "In the middle band, the model has already squeezed out everything the red flags can tell
> it. Flagging again on the same red flags would just double-count. We only flag there if
> something new turns up that the model couldn't see."

### D4. Did the thresholds hold up on unseen data?

**Technical answer:**

| | Validation | Test |
|---|---|---|
| Recall | 0.791 | 0.790 |
| Flag rate | 24.0% | 22.7% |

Calibration also held: test Brier 0.0488, against a base-rate predictor's 0.056.

**Say it simply:**

> "Yes. Recall was 79% on validation and 79% on test, so the threshold didn't overfit."

### D5. Why Platt calibration and not isotonic?

**Technical answer:**

- Class weights inflate raw scores: uncalibrated Brier 0.159.
- Platt: 0.0485; isotonic: 0.0500.
- Platt has 2 parameters, which is safer with only 139 validation frauds.
- Calibration was 5-fold cross-fitted, so the thresholds were picked on out-of-fold scores.

**Say it simply:**

> "Thresholds only make sense if '0.2' really means 20% risk. Calibration fixes that, and the
> simpler method was both more accurate and more stable with so few fraud cases."

### D6. What is the cost of your choice? Where does it fail?

**Technical answer:**

- **False positives:** 416 against 109 true positives; precision 0.21.
- **All Perils:** FPR 0.38.
- **Liability:** recall 0. These are 5 rare frauds at 0.6% prevalence.
- The agent's evidence and the adjuster are the mitigation.

**Say it simply:**

> "About four in five flags are honest claims. That's the price of catching 79% of fraud. The
> agent's job is to make those reviews fast, and we're upfront that Liability fraud is a blind
> spot."

---

## E. Explainability and fairness

### E1. How do you explain a score to an adjuster?

**Technical answer:**

- TreeSHAP gives exact, additive log-odds contributions per feature.
- One-hot levels are summed back to the original feature.
- Tool 2 returns the positive and negative factors with the base value. The red-flag score
  and Fault are the top global drivers.

**Say it simply:**

> "For every claim we can say which facts pushed the score up, which pushed it down, and by how
> much, and the pieces add up exactly to the score."

### E2. Is there anything in the SHAP results you can't explain?

**Technical answer:** Accident Month ranks 3rd in SHAP but was weak in the EDA (IV 0.064).
The model may be fitting month noise or interactions. A Month ablation on validation is the
planned follow-up; we did not revisit the test set.

**Say it simply:**

> "Yes: the model leans on the accident month more than we expected. We flagged it as an open
> question instead of hiding it."

### E3. Is the model fair?

**Technical answer:**

- Sex and MaritalStatus are not inputs.
- On test, flag rates and FPRs differ by under 1.5 points across these groups.
- Young age bands have higher FPRs. That is partly real (higher fraud rates), but each group
  has only 2–5 frauds, so it needs monitoring.

**Say it simply:**

> "Men and women, married and single, are flagged at almost the same rates. Young drivers get
> flagged more, partly because they really do have more fraud, but the samples are tiny, so
> we'd keep watching it."

---

## F. Agent design and orchestrator–tool interaction (deep dive)

### F1. Walk us through the graph.

**Technical answer:**

1. ingest
2. validate
3. score (Tool 1)
4. agent ⇄ tools (Tools 2–6, at most 8 steps)
5. complete_investigation
6. aggregate_evidence
7. synthesize
8. guardrails
9. human_review (interrupt)

Only `agent` and `synthesize` call an LLM; everything else is deterministic Python.

**Say it simply:**

> "It's a fixed pipeline with one flexible part in the middle. The steps that must always
> happen (validation, scoring, safety checks, human review) are hard-wired. Only the evidence
> gathering is left to the AI."

### F2. How does the orchestrator call tools? Can it call them wrongly?

**Technical answer:**

- Tools are **claim-bound**: the LLM picks the tool and arguments (e.g. `k`,
  `same_base_policy`, `top_k`) but can't change claim fields.
- **Dependencies are enforced in code:** `verify_policy` runs `retrieve_policy` first, and
  SHAP requires a fraud score.
- The prompt tells it not to repeat a call, and the loop stops at `max_steps = 8`.
- `complete_investigation` runs the required tools the LLM skipped (fraud score, SHAP,
  policy retrieval, verification, similar claims), and the trace logs them as `auto`.

**Say it simply:**

> "The AI decides what to look at, but it can't tamper with the claim, can't skip the
> essential checks, and can't loop forever."

### F3. Why a hybrid (ML + ReAct) instead of pure ML or a pure LLM?

**Technical answer:**

- **Pure ML:** a good score, but no investigation. No policy checks, no precedent, no
  rationale, and no way to use information outside the 89 features.
- **Pure LLM:**
  - LLMs are poorly calibrated on tabular risk;
  - they are non-deterministic;
  - they can hallucinate numbers;
  - they would have to read 15k claims to learn base rates.
- **The hybrid:** the ML does what it's best at (a calibrated, tested probability), and the
  LLM does what it's best at (planning, reading heterogeneous evidence, explaining in plain
  English).
- Guardrails tie the two together: the LLM can't approve a high-band claim or cite numbers
  that aren't in the tool outputs.

**Say it simply:**

> "The ML model is the calculator, the LLM is the investigator who writes the report, and the
> guardrails are the supervisor who checks the report against the facts. Each does the job
> it's good at."

### F4. Why LangGraph and not CrewAI, AutoGen or a plain LangChain agent?

**Technical answer:**

- We need deterministic nodes and LLM nodes in one graph, a hard step limit, and a native
  **interrupt + checkpointer** for human review.
- Multi-agent frameworks add conversational non-determinism we don't need: one orchestrator
  is enough.

**Say it simply:**

> "LangGraph lets us draw the exact flowchart and pause it for the human. Other frameworks are
> built for agents chatting with each other, which we didn't need."

### F5. What does evidence aggregation do, and why isn't the LLM doing it?

**Technical answer:**

- Code computes per-tool statuses, missing information, conflicts (e.g. high band but similar
  claims at base rate) and reason codes.
- These are output fields, so computing them in code means they can't be hallucinated. The
  LLM only writes the decision, rationale, evidence and questions.

**Say it simply:**

> "Facts are computed, never written by the AI. The AI only interprets them."

### F6. What did the pilots teach you?

**Technical answer:**

- **Pilot 1:** low-band claims were flagged because the policy document was missing (true for
  *every* claim). Fix: prompts explaining systemic gaps, plus the guardrail
  `unsupported_fraud_flag`.
- **Pilot 2:** every medium-band claim was flagged. Fix: the offline factor analysis, then
  out-of-model triggers only.
- **Pilot 3:** 10/10 agreement with the model policy in both simulated-data modes.

**Say it simply:**

> "The AI was too nervous at first: it treated 'we don't have the policy document' as
> suspicious, even though that's true for every claim. We measured it, fixed it with rules,
> and re-tested until it behaved."

---

## G. Guardrails, privacy and fallback (deep dive)

### G1. List your guardrails.

**Technical answer:**

1. Three decisions only; **never deny**.
2. A high band can never be APPROVE.
3. APPROVE needs all of: a fraud score, a consistent claim record, no policy-condition FAIL,
   and no failed required tool.
4. REQUEST_MORE_INFORMATION only with claim-level grounds (blocking validation, a PV02/PV04
   contradiction).
5. Unsupported flags in the low and medium bands are replaced.
6. Every cited number or ID must appear in the cited tool's output.
7. Every flag is tagged FRAUD_RISK, POLICY_CONDITION or SYSTEM_FAILURE.
8. Simulated data must be labelled.
9. After 2 failed LLM attempts, the rule-based fallback decides.

**Say it simply:**

> "The AI's answer is checked against hard rules before anyone sees it. If it breaks a rule,
> code replaces the decision and writes down why."

### G2. What happens if the LLM is down, slow or returns garbage?

**Technical answer:**

- Lenient JSON parsing, then a retry.
- After two failures, `rule_based_decision()` builds the decision from the same evidence
  record, with `decided_by = "rule_fallback"` and a note.
- Tools and scores never depend on the LLM, so the system **degrades gracefully**: the
  adjuster still gets a score, the evidence and a safe decision.

**Say it simply:**

> "If the AI fails, we fall back to a rulebook that uses the same evidence. Nothing crashes,
> and the human still reviews it."

### G3. Is it safe to send claim data to an external LLM API?

**Technical answer:**

- We send a compact evidence record. The Kaggle data has no names or addresses, only a
  PolicyNumber ID.
- With a hosted API, the data is processed under the provider's terms. Free tiers may log or
  train on inputs.
- For real claims, use **Ollama** (on-premises; nothing leaves the machine) or a contracted
  API with a no-retention agreement.
- Keys are only in `.env` or the environment settings, never in git.

**Say it simply:**

> "For this public dataset, a free API is fine. For real customers' claims, we'd run the model
> locally with Ollama so no data leaves the company."

### G4. What do you lose with a free API (Groq/Gemini) instead of Ollama + qwen2.5:72b?

**Technical answer:**

- **Privacy:** data goes to a third party.
- **Validation:** our pilot used qwen2.5:72b. A different model needs its own pilot, because
  tool-calling and JSON behaviour differ.
- **Rate limits and quotas:** a batch evaluation can be throttled.
- **No SLA:** models get deprecated.
- **What we keep:** scores, tools and guardrails are identical, and the fallback covers
  outages.
- Switching takes only env vars: `LLM_PROVIDER=openai`, `OPENAI_BASE_URL`, `MODEL_NAME`,
  `LLM_API_KEY`.

**Say it simply:**

> "Mainly privacy and reliability. The safety rules don't change, because they're in our code,
> not in the AI. But we'd re-test the new model before trusting it."

### G5. How do you stop the LLM hallucinating evidence?

**Technical answer:**

- Grounding check: every evidence item names a source tool, and its numbers and IDs must
  appear in that tool's output.
- An `unavailable` source can only be cited for that limitation, never as "no previous
  claims".
- Reason codes and conflicts are computed in code.
- Temperature is 0.

**Say it simply:**

> "Every fact the AI quotes has to be found, word for word, in what the tools returned.
> Otherwise it's rejected."

### G6. Why can't it deny claims?

**Technical answer:** Denial is a legal and financial decision with fairness implications.
At precision 0.21, most flags are honest claims. The system's role is triage: approve, flag,
or ask for information, with `requires_human_review = true` always.

**Say it simply:**

> "Four out of five flagged claims are honest. A tool like that should never be allowed to say
> no to a customer on its own."

---

## H. Simulated reference data

### H1. Why add simulated data? Isn't that faking results?

**Technical answer:**

- The Kaggle data has no policy documents and no linked history (PolicyNumber is unique), so
  Tools 3, 4 and 6 would always be `partial` or `unavailable`.
- The simulated DB demonstrates how the agent *uses* such evidence.
- Safeguards:
  - it is off by default;
  - every output is labelled "simulated reference data";
  - the generator reads only an allow-list of columns, and a test proves it never reads the
    label;
  - the evaluation reports off and on side by side.
- **Measured effect:** the same FRAUD_RISK recall (0.748 vs 0.748); one POLICY_CONDITION flag
  added.

**Say it simply:**

> "It's a labelled stand-in for the policy database a real insurer would have. It can't see
> who committed fraud, so it can't inflate our fraud numbers, and the results confirm that."

### H2. Why the 60-day reporting deadline?

**Technical answer:** Before finishing, we checked how many validation claims each clause
would fail. A clause failing more than about 5% of claims would generate flags from a
simulation artefact. The deadline was set to 60 days, with a conservative reporting lag
derived from the week/weekday fields.

**Say it simply:**

> "We tested the made-up rules first so they wouldn't flag lots of honest claims just because
> of how we invented the dates."

---

## I. Evaluation methodology

### I1. How do you evaluate the agent, not just the model?

**Technical answer:**

- A stratified sample: all test frauds plus 1 legit per fraud, re-weighted to the true
  prevalence.
- **Metrics:**
  - fraud → FRAUD_RISK flag (recall);
  - legit → APPROVE;
  - false-flag rate;
  - REQUEST_MORE_INFORMATION rate;
  - agreement with the model-only policy.
- Wilson CIs.
- Consistency re-runs (3 runs × 20 claims).
- An LLM-as-judge rubric on 40 rationales; 20 exported for manual review.
- One test run per model and mode, enforced with marker files.

**Say it simply:**

> "We check whether the agent catches fraud, whether it approves honest claims, whether it
> gives the same answer twice, and whether its explanations make sense — once, on data it's
> never seen."

### I2. Why count only FRAUD_RISK flags for fraud recall?

**Technical answer:** A SYSTEM_FAILURE flag (a tool crashed) or a POLICY_CONDITION flag (a
missed deadline) that happens to hit a fraud would inflate recall without being fraud
detection.

**Say it simply:**

> "If the system flags a claim because a tool broke, that's not catching fraud, so we don't
> give ourselves credit for it."

### I3. What evaluation results do you have so far?

**Technical answer:**

- **Offline rule-based baseline** on 278 validation claims: recall 0.748, legit-approve
  0.842.
- **LLM pilot** (qwen2.5:72b on the B200): 10/10 agreement.
- **The full live test-set run is still to be run.** Validation numbers are optimistic
  because the thresholds were chosen there.

**Say it simply:**

> "The pilot is clean, and the final run on the test set is the next step. We're not claiming
> numbers we haven't measured."

---

## J. Dependencies, versions and cross-platform wheels (deep dive)

### J1. Why pin exact versions?

**Technical answer:**

- The joblib artifacts contain pickled scikit-learn/LightGBM objects, which must load with the
  same versions that saved them.
- `requirements.txt` and `requirements-train.txt` share identical ML pins, so the B200 and a
  laptop are bit-compatible.
- Exact pins also make the build reproducible for examiners.

**Say it simply:**

> "The trained model is saved with specific library versions. If you load it with different
> ones, it might break or behave differently, so we lock them."

### J2. What was the PyYAML issue?

**Technical answer:**

- PyYAML 6.0.1 has no cp313 Windows wheel. On Python 3.13, pip fell back to building from
  source, which failed without a compiler and the right Cython.
- 6.0.2 ships cp313 wheels.
- After bumping, we verified **all 144 resolved packages** have Windows wheels for 3.11, 3.12
  and 3.13: `uv pip compile --python-platform x86_64-pc-windows-msvc --only-binary :all:`,
  plus a PyPI wheel-tag check.

**Say it simply:**

> "One small library had no ready-made installer for the newest Python on Windows. We updated
> it and then checked every other package the same way."

### J3. Why recommend Python 3.11/3.12 if 3.13 works?

**Technical answer:** Training ran on 3.11. The 3.11 and 3.12 wheels have the longest track
record. 3.13 works with the current pins, but any future upgrade risks hitting a package
without a cp313 wheel again.

**Say it simply:**

> "3.13 works today, but 3.11 and 3.12 are the safe, tested choice."

### J4. What was the langchain-core conflict?

**Technical answer:** `langchain-openai==1.7.2` requires `langchain-core>=1.6.10`, but we had
pinned 1.6.6, so the resolver failed on the B200. We bumped it to 1.6.11 and verified with a
clean install.

**Say it simply:**

> "Two packages wanted different versions of the same base library; we picked one both accept."

### J5. Why is torch not in the requirements?

**Technical answer:**

- The final model doesn't need it.
- On the B200, the preinstalled CUDA build of torch (Blackwell sm_100 needs CUDA 12.8+) must
  not be overwritten.
- CPU torch is an optional install, only for loading an MLP artifact.

**Say it simply:**

> "Our final model doesn't use PyTorch, and reinstalling it on the GPU server could break the
> GPU setup."

### J6. Why uv, if users install with pip?

**Technical answer:** pip can't reliably resolve for a different platform. Its cross-platform
mode mis-evaluated xgboost's Linux-only marker. `uv pip compile --python-platform` resolves
for Windows from Linux. uv is only a verification tool.

**Say it simply:**

> "We needed to check Windows compatibility from a Linux machine, and uv can do that."

---

## K. Performance, memory and scalability (deep dive)

### K1. Where is the bottleneck?

**Technical answer:**

- **The LLM**, by far: about 83 s per investigation with qwen2.5:72b in the pilot.
- **Everything else is milliseconds:** LightGBM scoring, TreeSHAP for one claim, a FAISS
  search over 10.8k vectors, SQLite.
- A cold first investigation (loading the model and index) took about 21 s in the cloud
  container. After that, the cost is the LLM calls.

**Say it simply:**

> "The maths is instant. Waiting for the AI to think is what takes time."

### K2. How much memory does it need? Will it fit free hosting?

**Technical answer:**

| Stage | RSS |
|---|---|
| After importing Streamlit | about 81 MB |
| After loading the model and scoring validation | about 261 MB |
| Peak | about 445 MB |

- The model file is 1.9 MB.
- A 512 MB free tier fits, but tightly.
- To reduce it: lazy-load SHAP and FAISS, avoid scoring the whole validation set at start-up,
  or use a slim deploy requirements file (no xgboost, catboost, optuna or jupyter).
- With Ollama, the LLM needs its own RAM/GPU: 7b about 5 GB, 72b about 45 GB+.

**Say it simply:**

> "The app itself needs under half a gigabyte, so it runs on a free server. The big local AI
> model is what needs a powerful machine, which is why the online-API option exists."

### K3. How would this scale to thousands of claims a day?

**Technical answer:**

- **Scoring** is vectorised. LightGBM scores thousands of rows per second on one CPU.
- **The agent step is embarrassingly parallel per claim.** Run workers concurrently,
  bounded by API rate limits or GPU throughput.
- **Production changes:**
  - a persistent checkpointer (PostgreSQL/Redis instead of `InMemorySaver`);
  - PostgreSQL for the audit store;
  - a queue (e.g. Celery/RQ);
  - batch-model routing (Haiku-class models for low-band claims);
  - vLLM for self-hosted throughput.
- **Cost lever:** skip the LLM for clear low-band claims.

**Say it simply:**

> "Scoring scales easily. For the AI part, we'd process many claims in parallel and only use
> the expensive AI on claims that actually need investigating."

### K4. Will FAISS scale?

**Technical answer:** At 10.8k vectors, exact `IndexFlatIP` is optimal. At millions of
vectors, switch to IVF or HNSW (approximate search). It is the same library and a one-line
change.

**Say it simply:**

> "For our size, exact search is instant. If the history grew to millions of claims, FAISS has
> faster approximate indexes built in."

### K5. What about concurrency in the dashboard?

**Technical answer:** Streamlit runs a session per user. SQLite handles a few concurrent
reviewers fine, and a claim can only be reviewed once. For many users, move to PostgreSQL and
a persistent checkpointer.

**Say it simply:**

> "It's fine for a small team. A big team would need a proper database server."

---

## L. Dashboard and audit

### L1. Why Streamlit?

**Technical answer:**

- Pure Python, calling the same `TriageRunner` as the batch scripts. No API layer, no
  JavaScript build.
- Testable headless with `AppTest`, and it runs on a laptop CPU.
- React/Flask would double the codebase for a prototype.

**Say it simply:**

> "We're ML engineers. Streamlit let us build a working review screen in Python, using the
> exact same code as the backend."

### L2. What is audited?

**Technical answer:**

- Each recommendation: decision, evidence, tool statuses, model version.
- Each adjuster decision.
- Overrides require a reason, and a second review of the same claim is refused.
- Stored in SQLite (`data/audit/audit.db`), which is git-ignored.

**Say it simply:**

> "Every AI suggestion and every human decision is recorded, and if the human disagrees, they
> have to say why."

---

## M. Testing and reproducibility

### M1. How do you test an LLM system?

**Technical answer:**

- Unit tests use a **stub LLM**, so the graph, tools and guardrails are tested
  deterministically without an API.
- Guardrail tests replay real pilot failures (e.g. PN-5736).
- There is a label-independence test for the simulated generator, and dashboard tests with
  `AppTest`.
- About 120 tests.
- LLM behaviour itself is measured by the pilot and evaluation runs, not by unit tests.

**Say it simply:**

> "We replace the AI with a fake during tests, so we can check that every rule and tool works
> exactly. Then we measure the real AI separately."

### M2. Can someone reproduce your results?

**Technical answer:**

- A fixed `random_state` of 42 and seeds 42/43/44.
- SHA-256 hashes of the inputs; pinned versions.
- Notebooks execute headless; `regenerate_model_report.py` recomputes the test metrics from
  the frozen artifacts and stops if they don't match `model_card.json`.

**Say it simply:**

> "Yes. Same data, same seeds, same versions, same numbers, and a script checks that they
> match."

---

## N. Limitations and future work

### N1. What are the main limitations?

**Technical answer:**

- **Liability fraud** recall is 0, and the **All Perils** FPR is 0.38.
- **LightGBM vs XGBoost** is within noise, and **CatBoost was not compared**.
- **The Month SHAP anomaly** is unexplained.
- **No real policy documents or history** exist.
- **The live LLM test run is pending.**
- **The 10:1 cost ratio** is an assumption.
- **The dataset is 1994–96 US data,** so concept drift is likely.

**Say it simply:**

> "We know where it's weak, and we've written it all down. A model that hides its weaknesses
> is more dangerous than one that admits them."

### N2. What would you do next?

**Technical answer:**

- Run the final live evaluation.
- Compare CatBoost.
- Run the Month ablation.
- A Liability-specific model or rules.
- Collect adjuster feedback for retraining and drift monitoring.
- Production hardening: PostgreSQL, a persistent checkpointer, auth, a slim deploy image.
- Pilot a free-API model before using it.

**Say it simply:**

> "Finish the final test, fix the blind spots, and learn from what the adjusters actually
> decide."

---

## O. Quick-fire questions

| Question | Short answer |
|---|---|
| Primary metric? | PR-AUC on real validation (imbalanced data) |
| Why bootstrap CIs? | Only about 139 frauds per split; point estimates alone mislead |
| Max agent steps? | 8 |
| Which tools are required? | fraud score, SHAP, policy retrieval, policy verification, similar claims |
| What's indexed in FAISS? | 10,793 real training claims only |
| Default LLMs? | Claude Sonnet (API), qwen2.5:7b (Ollama default), qwen2.5:72b (B200 pilot) |
| Temperature? | 0 |
| Ollama context window? | 16,384 (the 2,048 default truncates prompts) |
| Where are thresholds configured? | `config/settings.yaml` → `decision` |
| Can the system deny a claim? | Never |
| Where are API keys? | `.env` / environment settings, never committed |
| Who decides? | The human adjuster, always |
