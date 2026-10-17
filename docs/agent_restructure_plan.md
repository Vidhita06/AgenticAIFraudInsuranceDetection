# Six-tool agent restructure: implementation plan

This maps the team's *Agentic AI Insurance Fraud Detection — Project Plan* (one orchestrator,
six specialised tools, evidence aggregation, reasoning, final recommendation) onto the
existing LangGraph agent. It records what the data can and cannot support, so nothing in the
restructure invents evidence.

## 1. What changes, at a glance

| Plan component | Today | After the restructure |
|---|---|---|
| Layer 1: intake and validation | `ingest`, `validate` nodes | Unchanged. Each run also gets an `investigation_id`. |
| Layer 2: Orchestrator Agent | LLM ReAct loop over 2 tools; `mark_max_steps` auto-runs the policy check | LLM ReAct loop over 5 agent-callable tools. Code enforces tool dependencies. A `complete_investigation` node runs any *required* tool the agent skipped and logs it as `auto`. |
| Tool 1: Fraud Score | `score_claim` (probability, band, SHAP factors) | `score_claim`: predicted class, probability, risk band, threshold, model metadata, status. SHAP moves to Tool 2. The EDA red flags stay here as `risk_indicators`. |
| Tool 2: SHAP Explanation | inside `score_claim` | New `explain_claim`: signed SHAP values, positive/negative factors, base value, scale and limitations. |
| Tool 3: Policy Retrieval | none | New `retrieve_policy`: reads an actual policy store if one exists, else reports the recorded policy attributes and lists what is missing. |
| Tool 4: Policy Verification | `check_policy_rules` (rules + red flags) | New `verify_policy`: per-check `PASS` / `FAIL` / `UNKNOWN`, with rule, clause reference, reason and evidence. |
| Tool 5: Similar Claims | `find_similar_claims` | Adds provenance per match (genuine real claim, never synthetic), shared characteristics, differences, and an optional same-base-policy constraint. |
| Tool 6: Claim History | none | New `get_claim_history`: looks up an authorised history source. It reports `unavailable` when none exists, which is the case for this dataset (see §2). |
| Layer 4: Evidence Aggregation | none (synthesis read raw tool outputs) | New deterministic `aggregate_evidence` node: per-tool status and origin, findings, missing information, conflicts, warnings, completeness and investigation status. |
| Layer 5: Reasoning and Recommendation | `synthesize` + `guardrails` | `synthesize` reasons over the evidence record. Guardrails add two rules: no APPROVE on a policy FAIL, and no APPROVE without a fraud score. Reason codes are derived in code. |
| Layer 6: Output and audit | `TriageDecision`, trace, interrupt | `TriageDecision` gains the plan's final-output fields (§4). The trace records which tool was called by the agent, the orchestrator, a dependency or `auto`. |

Graph after the change:

```
START → ingest → validate → score (Tool 1) → agent ⇄ tools (Tools 2–6, ≤ max_steps)
      → complete_investigation → aggregate_evidence → synthesize → guardrails → human_review → END
```

## 2. Data facts that shape Tools 3, 4 and 6

The plan says these tools must use actual records and report clearly when records are
missing. The repository's data supports the following:

- **There are no policy documents.** The Kaggle data records only policy *attributes* on each
  claim: `BasePolicy`, `PolicyType`, `Deductible`, `AgentType`, and the binned
  `Days_Policy_Accident` / `Days_Policy_Claim`. There are no policy versions, validity dates,
  limits, exclusions, reporting requirements or clauses.
  - Tool 3 therefore reads a policy store, `data/policies/<PolicyNumber>.json`, when one exists.
    None is shipped and none will be generated.
  - Without a store, Tool 3 returns the recorded attributes with `status: partial`, labelled as
    claim-recorded attributes, and lists every missing category under `missing_information`.
- **There is no linked claim history.** `PolicyNumber` is unique across all 15,420 claims, so
  no policy or customer has more than one claim.
  - Tool 6 reads an authorised history file (`claim_history.source` in the settings) when
    one is configured.
  - Otherwise it returns `status: unavailable` and says so.
  - It still reports `PastNumberOfClaims` and `AddressChange_Claim`, but explicitly as summary
    fields recorded on this claim, not as previous claim records.
- **Policy Verification without policy text.** Checks that need policy terms return
  `UNKNOWN` with the reason; these are coverage conditions, limits, exclusions and reporting
  requirements. Checks the claim record can support return `PASS` / `FAIL`:
  - PolicyType consistency (V05);
  - claim timeline (V06, V09);
  - incident inside the coverage period (the `Days_Policy_Accident` bin; `none` → `UNKNOWN`).

  On real validation and test, the checks that can `FAIL` fire on 3 of 4,626 claims. So the
  new "no APPROVE on a policy FAIL" guardrail barely changes the decision mix.
- **An UNKNOWN is a limitation, not a red flag.** It goes into `missing_information` and the
  rationale. It does not block APPROVE on its own; otherwise every claim would need more
  information simply because the dataset has no policy text.

## 3. Orchestration rules

- **Tool 1** runs before the agent, as the plan's "initial model-based risk assessment".
- **Agent-callable tools:** `explain_fraud_score(top_k)`, `retrieve_policy()`, `verify_policy()`,
  `find_similar_claims(k, same_base_policy)` and `get_claim_history()`. They are claim-bound, so
  the LLM chooses tools and arguments but can never retype the claim.
- **Dependencies are enforced in code:**
  - `verify_policy` runs `retrieve_policy` first if it has not run (origin `dependency`).
  - `explain_fraud_score` needs a successful fraud score; otherwise it reports `failed`.
- **Required tools** (`agent.required_tools`, default: fraud score, SHAP explanation, policy
  retrieval, policy verification) are run by `complete_investigation` if the agent skipped them.
  - These are cheap, local and deterministic, and the adjuster needs them.
  - Similar claims and claim history remain the agent's choice.
  - Nothing runs automatically when the claim fails schema validation (V01/V02).
- **Statuses** are `completed`, `partial`, `unavailable`, `failed` and `not_run`. They keep a
  negative finding separate from a missing source or a failed lookup.

## 4. Final output (`TriageDecision`)

- **Existing fields, kept:**
  - `claim_id`, `decision`, `fraud_probability`, `risk_band`, `rationale`;
  - `evidence`, `validation_issues`, `similar_claim_ids`, `requires_human_review` (always true);
  - `decided_by`, `guardrail_notes`, `model_version`.
- **New fields:**
  - `investigation_id`, `investigation_status`, `reason_codes`, `predicted_class`;
  - `policy_findings`, `similar_claim_findings`, `claim_history_summary`;
  - `missing_information`, `conflicting_evidence`, `tool_statuses`, `unresolved_questions`.
- **Decision labels:** `APPROVE`, `FLAG_FOR_INVESTIGATION` and `REQUEST_MORE_INFORMATION`, as
  in the plan. The old `REQUEST_MORE_INFO` is still accepted when parsing model output.
- **Evidence sources:** `validation`, `fraud_score`, `shap_explanation`, `policy_retrieval`,
  `policy_verification`, `similar_claims` and `claim_history`. Every fact must still be
  grounded in its source tool's output. A source that reported `unavailable` can be cited only
  for that limitation, never as a negative finding (e.g. "no previous claims").
- **Fields filled in code:** the LLM writes only the decision, rationale, evidence and
  unresolved questions. Reason codes, missing information, conflicts and tool statuses come
  from the evidence record, so they cannot be hallucinated.

## 5. Conflict rules (evidence aggregation)

- A `high` band, but at most the base rate of the similar claims were fraudulent (k ≥ 5).
- A `low` band, but ≥ 40% of the similar claims were fraudulent.
- A `low` band, but one or more warning-level risk indicators (red flags) fired.
- Any policy-verification `FAIL` when the band is not `high`.

## 6. Files

- **New:**
  - `src/tools/shap_explanation_tool.py`, `policy_retrieval_tool.py`,
    `policy_verification_tool.py`, `claim_history_tool.py`;
  - `src/agent/evidence.py` (aggregation, conflicts and reason codes);
  - tests for each.
- **Changed:**
  - `src/tools/{__init__,fraud_score_tool,similar_claims_tool}.py`;
  - `src/agent/{nodes,graph,guardrails,runner}.py`;
  - `src/schemas/{decision,state}.py`;
  - `config/prompts/*.md`, `config/settings.yaml`;
  - `evaluation/agent_eval.py`, `scripts/run_batch_triage.py`;
  - `docs/make_architecture_diagram.py` (+ png), the README, and the tests.
- **Removed:** `src/tools/policy_check_tool.py`. Its rule checks are split between Tool 4
  (policy conditions) and Tool 1 (risk indicators).

## 7. Effect on the Step D pilot

- Prompts, tool outputs and the decision label all change. Pilot results gathered before this
  restructure (latency, tokens, tool-choice rates) do not carry over.
- Re-run `python scripts/run_batch_triage.py pilot` after pulling.
- The model artifacts and the FAISS index are unchanged, so no rebuild is needed.
