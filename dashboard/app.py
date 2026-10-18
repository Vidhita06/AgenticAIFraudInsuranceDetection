"""Claims adjuster dashboard.

    streamlit run dashboard/app.py

Submit a claim (form, or pick a real validation claim), run the investigation agent (LLM from
.env, or rule-based with no LLM), review the recommendation with its evidence, the six tool
results, guardrail notes and the SHAP chart, then approve or override it. The decision resumes
the paused LangGraph run and is written to the audit store (paths.audit_db). CPU only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import core  # noqa: E402

st.set_page_config(page_title="Claims Adjuster Dashboard", page_icon="🛡️", layout="wide")

DECISION_STYLE = {"APPROVE": ("✅", "green"), "FLAG_FOR_INVESTIGATION": ("🚩", "red"),
                  "REQUEST_MORE_INFORMATION": ("📝", "orange"), "DENY": ("⛔", "gray")}
STATUS_ICON = {"completed": "🟢", "partial": "🟡", "unavailable": "⚪", "failed": "🔴", "not_run": "⚫"}
RESULT_ICON = {"PASS": "🟢 PASS", "FAIL": "🔴 FAIL", "UNKNOWN": "⚪ UNKNOWN"}


# ---------------------------------------------------------------------------------------------
# Cached resources (one runner per agent mode and data mode: its checkpointer holds paused runs)
# ---------------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Scoring the validation claims…")
def load_validation() -> pd.DataFrame:
    return core.validation_claims()


@st.cache_resource(show_spinner="Preparing the agent…")
def get_runner(mode: str, simulated: bool):
    setup = core.build_agent(mode)
    if setup.error:
        return setup, None
    return setup, core.make_runner(setup, simulated)


def badge(decision: str | None) -> str:
    icon, color = DECISION_STYLE.get(decision or "", ("", "gray"))
    return f":{color}[**{icon} {decision}**]" if decision else "–"


# ---------------------------------------------------------------------------------------------
# Sidebar: agent mode, data mode, adjuster
# ---------------------------------------------------------------------------------------------

st.sidebar.title("🛡️ Adjuster dashboard")
mode_label = st.sidebar.radio("Agent", ["Rule-based (no LLM)", "LLM from .env"], index=0,
                              help="Rule-based needs no model or GPU. LLM mode uses LLM_PROVIDER / MODEL_NAME "
                                   "from .env (Ollama, an OpenAI-compatible API, or Anthropic).")
mode = "llm" if mode_label.startswith("LLM") else "rule_based"
sim_available = core.simulated_db_available()
simulated = st.sidebar.toggle("Simulated reference data", value=False, disabled=not sim_available,
                              help="Illustrative policies, clauses and prior claims for Tools 3, 4 and 6 "
                                   "(scripts/build_simulated_db.py). Off = the real data only.")
if not sim_available:
    st.sidebar.caption("Simulated database not built: `python scripts/build_simulated_db.py`")
adjuster_id = st.sidebar.text_input("Adjuster ID", value="adjuster-1")

setup, runner = get_runner(mode, simulated)
st.sidebar.markdown(f"**Agent:** {setup.description}")
if setup.error:
    st.sidebar.error(setup.error)
    if st.sidebar.button("Retry LLM connection"):
        get_runner.clear()
        st.rerun()
if not core.vector_index_available():
    st.sidebar.warning("Similar-claims index missing: run `python scripts/build_vector_index.py`.")

with st.sidebar.expander("Recent audit entries"):
    if runner is not None:
        recent = runner.audit.rows("adjuster_decisions", limit=10)
        st.caption(f"`{runner.audit.path.relative_to(ROOT) if runner.audit.path.is_relative_to(ROOT) else runner.audit.path}`")
        st.dataframe(pd.DataFrame(recent)[["recorded_at", "claim_id", "recommendation", "adjuster_decision",
                                           "overrides_recommendation", "override_reason"]] if recent else
                     pd.DataFrame(), hide_index=True)

st.title("Vehicle claim investigation")
st.caption("The agent recommends; you decide. It never denies a claim, and every recommendation needs your review.")

if runner is None:
    st.error("The LLM agent is not available (see the sidebar). Switch to **Rule-based (no LLM)** to continue.")
    st.stop()

# ---------------------------------------------------------------------------------------------
# 1. Claim
# ---------------------------------------------------------------------------------------------

st.header("1 · Claim")
source = st.radio("Claim source", ["Pick a validation claim", "New claim (blank form)"], horizontal=True)
truth = None
if source.startswith("Pick"):
    df = load_validation()
    bands = st.multiselect("Risk bands", ["low", "medium", "high"], default=["low", "medium", "high"],
                           help="The fraud model's band for each claim, to find interesting examples.")
    pool = df[df.risk_band.isin(bands)] if bands else df
    labels = {core.claim_label(r): i for i, r in pool.iterrows()}
    choice = st.selectbox(f"Validation claim ({len(pool)} available)", list(labels))
    row = df.loc[labels[choice]]
    base_claim, truth = core.claim_from_row(row), int(row[core.TARGET])
    form_key = f"PN-{int(row.PolicyNumber)}"
else:
    base_claim, form_key = core.blank_claim(), "blank"

# Collapse the form only while it shows the claim whose result is on screen.
has_result = bool(st.session_state.get("thread")) and st.session_state.get("run_form") == form_key
with st.expander("Claim fields (review or edit, then run)", expanded=not has_result), st.form(f"claim-{form_key}"):
    claim: dict = {}
    cols = st.columns(len(core.FIELD_GROUPS))
    for col, (group, fields) in zip(cols, core.FIELD_GROUPS.items()):
        with col:
            st.markdown(f"**{group}**")
            for f in fields:
                opts, val = core.field_options(f), base_claim.get(f)
                if opts:
                    claim[f] = st.selectbox(f, opts, index=opts.index(val) if val in opts else 0, key=f"{form_key}-{f}")
                elif f == "PolicyNumber":
                    claim[f] = st.number_input(f, min_value=0, value=int(val) if val is not None else 0,
                                               help="0 = no policy number", key=f"{form_key}-{f}") or None
                else:
                    claim[f] = st.number_input(f, min_value=0, max_value=120, value=int(val or 0),
                                               help="0 = age not recorded" if f == "Age" else None, key=f"{form_key}-{f}")
    run = st.form_submit_button("▶ Run investigation", type="primary")

if truth is not None:
    with st.expander("Known outcome (evaluation data; never shown to the agent)"):
        st.write("Fraud found" if truth else "No fraud found")

if run:
    with st.spinner("Investigating… (rule-based: a few seconds; LLM: up to a minute or more)"):
        try:
            tid, decision, trace = runner.run_claim(claim)
            st.session_state.update(thread=tid, run_mode=(mode, simulated), run_form=form_key)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"The investigation failed: {type(e).__name__}: {e}")
            st.stop()

thread = st.session_state.get("thread")
if not thread or st.session_state.get("run_mode") != (mode, simulated):
    st.info("Run an investigation to see the recommendation.")
    st.stop()

state = core.run_state(runner, thread)
d = state["decision"]
tr = state.get("tool_results") or {}
record = state.get("evidence_record") or {}

# ---------------------------------------------------------------------------------------------
# 2. Recommendation
# ---------------------------------------------------------------------------------------------

st.header("2 · Recommendation")
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"**Decision**  \n{badge(d['decision'])}")
c2.markdown(f"**Flag category**  \n{d.get('flag_category') or '–'}")
p = d.get("fraud_probability")
c3.metric("Fraud probability", f"{p:.3f}" if p is not None else "–", d.get("risk_band") or None,
          delta_color="off")
decided = {"llm": "LLM agent", "rule_fallback": "rule-based mode" if mode == "rule_based"
           else "rule-based fallback (the LLM failed)"}.get(d["decided_by"], d["decided_by"])
c4.markdown(f"**Decided by**  \n{decided} · investigation {d.get('investigation_status')}")
if d.get("simulated_data"):
    st.warning("Simulated reference data was used for policy and history evidence (illustrative, not real records).")
st.markdown("**Reason codes:** " + " ".join(f"`{c}`" for c in d.get("reason_codes", [])))
st.markdown("**Rationale**")
st.write(d["rationale"])

left, right = st.columns(2)
with left:
    st.markdown("**Evidence (each fact traced to its tool)**")
    for e in d.get("evidence", []):
        st.markdown(f"- `{e['source_tool']}` {e['fact']}")
    if not d.get("evidence"):
        st.caption("No evidence items.")
    if d.get("unresolved_questions"):
        st.markdown("**Unresolved questions**")
        for q in d["unresolved_questions"]:
            st.markdown(f"- {q}")
with right:
    if d.get("guardrail_notes"):
        st.markdown("**Guardrail notes**")
        for n in d["guardrail_notes"]:
            st.warning(n)
    if d.get("conflicting_evidence"):
        st.markdown("**Conflicting evidence**")
        for c in d["conflicting_evidence"]:
            st.markdown(f"- {c}")
    if d.get("missing_information"):
        with st.expander(f"Missing information ({len(d['missing_information'])})"):
            for m in d["missing_information"]:
                st.markdown(f"- {m}")

# ---------------------------------------------------------------------------------------------
# 3. Tool results
# ---------------------------------------------------------------------------------------------

st.header("3 · Tool results")
statuses = d.get("tool_statuses", {})
st.dataframe(pd.DataFrame([{"tool": label, "status": f"{STATUS_ICON.get(statuses.get(src, {}).get('status'), '')} "
                                                     f"{statuses.get(src, {}).get('status', '–')}",
                            "called by": statuses.get(src, {}).get("origin") or "–",
                            "data": statuses.get(src, {}).get("data_origin") or "real"}
                           for src, label in core.TOOL_TABS]), hide_index=True, width="stretch")
tabs = st.tabs([label for _, label in core.TOOL_TABS] + ["Trace"])
for (src, label), tab in zip(core.TOOL_TABS, tabs):
    out = tr.get(src)
    with tab:
        if out is None:
            st.caption("Not run for this claim.")
            continue
        if not out.get("ok"):
            st.error(out.get("error"))
            continue
        if src == "validation":
            st.write("Blocking problem" if out.get("blocking") else "No blocking problem")
            if out.get("failed_rules"):
                st.dataframe(pd.DataFrame(out["failed_rules"])[["rule_id", "severity", "message"]], hide_index=True)
        elif src == "fraud_score":
            a, b, c = st.columns(3)
            a.metric("Probability", out["fraud_probability"])
            b.metric("Risk band", out["risk_band"])
            c.metric("Predicted class", out["predicted_class"])
            st.caption(f"Thresholds: {out.get('thresholds')} · model {out.get('model_version')}")
            if out.get("risk_indicators"):
                st.markdown("**Red flags (rule-based risk indicators)**")
                st.dataframe(pd.DataFrame(out["risk_indicators"])[["rule_id", "severity", "condition", "support"]],
                             hide_index=True)
        elif src == "shap_explanation":
            sf = core.shap_frame(out)
            chart = alt.Chart(sf).mark_bar().encode(
                x=alt.X("shap_value:Q", title="SHAP contribution (log-odds)"),
                y=alt.Y("feature:N", sort=alt.EncodingSortField("shap_value", op="sum", order="descending"), title=None,
                        axis=alt.Axis(labelLimit=260, labelOverlap=False)),
                color=alt.Color("direction:N", scale=alt.Scale(domain=["raises fraud score", "lowers fraud score"],
                                                               range=["#d62728", "#1f77b4"]), legend=alt.Legend(orient="bottom")),
                tooltip=["feature", "value", "shap_value"]).properties(height=alt.Step(34))
            st.altair_chart(chart, width="stretch")
            st.dataframe(sf, hide_index=True)
            st.caption(" ".join(out.get("limitations", [])))
        elif src == "policy_retrieval":
            st.write(f"Status: **{out.get('status')}** · source: {out.get('source', 'claim record')}")
            if out.get("wording_note"):
                st.caption(out["wording_note"])
            st.json(out.get("recorded_policy_attributes", {}), expanded=False)
            if out.get("document"):
                st.json(out["document"], expanded=False)
            for m in out.get("missing_information", []):
                st.markdown(f"- {m}")
        elif src == "policy_verification":
            n = out["counts"]
            st.write(f"Overall: **{out['overall_result']}** · {n['PASS']} PASS · {n['FAIL']} FAIL · "
                     f"{n['UNKNOWN']} UNKNOWN · source: {out.get('policy_source')}")
            cf = core.checks_frame(out)
            cf["result"] = cf["result"].map(RESULT_ICON)
            st.dataframe(cf, hide_index=True, width="stretch")
        elif src == "similar_claims":
            st.write(f"{out['neighbour_frauds']} of {out['k']} most similar genuine past claims were fraudulent "
                     f"(historical base rate {out['historical_base_rate']:.1%}).")
            st.dataframe(core.neighbours_frame(out), hide_index=True, width="stretch")
            for k in out.get("key_differences", []):
                st.caption(k)
        elif src == "claim_history":
            st.write(f"Status: **{out['status']}** · previous claims: {out.get('previous_claim_count', '–')}")
            if out.get("limitation"):
                st.caption(out["limitation"])
            if out.get("note"):
                st.caption(out["note"])
            hf = core.history_frame(out)
            if len(hf):
                st.dataframe(hf, hide_index=True)
        with st.expander("Raw output"):
            st.json(out, expanded=False)
with tabs[-1]:
    st.json(state.get("trace", []), expanded=False)

# ---------------------------------------------------------------------------------------------
# 4. Adjuster review
# ---------------------------------------------------------------------------------------------

st.header("4 · Your decision")
if runner.pending_review(thread):
    with st.form(f"review-{thread}"):
        choice = st.radio("Decision", core.ADJUSTER_DECISIONS, index=core.ADJUSTER_DECISIONS.index(d["decision"]),
                          horizontal=True, help="Only you can deny a claim.")
        reason = st.text_area("Override reason (required if you change the recommendation)")
        notes = st.text_area("Notes (optional)")
        submit = st.form_submit_button("Submit decision", type="primary")
    if submit:
        if choice != d["decision"] and not reason.strip():
            st.error("Please give a reason for overriding the recommendation.")
        else:
            try:
                runner.resume(thread, {"decision": choice, "adjuster_id": adjuster_id,
                                       "override_reason": reason.strip() or None, "notes": notes.strip() or None})
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not record the decision: {type(e).__name__}: {e}")
else:
    adj = state.get("adjuster") or {}
    st.success(f"Reviewed by {adj.get('adjuster_id')}: {badge(adj.get('decision'))}"
               + (f" (override: {adj.get('override_reason')})" if adj.get("override_reason") else ""))
    rows = runner.audit.rows("adjuster_decisions", investigation_id=d.get("investigation_id"))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
