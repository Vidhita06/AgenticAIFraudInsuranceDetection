"""Insurance Fraud AI - Claims Adjuster Dashboard.

    streamlit run dashboard/app.py

Multi-page dashboard: Home, New Investigation, Claim Investigation, Claim History,
Reports, Settings. Styled with a dark sidebar and card-based layout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import core

st.set_page_config(
    page_title="Insurance Fraud AI",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Dark sidebar */
section[data-testid="stSidebar"] {
    background-color: #1a1f36;
}
section[data-testid="stSidebar"] * {
    color: #c8d0e0 !important;
}
section[data-testid="stSidebar"] .stRadio label span {
    color: #c8d0e0 !important;
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"] span {
    color: #ffffff !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] hr {
    border-color: #2d3555;
}

/* Card style */
div[data-testid="stContainer"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}

/* Metric value styling */
div[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Badge styles */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-high { background: #fee2e2; color: #dc2626; }
.badge-medium { background: #fef3c7; color: #d97706; }
.badge-low { background: #d1fae5; color: #059669; }
.badge-approve { background: #d1fae5; color: #059669; }
.badge-flag { background: #fee2e2; color: #dc2626; }
.badge-rmi { background: #fef3c7; color: #d97706; }
.badge-deny { background: #e5e7eb; color: #6b7280; }

/* Status dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot-green { background: #10b981; }
.status-dot-red { background: #ef4444; }
.status-dot-yellow { background: #f59e0b; }
.status-dot-gray { background: #9ca3af; }

/* Checklist item */
.checklist-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.9rem;
}
.checklist-item:last-child { border-bottom: none; }
.check-pass { color: #10b981; font-weight: 600; }
.check-fail { color: #ef4444; font-weight: 600; }
.check-unknown { color: #9ca3af; font-weight: 600; }
.check-missing { color: #f59e0b; font-weight: 600; }

/* KPI card */
.kpi-card {
    text-align: center;
    padding: 1.2rem;
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
}
.kpi-label {
    font-size: 0.85rem;
    color: #64748b;
    margin: 4px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    font-size: 1rem;
    font-weight: 600;
    color: #1e293b;
}

/* Claim bar */
.claim-bar {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 12px 20px;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.claim-bar-item {
    display: flex;
    flex-direction: column;
}
.claim-bar-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.claim-bar-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e293b;
}

/* Gauge */
.gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 0;
}
.gauge-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 4px;
}

/* Reason list */
.reason-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 6px 0;
}
.reason-number {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #3b82f6;
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Verification row */
.verify-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f1f5f9;
}
.verify-row:last-child { border-bottom: none; }

/* Nav item active highlight */
section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 8px 12px;
    border-radius: 8px;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.08);
}

/* Hide radio circle in sidebar nav */
.nav-radio .stRadio > div > label > div:first-child {
    display: none;
}

/* Disclaimer */
.disclaimer {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #1e40af;
    margin-top: 12px;
}

/* Thin scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* Fix stContainer nesting */
div[data-testid="stContainer"] div[data-testid="stContainer"] {
    border: none;
    box-shadow: none;
    padding: 0;
}

/* Table styling */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

NAV_ICONS = {
    "Home": "house",
    "New Investigation": "search",
    "Claim Investigation": "file-text",
    "Claim History": "clock",
    "Reports": "bar-chart-2",
    "Settings": "settings",
}

DECISION_BADGE = {
    "APPROVE": ("Approved", "badge-approve"),
    "FLAG_FOR_INVESTIGATION": ("Flag for Investigation", "badge-flag"),
    "REQUEST_MORE_INFORMATION": ("Request More Info", "badge-rmi"),
    "DENY": ("Denied", "badge-deny"),
}

RISK_BADGE = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}

STATUS_MAP = {
    "completed": ("Completed", "status-dot-green"),
    "partial": ("Partial", "status-dot-yellow"),
    "unavailable": ("Not Available", "status-dot-gray"),
    "failed": ("Failed", "status-dot-red"),
    "not_run": ("Not Run", "status-dot-gray"),
}

TOOL_LABELS = {
    "validation": "Data Validation",
    "fraud_score": "Fraud Prediction",
    "shap_explanation": "SHAP Analysis",
    "policy_retrieval": "Policy Retrieval",
    "policy_verification": "Policy Verification",
    "similar_claims": "Similar Claims Retrieval",
    "claim_history": "Claim History",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decision_badge_html(decision: str | None) -> str:
    if not decision:
        return ""
    label, cls = DECISION_BADGE.get(decision, (decision, "badge-deny"))
    return f'<span class="badge {cls}">{label.upper()}</span>'


def risk_badge_html(band: str | None) -> str:
    if not band:
        return ""
    cls = RISK_BADGE.get(band, "badge-low")
    return f'<span class="badge {cls}">{band.upper()}</span>'


def status_dot_html(status: str) -> str:
    label, dot_cls = STATUS_MAP.get(status, (status, "status-dot-gray"))
    return f'<span class="status-dot {dot_cls}"></span>{label}'


def fraud_gauge_svg(probability: float, size: int = 180) -> str:
    pct = min(max(probability * 100, 0), 100)
    angle = pct / 100 * 270
    r = size / 2 - 15
    cx, cy = size / 2, size / 2 + 10
    import math
    start_angle = 135
    end_angle = start_angle + angle
    bg_end_angle = start_angle + 270

    def arc_point(a):
        rad = math.radians(a)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    def arc_path(sa, ea, radius):
        sx, sy = arc_point(sa)
        ex, ey = arc_point(ea)
        large = 1 if (ea - sa) > 180 else 0
        return f"M {sx},{sy} A {radius},{radius} 0 {large} 1 {ex},{ey}"

    if pct < 30:
        color = "#10b981"
    elif pct < 60:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    bg_path = arc_path(start_angle, bg_end_angle, r)
    fg_path = arc_path(start_angle, end_angle, r) if angle > 0.5 else ""

    return f"""
    <div class="gauge-container">
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size + 20}">
        <path d="{bg_path}" fill="none" stroke="#e5e7eb" stroke-width="14" stroke-linecap="round"/>
        {"" if not fg_path else f'<path d="{fg_path}" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"/>'}
        <text x="{cx}" y="{cy - 5}" text-anchor="middle" font-size="32" font-weight="700" fill="{color}">{pct:.0f}%</text>
        <text x="{cx}" y="{cy + 18}" text-anchor="middle" font-size="12" fill="#64748b">Fraud Probability</text>
    </svg>
    </div>
    """


@st.cache_data(show_spinner="Loading validation claims...")
def load_validation() -> pd.DataFrame:
    return core.validation_claims()


@st.cache_resource(show_spinner="Preparing the agent...")
def get_runner(mode: str, simulated: bool):
    setup = core.build_agent(mode)
    if setup.error:
        return setup, None
    return setup, core.make_runner(setup, simulated)


def load_model_card() -> dict:
    try:
        with open(ROOT / "models" / "model_card.json") as f:
            return json.load(f)
    except Exception:
        return {}


def get_audit_rows(runner, table: str, limit: int = 200) -> list[dict]:
    if runner and runner.audit:
        try:
            return runner.audit.rows(table, limit=limit)
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "agent_mode" not in st.session_state:
    st.session_state.agent_mode = "rule_based"
if "simulated" not in st.session_state:
    st.session_state.simulated = False
if "adjuster_id" not in st.session_state:
    st.session_state.adjuster_id = "adjuster-1"

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 20px 0;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:36px; height:36px; background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                        border-radius:8px; display:flex; align-items:center; justify-content:center;">
                <span style="color:white; font-size:18px;">&#x1F6E1;</span>
            </div>
            <div>
                <div style="font-size:1.05rem; font-weight:700; color:#ffffff !important;">Insurance Fraud AI</div>
                <div style="font-size:0.72rem; color:#94a3b8 !important;">Smarter Claims. Safer Tomorrow.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = ["Home", "New Investigation", "Claim Investigation", "Claim History", "Reports", "Settings"]
    nav_choice = st.radio(
        "Navigation",
        pages,
        index=pages.index(st.session_state.page) if st.session_state.page in pages else 0,
        label_visibility="collapsed",
    )
    if nav_choice != st.session_state.page:
        st.session_state.page = nav_choice
        st.rerun()

    st.markdown("---")

    mode_label = st.radio(
        "Agent Mode",
        ["Rule-based (no LLM)", "LLM from .env"],
        index=0 if st.session_state.agent_mode == "rule_based" else 1,
        help="Rule-based needs no model or GPU. LLM mode uses LLM_PROVIDER / MODEL_NAME "
             "from .env (Ollama, an OpenAI-compatible API, or Anthropic).",
    )
    new_mode = "llm" if mode_label.startswith("LLM") else "rule_based"
    if new_mode != st.session_state.agent_mode:
        st.session_state.agent_mode = new_mode
        get_runner.clear()
        st.rerun()

    sim_avail = core.simulated_db_available()
    new_sim = st.toggle(
        "Simulated reference data",
        value=st.session_state.simulated,
        disabled=not sim_avail,
        help="Illustrative policies, clauses and prior claims for Tools 3, 4, 6.",
    )
    if not sim_avail:
        st.caption("Simulated DB not built")
    if new_sim != st.session_state.simulated:
        st.session_state.simulated = new_sim
        get_runner.clear()
        st.rerun()

    mode = st.session_state.agent_mode
    sim = st.session_state.simulated
    setup, runner = get_runner(mode, sim)

    online = not setup.error
    dot = "green" if online else "red"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:6px; padding:4px 0;">
        <span class="status-dot status-dot-{dot}"></span>
        <span style="font-size:0.82rem;">System {'Online' if online else 'Offline'}</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Agent: {setup.description}")
    if setup.error:
        st.error(setup.error)
        if st.button("Retry connection"):
            get_runner.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="padding:8px 0;">
        <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:30px; height:30px; background:#2d3555; border-radius:6px;
                        display:flex; align-items:center; justify-content:center;">
                <span style="font-size:14px;">&#x1F6E1;</span>
            </div>
            <div>
                <div style="font-size:0.78rem; font-weight:600; color:#e2e8f0 !important;">AI-Powered</div>
                <div style="font-size:0.68rem; color:#94a3b8 !important;">Fraud Detection</div>
            </div>
        </div>
        <div style="font-size:0.68rem; color:#64748b !important; margin-top:8px;">
            Helping you reduce fraud, protect claims and improve efficiency.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: Home
# ---------------------------------------------------------------------------

def page_home():
    st.markdown("## Dashboard Overview")
    st.caption("Real-time fraud detection insights and recent activity.")

    recs = get_audit_rows(runner, "recommendations")
    decisions = get_audit_rows(runner, "adjuster_decisions")

    total = len(recs)
    flagged = sum(1 for r in recs if r.get("decision") == "FLAG_FOR_INVESTIGATION")
    approved_adj = sum(1 for d in decisions if d.get("adjuster_decision") == "APPROVE")
    overrides = sum(1 for d in decisions if d.get("overrides_recommendation"))

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, str(total), "Total Investigations"),
        (c2, f"{flagged / total * 100:.1f}%" if total else "0%", "Fraud Detection Rate"),
        (c3, f"{approved_adj / len(decisions) * 100:.1f}%" if decisions else "0%", "Approval Rate"),
        (c4, str(overrides), "Overrides"),
    ]:
        with col:
            with st.container():
                st.markdown(f"""
                <div class="kpi-card">
                    <p class="kpi-value">{val}</p>
                    <p class="kpi-label">{label}</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("### System Status")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container():
            st.markdown("**Agent Mode**")
            st.write("Rule-based (no LLM)" if mode == "rule_based" else "LLM from .env")
            if setup.error:
                st.error(setup.error)
    with s2:
        with st.container():
            st.markdown("**Data Mode**")
            st.write("Simulated data ON" if sim else "Real data only")
            if not core.simulated_db_available():
                st.caption("Simulated DB not built")
    with s3:
        with st.container():
            st.markdown("**Components**")
            vi = core.vector_index_available()
            st.markdown(f"Vector index: {'Available' if vi else 'Missing'}")
            st.markdown(f"Audit store: {'Ready' if runner and runner.audit else 'Unavailable'}")

    st.markdown("### Recent Investigations")
    if decisions:
        df = pd.DataFrame(decisions[:20])
        display_cols = [c for c in ["recorded_at", "claim_id", "recommendation",
                                     "adjuster_decision", "overrides_recommendation",
                                     "override_reason"] if c in df.columns]
        st.dataframe(df[display_cols], hide_index=True, use_container_width=True)
    else:
        st.info("No investigations recorded yet. Start a new investigation from the sidebar.")


# ---------------------------------------------------------------------------
# PAGE: New Investigation
# ---------------------------------------------------------------------------

def page_new_investigation():
    st.markdown("## New Investigation")
    st.caption("Submit a claim for AI-powered fraud analysis.")

    if runner is None:
        st.error("Agent unavailable. Go to Settings to check configuration.")
        return

    source = st.radio("Claim source", ["Pick a validation claim", "New claim (blank form)"],
                       horizontal=True)
    truth = None
    if source.startswith("Pick"):
        df = load_validation()
        bands = st.multiselect("Risk bands", ["low", "medium", "high"],
                                default=["low", "medium", "high"])
        pool = df[df.risk_band.isin(bands)] if bands else df
        labels = {core.claim_label(r): i for i, r in pool.iterrows()}
        choice = st.selectbox(f"Validation claim ({len(pool)} available)", list(labels))
        row = df.loc[labels[choice]]
        base_claim = core.claim_from_row(row)
        truth = int(row[core.TARGET])
        form_key = f"PN-{int(row.PolicyNumber)}"
    else:
        base_claim = core.blank_claim()
        form_key = "blank"

    if truth is not None:
        with st.expander("Known outcome (never shown to the agent)"):
            st.write("**Fraud found**" if truth else "**No fraud found**")

    st.markdown("### Claim Details")
    with st.form(f"claim-{form_key}"):
        claim: dict = {}
        cols = st.columns(len(core.FIELD_GROUPS))
        for col, (group, fields) in zip(cols, core.FIELD_GROUPS.items()):
            with col:
                st.markdown(f"**{group}**")
                for f in fields:
                    opts = core.field_options(f)
                    val = base_claim.get(f)
                    if opts:
                        claim[f] = st.selectbox(
                            f, opts,
                            index=opts.index(val) if val in opts else 0,
                            key=f"{form_key}-{f}",
                        )
                    elif f == "PolicyNumber":
                        claim[f] = st.number_input(
                            f, min_value=0, value=int(val) if val is not None else 0,
                            help="0 = no policy number", key=f"{form_key}-{f}",
                        ) or None
                    else:
                        claim[f] = st.number_input(
                            f, min_value=0, max_value=120, value=int(val or 0),
                            help="0 = age not recorded" if f == "Age" else None,
                            key=f"{form_key}-{f}",
                        )

        run_btn = st.form_submit_button("Run Investigation", type="primary",
                                         use_container_width=True)

    if run_btn:
        with st.spinner("Investigating... (rule-based: seconds; LLM: up to a minute)"):
            try:
                tid, decision, trace = runner.run_claim(claim)
                st.session_state.update(
                    thread=tid,
                    run_mode=(mode, sim),
                    run_form=form_key,
                    run_claim=claim,
                    run_truth=truth,
                    page="Claim Investigation",
                )
                st.rerun()
            except Exception as e:
                st.error(f"Investigation failed: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# PAGE: Claim Investigation (detail view)
# ---------------------------------------------------------------------------

def page_claim_investigation():
    thread = st.session_state.get("thread")
    if not thread or st.session_state.get("run_mode") != (mode, sim):
        st.info("No active investigation. Go to **New Investigation** to run one.")
        if st.button("Go to New Investigation"):
            st.session_state.page = "New Investigation"
            st.rerun()
        return

    state = core.run_state(runner, thread)
    d = state["decision"]
    tr = state.get("tool_results") or {}
    claim = st.session_state.get("run_claim") or {}

    claim_id = d.get("claim_id") or f"PN-{claim.get('PolicyNumber', '?')}"
    prob = d.get("fraud_probability")
    band = d.get("risk_band")

    # --- Top claim bar ---
    bar_items = [
        ("Claim ID", claim_id),
        ("Policy Holder Age", str(claim.get("AgeOfPolicyHolder", claim.get("Age", "N/A")))),
        ("Vehicle Category", str(claim.get("VehicleCategory", "N/A"))),
        ("Vehicle Price", str(claim.get("VehiclePrice", "N/A"))),
        ("Accident Area", str(claim.get("AccidentArea", "N/A"))),
        ("Base Policy", str(claim.get("BasePolicy", "N/A"))),
        ("Fault", str(claim.get("Fault", "N/A"))),
    ]
    bar_html = "".join(
        f'<div class="claim-bar-item"><span class="claim-bar-label">{lbl}</span>'
        f'<span class="claim-bar-value">{val}</span></div>'
        for lbl, val in bar_items
    )
    st.markdown(f'<div class="claim-bar">{bar_html}</div>', unsafe_allow_html=True)

    # --- Main layout: left (2/3) + right (1/3) ---
    left, right = st.columns([2, 1], gap="medium")

    with left:
        # -- Fraud Probability + Risk Level --
        r1c1, r1c2 = st.columns([1, 2])
        with r1c1:
            with st.container():
                if prob is not None:
                    st.markdown(fraud_gauge_svg(prob, 180), unsafe_allow_html=True)
                    st.markdown(
                        f'<div style="text-align:center;">{risk_badge_html(band)}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.metric("Fraud Probability", "N/A")
                decided = {
                    "llm": "LLM Agent",
                    "rule_fallback": "Rule-based",
                }.get(d["decided_by"], d["decided_by"])
                st.caption(f"Model: {d.get('model_version', 'N/A')}")
                st.caption(f"Decided by: {decided}")

        with r1c2:
            # Key Factors (SHAP)
            with st.container():
                st.markdown('<div class="section-header">Key Factors (SHAP Analysis)</div>',
                            unsafe_allow_html=True)
                shap_out = tr.get("shap_explanation")
                sf = core.shap_frame(shap_out)
                if len(sf):
                    chart = (
                        alt.Chart(sf)
                        .mark_bar(cornerRadiusEnd=4)
                        .encode(
                            x=alt.X("shap_value:Q", title=None, axis=alt.Axis(labels=True, grid=False)),
                            y=alt.Y("feature:N",
                                    sort=alt.EncodingSortField("shap_value", op="sum", order="descending"),
                                    title=None,
                                    axis=alt.Axis(labelLimit=200, labelOverlap=False)),
                            color=alt.Color(
                                "direction:N",
                                scale=alt.Scale(
                                    domain=["raises fraud score", "lowers fraud score"],
                                    range=["#ef4444", "#3b82f6"],
                                ),
                                legend=None,
                            ),
                            tooltip=[
                                alt.Tooltip("feature:N"),
                                alt.Tooltip("value:N"),
                                alt.Tooltip("shap_value:Q", format=".3f"),
                            ],
                        )
                        .properties(height=alt.Step(28))
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("SHAP analysis not available for this claim.")

        # -- Policy Verification --
        pv_out = tr.get("policy_verification")
        with st.container():
            st.markdown('<div class="section-header">Policy Verification</div>',
                        unsafe_allow_html=True)
            if pv_out and pv_out.get("ok"):
                checks = pv_out.get("checks", [])
                for ck in checks:
                    result = ck.get("result", "UNKNOWN")
                    if result == "PASS":
                        icon, cls = "&#10004;", "check-pass"
                    elif result == "FAIL":
                        icon, cls = "&#10006;", "check-fail"
                    else:
                        icon, cls = "&#63;", "check-unknown"
                    reason = ck.get("reason", "")
                    label = ck.get("check_id", "").replace("_", " ").title()
                    st.markdown(
                        f'<div class="verify-row">'
                        f'<span>{label}</span>'
                        f'<span class="{cls}">{icon} {result}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if result == "FAIL" and reason:
                        st.caption(f"  {reason}")
            else:
                st.caption("Policy verification not available.")

        # -- Row: Similar Claims + Claim History + Conclusion --
        sc1, sc2 = st.columns(2)
        with sc1:
            # Similar Claims
            with st.container():
                st.markdown('<div class="section-header">Similar Claims</div>',
                            unsafe_allow_html=True)
                sim_out = tr.get("similar_claims")
                if sim_out and sim_out.get("ok"):
                    nf = core.neighbours_frame(sim_out)
                    if len(nf):
                        st.dataframe(nf, hide_index=True, use_container_width=True)
                    st.caption(
                        f"{sim_out.get('neighbour_frauds', 0)} of {sim_out.get('k', 5)} "
                        f"similar claims were fraudulent"
                    )
                else:
                    st.caption("Similar claims data not available.")

            # Claim History
            with st.container():
                st.markdown('<div class="section-header">Claim History</div>',
                            unsafe_allow_html=True)
                ch_out = tr.get("claim_history")
                if ch_out and ch_out.get("ok"):
                    st.write(f"Status: **{ch_out.get('status')}**")
                    hf = core.history_frame(ch_out)
                    if len(hf):
                        st.dataframe(hf, hide_index=True, use_container_width=True)
                    if ch_out.get("limitation"):
                        st.caption(ch_out["limitation"])
                    if ch_out.get("note"):
                        st.caption(ch_out["note"])
                else:
                    st.info("Historical claim information is not available for this policy holder.")
                    st.caption("The dataset does not contain previous claim history for this policy holder.")

        with sc2:
            # Conclusion & Recommendation
            with st.container():
                st.markdown('<div class="section-header">Conclusion &amp; Recommendation</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div style="margin: 8px 0;">{decision_badge_html(d["decision"])}</div>',
                    unsafe_allow_html=True,
                )
                st.write(d["rationale"])

                evidence_list = d.get("evidence", [])
                if evidence_list:
                    st.markdown("**Key Reasons**")
                    for i, e in enumerate(evidence_list[:5], 1):
                        st.markdown(
                            f'<div class="reason-item">'
                            f'<span class="reason-number">{i}</span>'
                            f'<span>{e["fact"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                if d.get("guardrail_notes"):
                    st.markdown("**Guardrail Notes**")
                    for n in d["guardrail_notes"]:
                        st.warning(n)

        # -- Detailed tool output tabs --
        with st.expander("Detailed Tool Outputs", expanded=False):
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
                            st.dataframe(
                                pd.DataFrame(out["failed_rules"])[["rule_id", "severity", "message"]],
                                hide_index=True,
                            )
                    elif src == "fraud_score":
                        a, b, c = st.columns(3)
                        a.metric("Probability", out["fraud_probability"])
                        b.metric("Risk band", out["risk_band"])
                        c.metric("Predicted class", out["predicted_class"])
                        st.caption(f"Thresholds: {out.get('thresholds')} | model {out.get('model_version')}")
                        if out.get("risk_indicators"):
                            st.dataframe(
                                pd.DataFrame(out["risk_indicators"])[["rule_id", "severity", "condition", "support"]],
                                hide_index=True,
                            )
                    elif src == "shap_explanation":
                        sf2 = core.shap_frame(out)
                        st.dataframe(sf2, hide_index=True)
                    elif src == "policy_retrieval":
                        st.write(f"Status: **{out.get('status')}** | source: {out.get('source', 'claim record')}")
                        st.json(out.get("recorded_policy_attributes", {}), expanded=False)
                    elif src == "policy_verification":
                        cf = core.checks_frame(out)
                        st.dataframe(cf, hide_index=True, use_container_width=True)
                    elif src == "similar_claims":
                        st.dataframe(core.neighbours_frame(out), hide_index=True, use_container_width=True)
                    elif src == "claim_history":
                        hf2 = core.history_frame(out)
                        if len(hf2):
                            st.dataframe(hf2, hide_index=True)
                        else:
                            st.caption("No history records.")
                    with st.expander("Raw JSON"):
                        st.json(out, expanded=False)
            with tabs[-1]:
                st.json(state.get("trace", []), expanded=False)

    # -- Right column --
    with right:
        # Claim Summary
        with st.container():
            st.markdown('<div class="section-header">Claim Summary</div>',
                        unsafe_allow_html=True)
            summary_fields = [
                ("Claim ID", claim_id),
                ("Policy Holder Age", claim.get("AgeOfPolicyHolder", claim.get("Age", "N/A"))),
                ("Vehicle Category", claim.get("VehicleCategory", "N/A")),
                ("Vehicle Price", claim.get("VehiclePrice", "N/A")),
                ("Accident Area", claim.get("AccidentArea", "N/A")),
                ("Base Policy", claim.get("BasePolicy", "N/A")),
                ("Fault", claim.get("Fault", "N/A")),
                ("Deductible", claim.get("Deductible", "N/A")),
                ("Police Report", claim.get("PoliceReportFiled", "N/A")),
                ("Witness", claim.get("WitnessPresent", "N/A")),
                ("Make", claim.get("Make", "N/A")),
                ("Agent Type", claim.get("AgentType", "N/A")),
            ]
            for lbl, val in summary_fields:
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; padding:4px 0; '
                    f'border-bottom:1px solid #f1f5f9; font-size:0.88rem;">'
                    f'<span style="color:#64748b;">{lbl}</span>'
                    f'<span style="font-weight:600;">{val}</span></div>',
                    unsafe_allow_html=True,
                )

        # Investigation Checklist
        with st.container():
            st.markdown('<div class="section-header">Investigation Checklist</div>',
                        unsafe_allow_html=True)
            statuses = d.get("tool_statuses", {})
            for src, label in TOOL_LABELS.items():
                ts = statuses.get(src, {})
                status = ts.get("status", "not_run")
                st.markdown(
                    f'<div class="checklist-item">'
                    f'<span>{status_dot_html(status)} {label}</span>'
                    f'<span style="font-size:0.8rem; color:#64748b;">'
                    f'{STATUS_MAP.get(status, (status,))[0]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if d.get("simulated_data"):
            st.warning("Simulated reference data was used (illustrative, not real records).")

        st.markdown(
            '<div class="disclaimer">This analysis is based on AI-powered insights '
            'and should be reviewed by a human investigator for final decision.</div>',
            unsafe_allow_html=True,
        )

        # Known outcome
        truth = st.session_state.get("run_truth")
        if truth is not None:
            with st.expander("Known outcome (hidden from agent)"):
                st.write("**Fraud found**" if truth else "**No fraud found**")

    # -- Adjuster decision form --
    st.markdown("---")
    st.markdown("## Your Decision")
    if runner.pending_review(thread):
        with st.form(f"review-{thread}"):
            adjuster_id = st.session_state.adjuster_id
            c1, c2 = st.columns([1, 1])
            with c1:
                choice = st.radio(
                    "Decision",
                    core.ADJUSTER_DECISIONS,
                    index=core.ADJUSTER_DECISIONS.index(d["decision"]),
                    horizontal=True,
                    help="Only you can deny a claim.",
                )
            with c2:
                reason = st.text_area("Override reason (required if changed)")
                notes = st.text_area("Notes (optional)")
            submit = st.form_submit_button("Submit Decision", type="primary",
                                            use_container_width=True)
        if submit:
            if choice != d["decision"] and not reason.strip():
                st.error("Please give a reason for overriding the recommendation.")
            else:
                try:
                    runner.resume(thread, {
                        "decision": choice,
                        "adjuster_id": adjuster_id,
                        "override_reason": reason.strip() or None,
                        "notes": notes.strip() or None,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not record: {type(e).__name__}: {e}")
    else:
        adj = state.get("adjuster") or {}
        adj_decision = adj.get("decision")
        st.success(
            f"Reviewed by {adj.get('adjuster_id', 'unknown')}: "
            f"**{adj_decision}**"
            + (f" (override: {adj.get('override_reason')})" if adj.get("override_reason") else "")
        )


# ---------------------------------------------------------------------------
# PAGE: Claim History
# ---------------------------------------------------------------------------

def page_claim_history():
    st.markdown("## Claim History")
    st.caption("Browse past investigations and adjuster decisions.")

    if runner is None or runner.audit is None:
        st.info("Audit store not available. Run an investigation first.")
        return

    tab1, tab2 = st.tabs(["Adjuster Decisions", "Recommendations"])

    with tab1:
        decisions = get_audit_rows(runner, "adjuster_decisions")
        if decisions:
            df = pd.DataFrame(decisions)
            # Filters
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                dec_filter = st.multiselect(
                    "Decision",
                    options=df["adjuster_decision"].unique().tolist() if "adjuster_decision" in df.columns else [],
                    default=[],
                )
            with fc2:
                override_filter = st.selectbox("Overrides only?", ["All", "Yes", "No"])
            with fc3:
                limit = st.number_input("Show", min_value=10, max_value=500, value=50, step=10)

            filtered = df
            if dec_filter and "adjuster_decision" in filtered.columns:
                filtered = filtered[filtered["adjuster_decision"].isin(dec_filter)]
            if override_filter == "Yes" and "overrides_recommendation" in filtered.columns:
                filtered = filtered[filtered["overrides_recommendation"] == 1]
            elif override_filter == "No" and "overrides_recommendation" in filtered.columns:
                filtered = filtered[filtered["overrides_recommendation"] == 0]

            display_cols = [c for c in ["recorded_at", "claim_id", "recommendation",
                                         "adjuster_decision", "overrides_recommendation",
                                         "override_reason", "notes"] if c in filtered.columns]
            st.dataframe(filtered[display_cols].head(int(limit)),
                         hide_index=True, use_container_width=True)
        else:
            st.info("No adjuster decisions recorded yet.")

    with tab2:
        recs = get_audit_rows(runner, "recommendations")
        if recs:
            df_r = pd.DataFrame(recs)
            display_cols = [c for c in ["recorded_at", "claim_id", "decision", "flag_category",
                                         "fraud_probability", "risk_band", "decided_by",
                                         "agent_mode", "simulated_data"] if c in df_r.columns]
            st.dataframe(df_r[display_cols], hide_index=True, use_container_width=True)
        else:
            st.info("No recommendations recorded yet.")


# ---------------------------------------------------------------------------
# PAGE: Reports
# ---------------------------------------------------------------------------

def page_reports():
    st.markdown("## Reports & Analytics")
    st.caption("Model performance and investigation analytics.")

    mc = load_model_card()

    # Model card summary
    with st.container():
        st.markdown("### Model Performance")
        if mc:
            m1, m2, m3, m4 = st.columns(4)
            val_metrics = mc.get("validation_metrics", {})
            with m1:
                st.metric("ROC AUC", f"{val_metrics.get('roc_auc', 0):.4f}")
            with m2:
                st.metric("PR AUC", f"{val_metrics.get('pr_auc', 0):.4f}")
            with m3:
                st.metric("Brier Score", f"{val_metrics.get('brier', 0):.4f}")
            with m4:
                st.metric("Model", mc.get("family", "N/A").upper())

            st.markdown("#### Test Set Evaluation")
            eval_data = mc.get("evaluation", {})
            if eval_data.get("metrics"):
                metrics_df = pd.DataFrame(eval_data["metrics"])
                key_metrics = metrics_df[metrics_df["metric"].isin([
                    "roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy",
                ])].copy()
                key_metrics["value"] = key_metrics["value"].apply(lambda x: f"{x:.4f}")
                key_metrics["95% CI"] = key_metrics.apply(
                    lambda r: f"[{r['lo']:.4f}, {r['hi']:.4f}]"
                    if pd.notna(r.get("lo")) and pd.notna(r.get("hi")) else "N/A",
                    axis=1,
                )
                st.dataframe(key_metrics[["metric", "value", "95% CI"]],
                             hide_index=True, use_container_width=True)

            st.markdown("#### Risk Band Thresholds")
            thresholds = mc.get("thresholds", {})
            if thresholds:
                rows = []
                for name, vals in thresholds.items():
                    rows.append({
                        "Threshold": name.replace("_", " ").title(),
                        "Value": f"{vals.get('threshold', 0):.4f}",
                        "Precision": f"{vals.get('precision', 0):.3f}",
                        "Recall": f"{vals.get('recall', 0):.3f}",
                        "Flag Rate": f"{vals.get('flag_rate', 0):.1%}",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            with st.expander("Model Card Details"):
                st.json(mc, expanded=False)
        else:
            st.info("Model card not found at models/model_card.json.")

    # Investigation analytics
    st.markdown("### Investigation Analytics")
    recs = get_audit_rows(runner, "recommendations")
    decisions = get_audit_rows(runner, "adjuster_decisions")

    if recs:
        df_r = pd.DataFrame(recs)

        c1, c2 = st.columns(2)
        with c1:
            with st.container():
                st.markdown("**Decision Distribution (Agent)**")
                if "decision" in df_r.columns:
                    dec_counts = df_r["decision"].value_counts().reset_index()
                    dec_counts.columns = ["Decision", "Count"]
                    chart = (
                        alt.Chart(dec_counts)
                        .mark_bar(cornerRadiusEnd=6)
                        .encode(
                            x=alt.X("Count:Q", title="Count"),
                            y=alt.Y("Decision:N", sort="-x", title=None),
                            color=alt.Color(
                                "Decision:N",
                                scale=alt.Scale(
                                    domain=["APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFORMATION"],
                                    range=["#10b981", "#ef4444", "#f59e0b"],
                                ),
                                legend=None,
                            ),
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart, use_container_width=True)

        with c2:
            with st.container():
                st.markdown("**Risk Band Distribution**")
                if "risk_band" in df_r.columns:
                    band_counts = df_r["risk_band"].value_counts().reset_index()
                    band_counts.columns = ["Band", "Count"]
                    chart2 = (
                        alt.Chart(band_counts)
                        .mark_arc(innerRadius=50, cornerRadius=4)
                        .encode(
                            theta="Count:Q",
                            color=alt.Color(
                                "Band:N",
                                scale=alt.Scale(
                                    domain=["low", "medium", "high"],
                                    range=["#10b981", "#f59e0b", "#ef4444"],
                                ),
                            ),
                            tooltip=["Band", "Count"],
                        )
                        .properties(height=200)
                    )
                    st.altair_chart(chart2, use_container_width=True)

        if decisions:
            df_d = pd.DataFrame(decisions)
            st.markdown("**Override Statistics**")
            o1, o2, o3 = st.columns(3)
            total_d = len(df_d)
            overrides = df_d["overrides_recommendation"].sum() if "overrides_recommendation" in df_d.columns else 0
            with o1:
                st.metric("Total Reviews", total_d)
            with o2:
                st.metric("Overrides", int(overrides))
            with o3:
                st.metric("Override Rate", f"{overrides / total_d * 100:.1f}%" if total_d else "0%")
    else:
        st.info("No investigations recorded yet. Run some investigations to see analytics.")


# ---------------------------------------------------------------------------
# PAGE: Settings
# ---------------------------------------------------------------------------

def page_settings():
    st.markdown("## Settings")
    st.caption("Configure the agent, data source, and adjuster identity.")

    with st.container():
        st.markdown("### Agent Configuration")
        mode_choice = st.radio(
            "Agent Mode",
            ["Rule-based (no LLM)", "LLM from .env"],
            index=0 if st.session_state.agent_mode == "rule_based" else 1,
            help="Rule-based needs no model or GPU. LLM mode uses LLM_PROVIDER / MODEL_NAME "
                 "from .env (Ollama, an OpenAI-compatible API, or Anthropic).",
        )
        new_mode = "llm" if mode_choice.startswith("LLM") else "rule_based"
        if new_mode != st.session_state.agent_mode:
            st.session_state.agent_mode = new_mode
            get_runner.clear()
            st.rerun()

    with st.container():
        st.markdown("### Data Source")
        sim_avail = core.simulated_db_available()
        new_sim = st.toggle(
            "Use simulated reference data",
            value=st.session_state.simulated,
            disabled=not sim_avail,
            help="Illustrative policies, clauses and prior claims for Tools 3, 4, 6.",
        )
        if not sim_avail:
            st.caption("Simulated database not built: `python scripts/build_simulated_db.py`")
        if new_sim != st.session_state.simulated:
            st.session_state.simulated = new_sim
            get_runner.clear()
            st.rerun()

    with st.container():
        st.markdown("### Adjuster Identity")
        new_id = st.text_input("Adjuster ID", value=st.session_state.adjuster_id)
        if new_id != st.session_state.adjuster_id:
            st.session_state.adjuster_id = new_id

    with st.container():
        st.markdown("### System Information")
        si1, si2 = st.columns(2)
        with si1:
            st.markdown("**Component Status**")
            vi = core.vector_index_available()
            sd = core.simulated_db_available()
            st.markdown(f"- Vector index: {'Available' if vi else 'Missing (`python scripts/build_vector_index.py`)'}")
            st.markdown(f"- Simulated DB: {'Available' if sd else 'Missing (`python scripts/build_simulated_db.py`)'}")
            st.markdown(f"- Audit store: {'Ready' if runner and runner.audit else 'Unavailable'}")
            if setup.error:
                st.error(f"Agent error: {setup.error}")
                if st.button("Retry LLM connection"):
                    get_runner.clear()
                    st.rerun()
        with si2:
            st.markdown("**Model Card**")
            mc = load_model_card()
            if mc:
                st.write(f"Model: {mc.get('family', 'N/A')} ({mc.get('model_version', 'N/A')})")
                st.write(f"Trained: {mc.get('trained_at', 'N/A')}")
                env = mc.get("environment", {})
                st.write(f"Device: {env.get('gpu', 'CPU')}")
                st.write(f"Python: {env.get('python', 'N/A')}")
                st.write(f"Features: {len(mc.get('features', []))}")
            else:
                st.caption("Model card not found.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGE_MAP = {
    "Home": page_home,
    "New Investigation": page_new_investigation,
    "Claim Investigation": page_claim_investigation,
    "Claim History": page_claim_history,
    "Reports": page_reports,
    "Settings": page_settings,
}

page_fn = PAGE_MAP.get(st.session_state.page, page_home)
page_fn()
