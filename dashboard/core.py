"""Dashboard logic without Streamlit: claim sources, agent construction, and views of the
tool outputs. Everything runs on CPU; an LLM is optional (rule-based mode needs none)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import path_for  # noqa: E402
from src.ml.data import CATEGORIES, INTEGER_DOMAINS, RAW_COLUMNS, TARGET  # noqa: E402

DECISIONS = ["APPROVE", "FLAG_FOR_INVESTIGATION", "REQUEST_MORE_INFORMATION"]
ADJUSTER_DECISIONS = DECISIONS + ["DENY"]          # only a human may deny a claim
TOOL_TABS = [("validation", "Validation"), ("fraud_score", "1 · Fraud score"), ("shap_explanation", "2 · SHAP"),
             ("policy_retrieval", "3 · Policy retrieval"), ("policy_verification", "4 · Policy verification"),
             ("similar_claims", "5 · Similar claims"), ("claim_history", "6 · Claim history")]
CLAIM_FIELDS = [c for c in RAW_COLUMNS if c != TARGET]
FIELD_GROUPS = {
    "Incident": ["Month", "WeekOfMonth", "DayOfWeek", "Year", "AccidentArea", "Fault", "PoliceReportFiled",
                 "WitnessPresent"],
    "Claim": ["MonthClaimed", "WeekOfMonthClaimed", "DayOfWeekClaimed", "Days_Policy_Accident", "Days_Policy_Claim",
              "PastNumberOfClaims", "NumberOfSuppliments", "AddressChange_Claim"],
    "Policy & vehicle": ["PolicyNumber", "BasePolicy", "PolicyType", "VehicleCategory", "Make", "VehiclePrice",
                         "AgeOfVehicle", "Deductible", "AgentType", "RepNumber", "NumberOfCars"],
    "Policyholder": ["Age", "AgeOfPolicyHolder", "Sex", "MaritalStatus", "DriverRating"],
}
assert sorted(sum(FIELD_GROUPS.values(), [])) == sorted(CLAIM_FIELDS), "every claim field is in one group"


def field_options(field: str) -> Optional[list]:
    """Allowed values for a categorical / small-integer field, or None for free numbers."""
    if field in CATEGORIES:
        return list(CATEGORIES[field])
    dom = INTEGER_DOMAINS.get(field)
    return sorted(dom) if dom else None


def blank_claim() -> dict[str, Any]:
    """A complete, typical claim to start a new form from (first allowed value per field)."""
    claim = {f: (field_options(f) or [None])[0] for f in CLAIM_FIELDS}
    claim.update(Make="Toyota", AccidentArea="Urban", Sex="Male", MaritalStatus="Married", DriverRating=2,
                 Age=40, PolicyNumber=None, Fault="Third Party", Days_Policy_Accident="more than 30",
                 Days_Policy_Claim="more than 30", PastNumberOfClaims="none", VehiclePrice="20000 to 29000",
                 AgeOfPolicyHolder="36 to 40", AddressChange_Claim="no change", NumberOfCars="1 vehicle",
                 AgeOfVehicle="7 years", NumberOfSuppliments="none", Deductible=400, BasePolicy="Collision",
                 VehicleCategory="Sedan", PolicyType="Sedan - Collision")
    return claim


def validation_claims() -> pd.DataFrame:
    """Real validation claims with the model's risk band (for picking demo claims). The label
    stays in the frame for the 'known outcome' panel but is never passed to the agent."""
    from src.ml.predict import load_artifacts, predict_proba, risk_band
    df = pd.read_csv(path_for("real_validation"))
    art = load_artifacts()
    df["fraud_probability"] = predict_proba(df.drop(columns=[TARGET]), art)
    df["risk_band"] = [risk_band(p, art.bands) for p in df.fraud_probability]
    return df


def claim_label(row: pd.Series) -> str:
    return (f"PN-{int(row.PolicyNumber)} · {row.risk_band} band · {row.BasePolicy} · "
            f"{row.Fault} at fault · {row.VehicleCategory}")


def claim_from_row(row: pd.Series) -> dict[str, Any]:
    return {f: (row[f].item() if hasattr(row[f], "item") else row[f]) for f in CLAIM_FIELDS}


@dataclass
class AgentSetup:
    mode: str                         # "rule_based" | "llm"
    llm: Any = None
    synth_llm: Any = None
    description: str = ""
    error: Optional[str] = None


def build_agent(mode: str) -> AgentSetup:
    """Rule-based (no LLM, always available) or the LLM configured in .env. Errors are
    returned, not raised, so the dashboard can explain them and fall back."""
    if mode != "llm":
        return AgentSetup("rule_based", description="Rule-based mode: no LLM; all six tools run, deterministic decision.")
    from src.agent.llm import check_ollama, get_llm, llm_settings, needs_api_key
    s = llm_settings()
    desc = f"{s['provider']} · {s['model']}" + (f" · {s['base_url']}" if s.get("base_url") else "")
    try:
        if s["provider"] == "ollama":
            check_ollama(s["model"], s["base_url"] or "http://localhost:11434")
        elif needs_api_key(s) and not s["api_key"]:
            return AgentSetup("llm", description=desc, error="No API key: set LLM_API_KEY in .env (see .env.example) "
                                                              "or use LLM_PROVIDER=ollama.")
        llm = get_llm()
        synth = get_llm(json_mode=True) if s["provider"] == "ollama" else None
        return AgentSetup("llm", llm=llm, synth_llm=synth, description=desc)
    except Exception as e:  # noqa: BLE001
        return AgentSetup("llm", description=desc, error=f"{type(e).__name__}: {e}")


def make_runner(setup: AgentSetup, simulated: bool, audit_db: Any = True):
    from src.agent.nodes import NodeDeps
    from src.agent.runner import TriageRunner
    deps = NodeDeps(llm=setup.llm, synth_llm=setup.synth_llm, simulated=simulated)
    return TriageRunner(deps=deps, audit_db=audit_db)


def run_state(runner, thread_id: str) -> dict[str, Any]:
    return runner.graph.get_state({"configurable": {"thread_id": thread_id}}).values


def shap_frame(shap: Optional[dict]) -> pd.DataFrame:
    if not shap or not shap.get("ok"):
        return pd.DataFrame(columns=["feature", "value", "shap_value", "direction"])
    return pd.DataFrame([{"feature": f["feature"], "value": str(f.get("value")), "shap_value": f["shap_value"],
                          "direction": "raises fraud score" if f["shap_value"] > 0 else "lowers fraud score"}
                         for f in shap.get("top_factors", [])])


def checks_frame(pv: Optional[dict]) -> pd.DataFrame:
    if not pv or not pv.get("ok"):
        return pd.DataFrame()
    return pd.DataFrame([{"check": c["check_id"], "result": c["result"], "level": c.get("level"),
                          "category": c.get("category"), "reason": c.get("reason"), "source": c.get("source")}
                         for c in pv.get("checks", [])])


def neighbours_frame(sim: Optional[dict]) -> pd.DataFrame:
    if not sim or not sim.get("ok"):
        return pd.DataFrame()
    rows = []
    for n in sim.get("neighbours", []):
        rows.append({"claim": n.get("claim_id"), "similarity": n.get("similarity"), "outcome": n.get("outcome"),
                     "differences": ", ".join(n.get("differences", {}) or {}) or "none"})
    return pd.DataFrame(rows)


def history_frame(ch: Optional[dict]) -> pd.DataFrame:
    if not ch or not ch.get("ok"):
        return pd.DataFrame()
    return pd.DataFrame(ch.get("previous_claims") or [])


def simulated_db_available() -> bool:
    from src.simulated.db import db_path
    return db_path().exists()


def vector_index_available() -> bool:
    return (path_for("vector_store_dir") / "claims.faiss").exists()
