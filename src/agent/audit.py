"""Audit store (SQLite): every recommendation and every adjuster decision.

Independent of the simulated reference database, so it works with simulated data on or off.
Path: `paths.audit_db` in config/settings.yaml (default data/audit/audit.db, git-ignored).

    recommendations     one row per agent recommendation (the paused graph's output)
    adjuster_decisions  one row per human review: the adjuster's decision, whether it
                        overrides the recommendation, and the reason
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from src.config import PROJECT_ROOT, load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT, investigation_id TEXT, thread_id TEXT, claim_id TEXT,
  decision TEXT, flag_category TEXT, decided_by TEXT, fraud_probability REAL, risk_band TEXT,
  reason_codes TEXT, guardrail_notes TEXT, simulated_data INTEGER, model_version TEXT, agent_mode TEXT);
CREATE TABLE IF NOT EXISTS adjuster_decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, recorded_at TEXT, investigation_id TEXT, thread_id TEXT, claim_id TEXT,
  recommendation TEXT, recommendation_flag_category TEXT, adjuster_decision TEXT, adjuster_id TEXT,
  overrides_recommendation INTEGER, override_reason TEXT, notes TEXT);
"""


def default_audit_path() -> Path:
    return PROJECT_ROOT / load_config()["paths"].get("audit_db", "data/audit/audit.db")


class AuditStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_audit_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = self._con()
        try:
            con.executescript(SCHEMA)
        finally:
            con.close()

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def _insert(self, table: str, row: Mapping[str, Any]) -> None:
        row = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
        con = self._con()
        try:
            con.execute(f"INSERT INTO {table} ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                        tuple(row.values()))
            con.commit()
        finally:
            con.close()

    def record_recommendation(self, thread_id: str, decision: Mapping[str, Any], agent_mode: str) -> None:
        self._insert("recommendations", {
            "investigation_id": decision.get("investigation_id"), "thread_id": thread_id,
            "claim_id": decision.get("claim_id"), "decision": decision.get("decision"),
            "flag_category": decision.get("flag_category"), "decided_by": decision.get("decided_by"),
            "fraud_probability": decision.get("fraud_probability"), "risk_band": decision.get("risk_band"),
            "reason_codes": json.dumps(decision.get("reason_codes", [])),
            "guardrail_notes": json.dumps(decision.get("guardrail_notes", [])),
            "simulated_data": int(bool(decision.get("simulated_data"))), "model_version": decision.get("model_version"),
            "agent_mode": agent_mode})

    def record_adjuster_decision(self, thread_id: str, recommendation: Mapping[str, Any],
                                 adjuster: Mapping[str, Any]) -> None:
        self._insert("adjuster_decisions", {
            "investigation_id": recommendation.get("investigation_id"), "thread_id": thread_id,
            "claim_id": recommendation.get("claim_id"), "recommendation": recommendation.get("decision"),
            "recommendation_flag_category": recommendation.get("flag_category"),
            "adjuster_decision": adjuster.get("decision"), "adjuster_id": adjuster.get("adjuster_id"),
            "overrides_recommendation": int(adjuster.get("decision") != recommendation.get("decision")),
            "override_reason": adjuster.get("override_reason"), "notes": adjuster.get("notes")})

    def rows(self, table: str, limit: int = 50, investigation_id: Optional[str] = None) -> list[dict]:
        sql, args = f"SELECT * FROM {table}", ()
        if investigation_id:
            sql, args = sql + " WHERE investigation_id = ?", (investigation_id,)
        con = self._con()
        try:
            return [dict(r) for r in con.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, int(limit)))]
        finally:
            con.close()
