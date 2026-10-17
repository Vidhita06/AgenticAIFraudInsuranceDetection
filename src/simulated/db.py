"""Read access to the SIMULATED reference database and the run-time write tables.

All reads return plain dicts carrying `data_origin = "simulated reference data"`. The
run-time tables (verification_results, investigation_audit) hold real agent outputs.
"""
from __future__ import annotations

import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from src.config import PROJECT_ROOT, load_config
from src.simulated import DATA_ORIGIN


def settings() -> dict[str, Any]:
    return dict(load_config().get("simulated_data") or {})


def enabled(override: Optional[bool] = None) -> bool:
    return bool(settings().get("enabled", False)) if override is None else bool(override)


def db_path() -> Path:
    return PROJECT_ROOT / settings().get("path", "data/simulated/insurance.db")


class SimulatedDBMissing(RuntimeError):
    pass


class SimulatedDB:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else db_path()
        if not self.path.exists():
            raise SimulatedDBMissing(f"simulated reference database not found at {self.path}; "
                                     "run: python scripts/build_simulated_db.py")

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        con = self._con()
        try:
            return [dict(r) for r in con.execute(sql, args)]
        finally:
            con.close()

    def policy(self, policy_number: Any) -> Optional[dict]:
        if policy_number is None:
            return None
        rows = self._rows("SELECT * FROM policies WHERE policy_number = ?", (int(policy_number),))
        return rows[0] if rows else None

    def clauses(self, base_policy: str) -> list[dict]:
        out = self._rows("SELECT * FROM policy_rules WHERE base_policy = ? ORDER BY clause_id", (base_policy,))
        for c in out:
            c["test"] = json.loads(c.pop("test_json")) if c.get("test_json") else None
        return out

    def history(self, customer_id: str, start: Optional[str], end: Optional[str]) -> list[dict]:
        sql, args = "SELECT * FROM claim_history WHERE customer_id = ?", [customer_id]
        if start:
            sql += " AND claim_date >= ?"
            args.append(start)
        if end:
            sql += " AND claim_date < ?"
            args.append(end)
        return self._rows(sql + " ORDER BY claim_date", tuple(args))

    def write_verification(self, investigation_id: str, claim_id: str, checks: list[Mapping[str, Any]]) -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        con = self._con()
        try:
            con.executemany(
                "INSERT INTO verification_results (recorded_at, investigation_id, claim_id, check_id, level, category, "
                "result, reason, clause_reference, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(now, investigation_id, claim_id, c.get("check_id"), c.get("level"), c.get("category"),
                  c.get("result"), c.get("reason"), c.get("clause_reference"), c.get("source")) for c in checks])
            con.commit()
        finally:
            con.close()

    def write_audit(self, record: Mapping[str, Any]) -> None:
        con = self._con()
        try:
            con.execute(
                "INSERT INTO investigation_audit (recorded_at, investigation_id, claim_id, decision, flag_category, "
                "decided_by, reason_codes, tool_statuses, simulated_data, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.strftime("%Y-%m-%dT%H:%M:%S"), record.get("investigation_id"), record.get("claim_id"),
                 record.get("decision"), record.get("flag_category"), record.get("decided_by"),
                 json.dumps(record.get("reason_codes", [])), json.dumps(record.get("tool_statuses", {}), default=str),
                 int(bool(record.get("simulated_data"))), record.get("model_version")))
            con.commit()
        finally:
            con.close()


@lru_cache(maxsize=4)
def get_db(path: str | None = None) -> SimulatedDB:
    return SimulatedDB(path)


__all__ = ["DATA_ORIGIN", "SimulatedDB", "SimulatedDBMissing", "get_db", "enabled", "settings", "db_path"]
