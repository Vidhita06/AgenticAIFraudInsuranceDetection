"""Phase 1: data audit and validation-rule report -> reports/01_data_audit.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import run_audit, write_report  # noqa: E402

if __name__ == "__main__":
    write_report(run_audit())
    print("Wrote reports/01_data_audit.md and reports/tables/01_*.csv")
