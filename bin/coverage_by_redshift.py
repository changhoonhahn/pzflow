#!/usr/bin/env python3
"""Run pzflow.coverage_by_redshift."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    runpy.run_module("pzflow.coverage_by_redshift", run_name="__main__", alter_sys=True)
