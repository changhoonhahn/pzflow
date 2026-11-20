#!/usr/bin/env python3
"""Run pzflow.flow_experiment_data."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    runpy.run_module("pzflow.flow_experiment_data", run_name="__main__", alter_sys=True)
