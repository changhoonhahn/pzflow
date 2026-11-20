#!/usr/bin/env python3
"""Run pzflow.plot_bootstrap_workflow."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    runpy.run_module("pzflow.plot_bootstrap_workflow", run_name="__main__", alter_sys=True)
