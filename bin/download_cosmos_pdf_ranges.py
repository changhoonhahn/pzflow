#!/usr/bin/env python3
"""Run pzflow.download_cosmos_pdf_ranges."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    runpy.run_module("pzflow.download_cosmos_pdf_ranges", run_name="__main__", alter_sys=True)
