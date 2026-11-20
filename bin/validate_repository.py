#!/usr/bin/env python3
"""Run the source-layout and compact-result checks."""
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__=='__main__':
    runpy.run_module('pzflow.validate_repository',run_name='__main__',alter_sys=True)
