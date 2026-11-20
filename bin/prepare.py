#!/usr/bin/env python3
"""Prepare the spatially split, magnitude-limited COSMOS experiment."""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from pzflow.data import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual-rubin-csv',type=Path)
    p.add_argument('--i-limit',type=float,default=25.);p.add_argument('--i-snr-min',type=float,default=5.)
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/flow_magnitude_limited')
    a=p.parse_args();prepare(a.actual_rubin_csv,a.i_limit,a.i_snr_min,a.output_dir)
