"""Regenerate all existing diagnostics from saved data, without retraining."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def main():
    logs=ROOT/'tmp/style_logs';logs.mkdir(parents=True,exist_ok=True)
    scripts=['plot_cosmos2020_specz.py','plot_r_magnitude_redshift.py',
             'cosmos2020_coverage.py','coverage_by_redshift.py',
             'audit_magnitude_selection.py','compare_template_baselines.py',
             'plot_coverage_prior_correction.py','plot_coverage_prior_all_models.py',
             'plot_prior_ratio.py']
    for script in scripts:
        print(f'Rendering {script}',flush=True)
        with (logs/(script+'.log')).open('w') as log:
            proc=subprocess.run([sys.executable,str(ROOT/'bin'/script)],cwd=ROOT,
                                stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError((logs/(script+'.log')).read_text()[-8000:])
    from . import cosmos_lsst_flow as flow
    for folder in ['flow_magnitude_limited','flow_experiment']:
        print(f'Rendering saved {folder} predictions',flush=True)
        flow.OUT=ROOT/'results'/folder
        with np.load(flow.OUT/'heldout_predictions.npz') as f:a={k:f[k] for k in f.files}
        result=json.loads((flow.OUT/'comparison_metrics.json').read_text())
        flow.make_plots(a,result)
    print('All existing diagnostic figures regenerated; no model training.',flush=True)

if __name__=='__main__':main()
