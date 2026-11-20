"""Additional held-out diagnostics; this does not fit or modify either model."""
import csv
import json
import argparse
from pathlib import Path
import numpy as np
from .cosmos_lsst_flow import metrics

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'results/flow_magnitude_limited')
OUT=p.parse_args().output_dir
a=np.load(OUT/'heldout_predictions.npz')
report=json.loads((OUT/'comparison_metrics.json').read_text())
z=a['zspec'];f=a['observed_flux'][:,3]
imag=a['imag'] if 'imag' in a else np.where(f>0,23.9-2.5*np.log10(np.maximum(f,1e-30)),np.nan)
groups=[('all',np.ones(len(z),bool))]
groups += [(f'{lo}<=zspec<{hi}',(z>=lo)&(z<hi)) for lo,hi in [(0,.5),(.5,1),(1,2),(2,3),(3,10)]]
groups += [(f'{lo}<=i<{hi}',(imag>=lo)&(imag<hi)) for lo,hi in [(0,22.5),(22.5,24),(24,25),(25,40)]]
rows=[]
for label,mask in groups:
    if mask.sum()<20:continue
    for name in report['models']:
        row=dict(group=label,model=name,**metrics(z[mask],a[f'{name}_quantiles'][mask],a[f'{name}_pit'][mask],a[f'{name}_nll'][mask]))
        rows.append(row)
with (OUT/'binned_metrics.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# CSV with individual predictions makes independent checks straightforward.
with (OUT/'heldout_predictions.csv').open('w') as f:
    w=csv.writer(f)
    columns=['id_classic','zspec']
    for name in report['models']:
        columns += [f'{name}_{q}' for q in ['z025','z160','z500','z840','z975','pit']]
    w.writerow(columns)
    for i in range(len(z)):
        row=[a['id'][i],z[i]]
        for name in report['models']:row+=a[f'{name}_quantiles'][i].tolist()+[float(a[f'{name}_pit'][i])]
        w.writerow(row)
print('Wrote binned metrics and held-out prediction CSV',flush=True)
