#!/usr/bin/env python3
"""Apply an experiment checkpoint to six-band fluxes/errors (microJy).

CSV schema is documented in FLOW_EXPERIMENT.txt. Only checkpoints generated
locally by this experiment are loaded. This does not convert filter systems.
"""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import csv
import json
import numpy as np
import torch
from pzflow.flow import ROOT,make_flow,context,t_to_z

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--model',choices=['cosmos_pdf_flow','cosmos_pdf_matched_flow','broadband_spec_flow'],default='cosmos_pdf_flow')
    p.add_argument('--model-dir',type=Path,default=ROOT/'results/flow_magnitude_limited')
    args=p.parse_args();torch.set_num_threads(4)
    folder=args.model_dir
    manifest=json.loads((folder/'comparison_metrics.json').read_text())['data']
    # This known local checkpoint contains numpy preprocessing arrays as well as tensors.
    ckpt=torch.load(folder/f'{args.model}.pt',map_location='cpu',weights_only=False)
    model=make_flow();model.load_state_dict(ckpt['state_dict']);model.eval()
    d=np.genfromtxt(args.input,delimiter=',',names=True,dtype=None,encoding='utf8',filling_values=np.nan)
    d=np.atleast_1d(d)
    flux=np.stack([d[f'flux_{b}'] for b in 'ugrizy'],axis=1).astype(np.float32)
    err=np.stack([d[f'fluxerr_{b}'] for b in 'ugrizy'],axis=1).astype(np.float32)
    if not (np.isfinite(flux).all() and np.isfinite(err).all() and (err>0).all()):
        raise ValueError('All fluxes must be finite and all errors positive; negative fluxes are allowed')
    x=context(flux,err,ckpt['feature_mean'],ckpt['feature_scale'])
    noise=torch.distributions.Normal(0.,1.).icdf(torch.tensor([.025,.16,.5,.84,.975]))
    q=[]
    with torch.no_grad():
        for i in range(0,len(x),512):
            xx=x[i:i+512];n=len(xx)
            t,_=model._transform.inverse(noise.repeat(n)[:,None],context=xx.repeat_interleave(5,0))
            q.append(t_to_z(t).reshape(n,5).numpy())
    q=np.concatenate(q)
    with args.output.open('w') as f:
        w=csv.writer(f);w.writerow(['id','z025','z160','z500','z840','z975'])
        w.writerows([[id,*values] for id,values in zip(d['id'],q)])
    print(f'Predicted {len(q):,} objects; checkpoint photometry mode: {manifest["mode"]}')
    if manifest['filter_warning']:print(manifest['filter_warning'])

if __name__=='__main__':main()
