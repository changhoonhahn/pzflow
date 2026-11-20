#!/usr/bin/env python3
"""Train the COSMOS-PDF, matched-PDF, and spec-z flow controls."""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse, json
import numpy as np
import torch
from pzflow.paths import ROOT
from pzflow import flow
from pzflow.flow import SEED,raw_features,PDFSampler,context,fit,posterior_summary,metrics
def main(epochs=200,max_steps=20000,data_dir=None,output_dir=None):
    global OUT
    data_dir=data_dir or ROOT/'data/flow_magnitude_limited'
    OUT=output_dir or ROOT/'results/flow_magnitude_limited'
    OUT.mkdir(parents=True,exist_ok=True)
    flow.OUT=OUT
    torch.set_num_threads(4);torch.manual_seed(SEED)
    d=np.load(data_dir/'prepared.npz')
    ids=d['id'];flux=d['flux'];err=d['err'];extra=d['extra'];zspec=d['zspec'];splits=d['split'];cell=d['cell']
    for a in range(3):
        for b in range(a):assert not np.intersect1d(cell[splits==a],cell[splits==b]).size
    base=raw_features(flux[splits==0],err[splits==0]);mean=base.mean(axis=0);scale=np.maximum(base.std(axis=0),.1)
    sampler=PDFSampler(d['grid'],d['pdf'])
    test=np.flatnonzero((splits==2)&np.isfinite(zspec))
    assert len(test)>100
    rng=np.random.default_rng(SEED+1)
    observed=flux[test]+rng.normal(size=(len(test),6)).astype(np.float32)*extra[test]
    ctx=context(observed,err[test],mean,scale)
    outputs={};report={}
    for name,teacher,matched in [('cosmos_pdf_flow',True,False),('cosmos_pdf_matched_flow',True,True),('broadband_spec_flow',False,True)]:
        model,training=fit(name,teacher,flux,err,extra,zspec,splits,sampler,mean,scale,epochs,max_steps,matched)
        q,pit,nll=posterior_summary(model,ctx,zspec[test])
        outputs[name]=dict(quantiles=q,pit=pit,nll=nll)
        report[name]=dict(training=training,all=metrics(zspec[test],q,pit,nll),
                          z_gt2=metrics(zspec[test][zspec[test]>2],q[zspec[test]>2],pit[zspec[test]>2],nll[zspec[test]>2]))
        print(name,json.dumps(report[name]),flush=True)
    # Paired resampling of common test galaxies; no metric is selected on test.
    paired={};rng=np.random.default_rng(SEED+2)
    for teacher_name in ['cosmos_pdf_flow','cosmos_pdf_matched_flow']:
        paired[teacher_name]={}
        for key in ['sigma_nmad','outlier_fraction','mean_negative_log_density']:
            differences=[]
            for _ in range(200):
                ix=rng.integers(0,len(test),len(test));vv=[]
                for name in [teacher_name,'broadband_spec_flow']:
                    o=outputs[name];vv.append(metrics(zspec[test][ix],o['quantiles'][ix],o['pit'][ix],o['nll'][ix])[key])
                differences.append(vv[0]-vv[1])
            paired[teacher_name][key]=dict(flow_minus_baseline_95ci=np.quantile(differences,[.025,.975]).tolist())
    manifest=json.loads((data_dir/'manifest.json').read_text())
    result=dict(data=manifest,models=report,paired_bootstrap=paired,
                run_settings=dict(seed=SEED,max_epochs=epochs,max_steps=max_steps,n_training_seeds=1),
                limitations=['Forecast uses native COSMOS filters, not measured LSST bandpasses' if manifest['mode']!='actual_rubin' else 'Matched Rubin fluxes supplied by user',
                             'Spectroscopic test selection is not representative of the full photometric population',
                             'Held-out IDs are excluded from our model fitting; independence from the original COSMOS template/zero-point calibration has not been audited',
                             'COSMOS redshift distributions are assumed unbiased; their supplied widths and tails are retained without recalibration',
                             'Baseline is a spec-supervised six-band flow, not an official Rubin photo-z product or a template-fitting code'])
    (OUT/'comparison_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    arrays=dict(id=ids[test],zspec=zspec[test],imag=d['imag'][test],observed_flux=observed,fluxerr=err[test])
    for name,o in outputs.items():
        for k,v in o.items():arrays[f'{name}_{k}']=v
    np.savez_compressed(OUT/'heldout_predictions.npz',**arrays)
    np.savez_compressed(OUT/'split_ids.npz',**{f'split_{s}':ids[splits==s] for s in range(3)})
    np.savez_compressed(OUT/'matched_training_ids.npz',**{f'split_{s}':ids[(splits==s)&np.isfinite(zspec)] for s in [0,1]})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=200);parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/flow_magnitude_limited')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/flow_magnitude_limited')
    args=parser.parse_args();main(args.epochs,args.max_steps,args.data_dir,args.output_dir)
