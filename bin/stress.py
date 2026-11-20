#!/usr/bin/env python3
"""Run the fixed random teacher-PDF contamination sweep; resume completed cases."""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse,contextlib,hashlib,json
import numpy as np
import torch
from pzflow import flow
from pzflow.stress import ROOT,OUT,FRACTIONS,WEIGHTS,make_pattern,sampler_checks,CorruptedSampler,evaluate,scenario_name,collect_results
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fractions',nargs='+',type=float,default=FRACTIONS)
    parser.add_argument('--weights',nargs='+',type=float,default=WEIGHTS)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-steps',type=int,default=20000)
    parser.add_argument('--epochs',type=int,default=200)
    args=parser.parse_args()
    assert all(0<f<=.1 for f in args.fractions)
    assert all(0<w<=1 for w in args.weights)
    assert args.max_steps==20000 and args.epochs==200,'Use the matched original training budget.'
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    source=ROOT/'data/flow_magnitude_limited/prepared.npz'
    with np.load(source) as f:d={k:f[k] for k in f.files}
    with np.load(ROOT/'results/flow_magnitude_limited/heldout_predictions.npz') as f:a={k:f[k] for k in f.files}
    ckpt_path=ROOT/'results/flow_magnitude_limited/cosmos_pdf_flow.pt'
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    mean,scale=ckpt['feature_mean'],ckpt['feature_scale']
    test=(d['split']==2)&np.isfinite(d['zspec'])
    np.testing.assert_array_equal(a['id'],d['id'][test])
    assert not np.intersect1d(d['cell'][d['split']==0],d['cell'][test]).size
    ctx=flow.context(a['observed_flux'],a['fluxerr'],mean,scale)
    base=flow.PDFSampler(d['grid'],d['pdf'])
    del d['pdf']
    pattern=make_pattern(d['id'],d['split'],base)
    checks=sampler_checks(base,pattern,d['split'])
    np.savez_compressed(OUT/'corruption_pattern.npz',**pattern)
    config=dict(experiment='Random catastrophic teacher-PDF mixture injection',
        fractions=args.fractions,wrong_mode_weights=args.weights,
        training_seed=flow.SEED,pattern_seed=PATTERN_SEED,injection_seed=INJECTION_SEED,
        n_training_seeds=1,n_contamination_realizations=1,
        max_steps=args.max_steps,max_epochs=args.epochs,
        n_train=int(np.sum(d['split']==0)),n_validation=int(np.sum(d['split']==1)),n_test=int(test.sum()),
        affected_selection='Fixed nested random subset of training galaxies; never reselected between epochs.',
        wrong_mode='Truncated Normal(mu, sigma=0.01*(1+mu)) on [0,10]; mu is a randomly drawn training-galaxy teacher median, rejected unless |mu-source_median|/(1+source_median)>0.30.',
        labels='p_tilde=(1-lambda)*original_PDF+lambda*wrong_mode for selected training galaxies only; original labels retained for all other galaxies.',
        validation='Original clean validation PDFs and original early-stopping rules; validation and test never corrupted.',
        pairing='Same model initialization, shuffled training batches and photometric augmentation RNG as the original fit; independent shared injection RNG across scenarios.',
        reference='Reused original no-injection checkpoint; predictions recomputed and checked against saved test results.',
        prepared_data_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        baseline_checkpoint_sha256=hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),checks=checks,
        limitations=['Original COSMOS PDFs already contain errors: this is additional teacher-relative corruption, not known true-redshift contamination.',
                    'One paired training seed and one nested random corruption realization; test bootstrap intervals exclude training and contamination-realization variance.',
                    'Selected secure spectroscopy is not representative of the magnitude-limited parent.',
                    'No n(z) correction, teacher recalibration or removal of outliers; native CFHT/HSC filters with the existing LSST-like noise proxy.'])
    (OUT/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    baseline=flow.make_flow();baseline.load_state_dict(ckpt['state_dict']);baseline.eval()
    baseline_metrics,pred=evaluate(baseline,ctx,a['zspec'],a['imag'])
    np.testing.assert_allclose(pred['quantiles'],a['cosmos_pdf_flow_quantiles'],atol=1e-6)
    np.testing.assert_allclose(pred['pit'],a['cosmos_pdf_flow_pit'],atol=1e-6)
    np.savez_compressed(OUT/'baseline_predictions.npz',id=a['id'],zspec=a['zspec'],imag=a['imag'],**pred)
    original=json.loads((ROOT/'results/flow_magnitude_limited/comparison_metrics.json').read_text())
    (OUT/'baseline.json').write_text(json.dumps(dict(fraction=0.,wrong_mode_weight=0.,n_affected=0,
        metrics=baseline_metrics,training=original['models']['cosmos_pdf_flow']['training'],reused_checkpoint=True),indent=2)+'\n')
    flow.OUT=OUT
    for fraction in args.fractions:
        for weight in args.weights:
            name=scenario_name(fraction,weight)
            record=OUT/f'{name}.json'
            if record.exists() and (OUT/f'{name}_predictions.npz').exists():
                print(f'Skipping completed {name}',flush=True)
                continue
            print(f'Starting {name}: fraction={fraction}, wrong-mode mass={weight}',flush=True)
            sampler=CorruptedSampler(base,len(d['id']),pattern,fraction,weight,d['split'])
            with (OUT/f'{name}_training.log').open('w') as log,contextlib.redirect_stdout(log):
                model,training=flow.fit(name,True,d['flux'],d['err'],d['extra'],d['zspec'],
                    d['split'],sampler,mean,scale,args.epochs,args.max_steps)
            metrics,pred=evaluate(model,ctx,a['zspec'],a['imag'])
            np.savez_compressed(OUT/f'{name}_predictions.npz',id=a['id'],zspec=a['zspec'],imag=a['imag'],**pred)
            result=dict(fraction=fraction,wrong_mode_weight=weight,n_affected=int(sampler.affected.sum()),
                        effective_wrong_probability_mass=float(sampler.affected.mean()*weight),
                        effective_training_wrong_probability_mass=float(sampler.affected[d['split']==0].mean()*weight),
                        metrics=metrics,training=training)
            record.write_text(json.dumps(result,indent=2)+'\n')
            print(f'Finished {name}: '+json.dumps(metrics['all']),flush=True)
    expected=[scenario_name(f,w) for f in args.fractions for w in args.weights]
    assert all((OUT/f'{n}.json').exists() for n in expected)
    (OUT/'completion.json').write_text(json.dumps(dict(scenarios=['baseline']+expected,
        n_trained=len(expected),n_reused_baselines=1,completed=True),indent=2)+'\n')
    collect_results()
    print('Completed all requested corruption scenarios.',flush=True)


if __name__=='__main__':main()
