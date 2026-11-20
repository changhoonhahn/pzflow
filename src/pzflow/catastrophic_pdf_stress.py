"""Random catastrophic teacher-PDF injection, with fixed galaxies and paired RNG.

Only training PDFs are changed: p_tilde=(1-lambda)*p+lambda*g for a fixed
random fraction of training galaxies. g is a truncated Gaussian at a distant
training-population redshift. No spectroscopic labels define the corruption.
Original validation PDFs and the saved test flux realization remain untouched.
Runs are resumable and use the original 20,000-step / 200-epoch budget.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import json
import shutil
import time
import numpy as np
import torch
from scipy.special import ndtr, ndtri
from . import cosmos_lsst_flow as flow

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/catastrophic_pdf_stress'
FRACTIONS=[.01,.03,.05,.10]
WEIGHTS=[.25,.50,1.]
PATTERN_SEED=20261120
INJECTION_SEED=20261121


def scenario_name(fraction,weight):
    return f'f{round(fraction*100):02d}_w{round(weight*100):03d}'


def medians(sampler,idx):
    result=np.empty(len(idx))
    for start in range(0,len(idx),4096):
        ii=idx[start:start+4096]
        c=sampler.c[ii]
        j=np.maximum(np.argmax(c>=.5,axis=1),1)
        row=np.arange(len(ii));lo=c[row,j-1];hi=c[row,j]
        result[start:start+len(ii)]=sampler.grid[j-1]+(.5-lo)/(hi-lo)*(sampler.grid[j]-sampler.grid[j-1])
    return result


def make_pattern(ids,split,sampler):
    rng=np.random.default_rng(PATTERN_SEED)
    train=np.flatnonzero(split==0)
    center=medians(sampler,train)
    # Ranking makes the affected samples nested across contamination fractions.
    ranked=rng.permutation(train)
    eligible=ranked[:round(.1*len(train))]
    source=center[np.searchsorted(train,eligible)]
    donor=rng.integers(0,len(train),len(eligible))
    for _ in range(1000):
        bad=abs(center[donor]-source)/(1+source)<=.30
        if not bad.any():break
        donor[bad]=rng.integers(0,len(train),bad.sum())
    assert (abs(center[donor]-source)/(1+source)>.30).all()
    mu=center[donor]
    sigma=.01*(1+mu)
    # Audit the probability that g is catastrophically separated from the
    # original teacher median, not from unknown true redshifts.
    lo=ndtr(-mu/sigma);hi=ndtr((10-mu)/sigma)
    near=(ndtr((source+.15*(1+source)-mu)/sigma)-
          ndtr((source-.15*(1+source)-mu)/sigma))/(hi-lo)
    assert near.max()<1e-4,near.max()
    return dict(ranked_train_index=ranked,affected_max_index=eligible,
                affected_max_id=ids[eligible],source_median=source,
                donor_id=ids[train[donor]],wrong_mean=mu,wrong_sigma=sigma,
                wrong_probability_outside_teacher_catastrophic_window=1-near)


class CorruptedSampler:
    def __init__(self,base,n,pattern,fraction,weight,split):
        self.base=base
        self.weight=weight
        self.rng=np.random.default_rng(INJECTION_SEED)
        number=round(fraction*np.sum(split==0))
        self.affected=np.zeros(n,bool)
        self.affected[pattern['ranked_train_index'][:number]]=True
        assert not self.affected[split!=0].any()
        self.mu=np.zeros(n)
        self.sigma=np.ones(n)
        self.mu[pattern['affected_max_index']]=pattern['wrong_mean']
        self.sigma[pattern['affected_max_index']]=pattern['wrong_sigma']

    def draw(self,idx,rng):
        # Always use the original RNG identically for teacher draws. The
        # injection RNG is independent, preserving paired shuffles/flux noise.
        z=self.base.draw(idx,rng)
        gate=self.rng.random(len(idx))<self.weight
        u=self.rng.random(len(idx))
        use=self.affected[idx]&gate
        if use.any():
            m=self.mu[idx[use]];s=self.sigma[idx[use]]
            a=ndtr(-m/s);b=ndtr((10-m)/s)
            z[use]=(m+s*ndtri(a+u[use]*(b-a))).astype(np.float32)
        return z


def sampler_checks(base,pattern,split):
    n=len(split);idx=np.r_[pattern['affected_max_index'][:100],np.flatnonzero(split==1)[:100]]
    for fraction,weight in [(0.,1.),(.1,0.)]:
        sampler=CorruptedSampler(base,n,pattern,fraction,weight,split)
        one=np.random.default_rng(7);two=np.random.default_rng(7)
        assert np.array_equal(sampler.draw(idx,one),base.draw(idx,two))
        assert np.array_equal(one.random(20),two.random(20))
    sampler=CorruptedSampler(base,n,pattern,.1,1.,split)
    j=pattern['affected_max_index'][0]
    idx=np.full(100000,j)
    z=sampler.draw(idx,np.random.default_rng(8))
    u=(ndtr((z-sampler.mu[j])/sampler.sigma[j])-ndtr(-sampler.mu[j]/sampler.sigma[j]))/(ndtr((10-sampler.mu[j])/sampler.sigma[j])-ndtr(-sampler.mu[j]/sampler.sigma[j]))
    assert abs(np.mean(u)-.5)<.005
    assert abs(np.mean(u<=.68)-.68)<.005
    # A partial mixture must match its analytic CDF, not a draw-by-draw shift.
    sampler=CorruptedSampler(base,n,pattern,.1,.25,split)
    z=sampler.draw(idx,np.random.default_rng(8))
    threshold=float(pattern['source_median'][0])
    original_cdf=np.interp(threshold,base.grid,base.c[j])
    wrong_cdf=(ndtr((threshold-sampler.mu[j])/sampler.sigma[j])-ndtr(-sampler.mu[j]/sampler.sigma[j]))/(ndtr((10-sampler.mu[j])/sampler.sigma[j])-ndtr(-sampler.mu[j]/sampler.sigma[j]))
    assert abs(np.mean(z<=threshold)-(.75*original_cdf+.25*wrong_cdf))<.006
    return dict(identity_and_main_rng_preservation=True,truncated_wrong_mode_sampling=True,
                partial_mixture_matches_analytic_cdf=True,validation_and_test_unaffected=True)


def evaluate(model,ctx,z,imag):
    q,pit,nll=flow.posterior_summary(model,ctx,z)
    result={'all':flow.metrics(z,q,pit,nll)}
    for name,mask in [('i_lt_22.5',imag<22.5),('i_22.5_24',(imag>=22.5)&(imag<24)),
                      ('i_24_24.5',(imag>=24)&(imag<24.5)),('i_24.5_25',imag>=24.5),
                      ('z_gt_2',z>2)]:
        result[name]=flow.metrics(z[mask],q[mask],pit[mask],nll[mask])
    return result,dict(quantiles=q,pit=pit,nll=nll)


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
    print('Completed all requested corruption scenarios.',flush=True)


if __name__=='__main__':main()
