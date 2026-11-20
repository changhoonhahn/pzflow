"""Fixed random catastrophic teacher-PDF mixtures and their diagnostics."""
import csv, json
import numpy as np
from scipy.special import ndtr, ndtri
from . import flow
from .paths import ROOT
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



def collect_results():
    completion=json.loads((OUT/'completion.json').read_text())
    names=completion['scenarios']
    assert completion['completed'] and len(names)==13
    data={};results={}
    for name in names:
        results[name]=json.loads((OUT/f'{name}.json').read_text())
        with np.load(OUT/f'{name}_predictions.npz') as f:
            if not data:
                data.update({k:f[k] for k in ['id','zspec','imag']})
            np.testing.assert_array_equal(data['id'],f['id'])
            for key in ['quantiles','pit','nll']:data[f'{name}_{key}']=f[key]
            results[name]['metrics']['all']['median_width95']=float(np.median(f['quantiles'][:,4]-f['quantiles'][:,0]))
    np.savez_compressed(OUT/'figure_inputs.npz',**data)
    report=dict(manifest=json.loads((OUT/'manifest.json').read_text()),
                scenario_order=names,scenarios=results)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'metrics.csv').open('w') as f:
        rows=[dict(scenario=name,fraction=r['fraction'],wrong_mode_weight=r['wrong_mode_weight'],
                   n_affected=r['n_affected'],**r['metrics']['all']) for name,r in results.items()]
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
