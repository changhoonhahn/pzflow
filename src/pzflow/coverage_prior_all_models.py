"""Apply the same independently fitted COSMOS/spec n(z) ratio to all methods.

For EAZY this is a common reweighting sensitivity test, not an exact replacement
of its flat or magnitude-dependent original prior. No models are retrained.
"""
from pathlib import Path
import json
import numpy as np
import torch
from .cosmos2020_coverage import GridPDF
from .cosmos_lsst_flow import make_flow, context, t_to_z
from .coverage_prior_correction import ROOT, WIDTHS, SEED, get_summary

OUT = ROOT/'results/coverage_prior_correction'


def flow_grid(a):
    path = OUT/'cosmos_full_flow_probability_grid.npz'
    if path.exists():
        with np.load(path) as f:
            np.testing.assert_array_equal(f['id'],a['id'])
            return f['u'], f['z_quantiles']
    torch.set_num_threads(4)
    ckpt = torch.load(ROOT/'results/flow_magnitude_limited/cosmos_pdf_flow.pt',
                      map_location='cpu', weights_only=False)
    model = make_flow();model.load_state_dict(ckpt['state_dict']);model.eval()
    ctx = context(a['observed_flux'],a['fluxerr'],ckpt['feature_mean'],ckpt['feature_scale'])
    # Integrate reweighting factors in original posterior probability space.
    # This resolves arbitrarily narrow peaks without a uniform-redshift grid.
    u=np.linspace(0,1,4097)
    normal=torch.distributions.Normal(0.,1.).icdf(torch.tensor(np.clip(u,1e-7,1-1e-7),dtype=torch.float32))
    p = np.empty((len(ctx),len(u)),dtype=np.float32)
    for start in range(0,len(ctx),8):
        x = ctx[start:start+8]
        with torch.no_grad():
            q,_=model._transform.inverse(normal.repeat(len(x))[:,None],context=x.repeat_interleave(len(u),0))
            p[start:start+len(x)] = t_to_z(q).numpy().reshape(len(x),len(u))
        if start % 800 == 0:
            print(f'Flow probability grid: {start}/{len(ctx)}',flush=True)
    p[:,0]=0;p[:,-1]=10
    assert np.isfinite(p).all() and (np.diff(p,axis=1)>=-1e-6).all()
    np.savez_compressed(path,id=a['id'],u=u,z_quantiles=p)
    return u,p


def reweight_flow(u,zq,pit,ratio_grid,ratio):
    w=np.interp(zq,ratio_grid,ratio)
    du=np.diff(u)
    c=np.c_[np.zeros(len(zq)),np.cumsum(.5*(w[:,:-1]+w[:,1:])*du,axis=1)]
    j=np.clip(np.searchsorted(u,pit,side='right')-1,0,len(u)-2)
    row=np.arange(len(pit));delta=pit-u[j]
    val=c[row,j]+w[row,j]*delta+.5*(w[row,j+1]-w[row,j])*delta**2/du[j]
    return np.clip(val/c[:,-1],0,1)


def pit_summary(pit):
    return dict(n=len(pit),coverage68=float(np.mean(abs(2*pit-1)<=.68)),
                coverage95=float(np.mean(abs(2*pit-1)<=.95)),pit_mean=float(pit.mean()))


def main():
    with np.load(OUT/'diagnostics.npz') as f:d={k:f[k] for k in f.files}
    with np.load(ROOT/'results/flow_magnitude_limited/heldout_predictions.npz') as f:
        a={k:f[k] for k in f.files}
    np.testing.assert_array_equal(d['id'],a['id'])
    z=a['zspec'];old=json.loads((OUT/'summary.json').read_text())
    report=dict(method='Same n_spec/n_parent ratio as the LePhare experiment, independently fitted on spatial train/validation cells; each reweighted PDF normalized.',
                interpretation='Common prior-shift sensitivity test. EAZY original priors differ from the COSMOS parent stack; this is not an exact replacement of each estimator\'s implicit prior.',
                n_test=len(z),n_parent_calibration=old['n_parent_calibration'],
                n_spec_calibration=old['n_spec_calibration'],
                baseline_bandwidth_log1pz=.03,
                uncertainty=old['uncertainty'],models={})
    saved={'id':a['id'],'zspec':z,'imag':a['imag']}
    rng=np.random.default_rng(SEED)
    boot=rng.integers(0,len(z),size=(1000,len(z)))
    methods=['lephare_full_cosmos','eazy_six_band','eazy_rprior','cosmos_pdf_flow']
    for name in methods:
        if name=='lephare_full_cosmos':
            report['models'][name]=old['metrics']
            for key in ['original']+[f'h_{h:.3f}' for h in WIDTHS]:
                saved[f'{name}_pit_{key}']=d[f'pit_{key}']
            continue
        if name.startswith('eazy'):
            with np.load(ROOT/f'results/template_comparison/{name}_posteriors.npz') as f:
                np.testing.assert_array_equal(a['id'],f['id'])
                grid,p=f['grid'],f['pdf']
        else:
            grid,p=flow_grid(a)
        metrics={};pits={}
        if name=='cosmos_pdf_flow':
            pits['original']=a['cosmos_pdf_flow_pit'].astype(float)
            metrics['original']=pit_summary(pits['original'])
            report['flow_integration']='Trapezoidal quadrature of ratio at 4097 inverse-CDF nodes in original posterior probability; exact analytic original PIT; quadratic interpolation of integrated weights.'
        else:
            metrics['original'],pits['original']=get_summary(z,grid,p)
            with np.load(ROOT/'results/template_comparison/heldout_predictions.npz') as f:
                np.testing.assert_allclose(pits['original'],f[f'{name}_pit'],atol=1e-10)
        for h in WIDTHS:
            key=f'h_{h:.3f}'
            if name=='cosmos_pdf_flow':
                pits[key]=reweight_flow(grid,p,pits['original'],d['grid'],d[f'ratio_{key}'])
                metrics[key]=pit_summary(pits[key])
            else:
                ratio=np.interp(grid,d['grid'],d[f'ratio_{key}'])
                metrics[key],pits[key]=get_summary(z,grid,p.astype(float)*ratio[None,:])
        if name=='cosmos_pdf_flow':
            coarse=reweight_flow(grid[::2],p[:,::2],pits['original'],d['grid'],d['ratio_h_0.030'])
            err=np.max(abs(coarse-pits['h_0.030']))
            assert err<.002,err
            report['flow_corrected_grid_max_pit_change_halving_resolution']=float(err)
            changes={str(alpha):float(np.mean(abs(2*coarse-1)<=alpha)-
                         np.mean(abs(2*pits['h_0.030']-1)<=alpha)) for alpha in [.68,.95]}
            assert max(abs(v) for v in changes.values())<.001,changes
            report['flow_coverage_change_halving_resolution']=changes
        for key,pit in pits.items():
            saved[f'{name}_pit_{key}']=pit
            for alpha in [.68,.95]:
                label=f'coverage{int(alpha*100)}'
                hit=abs(2*pit-1)<=alpha
                base=abs(2*pits['original']-1)<=alpha
                metrics[key][label+'_95ci']=np.quantile(hit[boot].mean(axis=1),[.025,.975]).tolist()
                delta=(hit.astype(float)-base)[boot].mean(axis=1)
                metrics[key][label+'_change']=float(hit.mean()-base.mean())
                metrics[key][label+'_change_95ci']=np.quantile(delta,[.025,.975]).tolist()
        report['models'][name]=metrics
        print(name,json.dumps({k:{v:m[v] for v in ['coverage68','coverage95']} for k,m in metrics.items()}),flush=True)
    # The supervised baseline already learns the selected spectroscopic
    # population. Applying the parent-to-spec ratio again would double-count
    # that prior shift, so include its original predictions as the control.
    name='broadband_spec_flow'
    pit=a[f'{name}_pit'].astype(float)
    saved[f'{name}_pit_original']=pit
    m=pit_summary(pit)
    for alpha in [.68,.95]:
        hit=abs(2*pit-1)<=alpha
        m[f'coverage{int(alpha*100)}_95ci']=np.quantile(hit[boot].mean(axis=1),[.025,.975]).tolist()
    training=json.loads((ROOT/'results/flow_magnitude_limited/comparison_metrics.json').read_text())['models'][name]
    m.update({k:training['all'][k] for k in ['bias','sigma_nmad','outlier_fraction']})
    report['models'][name]={'original':m}
    report['spectroscopic_baseline']=dict(n_train=training['training']['n_train'],
        n_validation=training['training']['n_validation'],
        objective='Conditional flow p(z|six broadband fluxes and errors), trained by maximum likelihood on secure spectroscopic redshifts.',
        reweighting='None: the supervised baseline already learns the spectroscopically selected training population; multiplying by the parent-to-spec ratio would count the shift again.')
    print(name,json.dumps(m),flush=True)
    (OUT/'all_models_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'all_models_diagnostics.npz',**saved)


if __name__=='__main__':main()
