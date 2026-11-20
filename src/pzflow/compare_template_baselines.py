"""Same-object comparisons against released multiband LePhare and optional six-band EAZY."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plot_style import set_style, COLORS
set_style()
from .cosmos2020_coverage import GridPDF
from .cosmos_lsst_flow import metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/template_comparison'
FLOW=ROOT/'results/flow_magnitude_limited'

def summarize_grid(grid,pdf,z):
    g=GridPDF(grid,pdf)
    q=np.stack([g.quantile(u) for u in [.025,.16,.5,.84,.975]],axis=1)
    return q,g.cdf(z)

def score(z,q,pit,point=None):
    qq=q.copy()
    if point is not None:qq[:,2]=point
    m=metrics(z,qq,pit,np.zeros(len(z)))
    del m['mean_negative_log_density']
    return m

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=np.load(FLOW/'heldout_predictions.npz');ids=a['id'];z=a['zspec'];imag=a['imag']
    t=np.load(ROOT/'data/cosmos2020_specz/classic_matched_pdfs.npz')
    j=np.searchsorted(t['ids'],ids);np.testing.assert_array_equal(t['ids'][j],ids)
    q,pit=summarize_grid(t['grid'],t['pdf'][j],z)
    outputs={'cosmos_pdf_flow':dict(q=a['cosmos_pdf_flow_quantiles'],pit=a['cosmos_pdf_flow_pit']),
             'broadband_spec_flow':dict(q=a['broadband_spec_flow_quantiles'],pit=a['broadband_spec_flow_pit']),
             'lephare_full_cosmos':dict(q=q,pit=pit)}
    labels={'cosmos_pdf_flow':'COSMOS-PDF-trained flow (six bands)',
            'broadband_spec_flow':'Spec-trained flow (six bands)',
            'lephare_full_cosmos':'Released LePhare (full COSMOS photometry)',
            'eazy_six_band':'EAZY (six bands, flat prior)',
            'eazy_rprior':'EAZY (six bands, supplied R prior)'}
    eazy_path=OUT/'eazy_six_band_posteriors.npz'
    eazy_point=None
    if eazy_path.exists():
        e=np.load(eazy_path);np.testing.assert_array_equal(e['id'],ids)
        eq,epit=summarize_grid(e['grid'],e['pdf'],z)
        outputs['eazy_six_band']=dict(q=eq,pit=epit)
        eazy_point=score(z,eq,epit,e['zml'])
    prior_path=OUT/'eazy_rprior_posteriors.npz'
    eazy_prior_point=None
    if prior_path.exists():
        e=np.load(prior_path);np.testing.assert_array_equal(e['id'],ids)
        eq,epit=summarize_grid(e['grid'],e['pdf'],z)
        outputs['eazy_rprior']=dict(q=eq,pit=epit)
        eazy_prior_point=score(z,eq,epit,e['zml'])
    summary={name:score(z,o['q'],o['pit']) for name,o in outputs.items()}
    meta=np.genfromtxt(ROOT/'data/cosmos2020_specz/classic_magnitude_intervals.csv',delimiter=',',names=True)
    meta=meta[np.argsort(meta['id'])];ix=np.searchsorted(meta['id'],ids)
    np.testing.assert_array_equal(meta['id'][ix],ids)
    published=score(z,q,pit,meta['lp_zpdf'][ix])
    # Paired galaxy bootstrap, holding models and test photometry fixed.
    rng=np.random.default_rng(20261116);paired={}
    for name in outputs:
        if name=='cosmos_pdf_flow':continue
        paired[name]={}
        for key in ['sigma_nmad','outlier_fraction','coverage68']:
            values=[]
            for _ in range(300):
                k=rng.integers(0,len(z),len(z))
                f=outputs['cosmos_pdf_flow'];o=outputs[name]
                values.append(score(z[k],f['q'][k],f['pit'][k])[key]-score(z[k],o['q'][k],o['pit'][k])[key])
            paired[name][key]=dict(flow_minus_baseline_95ci=np.quantile(values,[.025,.975]).tolist())
    rows=[]
    groups=[('all',np.ones(len(z),bool))]
    groups += [(f'{lo}<=i<{hi}',(imag>=lo)&(imag<hi)) for lo,hi in [(0,22.5),(22.5,24),(24,25)]]
    groups += [(f'{lo}<=zspec<{hi}',(z>=lo)&(z<hi)) for lo,hi in [(0,.5),(.5,1),(1,2),(2,10)]]
    for group,m in groups:
        for name,o in outputs.items():rows.append(dict(group=group,model=name,**score(z[m],o['q'][m],o['pit'][m])))
    with (OUT/'binned_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(n_test=len(ids),selection='Same 5083 secure-spec test IDs from the i<25 magnitude-limited experiment',
                point_estimate='Posterior median for every model in models; published LePhare catalog point estimate also reported separately',
                models=summary,lephare_published_catalog_point=published,eazy_mode_point=eazy_point,eazy_rprior_mode_point=eazy_prior_point,paired_bootstrap=paired,
                caveats=['Released LePhare sees full deep multiband COSMOS photometry, not just the six noisy inputs. It is the training teacher, not an independent equal-input competitor.',
                         'The teacher is assumed unbiased during training but evaluated as released against spectroscopy.',
                         'LePhare PDFs use dz=0.01; piecewise-linear reconstruction limits sub-grid uncertainty accuracy.',
                         'Test spectroscopy is selected and much brighter than the parent photometric population.',
                         'Independence from the released COSMOS template/zero-point calibration has not been audited.',
                         'Flow results use one seed; bootstrap excludes training variance and spatial covariance.',
                         'Native CFHT u/HSC grizy with approximate LSST-like noise, not measured Rubin photometry.'])
    if eazy_path.exists():
        report['eazy_configuration']=json.loads((OUT/'eazy_manifest.json').read_text())
        report['caveats'].append('Six-band EAZY is an untuned default-template configuration with a flat redshift prior, not a calibrated survey pipeline or a claim about the best achievable template-fitting performance.')
    if prior_path.exists():
        report['eazy_rprior_configuration']=json.loads((OUT/'eazy_rprior_manifest.json').read_text())
        report['caveats'].append('R-prior sensitivity uses the supplied standard R magnitude prior with HSC r and catalog total-flux correction; it has not been calibrated for this population.')
    (OUT/'comparison_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    save=dict(id=ids,zspec=z,imag=imag,lephare_catalog_point=meta['lp_zpdf'][ix])
    for name,o in outputs.items():save[f'{name}_quantiles']=o['q'];save[f'{name}_pit']=o['pit']
    np.savez_compressed(OUT/'heldout_predictions.npz',**save)
    colors={'cosmos_pdf_flow':'C0','broadband_spec_flow':'C1','lephare_full_cosmos':'C3','eazy_six_band':'C4','eazy_rprior':'C2'}
    names=['cosmos_pdf_flow','lephare_full_cosmos']+(['eazy_rprior'] if 'eazy_rprior' in outputs else (['eazy_six_band'] if 'eazy_six_band' in outputs else []))
    fig,axs=plt.subplots(1,len(names)+1,figsize=(5*(len(names)+1),5.2),layout='constrained')
    for ax,name in zip(axs,names):
        o=outputs[name];m=summary[name]
        ax.hexbin(z,o['q'][:,2],gridsize=100,bins='log',mincnt=1,cmap='viridis',linewidths=0,rasterized=True)
        ax.plot([0,5],[0,5],color='C0',lw=1)
        ax.set(xlim=(0,5),ylim=(0,5),xlabel='Spectroscopic redshift',ylabel='Posterior median',title=labels[name],aspect='equal')
        ax.text(.04,.96,f"N={len(z):,}\nNMAD={m['sigma_nmad']:.4f}\nOutliers={100*m['outlier_fraction']:.2f}%",va='top',transform=ax.transAxes,bbox=dict(facecolor='white',alpha=.9,edgecolor='.8'))
    alpha=np.linspace(0,1,101)
    for name,o in outputs.items():
        rank=abs(2*o['pit']-1)
        axs[-1].plot(alpha,[(rank<=v).mean() for v in alpha],label=labels[name],color=colors[name])
    axs[-1].plot([0,1],[0,1],'k--',lw=1);axs[-1].legend(fontsize=7)
    axs[-1].set(xlabel='Nominal credibility',ylabel='Equal-tailed coverage',xlim=(0,1),ylim=(0,1),title='Same held-out galaxies',aspect='equal')
    fig.suptitle('Magnitude-limited COSMOS experiment: flow versus template fitting',fontsize=16)
    fig.supxlabel('LePhare uses full deep COSMOS photometry; flows and six-band EAZY use native CFHT u/HSC grizy with LSST-like noise.\nOnly secure-spec overlap is scored; all point estimates here are posterior medians. Scatter panels zoom to z<5; metrics use all objects.',fontsize=10)
    fig.savefig(OUT/'flow_vs_templates.png',dpi=180);fig.savefig(OUT/'flow_vs_templates.pdf');plt.close(fig)
    print(json.dumps(dict(models=summary,lephare_catalog_point=published,eazy_mode_point=eazy_point,eazy_rprior_mode_point=eazy_prior_point),indent=2))

if __name__=='__main__':main()
