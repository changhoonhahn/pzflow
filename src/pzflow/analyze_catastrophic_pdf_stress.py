"""Collect the completed corruption sweep and author its figure notebook."""
from pathlib import Path
import csv
import json
import zipfile
import numpy as np
import nbformat
from .catastrophic_pdf_stress import ROOT, OUT, FRACTIONS, WEIGHTS, scenario_name

CODE=r'''
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt

ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents]
          if (p/'paper/data/predictions.npz').exists())
OUT=ROOT/'results/catastrophic_pdf_stress'
sys.path.insert(0,str(ROOT/'src'))
from pzflow.project_plot_style import set_style
from pzflow.paper_utils import summary
set_style()
report=json.loads((OUT/'summary.json').read_text())
with np.load(OUT/'figure_inputs.npz') as f:d={k:f[k] for k in f.files}
z=d['zspec'];imag=d['imag'];n=len(z)
assert n==5083 and len(np.unique(d['id']))==n
names=report['scenario_order']
keys=['sigma_nmad','outlier_fraction','mean_negative_log_density',
      'coverage68','coverage95','median_width95']
rng=np.random.default_rng(20261122)
boot=rng.integers(0,n,size=(300,n))
stats={}
base_hits={a:abs(2*d['baseline_pit']-1)<=a for a in [.68,.95]}
for name in names:
    q=d[f'{name}_quantiles'];pit=d[f'{name}_pit'];nll=d[f'{name}_nll']
    r=(q[:,2]-z)/(1+z)
    vals=[]
    for ix in boot:
        rr=r[ix]
        vals.append([1.4826*np.median(abs(rr-np.median(rr))),
                     np.mean(abs(rr)>.15),np.mean(nll[ix]),
                     np.mean(abs(2*pit[ix]-1)<=.68),
                     np.mean(abs(2*pit[ix]-1)<=.95),
                     np.median(q[ix,4]-q[ix,0])])
    stats[name]={'95ci':dict(zip(keys,np.quantile(vals,[.025,.975],axis=0).T.tolist()))}
    base_q=d['baseline_quantiles'];base_r=(base_q[:,2]-z)/(1+z)
    base_nll=d['baseline_nll']
    diffs={}
    diffs['outlier_fraction']=np.mean((abs(r[boot])>.15).astype(float)-
                                    (abs(base_r[boot])>.15),axis=1)
    diffs['mean_negative_log_density']=np.mean((nll-base_nll)[boot],axis=1)
    for a in [.68,.95]:
        diff=(abs(2*pit-1)<=a).astype(float)-base_hits[a]
        diffs[f'coverage{int(a*100)}']=diff[boot].mean(axis=1)
    for key,values in diffs.items():
        stats[name][f'{key}_change_95ci']=np.quantile(values,[.025,.975]).tolist()
(OUT/'bootstrap_summary.json').write_text(json.dumps(dict(
    n_bootstrap=300,seed=20261122,
    interpretation='Paired test-galaxy bootstrap, conditional on a single training seed and a single fixed corruption pattern. Excludes training variance, spatial covariance, and uncertainty in the spectroscopic selection.',
    scenarios=stats),indent=2)+'\n')

fig,axs=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
ylabels=[r'$\sigma_{\rm NMAD}$','Catastrophic outliers [percent]',
         'Mean negative log density','68 percent interval coverage [percent]',
         '95 percent interval coverage [percent]','Median 95 percent interval width']
scales=[1,100,1,100,100,1]
for ax,key,label,scale in zip(axs.flat,keys,ylabels,scales):
    baseline=report['scenarios']['baseline']['metrics']['all'][key]*scale
    ax.axhline(baseline,color='black',ls=':',lw=1,label='No added corruption')
    for j,w in enumerate([.25,.5,1.]):
        series=['baseline']+[f'f{round(f*100):02d}_w{round(w*100):03d}' for f in [.01,.03,.05,.10]]
        x=np.array([0,1,3,5,10])
        y=np.array([report['scenarios'][s]['metrics']['all'][key] for s in series])*scale
        ci=np.array([stats[s]['95ci'][key] for s in series])*scale
        ax.fill_between(x,ci[:,0],ci[:,1],color=f'C{j}',alpha=.2,lw=0)
        ax.plot(x,y,'o-',color=f'C{j}',label=f'{round(w*100)} percent wrong-mode mass')
    if key=='coverage68':ax.axhline(68,color='0.5',ls='--',lw=1)
    if key=='coverage95':ax.axhline(95,color='0.5',ls='--',lw=1)
    ax.set(xlabel='Affected training galaxies [percent]',ylabel=label,xticks=[0,1,3,5,10])
axs[0,0].legend(fontsize=8)
fig.suptitle('Impact of additional catastrophic teacher-PDF contamination',fontsize=15)
fig.supxlabel('Same 111,952 training objects, untouched validation PDFs, and 5,083 held-out spectroscopic test galaxies.\n'
              'Shading: 95 percent paired test-bootstrap intervals; one paired training seed and one nested random corruption pattern.',fontsize=9)
fig.savefig(OUT/'contamination_response.png',dpi=180)
plt.show();plt.close(fig)

edges=[-np.inf,22.5,24,24.5,25]
masks=[(imag>=lo)&(imag<hi) for lo,hi in zip(edges[:-1],edges[1:])]
x=np.array([np.median(imag[m]) for m in masks]);counts=[int(m.sum()) for m in masks]
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
series=['baseline','f01_w100','f03_w100','f05_w100','f10_w100']
labels=['No added corruption','1 percent affected','3 percent affected','5 percent affected','10 percent affected']
magstats={}
for j,(name,label) in enumerate(zip(series,labels)):
    rows=[];intervals=[]
    for b,m in enumerate(masks):
        q=d[f'{name}_quantiles'][m];pit=d[f'{name}_pit'][m];zz=z[m]
        r=(q[:,2]-zz)/(1+zz)
        rows.append([1.4826*np.median(abs(r-np.median(r))),np.mean(abs(2*pit-1)<=.68)])
        ix=np.random.default_rng(20261122+b).integers(0,len(zz),size=(300,len(zz)))
        rr=r[ix]
        vv=np.c_[1.4826*np.median(abs(rr-np.median(rr,axis=1)[:,None]),axis=1),
                 np.mean(abs(2*pit[ix]-1)<=.68,axis=1)]
        intervals.append(np.quantile(vv,[.025,.975],axis=0))
    rows=np.array(rows);intervals=np.array(intervals)
    magstats[name]=dict(values=rows.tolist(),intervals=intervals.tolist())
    for k,ax in enumerate(axs):
        scale=1 if k==0 else 100
        ax.fill_between(x,intervals[:,0,k]*scale,intervals[:,1,k]*scale,color=f'C{j}',alpha=.5,lw=0)
        ax.plot(x,rows[:,k]*scale,'o-',color=f'C{j}',label=label)
for ax in axs:
    for xx,count in zip(x,counts):
        ax.text(xx,.97,f'$N={count:,}$',transform=ax.get_xaxis_transform(),ha='center',va='top')
    ax.set(xlim=(21.3,25),xticks=np.arange(21.5,25.01,.5),xlabel=r'Native HSC $i_{\rm 2\,arcsec}$ [AB]')
top=max(np.asarray(v['intervals'])[:,1,0].max() for v in magstats.values())
axs[0].set(ylabel=r'$\sigma_{\rm NMAD}$',ylim=(0,top*1.2))
axs[0].legend(loc='upper left',bbox_to_anchor=(0,.88),fontsize=8)
axs[1].axhline(68,color='black',ls=':',lw=1)
axs[1].set(ylabel='68 percent interval coverage [percent]',ylim=(0,105))
fig.suptitle('Magnitude dependence: complete wrong-mode replacement',fontsize=15)
fig.supxlabel('Points at median magnitude in each bin. Shading: 95 percent galaxy-bootstrap intervals, alpha=0.5.\n'
              'Contamination occurs only in training; all test objects remain unchanged.',fontsize=9)
fig.savefig(OUT/'contamination_by_magnitude.png',dpi=180)
plt.show();plt.close(fig)
(OUT/'magnitude_summary.json').write_text(json.dumps(dict(effective_magnitudes=x.tolist(),counts=counts,scenarios=magstats),indent=2)+'\n')

fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
alpha=np.linspace(0,1,201)
for j,(name,label) in enumerate(zip(series,labels)):
    pit=d[f'{name}_pit']
    axs[0].hist(pit,bins=np.linspace(0,1,21),density=True,histtype='step',
                color=f'C{j}',lw=1.5,label=label)
    rank=abs(2*pit-1)
    cov=(rank[:,None]<=alpha[None,:]).mean(axis=0)
    cov[0]=0;cov[-1]=1
    axs[1].plot(alpha,cov,color=f'C{j}',label=label)
axs[0].axhline(1,color='black',ls=':',lw=1)
axs[0].set(xlim=(0,1),xlabel='PIT',ylabel='Probability density')
axs[0].legend(fontsize=8)
axs[1].plot([0,1],[0,1],color='black',ls=':',lw=1)
axs[1].set(xlim=(0,1),ylim=(0,1),xlabel='Nominal interval probability',ylabel='Empirical coverage')
fig.suptitle('PDF calibration under complete wrong-mode replacement',fontsize=15)
fig.supxlabel('A wider posterior can improve aggregate coverage while adding probability at wrong redshifts.\n'
              'Compare predictive likelihood as well as coverage; these diagnostics use the selected spectroscopic test sample.',fontsize=9)
fig.savefig(OUT/'contamination_pit_coverage.png',dpi=180)
plt.show();plt.close(fig)
'''


def main():
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
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('''# Catastrophic COSMOS PDF injection: controlled retraining

This notebook reads saved predictions from 12 retrained flows plus the original
no-injection checkpoint. It computes paired test-galaxy bootstrap intervals and
generates response curves and magnitude-dependent diagnostics. No retraining or
network access is needed to regenerate the figures.

Only a fixed random subset of training PDFs was modified. Each selected PDF is
replaced by a mixture of its original density and a distant narrow truncated
Gaussian. Validation PDFs and all test photometry/redshifts remain untouched.
Affected samples are nested across rates; the same pattern is used across mass
weights. The flow initialization and photometric-augmentation streams are paired.

The zero-injection baseline retains existing COSMOS errors. Injection is defined
relative to the teacher median, without using true redshifts, so it measures
additional teacher-label corruption rather than a known true-redshift error rate.
One training seed and one corruption realization are used. Shaded bootstrap
bands omit training variance, spatial covariance, and selection uncertainty.

Requires `requirements/notebooks.txt` plus a TeX installation (`latex`, `dvipng`).
'''),nbformat.v4.new_code_cell('%matplotlib inline\n'+CODE)])
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(nb,ROOT/'nb/figure_catastrophic_pdf_stress.ipynb')
    print('Collected 13 conditions and authored the figure notebook.',flush=True)


if __name__=='__main__':main()
