"""Diagnostic redshift trends from the existing PDF-coverage arrays."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plot_style import set_style, COLORS
set_style()

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cosmos2020_coverage'
d=np.load(OUT/'coverage_arrays.npz')
rank=abs(2*d['pit']-1)
edges=[0,.5,1,1.5,2,3,10]
report={}
fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for ax,key,title in zip(axs,['zspec','photoz'],['Binned by spectroscopic redshift','Binned by photometric point estimate']):
    rows=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(d[key]>=lo)&(d[key]<hi)
        n=int(m.sum())
        mag=d['mag_i'][m]; mag=mag[np.isfinite(mag)&(mag>0)&(mag<40)]
        row=dict(lo=lo,hi=hi,n=n,median_i=float(np.median(mag)),coverage68=float(np.mean(rank[m]<=.68)),coverage95=float(np.mean(rank[m]<=.95)))
        # Wilson 95% intervals: descriptive sampling uncertainty, not systematics.
        for alpha,k in [(.68,'coverage68'),(.95,'coverage95')]:
            p=row[k];den=1+1.96**2/n
            mid=(p+1.96**2/(2*n))/den
            half=1.96*np.sqrt(p*(1-p)/n+1.96**2/(4*n*n))/den
            row[k+'_95ci']=[mid-half,mid+half]
        rows.append(row)
    report[key]=rows
    xx=np.arange(len(rows))
    for k,color,label,nominal in [('coverage68','C0','68% intervals',.68),('coverage95','C1','95% intervals',.95)]:
        yy=np.array([r[k] for r in rows]);ci=np.array([r[k+'_95ci'] for r in rows]).T
        ax.errorbar(xx,yy,yerr=np.vstack([yy-ci[0],ci[1]-yy]),color=color,marker='o',capsize=3,label=label)
        ax.axhline(nominal,color=color,ls='--',alpha=.6,lw=1)
    ax.set_xticks(xx,[f'{r["lo"]:g}–{r["hi"]:g}\nN={r["n"]:,}' for r in rows],fontsize=9)
    ax.set(title=title,ylabel='Equal-tailed interval coverage',xlabel='Redshift bin',ylim=(0,1))
    ax.legend(fontsize=9);ax.grid(alpha=.15)
fig.suptitle('COSMOS2020 Classic / LePhare: redshift dependence of coverage',fontsize=15,fontweight='bold')
fig.supxlabel('Dashed lines: nominal coverage. Error bars: 95% Wilson sampling intervals.\nTrue-redshift binning is diagnostic: even calibrated posteriors need not have nominal coverage within each true-redshift bin. Spectroscopic selection remains.',fontsize=9)
fig.savefig(OUT/'coverage_by_redshift.png',dpi=180)
fig.savefig(OUT/'coverage_by_redshift.pdf')
(OUT/'coverage_by_redshift.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
