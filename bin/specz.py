#!/usr/bin/env python3
"""Compare public DR1.1 spectroscopy with supplied COSMOS2020 Classic LePhare photo-z.

Requirements: numpy, astropy, matplotlib. Run from any directory.
Raw catalog: https://media.githubusercontent.com/media/cosmosastro/speczcompilation/main/specz_compilation/specz_compilation_COSMOS_DR1.1_unique.fits
Catalog reference: Khostovan et al. (2025), arXiv:2503.00120.
"""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import csv
import hashlib
import json
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pzflow.plotting import set_style, COLORS
set_style()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/cosmos2020_specz'
OUT = ROOT / 'results/cosmos2020_specz'
OUT.mkdir(parents=True, exist_ok=True)
raw = DATA / 'specz_compilation_COSMOS_DR1.1_unique.fits'
d = fits.getdata(raw, 1)
mask = (np.isin(d['flag'], [3, 4, 13, 14])
        & (d['photoz_type'] == 0) & (d['Id_COS20_Classic'] > 0)
        & np.isfinite(d['specz']) & (d['specz'] > 0)
        & np.isfinite(d['photoz']) & (d['photoz'] >= 0)
        & (d['public_or_private'] == 1))
s = d[mask]
assert len(np.unique(s['Id_COS20_Classic'])) == len(s)
z = np.asarray(s['specz'], float)
p = np.asarray(s['photoz'], float)
r = (p-z)/(1+z)

def stats(m):
    a = r[m]
    return dict(n=int(len(a)), median_normalized_bias=float(np.median(a)),
                sigma_nmad= float(1.4826*np.median(np.abs(a-np.median(a)))),
                outlier_fraction=float(np.mean(np.abs(a)>0.15)))

surveys = {}
for line in (DATA/'specz_compilation_COSMOS_DR1.1_surveys.list').read_text().splitlines():
    if not line.strip() or line.startswith('#'): continue
    fields = line.split()
    surveys[int(float(fields[0]))] = dict(name=fields[1], reference=fields[2], status=fields[3])
ids, counts = np.unique(s['survey'], return_counts=True)
by_survey = sorted([dict(id=int(i), count=int(n), **surveys[int(i)])
                    for i,n in zip(ids,counts)], key=lambda a:-a['count'])
report = dict(source_url='https://media.githubusercontent.com/media/cosmosastro/speczcompilation/main/specz_compilation/specz_compilation_COSMOS_DR1.1_unique.fits',
              sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
              photoz='Supplied photoz: COSMOS2020 Classic LePhare (not Farmer or EAZY)',
              selection='unique catalog; public_or_private=1; flag in [3,4,13,14]; photoz_type=0; Id_COS20_Classic>0; finite specz>0 and photoz>=0. No cut on photoz-specz agreement, magnitude, or photometric quality.',
              nmad_definition='1.4826 median(abs(r-median(r))), r=(photoz-specz)/(1+specz)',
              all=stats(np.ones(len(z),bool)),
              redshift_bins={f'{lo:g} <= zspec < {hi:g}':stats((z>=lo)&(z<hi)) for lo,hi in [(0,0.5),(0.5,1),(1,2),(2,3),(3,10)]},
              survey_assignment_note='Counts use the preferred measurement survey; galaxies can have measurements in other surveys.',
              surveys=by_survey)
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
cols=['Id_specz','Id_COS20_Classic','specz','photoz','flag','photoz_type','survey']
with (OUT/'matched_galaxies.csv').open('w') as f:
    w=csv.writer(f); w.writerow(cols+['normalized_residual'])
    for row,ri in zip(s,r): w.writerow([row[c] for c in cols]+[ri])

set_style()
fig, axes=plt.subplots(1,3,figsize=(16,5.4),layout='constrained')
fig.suptitle('COSMOS2020 Classic / LePhare vs. reliable spectroscopy',fontsize=18,fontweight='bold')
for ax,lim,title in [(axes[0],3,'Low-redshift zoom'),(axes[1],10,'Full redshift range')]:
    h=ax.hexbin(z,p,gridsize=140 if lim==3 else 110,extent=(0,lim,0,lim),
                bins='log',mincnt=1,cmap='viridis',linewidths=0,rasterized=True)
    xx=np.linspace(0,lim,300)
    ax.plot(xx,xx,color='C1',lw=1.2,label='One-to-one')
    ax.plot(xx,xx+0.15*(1+xx),'--',color='C1',lw=0.8)
    ax.plot(xx,xx-0.15*(1+xx),'--',color='C1',lw=0.8,label=r'$|\Delta z|/(1+z_{\rm spec})=0.15$')
    ax.set(xlim=(0,lim),ylim=(0,lim),xlabel=r'Spectroscopic redshift $z_{\rm spec}$',ylabel=r'Photometric redshift $z_{\rm phot}$',title=title)
    ax.set_aspect('equal')
    if lim==3: ax.legend(loc='upper left',fontsize=8,frameon=True)
    cb=fig.colorbar(h,ax=ax,shrink=0.75,pad=0.02);cb.set_label('Galaxies per hexagon',fontsize=9)
ax=axes[2]
ax.hexbin(z,r,gridsize=110,extent=(0,8,-1,1),bins='log',mincnt=1,cmap='viridis',linewidths=0,rasterized=True)
ax.axhline(0,color='C1',lw=1)
for rr in [-0.15,0.15]:ax.axhline(rr,color='C1',ls='--',lw=0.8)
ax.set(xlim=(0,8),ylim=(-1,1),xlabel=r'Spectroscopic redshift $z_{\rm spec}$',ylabel=r'$(z_{\rm phot}-z_{\rm spec})/(1+z_{\rm spec})$',title='Normalized residuals')
st=report['all']
ax.text(0.04,0.96,f"N = {len(z):,}\nMedian bias = {st['median_normalized_bias']:+.4f}\nNMAD = {st['sigma_nmad']:.4f}\nOutliers = {100*st['outlier_fraction']:.2f}%",transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',alpha=0.93,edgecolor='0.8'))
fig.supxlabel('DR1.1 unique catalog; reliable flags 3/4/13/14; galaxy templates; valid Classic matches.\nNo cut on redshift agreement. All-sample metrics include points outside displayed residual limits. Spectroscopic selection is heterogeneous.',fontsize=10)
fig.savefig(OUT/'specz_vs_photoz.png',dpi=180)
fig.savefig(OUT/'specz_vs_photoz.pdf')
print(json.dumps(report,indent=2))
