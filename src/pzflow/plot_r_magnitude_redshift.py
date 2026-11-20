"""HSC r-band aperture magnitude vs spec-z for the existing secure overlap sample."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plot_style import set_style, COLORS
set_style()

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/cosmos2020_specz'
OUT=ROOT/'results/cosmos2020_specz'
s=np.genfromtxt(OUT/'matched_galaxies.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
p=np.genfromtxt(DATA/'classic_r_photometry.csv',delimiter=',',names=True,dtype=None,encoding='utf8',filling_values=np.nan)
p=p[np.argsort(p['id'])]
j=np.searchsorted(p['id'],s['Id_COS20_Classic'])
assert np.all(j<len(p)) and np.array_equal(p['id'][j],s['Id_COS20_Classic'])
p=p[j]
mag=p['hsc_r_mag_aper2'].astype(float)
flux=p['hsc_r_flux_aper2'].astype(float)
err=p['hsc_r_fluxerr_aper2'].astype(float)
snr=np.divide(flux,err,out=np.full(len(flux),np.nan),where=err>0)
good=np.isfinite(mag)&(mag>0)&(mag<40)&(flux>0)
z=s['specz'].astype(float)
stats=[]
edges=[0,.25,.5,.75,1,1.25,1.5,2,2.5,3,4,5,6,8]
for lo,hi in zip(edges[:-1],edges[1:]):
    use=good&(z>=lo)&(z<hi)
    if use.sum()>=20:
        q=np.quantile(mag[use],[.16,.5,.84])
        stats.append(dict(z_lo=lo,z_hi=hi,z_median=float(np.median(z[use])),n=int(use.sum()),r_p16=float(q[0]),r_median=float(q[1]),r_p84=float(q[2])))
report=dict(n_overlap=len(s),n_plotted=int(good.sum()),n_invalid_magnitude=int((~good).sum()),n_snr_lt3=int(np.sum(good&(snr<3))),
            magnitude='COSMOS2020 Classic HSC_r_MAG_APER2, AB, 2 arcsec diameter; no total correction or S/N cut',redshift='Preferred reliable spectroscopic redshift from COSMOS DR1.1 compilation',bins=stats)
set_style()
fig,ax=plt.subplots(figsize=(10,6.4),layout='constrained')
h=ax.hexbin(z[good],mag[good],gridsize=(145,100),bins='log',mincnt=1,cmap='viridis',linewidths=0,rasterized=True)
cb=fig.colorbar(h,ax=ax,pad=.02);cb.set_label('Galaxies per hexagon')
x=np.array([t['z_median'] for t in stats]);q=np.array([[t['r_p16'],t['r_median'],t['r_p84']] for t in stats])
ax.fill_between(x,q[:,0],q[:,2],color='C0',alpha=.13)
ax.plot(x,q[:,1],color='C0',lw=2.2,marker='o',ms=4,label='Median r; shading: 16–84%')
ax.set(xlabel=r'Spectroscopic redshift $z_{\rm spec}$',ylabel=r'HSC $r$ magnitude (AB, 2″ aperture)',xlim=(0,8),title=f'COSMOS2020 spectroscopic overlap · {good.sum():,} galaxies with valid r magnitudes')
ax.invert_yaxis();ax.grid(alpha=.15)
ax.legend(loc='upper right',fontsize=10)
fig.supxlabel(f'Classic photometry; same {len(s):,}-galaxy reliable spec-z sample. {np.sum(~good):,} unavailable/invalid magnitudes omitted.\nNo r-band S/N cut; positive-flux low-S/N measurements retained. This is a selected spectroscopic sample.',fontsize=10)
fig.savefig(OUT/'r_magnitude_vs_specz.png',dpi=180)
fig.savefig(OUT/'r_magnitude_vs_specz.pdf')
(OUT/'r_magnitude_vs_specz_summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'r_magnitude_vs_specz.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['id_classic','zspec','r_mag_aper2','r_snr','valid_r_magnitude'])
    w.writerows(zip(s['Id_COS20_Classic'],z,mag,snr,good))
print(json.dumps(report,indent=2))
