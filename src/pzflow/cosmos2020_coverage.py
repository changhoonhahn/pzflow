"""Coverage, PIT, and 1D TARP diagnostics for the existing secure-spec-z sample.

TARP: Lemos et al. 2023, https://proceedings.mlr.press/v202/lemos23a.html.
We evaluate posterior mass in distance-to-random-point balls by exact integration
of piecewise-linear grid PDFs, rather than Monte Carlo posterior draws.
No teacher recalibration, outlier clipping, or additional quality cuts are applied.
Dependencies: numpy, matplotlib, astropy.
"""
from pathlib import Path
import csv
import gzip
import json
import hashlib
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plot_style import set_style, COLORS

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/cosmos2020_specz'
OUT=ROOT/'results/cosmos2020_coverage'
SEED=20261023
ALPHA=np.unique(np.r_[np.linspace(0,1,101),.68,.95])

def extract():
    sample=np.genfromtxt(ROOT/'results/cosmos2020_specz/matched_galaxies.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    order=np.argsort(sample['Id_COS20_Classic'])
    sample=sample[order]
    ids=sample['Id_COS20_Classic'].astype(int)
    assert len(np.unique(ids))==len(ids)
    cache=DATA/'classic_matched_pdfs.npz'
    if cache.exists():
        with np.load(cache) as c:
            assert np.array_equal(ids,c['ids'])
            return sample,c['grid'],c['pdf']
    raw=DATA/'classic_lephare_complete.fits.gz'
    if not raw.exists():raw=DATA/'COSMOS2020_CLASSIC_R1_v2.1_LEPHARE_PZ_p3.fits.gz'
    with gzip.open(raw,'rb') as f:
        h=fits.Header.fromfile(f)
        assert h['NAXIS1']==1002 and h['BITPIX']==-32
        nrows=h['NAXIS2']
        grid=np.frombuffer(f.read(4*1002),dtype='>f4')[1:].astype(float)
        assert len(grid)==1001 and np.all(np.diff(grid)>0)
        assert np.allclose(grid,np.linspace(0,10,1001),atol=1e-6)
        pdf=np.full((len(ids),len(grid)),np.nan,dtype=np.float32)
        seen=np.zeros(len(ids),int)
        for start in range(1,nrows,16384):
            n=min(16384,nrows-start)
            buf=f.read(n*1002*4)
            assert len(buf)==n*1002*4
            a=np.frombuffer(buf,dtype='>f4').reshape(n,1002)
            rid=a[:,0].astype(int)
            ii=np.searchsorted(ids,rid)
            sel=(ii<len(ids))
            sel[sel]&=ids[ii[sel]]==rid[sel]
            pdf[ii[sel]]=a[sel,1:]
            seen[ii[sel]]+=1
        f.read() # finish stream, verifying gzip CRC
    assert np.all(seen==1),np.unique(seen,return_counts=True)
    np.savez_compressed(cache,ids=ids,grid=grid,pdf=pdf)
    print(f'Extracted PDFs for {len(ids):,} uniquely matched galaxies',flush=True)
    return sample,grid,pdf

class GridPDF:
    def __init__(self,grid,pdf):
        self.grid=np.asarray(grid,float)
        self.p=np.asarray(pdf,float)
        masses=np.diff(self.grid)[None,:]*(self.p[:,:-1]+self.p[:,1:])/2
        self.c=np.c_[np.zeros(len(pdf)),np.cumsum(masses,axis=1)]
        self.norm=self.c[:,-1].copy()
        assert np.all(np.isfinite(self.p)) and np.all(self.p>=0)
        assert np.all(self.norm>0)
        self.p/=self.norm[:,None];self.c/=self.norm[:,None]
    def cdf(self,x):
        x=np.asarray(x,float)
        assert x.shape==(len(self.p),)
        xclip=np.clip(x,self.grid[0],self.grid[-1])
        j=np.clip(np.searchsorted(self.grid,xclip,side='right')-1,0,len(self.grid)-2)
        row=np.arange(len(self.p));t=xclip-self.grid[j]
        a=self.p[row,j]; b=self.p[row,j+1]
        result=self.c[row,j]+a*t+.5*(b-a)/np.diff(self.grid)[j]*t*t
        return np.clip(result,0,1)
    def quantile(self,u):
        u=np.broadcast_to(u,(len(self.p),))
        lo=np.zeros(len(self.p));hi=np.full(len(self.p),self.grid[-1])
        for _ in range(40):
            mid=(lo+hi)/2
            lower=self.cdf(mid)<u
            lo=np.where(lower,mid,lo);hi=np.where(lower,hi,mid)
        return (lo+hi)/2

def tarp_masses(g,z,center,refs,seed):
    rng=np.random.default_rng(seed)
    out=[]
    for _ in range(refs):
        global_ref=rng.uniform(0,10,len(z))
        local_ref=rng.uniform(np.maximum(0,center-.25*(1+center)),np.minimum(10,center+.25*(1+center)))
        both=[]
        for ref in [global_ref,local_ref]:
            radius=np.abs(z-ref)
            both.append(np.clip(g.cdf(ref+radius)-g.cdf(ref-radius),0,1))
        out.append(both)
    return np.asarray(out).transpose(1,2,0) # scheme, galaxy, random-reference

def contributions(mass):
    if mass.ndim==1:mass=mass[:,None]
    a=np.empty((len(mass),len(ALPHA)),np.float32)
    for k,alpha in enumerate(ALPHA):a[:,k]=(mass<=alpha).mean(axis=1)
    # Set endpoints for display; finite-precision exact zero/one tails are audited separately.
    a[:,0]=0;a[:,-1]=1
    return a

def bootstrap(curves,mask,seed):
    x=curves[mask]
    rng=np.random.default_rng(seed)
    # Resample galaxies, retaining all reference draws for each selected galaxy.
    draws=rng.multinomial(len(x),np.full(len(x),1/len(x)),size=200).astype(np.float32)
    boots=draws@x/len(x)
    return x.mean(axis=0),np.quantile(boots,[.025,.975],axis=0)

def controls():
    rng=np.random.default_rng(SEED)
    n=5000;grid=np.linspace(0,10,1001)
    mu=rng.uniform(.5,3,n)
    sigma=rng.uniform(.04,.3,n)
    pdf=np.exp(-.5*((grid[None,:]-mu[:,None])/sigma[:,None])**2)
    g=GridPDF(grid,pdf)
    u=rng.random(n)
    z=g.quantile(u)
    pit=g.cdf(z)
    assert np.max(np.abs(pit-u))<1e-8,'CDF/quantile inversion failed'
    m=tarp_masses(g,z,mu,8,SEED+1)
    curves=np.array([contributions(x).mean(axis=0) for x in m])
    assert np.max(abs(curves-ALPHA))<.035,'Calibrated synthetic control failed'
    narrow=GridPDF(grid,pdf**4)
    nc=contributions(np.abs(2*narrow.cdf(z)-1)).mean(axis=0)
    assert nc[np.argmin(abs(ALPHA-.68))]<.5,'Overconfidence control failed'
    return dict(calibrated_tarp_max_deviation=float(np.max(abs(curves-ALPHA))),
                narrowed_equal_tailed_68=float(nc[np.argmin(abs(ALPHA-.68))]))

def main():
    set_style()
    OUT.mkdir(exist_ok=True,parents=True)
    print('Validating statistic on synthetic calibrated and overconfident controls',flush=True)
    control=controls()
    sample,grid,p=extract()
    n_input=len(sample)
    valid=np.isfinite(p).all(axis=1)&(p>=0).all(axis=1)&(np.sum(p,axis=1)>0)
    rejected=sample['Id_COS20_Classic'][~valid]
    sample=sample[valid];p=p[valid]
    g=GridPDF(grid,p)
    z=sample['specz'].astype(float)
    assert np.all((z>=grid[0])&(z<=grid[-1]))
    center=g.quantile(.5)
    pit=g.cdf(z)
    masses=tarp_masses(g,z,center,64,SEED+2)
    curves=[contributions(masses[0]),contributions(masses[1]),contributions(abs(2*pit-1))]
    meta=np.genfromtxt(DATA/'classic_magnitude_intervals.csv',delimiter=',',names=True,dtype=None,encoding='utf8',filling_values=np.nan)
    oi=np.argsort(meta['id']);meta=meta[oi]
    ix=np.searchsorted(meta['id'],sample['Id_COS20_Classic'])
    assert np.all(ix<len(meta)) and np.array_equal(meta['id'][ix],sample['Id_COS20_Classic'])
    meta=meta[ix]
    mag=meta['hsc_i_mag_aper2'].astype(float) # stated aperture magnitude; no total correction
    goodmag=np.isfinite(mag)&(mag>0)&(mag<40)
    meddiff=np.abs(center-meta['lp_zpdf'])
    groups=[('All galaxies',np.ones(len(z),bool)),
            ('i < 22.5',goodmag&(mag<22.5)),
            ('22.5 ≤ i < 24',goodmag&(mag>=22.5)&(mag<24)),
            ('24 ≤ i < 25',goodmag&(mag>=24)&(mag<25)),
            ('i ≥ 25',goodmag&(mag>=25))]
    curve_results={};rows=[]
    for j,(name,mask) in enumerate(groups):
        if mask.sum()<20:continue
        curve_results[name]=[]
        for k,c in enumerate(curves):
            mean,band=bootstrap(c,mask,SEED+100+j*3+k)
            curve_results[name].append((mean,band))
        row=dict(group=name,n=int(mask.sum()),pit_near_zero_fraction=float(np.mean(pit[mask]<.001)),pit_near_one_fraction=float(np.mean(pit[mask]>.999)))
        for alpha in [.5,.68,.95]:
            key=f'{alpha:.2f}'
            for k,method in enumerate(['tarp_global','tarp_local','equal_tailed']):
                ai=np.argmin(abs(ALPHA-alpha));mean,band=curve_results[name][k]
                row[f'{method}_{key}']=float(mean[ai])
                row[f'{method}_{key}_95ci']=[float(v) for v in band[:,ai]]
        rows.append(row)
    # Published interval summaries, evaluated separately from the gridded PDFs.
    lo=meta['lp_zpdf_l68'];hi=meta['lp_zpdf_u68'];zp=meta['lp_zpdf']
    interval_ok=np.isfinite(lo)&np.isfinite(hi)&(lo>=0)&(hi>=lo)
    sigma=np.maximum(zp-lo,hi-zp)
    for row,(name,m) in zip(rows,groups):
        use=m&interval_ok
        row['catalog_interval_n']=int(use.sum())
        row['catalog_68_interval_coverage']=float(np.mean((z[use]>=lo[use])&(z[use]<=hi[use])))
        row['weaver_symmetric_max_width_coverage']=float(np.mean(abs(z[use]-zp[use])<sigma[use]))

    report=dict(n_input=n_input,n_valid_pdf=len(z),n_rejected_pdf=int((~valid).sum()),rejected_ids=rejected.tolist(),
                pdf_source_url='https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/PZ/COSMOS2020_CLASSIC_R1_v2.1_LEPHARE_PZ_p3.fits.gz',
                matched_pdf_cache_sha256=hashlib.sha256((DATA/'classic_matched_pdfs.npz').read_bytes()).hexdigest(),
                seed=SEED,n_random_references_per_galaxy=64,n_bootstrap=200,
                magnitude='HSC i magnitude in 2 arcsec aperture, no total_off2 correction',
                pdf='COSMOS2020 Classic v2.1 LePhare grid; normalized independently, piecewise-linear density; no recalibration',
                tarp='Mass P(|Z-r|<=|zspec-r|); empirical CDF of masses vs nominal credibility',
                references_global='Uniform(0,10)',
                references_local='Uniform(max(0,PDF_median-0.25*(1+PDF_median)), min(10,PDF_median+0.25*(1+PDF_median)))',
                bands='95% pointwise galaxy-bootstrap intervals; do not include spatial dependence, selection or spec-z systematics',
                pdf_median_minus_catalog_median_abs_percentiles=dict(zip(['50','90','99','100'],np.quantile(meddiff,[.5,.9,.99,1]).tolist())),
                n_missing_magnitude=int((~goodmag).sum()),synthetic_controls=control,groups=rows,
                caveats=['Validation conditional on heterogeneous successful spectroscopy; not a population-wide posterior correctness test.',
                         'z grid resolution is 0.01; sub-grid PDF shape unavailable and reconstruction may affect narrow PDFs.',
                         'Spec-z is treated as truth; remaining wrong redshifts or crossmatches contribute to failures.',
                         'Extreme PIT values include probabilities rounded to zero in released PDFs.'])
    (OUT/'coverage_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'coverage_arrays.npz',id=sample['Id_COS20_Classic'],zspec=z,photoz=sample['photoz'],mag_i=mag,pit=pit,tarp_global=masses[0],tarp_local=masses[1],alpha=ALPHA)
    with (OUT/'per_galaxy_coverage.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['id_classic','zspec','photoz','i_aper2','pit','equal_tailed_rank','tarp_global_mean_rank','tarp_local_mean_rank'])
        w.writerows(zip(sample['Id_COS20_Classic'],z,sample['photoz'],mag,pit,abs(2*pit-1),masses[0].mean(axis=1),masses[1].mean(axis=1)))
    set_style()
    colors=['C0','C1','C2','C3','C4']
    fig,axs=plt.subplots(1,3,figsize=(15,5.7),layout='constrained')
    for k,title in enumerate(['TARP: global reference points','TARP: local reference points','Equal-tailed credible intervals']):
        ax=axs[k];ax.plot([0,1],[0,1],'k--',lw=1,label='Calibrated')
        for j,(name,mask) in enumerate(groups):
            if name not in curve_results:continue
            mean,band=curve_results[name][k]
            ax.plot(ALPHA,mean,color=colors[j],lw=2 if j==0 else 1.5,label=f'{name} (N={mask.sum():,})')
            ax.fill_between(ALPHA,*band,color=colors[j],alpha=.10)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Nominal credibility',ylabel='Empirical coverage',title=title)
        ax.set_aspect('equal');ax.grid(alpha=.15)
        if k==0:ax.legend(fontsize=8,loc='upper left')
    fig.suptitle(f'COSMOS2020 Classic / LePhare PDF calibration · {len(z):,} secure spectroscopic matches',fontsize=16,fontweight='bold')
    fig.supxlabel('Below the diagonal: undercoverage. Shading: 95% pointwise galaxy-bootstrap intervals.\n64 random reference points per galaxy; PDF integrals evaluated directly. Magnitudes: HSC i in 2″ apertures. Selected spectroscopy; no recalibration.',fontsize=10)
    fig.savefig(OUT/'tarp_coverage.png',dpi=180);fig.savefig(OUT/'tarp_coverage.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for j,(name,mask) in enumerate(groups):
        if name not in curve_results:continue
        hist,edges=np.histogram(pit[mask],bins=20,range=(0,1),density=True)
        axs[0].stairs(hist,edges,color=colors[j],lw=1.6,label=name)
        mean,band=curve_results[name][1]
        axs[1].plot(ALPHA,mean-ALPHA,color=colors[j],lw=1.6,label=name)
        axs[1].fill_between(ALPHA,band[0]-ALPHA,band[1]-ALPHA,color=colors[j],alpha=.1)
    axs[0].axhline(1,color='k',ls='--',lw=1)
    axs[0].set(xlabel=r'PIT = $F_i(z_{\rm spec})$',ylabel='Probability density',title='PIT: uniform density = 1 if calibrated',xlim=(0,1))
    axs[0].legend(fontsize=9)
    axs[1].axhline(0,color='k',ls='--',lw=1)
    axs[1].set(xlabel='Nominal credibility',ylabel='Empirical minus nominal coverage',title='Local TARP coverage departure',xlim=(0,1))
    fig.suptitle('Posterior calibration diagnostics',fontsize=16,fontweight='bold')
    fig.supxlabel('Pointwise sampling bands omit spectroscopic selection and correlated systematics. PIT boundary excess can reflect bias, narrow PDFs, or catastrophic failures.',fontsize=9)
    fig.savefig(OUT/'pit_and_tarp_residuals.png',dpi=180);fig.savefig(OUT/'pit_and_tarp_residuals.pdf');plt.close(fig)
    for row in rows:print(json.dumps(row),flush=True)
    print('PDF median checks:',report['pdf_median_minus_catalog_median_abs_percentiles'],flush=True)
    print('Outputs:',OUT,flush=True)

if __name__=='__main__':main()
