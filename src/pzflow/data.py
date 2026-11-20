"""COSMOS selection and matched photometry/PDF preparation."""
from pathlib import Path
import gzip, json
import numpy as np
from astropy.io import fits
from .paths import ROOT
DATA=ROOT/'data/cosmos2020_specz'
BANDS='ugrizy'
DEPTHS=np.array([25.6,26.9,26.9,26.4,25.6,24.8])

def sample_pdf(grid,pdf,rng):
    """Draw one continuous sample per row from a piecewise-linear PDF."""
    p=np.asarray(pdf,float)
    mass=(p[:,:-1]+p[:,1:])*np.diff(grid)[None,:]/2
    c=np.c_[np.zeros(len(p)),np.cumsum(mass,axis=1)]
    u=rng.random(len(p))*c[:,-1]
    lo=np.zeros(len(p),int);hi=np.full(len(p),len(grid)-1,int)
    row=np.arange(len(p))
    for _ in range(int(np.ceil(np.log2(len(grid))))+1):
        mid=(lo+hi)//2
        right=c[row,mid]<=u
        lo=np.where(right,mid,lo);hi=np.where(right,hi,mid)
    j=np.minimum(lo,len(grid)-2)
    a=p[row,j];s=(p[row,j+1]-a)/np.diff(grid)[j]
    w=u-c[row,j]
    den=a+np.sqrt(np.maximum(a*a+2*s*w,0))
    t=np.divide(2*w,den,out=np.zeros_like(w),where=den>0)
    return grid[j]+np.clip(t,0,np.diff(grid)[j])


def extract_pdfs(ids,cache=None):
    path=cache or DATA/'flow_training_pdfs.npz'
    if path.exists():
        with np.load(path) as c:
            if np.array_equal(c['id'],ids):return c['grid'],c['pdf']
    pdf=np.full((len(ids),1001),np.nan,np.float32);seen=np.zeros(len(ids),int)
    with gzip.open(DATA/'classic_lephare_complete.fits.gz','rb') as f:
        h=fits.Header.fromfile(f)
        grid=np.frombuffer(f.read(4008),dtype='>f4')[1:].astype(float)
        for start in range(1,h['NAXIS2'],16384):
            n=min(16384,h['NAXIS2']-start)
            a=np.frombuffer(f.read(n*4008),dtype='>f4').reshape(n,1002)
            rid=a[:,0].astype(int);j=np.searchsorted(ids,rid)
            m=j<len(ids);m[m]&=ids[j[m]]==rid[m]
            pdf[j[m]]=a[m,1:];seen[j[m]]+=1
        f.read()
    assert np.all(seen==1),'PDF IDs missing or duplicated'
    np.savez_compressed(path,id=ids,grid=grid,pdf=pdf)
    return grid,pdf


def magnitude_parent(raw,flux,err,i_limit=25.,i_snr_min=5.):
    """Selection independent of spectroscopy; i limit uses native 2-inch aperture flux."""
    phot=np.isfinite(flux).all(axis=1)&np.isfinite(err).all(axis=1)&(err>0).all(axis=1)&(np.abs(flux)<1e8).all(axis=1)
    native_errors=np.stack([raw[f'{b}_fluxerr_aper2'] for b in ['cfht_u','hsc_g','hsc_r','hsc_i','hsc_z','hsc_y']],axis=1)
    phot&=np.isfinite(native_errors).all(axis=1)&(native_errors>0).all(axis=1)
    iflux=raw['hsc_i_flux_aper2'];ierr=raw['hsc_i_fluxerr_aper2']
    magnitude=np.isfinite(iflux)&(iflux>10**((23.9-i_limit)/2.5))
    snr=np.divide(iflux,ierr,out=np.zeros(len(raw)),where=np.isfinite(ierr)&(ierr>0))
    stages=[('catalog',np.ones(len(raw),bool)),('clean_footprint',raw['flag_combined']==0),
            ('galaxy_template',raw['lp_type']==0),('i_magnitude_limit',magnitude),
            ('i_snr',np.isfinite(snr)&(snr>=i_snr_min)),('six_band_measurements',phot)]
    keep=np.ones(len(raw),bool);counts={}
    for name,cut in stages:
        keep&=cut;counts[name]=int(keep.sum())
    return keep,counts


def prepare(actual_csv=None,i_limit=25.,i_snr_min=5.,output_dir=None):
    rng=np.random.default_rng(20261024)
    raw=np.genfromtxt(DATA/'classic_ugrizy.csv',delimiter=',',names=True,dtype=None,encoding='utf8',filling_values=np.nan)
    raw=raw[np.argsort(raw['id'])]
    if actual_csv:
        p=np.genfromtxt(actual_csv,delimiter=',',names=True,dtype=None,encoding='utf8',filling_values=np.nan)
        assert len(np.unique(p['id']))==len(p),'Duplicate Classic IDs in Rubin input'
        p=p[np.argsort(p['id'])]
        common,ir,ip=np.intersect1d(raw['id'],p['id'],return_indices=True)
        raw=raw[ir];p=p[ip]
        flux=np.stack([p[f'flux_{b}'] for b in BANDS],axis=1)
        err=np.stack([p[f'fluxerr_{b}'] for b in BANDS],axis=1)
        extra=np.zeros_like(err)
        mode='actual_rubin'
    else:
        prefixes=['cfht_u']+[f'hsc_{b}' for b in BANDS[1:]]
        flux=np.stack([raw[f'{b}_flux_aper2'] for b in prefixes],axis=1)
        deep_err=np.stack([raw[f'{b}_fluxerr_aper2'] for b in prefixes],axis=1)
        # Background-limited point-source depth used as an approximate aperture
        # noise scale. Never claim an exact LSST instrument simulation.
        target=10**((23.9-DEPTHS)/2.5)/5
        err=np.maximum(deep_err,target)
        extra=np.sqrt(np.maximum(err**2-deep_err**2,0))
        mode='native_filter_lsst_like_noise_proxy'
    spec=np.genfromtxt(ROOT/'results/cosmos2020_specz/matched_galaxies.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    spec=spec[np.argsort(spec['Id_COS20_Classic'])]
    j=np.searchsorted(spec['Id_COS20_Classic'],raw['id'])
    isspec=j<len(spec);isspec[isspec]&=spec['Id_COS20_Classic'][j[isspec]]==raw['id'][isspec]
    zspec=np.full(len(raw),np.nan);zspec[isspec]=spec['specz'][j[isspec]]
    # Fix the sky-cell map before magnitude selection. It cannot depend on spec-z.
    all_cell=np.floor(raw['ra']/.05).astype(np.int64)*100000+np.floor(raw['dec']/.05).astype(np.int64)
    cells=np.unique(all_cell);labels=rng.choice(3,len(cells),p=[.7,.15,.15])
    all_split=labels[np.searchsorted(cells,all_cell)]
    good,selection_counts=magnitude_parent(raw,flux,err,i_limit,i_snr_min)
    # No positive-flux requirement: retain broadband nondetections.
    for arr in [flux,err,extra]:assert arr.shape==(len(raw),6)
    raw=raw[good];flux=flux[good].astype(np.float32);err=err[good].astype(np.float32);extra=extra[good].astype(np.float32);zspec=zspec[good]
    cell=all_cell[good];split=all_split[good]
    # Census of the eligible catalog: inclusion probability one, no spec preference.
    out=output_dir or ROOT/'data/flow_magnitude_limited';out.mkdir(parents=True,exist_ok=True)
    ids=raw['id'].astype(int)
    grid,pdf=extract_pdfs(ids,out/'teacher_pdfs.npz')
    pdf_good=np.isfinite(pdf).all(axis=1)&(pdf>=0).all(axis=1)&(pdf.sum(axis=1)>0)
    imag=23.9-2.5*np.log10(raw['hsc_i_flux_aper2'])
    np.savez_compressed(out/'prepared.npz',id=ids[pdf_good],flux=flux[pdf_good],err=err[pdf_good],extra=extra[pdf_good],zspec=zspec[pdf_good],split=split[pdf_good],cell=cell[pdf_good],imag=imag[pdf_good],grid=grid,pdf=pdf[pdf_good])
    manifest=dict(mode=mode,actual_input=str(actual_csv) if actual_csv else None,
                  n_parent_rows=len(raw),n_valid_pdf=int(pdf_good.sum()),seed=20261024,selection_counts=selection_counts,
                  magnitude_limit=dict(band='HSC i',system='AB',aperture_diameter_arcsec=2,upper_exclusive=i_limit,minimum_snr=i_snr_min),
                  sampling='All eligible catalog objects retained, independently of spectroscopic availability; no cap or oversampling',
                  completeness='Catalog-defined population; no injection/recovery completeness measurement. Combined detection, galaxy classification and finite-measurement cuts remain selection effects.',
                  teacher_assumption='COSMOS2020 photo-z distributions treated as unbiased; no recalibration or prior division',
                  comparison='same conditional-flow architecture; COSMOS-PDF training vs spec-z training; both use only six broadband fluxes/errors at inference',
                  forecast_depths_5sigma_ab=DEPTHS.tolist() if not actual_csv else None,
                  filter_warning='CFHT u/HSC grizy are native COSMOS filters, NOT Rubin bandpasses' if not actual_csv else None,
                  noise='Add independent Gaussian noise with extra variance max(target_variance - reported_deep_variance,0); effective errors cannot improve over COSMOS errors' if not actual_csv else 'Measured fluxes and errors; no synthetic noise augmentation',
                  split='disjoint 0.05 degree sky cells; 70/15/15 train/validation/test; all photometric test objects held out, only secure-spec subset scored',
                  selection='clean combined footprint; galaxy template lp_type=0; HSC i aperture magnitude limit and i SNR cut; finite six-band fluxes and positive errors; valid PDF; no other band SNR or redshift-agreement cut',
                  counts={str(s):dict(n=int(np.sum((split==s)&pdf_good)),n_spec=int(np.sum((split==s)&pdf_good&np.isfinite(zspec)))) for s in range(3)})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
    return out/'prepared.npz'
