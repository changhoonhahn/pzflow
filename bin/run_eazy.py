#!/usr/bin/env python3
"""Run untuned EAZY-py on the canonical parent broadband photometry.

Run from the repository root; default selection matches the flow test sample.
Use --sample test for all photometric test objects, or --sample all for the parent.

Requires public eazy-photoz resources in data/eazy-photoz-master and per-object
Galactic EBV values from IRSA. No spectroscopic redshift is passed to EAZY.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from astropy.table import Table
import eazy

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/parent.npz'))
    parser.add_argument('--assets',type=Path,default=Path('data/eazy-photoz-master'))
    parser.add_argument('--ebv',type=Path,default=Path('data/cosmos2020_specz/classic_i25_ebv.csv'))
    parser.add_argument('--magnitudes',type=Path,default=Path('data/cosmos2020_specz/classic_magnitude_intervals.csv'))
    parser.add_argument('--output',type=Path,help='Default: results/eazy/flat or results/eazy/rprior')
    parser.add_argument('--sample',choices=['spec-test','test','all'],default='spec-test')
    parser.add_argument('--limit',type=int,help='First N selected IDs, for smoke tests only')
    parser.add_argument('--r-prior',action='store_true')
    args=parser.parse_args()
    if args.limit is not None and args.limit < 1: parser.error('--limit must be positive')
    limit,r_prior=args.limit,args.r_prior
    ASSETS=args.assets.resolve()
    OUT=args.output or Path('results/eazy')/('rprior' if r_prior else 'flat')
    if OUT.exists() and any(OUT.iterdir()): parser.error('Output directory is nonempty; choose another --output')
    OUT.mkdir(parents=True,exist_ok=True)
    eazy.set_data_path(str(ASSETS))
    from eazy.photoz import PhotoZ
    from eazy.filters import FilterFile
    with np.load(args.data) as parent:
        if not json.loads(str(parent['metadata'])).get('canonical_parent'):
            raise ValueError('Use data/parent.npz from match_cosmos.py')
        keep=np.ones(len(parent['id']),bool) if args.sample=='all' else parent['split']==2
        if args.sample=='spec-test': keep &= parent['has_specz']
        ix=np.flatnonzero(keep)[:limit]
        ids=parent['id'][ix];flux=parent['flux'][ix].astype(float);err=parent['fluxerr'][ix].astype(float)
    if not len(ids): raise ValueError('Empty selected sample')
    original_flux,original_err=flux.copy(),err.copy()
    ebv=np.genfromtxt(args.ebv,delimiter=',',names=True)
    ebv=ebv[np.argsort(ebv['id'])];j=np.searchsorted(ebv['id'],ids)
    if len(np.unique(ebv['id'])) != len(ebv) or np.any(j==len(ebv)):
        raise ValueError('Missing or duplicated EBV IDs')
    np.testing.assert_array_equal(ebv['id'][j],ids)
    if not np.isfinite(ebv['ebv_mw'][j]).all(): raise ValueError('Nonfinite EBV')
    # Published Classic aperture preparation coefficients, no template/spec fit.
    # Apply the same multiplicative factor to flux AND uncertainty, including
    # nondetections. A common aperture-to-total factor is absorbed by SED amplitude.
    coeff=np.array([4.807,3.69,2.715,2.,1.515,1.298])
    correction=10**(.4*ebv['ebv_mw'][j,None]*coeff[None,:])
    flux*=correction;err*=correction
    if r_prior:
        meta=np.genfromtxt(args.magnitudes,delimiter=',',names=True)
        meta=meta[np.argsort(meta['id'])];j=np.searchsorted(meta['id'],ids)
        if len(np.unique(meta['id'])) != len(meta) or np.any(j==len(meta)):
            raise ValueError('Missing or duplicated magnitude IDs')
        np.testing.assert_array_equal(meta['id'][j],ids)
        factor=10**(-.4*meta['total_off2'][j])
        assert np.isfinite(factor).all() and (factor>0).all()
        flux*=factor[:,None];err*=factor[:,None]
    filter_ids=[409,314,315,316,317,318] # CFHT cfh9302 u, HSC g/r/i/z/y
    filters=FilterFile(str(ASSETS/'filters/FILTER.RES.latest'))
    catalog=Table({'id':ids})
    for k,fid in enumerate(filter_ids):catalog[f'F{fid}']=flux[:,k];catalog[f'E{fid}']=err[:,k]
    catalog_path=OUT/('eazy_smoke_catalog.fits' if limit else 'eazy_catalog.fits')
    catalog.write(catalog_path,overwrite=True)
    params=dict(CATALOG_FILE=str(catalog_path),CATALOG_FORMAT='fits',
                FILTERS_RES=str(ASSETS/'filters/FILTER.RES.latest'),
                TEMPLATES_FILE=str(ASSETS/'templates/fsps_full/tweak_fsps_QSF_12_v3.param'),
                TEMP_ERR_FILE=str(ASSETS/'templates/uvista_nmf/template_error_10.def'),
                WAVELENGTH_FILE=str(ASSETS/'templates/uvista_nmf/lambda.def'),
                PRIOR_FILE=str(ASSETS/'templates/prior_F160W_TAO.dat'),
                OUTPUT_DIRECTORY=str(OUT),MAIN_OUTPUT_FILE='eazy_six_band',
                PRIOR_ABZP=23.9,APPLY_PRIOR='n',FIX_ZSPEC='n',
                Z_MIN=.001,Z_MAX=10.,Z_STEP=.005,MW_EBV=0.,CAT_HAS_EXTCORR='y',
                SYS_ERR=.01,TEMP_ERR_A2=.20,VERBOSITY=0,ADD_CGM='n',
                NOT_OBS_THRESHOLD=-1.e10)
    if r_prior:
        # The original C-code prior has an uncommented header; EAZY-py needs '#'.
        original=(ASSETS/'templates/prior_R_zmax7.dat').read_text()
        prior_file=OUT/'prior_R_zmax7_py.dat';prior_file.write_text('# '+original)
        params.update(PRIOR_FILE=str(prior_file),PRIOR_FILTER=315,APPLY_PRIOR='y')
    start=time.monotonic()
    translate=OUT/'eazy.translate';translate.write_text('')
    p=PhotoZ(params=params,translate_file=str(translate),load_prior=r_prior,n_proc=0,random_seed=20261116)
    assert p.NOBJ==len(ids) and p.NFILT==6 and np.all(p.nusefilt==6)
    p.fit_catalog(n_proc=0,get_best_fit=False,prior=r_prior,beta_prior=False)
    p.compute_lnp(prior=r_prior,beta_prior=False,clip_wavelength=None)
    p.evaluate_zml(prior=r_prior,beta_prior=False,clip_wavelength=None)
    pdf=np.exp(p.lnp-p.lnp.max(axis=1)[:,None])
    assert np.isfinite(pdf).all() and (pdf.sum(axis=1)>0).all()
    pdf /= np.trapezoid(pdf,p.zgrid,axis=1)[:,None]
    stem=('eazy_rprior_smoke' if r_prior else 'eazy_smoke') if limit else ('eazy_rprior' if r_prior else 'eazy_six_band')
    np.savez_compressed(OUT/f'{stem}_posteriors.npz',id=ids,grid=p.zgrid,pdf=pdf,chi2=p.chi2_fit,zml=p.zml,
                        flux=original_flux,fluxerr=original_err,eazy_flux=flux,eazy_fluxerr=err)
    manifest=dict(code='EAZY-py',version=eazy.__version__,n_objects=len(ids),n_templates=p.NTEMP,
                  seconds=time.monotonic()-start,params=params,
                  filters=[dict(band=b,filter_id=fid,name=filters[fid].name) for b,fid in zip('ugrizy',filter_ids)],
                  galactic_extinction='Per-object catalog EBV; released COSMOS coefficients [4.807,3.69,2.715,2.0,1.515,1.298]; fluxes and errors corrected together',
                  same_input_check='Uses fixed canonical-parent fluxes and uncertainties, with deterministic Galactic correction; all six bands retained including negative fluxes',
                  aperture_correction='Catalog total_off2 common scale applied to all fluxes/errors to define total r magnitude for prior' if r_prior else 'Not applied: free template amplitude absorbs common scale and no magnitude prior is used',
                  prior='Supplied prior_R_zmax7.dat applied to extinction-corrected HSC r; approximate R-to-HSC-r mapping; no beta prior' if r_prior else 'Flat in redshift: no apparent-magnitude or beta prior. Default H160 prior cannot be applied to these six filters.',
                  calibration='Default 12 tweaked FSPS templates, 1% SYS_ERR and template error amplitude 0.20; no zero-point training or test-label tuning',
                  resource_source='https://github.com/gbrammer/eazy-photoz',
                  extinction_source='https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/misc/COSMOS2020_prepare_apertures.txt',
                  limitations=['Initial untuned EAZY configuration, not an optimized or survey-calibrated benchmark.',
                               'Library native-band curves are used; exact exposure-dependent HSC passbands are not modeled.',
                               'Six-band posterior degeneracies depend strongly on the chosen prior and templates.'])
    inputs={'parent':args.data,'ebv':args.ebv,
            'filters':ASSETS/'filters/FILTER.RES.latest',
            'template_list':ASSETS/'templates/fsps_full/tweak_fsps_QSF_12_v3.param',
            'template_error':ASSETS/'templates/uvista_nmf/template_error_10.def',
            'wavelength':ASSETS/'templates/uvista_nmf/lambda.def'}
    if r_prior:
        inputs.update(magnitudes=args.magnitudes,prior=ASSETS/'templates/prior_R_zmax7.dat')
    template_list=inputs['template_list']
    for row in template_list.read_text().splitlines():
        fields=row.split()
        if fields and not fields[0].startswith('#'):
            inputs['template_'+fields[0]]=ASSETS/fields[1]
    manifest.update(sample=args.sample,limit=limit,smoke_test=limit is not None,
                    pdf_normalization='Trapezoidal integral over saved grid equals one',inputs={})
    for name,path in inputs.items():
        with path.open('rb') as f:
            manifest['inputs'][name]=dict(path=str(path.resolve()),sha256=hashlib.file_digest(f,'sha256').hexdigest())
    (OUT/(f'{stem}_manifest.json' if limit else ('eazy_rprior_manifest.json' if r_prior else 'eazy_manifest.json'))).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
