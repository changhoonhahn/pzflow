"""Audit selection and quantify the remaining spectroscopic selection effects."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .plot_style import set_style, COLORS
set_style()
from .flow_experiment_data import ROOT,DATA,magnitude_parent

def main(data_dir,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    d=np.load(data_dir/'prepared.npz');manifest=json.loads((data_dir/'manifest.json').read_text())
    raw=np.genfromtxt(DATA/'classic_ugrizy.csv',delimiter=',',names=True)
    prefixes=['cfht_u','hsc_g','hsc_r','hsc_i','hsc_z','hsc_y']
    flux=np.stack([raw[f'{b}_flux_aper2'] for b in prefixes],axis=1)
    err=np.stack([raw[f'{b}_fluxerr_aper2'] for b in prefixes],axis=1)
    keep,counts=magnitude_parent(raw,flux,err,manifest['magnitude_limit']['upper_exclusive'],manifest['magnitude_limit']['minimum_snr'])
    # This run has no invalid PDFs: prove the entire eligible census was retained.
    np.testing.assert_array_equal(np.sort(raw['id'][keep].astype(int)),d['id'])
    assert len(np.unique(d['id']))==len(d['id'])
    for s in range(3):
        for t in range(s):assert not np.intersect1d(d['cell'][d['split']==s],d['cell'][d['split']==t]).size
    imag=d['imag'];spec=np.isfinite(d['zspec']);splits=d['split']
    rows=[]
    for lo,hi in [(0,21),(21,22),(22,23),(23,24),(24,24.5),(24.5,25)]:
        m=(imag>=lo)&(imag<hi)
        if not m.any():continue
        rows.append(dict(i_min=lo,i_max=hi,n_photometric=int(m.sum()),n_secure_spec=int((m&spec).sum()),
                         secure_spec_fraction=float(spec[m].mean()),n_test_photometric=int((m&(splits==2)).sum()),
                         n_test_spec=int((m&(splits==2)&spec).sum())))
    with (output_dir/'selection_by_magnitude.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(census_and_unique_ids_verified=True,disjoint_spatial_cells_verified=True,
                 magnitude_definition='Native HSC i AB, 2 arcsec aperture, before forecast noise',
                 input_catalog_rows=len(raw),eligible_rows=int(keep.sum()),
                 fraction_with_six_band_measurements=counts['six_band_measurements']/counts['i_snr'],
                 full_train_median_i=float(np.median(imag[splits==0])),
                 spec_train_median_i=float(np.median(imag[(splits==0)&spec])),
                 test_photometric_median_i=float(np.median(imag[splits==2])),
                 test_spec_median_i=float(np.median(imag[(splits==2)&spec])),
                 secure_spec_fraction=float(spec.mean()),magnitude_bins=rows,
                 limitations=['Finite-measurement retention is not detection completeness.',
                              'The secure-spec subset is not a random subset of the parent, even after the magnitude limit.',
                              'No inverse selection weights or photometric-only truth labels are available.'])
    (output_dir/'selection_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    bins=np.linspace(18,25,36)
    for mask,label,color in [(splits==0,'Full COSMOS training','C0'),((splits==0)&spec,'Spec training / matched PDF training','C1'),((splits==2)&spec,'Secure-spec test','C3')]:
        vals=imag[mask];weights=np.full(len(vals),1/len(vals)/(bins[1]-bins[0]))
        axs[0].hist(vals,bins=bins,weights=weights,histtype='step',lw=1.8,label=f'{label} (N={len(vals):,})',color=color)
    axs[0].set(xlabel='Native HSC i, 2 arcsec aperture (AB)',ylabel='Fraction per magnitude',title='Remaining spectroscopic selection')
    axs[0].legend(fontsize=8)
    centers=[(r['i_min']+r['i_max'])/2 if r['i_min'] else 20 for r in rows]
    axs[1].plot(centers,[100*r['secure_spec_fraction'] for r in rows],'o-',color='C1')
    axs[1].set(xlabel='Native HSC i, 2 arcsec aperture (AB)',ylabel='Secure spectroscopy available (%)',ylim=(0,100),title='Spectroscopy within selected parent')
    fig.suptitle('Magnitude-limited COSMOS census: i < 25, i S/N ≥ 5',fontsize=14)
    fig.supxlabel('All eligible catalog objects retained independently of spectroscopy. Full photometric test regions held out.\nNo claim of image-level completeness or representative spectroscopic evaluation.',fontsize=9)
    fig.savefig(output_dir/'sample_selection.png',dpi=180);fig.savefig(output_dir/'sample_selection.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data/flow_magnitude_limited')
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/flow_magnitude_limited')
    a=p.parse_args();main(a.data_dir,a.output_dir)
