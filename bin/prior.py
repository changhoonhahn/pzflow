#!/usr/bin/env python3
"""Fit an independent-split redshift-prior ratio and test LePhare coverage."""

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import numpy as np
from pzflow.paths import ROOT
from pzflow.prior import WIDTHS,EPS,SEED,smooth_density,get_summary
OUT=ROOT/'results/coverage_prior_correction'
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT/'data/flow_magnitude_limited/prepared.npz') as f:
        ids, grid, p, z, split, imag, cell = [f[k] for k in ['id','grid','pdf','zspec','split','imag','cell']]
    cal = split != 2
    spec_cal = cal & np.isfinite(z)
    test = (split == 2) & np.isfinite(z)
    assert test.sum() == 5083
    assert not np.intersect1d(cell[cal], cell[test]).size
    with np.load(ROOT/'results/figure_data/predictions.npz') as f:
        assert np.array_equal(ids[test], f['id'])
    # Stack independently normalized parent PDFs in batches to limit memory.
    stack = np.zeros(len(grid), dtype=float)
    idx = np.flatnonzero(cal)
    for batch in np.array_split(idx, 64):
        q = p[batch].astype(float)
        q /= np.trapezoid(q, grid, axis=1)[:, None]
        stack += q.sum(axis=0)
    stack /= len(idx)
    assert abs(np.trapezoid(stack, grid)-1) < 1e-8
    # Equal-width log(1+z) bins produce redshift-dependent smoothing in z.
    edges = np.linspace(0, np.log1p(grid[-1]), 2401)
    t = (edges[:-1]+edges[1:])/2
    zg = np.expm1(t)
    parent_raw = np.interp(zg, grid, stack)*np.exp(t)
    hist = np.histogram(np.log1p(z[spec_cal]), edges)[0].astype(float)
    hist /= hist.sum()*(edges[1]-edges[0])
    summary, original = get_summary(z[test], grid, p[test])
    metrics = {'original': summary}
    pits = {'original': original}
    ratios, parent, spec = {}, {}, {}
    for h in WIDTHS:
        key = f'h_{h:.3f}'
        parent[key] = smooth_density(t, parent_raw.copy(), h)
        spec[key] = smooth_density(t, hist.copy(), h)
        ratios[key] = np.interp(np.log1p(grid), t, spec[key]/parent[key])
        corrected = p[test].astype(float)*ratios[key][None, :]
        metrics[key], pits[key] = get_summary(z[test], grid, corrected)
    # Paired bootstrap: PDFs and calibration-density estimates are held fixed.
    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, test.sum(), size=(1000, test.sum()))
    for key, pit in pits.items():
        for a in [.68, .95]:
            hit = abs(2*pit-1) <= a
            base = abs(2*original-1) <= a
            boots = hit[boot_idx].mean(axis=1)
            delta = (hit.astype(float)-base)[boot_idx].mean(axis=1)
            label = f'coverage{int(a*100)}'
            metrics[key][label+'_95ci'] = np.quantile(boots, [.025,.975]).tolist()
            metrics[key][label+'_change'] = float(hit.mean()-base.mean())
            metrics[key][label+'_change_95ci'] = np.quantile(delta,[.025,.975]).tolist()
    # Descriptive magnitude subsets, with no separate prior fitting.
    subgroups = {}
    for name, mask in [('i_lt_22.5',imag[test]<22.5),
                       ('i_22.5_24',(imag[test]>=22.5)&(imag[test]<24)),
                       ('i_24_25',imag[test]>=24), ('z_gt_2',z[test]>2)]:
        subgroups[name] = {'n':int(mask.sum())}
        for key, pit in pits.items():
            subgroups[name][key] = {f'coverage{int(a*100)}':float(np.mean(abs(2*pit[mask]-1)<=a)) for a in [.68,.95]}
    report = dict(n_parent_calibration=int(cal.sum()),n_spec_calibration=int(spec_cal.sum()),
                  n_test=int(test.sum()),seed=SEED,baseline_bandwidth_log1pz=.03,
                  bandwidth_sensitivity_log1pz=WIDTHS,uniform_regularization=EPS,
                  method='p_corrected(z|x) proportional to p_original(z|x) * n_spec(z)/n_parent(z), each PDF renormalized; both n(z) estimated on spatial train+validation only',
                  parent_estimate='Stack of independently normalized COSMOS2020 LePhare PDFs for all eligible calibration galaxies',
                  spec_estimate='Secure spectroscopic redshift histogram in the same calibration population',
                  uncertainty='1000 paired galaxy bootstraps on test objects; fixed fitted density ratio; excludes density-estimation uncertainty, spatial covariance and systematic errors',
                  caveats=['Redshift-only selection/label-shift approximation; no correction for selection depending on photometry or galaxy properties at fixed redshift.',
                           'Stacked PDFs are an approximate parent n(z), not independent truth or necessarily the implicit prior of individual LePhare fits.',
                           'No test spectroscopic labels used to estimate or choose the density ratio or bandwidth.',
                           'Reweighting cannot restore posterior modes assigned exactly zero density.'],
                  metrics=metrics,subgroups=subgroups)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'diagnostics.npz',id=ids[test],zspec=z[test],imag=imag[test],
                        t=t,z_density=zg,grid=grid,**{f'pit_{k}':v for k,v in pits.items()},
                        **{f'ratio_{k}':v for k,v in ratios.items()},
                        **{f'parent_{k}':v for k,v in parent.items()},
                        **{f'spec_{k}':v for k,v in spec.items()})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
