#!/usr/bin/env python3
"""Select the magnitude-limited COSMOS population and add LSST-like noise.

Run from the repository root. Fluxes are microJy in native CFHT/HSC filters;
this is a depth forecast, not a Rubin bandpass or image simulation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--photometry', type=Path, default=Path('data/cosmos2020_specz/classic_ugrizy.csv')
    )
    p.add_argument('--output', type=Path, default=Path('data/lsst_like.npz'))
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--i-limit', type=float, default=25.0)
    p.add_argument('--i-snr-min', type=float, default=5.0)
    a = p.parse_args()
    # Sort by Classic ID so downstream joins and seeded realizations are reproducible.
    raw = np.genfromtxt(
        a.photometry, delimiter=',', names=True, dtype=None, encoding='utf8', filling_values=np.nan
    )
    raw = raw[np.argsort(raw['id'])]
    if len(np.unique(raw['id'])) != len(raw):
        raise ValueError('Duplicate Classic IDs')
    # Keep the six native bands in ugrizy order; each row is one galaxy.
    prefixes = ['cfht_u', 'hsc_g', 'hsc_r', 'hsc_i', 'hsc_z', 'hsc_y']
    flux = np.column_stack([raw[b + '_flux_aper2'] for b in prefixes])
    err = np.column_stack([raw[b + '_fluxerr_aper2'] for b in prefixes])
    # Split sky cells before selection so nearby objects share a partition.
    # Labels 0, 1, 2 denote training, validation, and test respectively.
    cell = np.floor(raw['ra'] / 0.05).astype(np.int64) * 100000 + np.floor(
        raw['dec'] / 0.05
    ).astype(np.int64)
    cells = np.unique(cell)
    labels = np.random.default_rng(a.seed).choice(3, len(cells), p=[0.7, 0.15, 0.15])
    split = labels[np.searchsorted(cells, cell)]
    # Define the parent using native photometry, independently of spectroscopy.
    # AB magnitudes obey m = 23.9 - 2.5 log10(flux / microJy).
    # Retain negative fluxes outside i: these still carry likelihood information.
    snr = np.divide(flux[:, 3], err[:, 3], out=np.zeros(len(raw)), where=err[:, 3] > 0)
    cuts = dict(
        clean_footprint=raw['flag_combined'] == 0,
        galaxy=raw['lp_type'] == 0,
        magnitude=flux[:, 3] > 10 ** ((23.9 - a.i_limit) / 2.5),
        i_snr=snr >= a.i_snr_min,
        six_bands=np.isfinite(flux).all(1)
        & (abs(flux) < 1e8).all(1)
        & np.isfinite(err).all(1)
        & (err > 0).all(1),
    )
    keep = np.ones(len(raw), bool)
    counts = {'input': len(raw)}
    for name, cut in cuts.items():
        keep &= cut
        counts[name] = int(keep.sum())
    native, native_err = flux[keep].astype('f4'), err[keep].astype('f4')
    # Convert forecast five-sigma depths to one-sigma flux errors. Independent
    # noise variances add; never reduce an existing COSMOS measurement error.
    depths = np.array([25.6, 26.9, 26.9, 26.4, 25.6, 24.8])
    target = 10 ** ((23.9 - depths) / 2.5) / 5
    effective = np.maximum(native_err, target).astype('f4')
    added = np.sqrt(np.maximum(effective**2 - native_err**2, 0)).astype('f4')
    # Perturb fluxes as well as uncertainties. This fixed all-parent realization
    # supplies validation/test inputs; training draws fresh noise separately.
    observed = (
        native + np.random.default_rng(a.seed + 1).normal(size=native.shape) * added
    ).astype('f4')
    # Record the exact input and selection, alongside both native and noisy fluxes.
    # Keeping native fluxes allows training augmentation without adding noise twice.
    with a.photometry.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    meta = dict(
        source=str(a.photometry.resolve()),
        sha256=digest,
        seed=a.seed,
        counts=counts,
        units='microJy; 2 arcsec aperture; AB magnitude',
        i_limit=a.i_limit,
        i_snr_min=a.i_snr_min,
        depths_5sigma_ab=depths.tolist(),
        split='0.05 degree sky cells, assigned before selection; 0=train, 1=validation, 2=test',
        selection='All eligible galaxies, independent of spectroscopy; negative flux allowed outside i',
        noise='Independent Gaussian flux perturbations; extra variance=max(target variance-native variance,0)',
        warning='CFHT u/HSC grizy proxy; not Rubin bandpasses or extended-source image simulation',
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        id=raw['id'][keep].astype('i8'),
        ra=raw['ra'][keep],
        dec=raw['dec'][keep],
        cell=cell[keep],
        split=split[keep],
        imag=23.9 - 2.5 * np.log10(native[:, 3]),
        native_flux=native,
        native_fluxerr=native_err,
        flux=observed,
        fluxerr=effective,
        sigma_added=added,
        metadata=json.dumps(meta),
    )
    a.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
