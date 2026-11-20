#!/usr/bin/env python3
"""Build the ONE analysis parent: LSST-like fluxes + COSMOS PDFs + secure spec-z.

NPZ PDF caches must contain id, grid, pdf. The default source is the public
compressed LePhare FITS image, streamed without loading the complete catalog.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pzflow.utils import PDFSampler


def read_pdfs(path, ids):
    if path.suffix == '.npz':
        with np.load(path) as c:
            source_ids = c['id']
            order = np.argsort(source_ids)
            source_ids = source_ids[order]
            if len(np.unique(source_ids)) != len(source_ids):
                raise ValueError('Duplicate PDF IDs')
            j = np.searchsorted(source_ids, ids)
            if np.any(j == len(source_ids)) or not np.array_equal(source_ids[j], ids):
                raise ValueError('Missing parent IDs in PDF cache')
            return c['grid'], c['pdf'][order[j]]
    seen = np.zeros(len(ids), int)
    with gzip.open(path, 'rb') as f:
        header = fits.Header.fromfile(f)
        width, height = header['NAXIS1'], header['NAXIS2']
        if header['BITPIX'] != -32:
            raise ValueError('Expected float32 PDF image')
        grid = np.frombuffer(f.read(width * 4), dtype='>f4')[1:].astype(float)
        pdf = np.empty((len(ids), width - 1), dtype='f4')
        for start in range(1, height, 16384):
            n = min(16384, height - start)
            block = np.frombuffer(f.read(n * width * 4), dtype='>f4').reshape(n, width)
            rid = block[:, 0].astype('i8')
            j = np.searchsorted(ids, rid)
            match = j < len(ids)
            match[match] &= ids[j[match]] == rid[match]
            np.add.at(seen, j[match], 1)
            pdf[j[match]] = block[match, 1:]
        f.read()  # consume padding and validate gzip checksum
    if not np.all(seen == 1):
        raise ValueError('Missing or duplicated parent IDs in PDF FITS')
    return grid, pdf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--photometry', type=Path, default=Path('data/lsst_like.npz'))
    p.add_argument(
        '--pdfs', type=Path, default=Path('data/cosmos2020_specz/classic_lephare_complete.fits.gz')
    )
    p.add_argument(
        '--spectroscopy',
        type=Path,
        default=Path('data/cosmos2020_specz/specz_compilation_COSMOS_DR1.1_unique.fits'),
    )
    p.add_argument('--output', type=Path, default=Path('data/parent.npz'))
    p.add_argument('--draws', type=int, default=32)
    p.add_argument('--seed', type=int, default=20261026)
    a = p.parse_args()
    if a.draws < 4:
        p.error('--draws must be at least 4 (fixed validation labels)')
    with np.load(a.photometry) as c:
        data = {k: c[k] for k in c.files if k != 'metadata'}
        meta = json.loads(str(c['metadata']))
    ids = data['id']
    if not np.all(np.diff(ids) > 0):
        raise ValueError('Photometry IDs must be unique and sorted')
    grid, pdf = read_pdfs(a.pdfs, ids)
    sampler = PDFSampler(
        grid, pdf
    )  # reject invalid PDFs rather than silently change the population
    rng = np.random.default_rng(a.seed)
    draws = np.empty((len(ids), a.draws), dtype='f4')
    for start in range(0, len(ids), 2048):
        draws[start : start + 2048] = sampler.draw(
            np.arange(start, min(start + 2048, len(ids))), rng, a.draws
        )
    with fits.open(a.spectroscopy, memmap=True) as h:
        s = h[1].data
        good = (
            (s['public_or_private'] == 1)
            & np.isin(s['flag'], [3, 4, 13, 14])
            & (s['photoz_type'] == 0)
            & (s['Id_COS20_Classic'] > 0)
            & np.isfinite(s['specz'])
            & (s['specz'] > 0)
            & np.isfinite(s['photoz'])
            & (s['photoz'] >= 0)
        )
        s = s[good]
        s = s[np.argsort(s['Id_COS20_Classic'])]
        sid = s['Id_COS20_Classic']
        if len(np.unique(sid)) != len(sid):
            raise ValueError('Ambiguous secure spectroscopic matches')
        j = np.searchsorted(sid, ids)
        match = j < len(sid)
        match[match] &= sid[j[match]] == ids[match]
        zspec = np.full(len(ids), np.nan)
        zspec[match] = s['specz'][j[match]]
        flag = np.zeros(len(ids), int)
        flag[match] = s['flag'][j[match]]
        survey = np.full(len(ids), '', dtype='U128')
        survey[match] = s['survey'][j[match]].astype(str)
    meta.update(
        canonical_parent=True,
        pdf_assumption='COSMOS PDFs treated as unbiased; no recalibration or prior division',
        pdf_sampling='32 cached draws by default; training resamples the full original PDF every batch',
        n_draws=a.draws,
        draw_seed=a.seed,
        spectroscopy_selection='public=1; flag=3,4,13,14; galaxy photoz_type=0; positive Classic ID and specz; finite nonnegative photoz; no photoz-specz agreement cut',
        counts_by_split={
            str(k): dict(
                total=int(np.sum(data['split'] == k)),
                spec=int(np.sum((data['split'] == k) & match)),
            )
            for k in range(3)
        },
    )
    meta['input_files'] = {}
    for name, path in [
        ('photometry', a.photometry),
        ('pdfs', a.pdfs),
        ('spectroscopy', a.spectroscopy),
    ]:
        with path.open('rb') as f:
            meta['input_files'][name] = dict(
                path=str(path.resolve()), sha256=hashlib.file_digest(f, 'sha256').hexdigest()
            )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.output,
        **data,
        grid=grid,
        pdf=pdf,
        z_draws=draws,
        zspec=zspec,
        has_specz=match,
        spec_flag=flag,
        spec_survey=survey,
        metadata=json.dumps(meta),
    )
    a.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta['counts_by_split']), flush=True)


if __name__ == '__main__':
    main()
