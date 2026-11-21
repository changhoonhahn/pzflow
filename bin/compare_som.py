#!/usr/bin/env python3
"""Compare a broadband SOM with the spec-z flow using identical labeled data.

This is a single-SOM cell-density baseline, not full deep-to-wide SOMPZ.
Only fixed degraded broadband fluxes/errors and secure spec-z are read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from importlib.metadata import version

import numpy as np
import torch
from minisom import MiniSom
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pzflow.utils import PDFSampler

# Reuse the actual flow implementation, rather than copying its architecture.
from train_flows import context, raw_features, train_specz, posterior_summary, z_to_t, metrics


def assign(x, weights):
    """Nearest-prototype assignments, with bounded temporary distance matrices."""
    return np.concatenate(
        [cdist(x[i : i + 1024], weights).argmin(1) for i in range(0, len(x), 1024)]
    )


def cell_pdfs(cells, z, weights, bandwidth, alpha):
    """Smoothed spec-z histograms per cell, with explicit empty-cell fallback."""
    # Smooth in log(1+z), giving approximately constant resolution in delta-z/(1+z).
    edges = np.linspace(0, np.log(11), 2001)
    grid = np.expm1(np.r_[0, (edges[1:] + edges[:-1]) / 2, edges[-1]])
    hist = np.zeros((len(weights), len(edges) - 1))
    bins = np.clip(np.searchsorted(edges, np.log1p(z)) - 1, 0, len(edges) - 2)
    np.add.at(hist, (cells, bins), 1)
    count = hist.sum(1)
    global_hist = hist.sum(0) / hist.sum()
    # Alpha is a validation-selected pseudo-count toward the training n(z).
    # Empty cells borrow the nearest populated prototype, never test redshifts.
    filled = np.flatnonzero(count > 0)
    fallback = np.arange(len(weights))
    empty = np.flatnonzero(count == 0)
    if len(empty):
        fallback[empty] = filled[cdist(weights[empty], weights[filled]).argmin(1)]
    hist = hist[fallback] + alpha * global_hist
    hist = gaussian_filter1d(hist, bandwidth / np.diff(edges)[0], axis=1, mode='reflect')
    hist /= hist.sum(1)[:, None]
    density = np.pad(hist, ((0, 0), (1, 1)), mode='edge') / (1 + grid)[None, :]
    return PDFSampler(grid, density), count, fallback


def density_at(sampler, z, cells):
    j = np.clip(np.searchsorted(sampler.grid, z) - 1, 0, len(sampler.grid) - 2)
    w = (z - sampler.grid[j]) / np.diff(sampler.grid)[j]
    return (1 - w) * sampler.p[cells, j] + w * sampler.p[cells, j + 1]


@torch.no_grad()
def flow_cdf(model, x, grid):
    """Evaluate the flow CDF on the same integration grid as the SOM."""
    result = np.empty((len(x), len(grid)), dtype='f4')
    for start in range(0, len(x), 64):
        ctx = x[start : start + 64]
        z = torch.tensor(grid, dtype=torch.float32).repeat(len(ctx)).reshape(-1, 1)
        u, _ = model._transform(z_to_t(z), context=ctx.repeat_interleave(len(grid), 0))
        result[start : start + len(ctx)] = (
            (0.5 * (1 + torch.erf(u / np.sqrt(2)))).reshape(len(ctx), -1).numpy()
        )
    result[:, 0], result[:, -1] = 0, 1
    return result


def score(z, q, pit, nll, cdf, grid):
    result = metrics(z, q, pit, nll)
    # CRPS integrates the squared difference between predictive and truth CDFs.
    result['crps'] = float(
        np.trapezoid((cdf - (grid[None, :] >= z[:, None])) ** 2, grid, axis=1).mean()
    )
    result['mean_redshift_error'] = float(np.trapezoid(1 - cdf.mean(0), grid) - z.mean())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/parent.npz'))
    p.add_argument('--output', type=Path, default=Path('results/som_comparison'))
    p.add_argument(
        '--seeds', type=int, nargs='+', default=[20261024, 20261025, 20261026, 20261027, 20261028]
    )
    p.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--iterations', type=int, default=60000)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Choose an empty output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    # Deliberately never load native_flux, COSMOS PDFs, or their posterior draws.
    with np.load(args.data) as f:
        if not json.loads(str(f['metadata'])).get('canonical_parent'):
            raise ValueError('Use the canonical parent dataset')
        data = {k: f[k] for k in ['id', 'flux', 'fluxerr', 'zspec', 'split', 'imag', 'cell']}
    masks = [(data['split'] == k) & np.isfinite(data['zspec']) for k in range(3)]
    train, valid, test = [np.flatnonzero(m) for m in masks]
    feature = raw_features(data['flux'][train], data['fluxerr'][train])
    mean, scale = feature.mean(0), np.maximum(feature.std(0), 0.1)
    x = context(data['flux'], data['fluxerr'], mean, scale).numpy()
    # Give both methods exactly the same fixed degraded training measurements.
    # The shared flow loop consumes noise draws, but zero extra variance disables augmentation.
    fit_data = dict(data, native_flux=data['flux'], sigma_added=np.zeros_like(data['flux']))
    grid = np.linspace(0, 10, 1001)
    run = dict(
        seeds=args.seeds,
        iterations=args.iterations,
        sample_counts=[len(t) for t in [train, valid, test]],
        inputs='fixed degraded ugrizy fluxes/errors; 12 common features; no augmentation or deep inputs',
        model='single rectangular MiniSom; cell PDFs, not full SOMPZ',
        minisom_version=version('minisom'),
        selection='Validation NLL; same 4096-object validation subset as seed-0 flow',
        trials=[],
        runs=[],
    )
    with args.data.open('rb') as f:
        run['parent_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    # Reproduce the first flow seed's validation subset; never optimize on test labels.
    vi = np.random.default_rng(args.seeds[0]).choice(valid, min(len(valid), 4096), replace=False)
    best = None
    for i, seed in enumerate(args.seeds):
        start = time.monotonic()
        candidates = args.sizes if i == 0 else [run['selected']['side']]
        for side in candidates:
            som = MiniSom(
                side,
                side,
                12,
                sigma=side / 2,
                learning_rate=0.5,
                random_seed=seed,
                sigma_decay_function='linear_decay_to_one',
            )
            som.random_weights_init(x[train])
            som.train_random(x[train], args.iterations)
            weights = som.get_weights().reshape(-1, 12)
            tc, vc = assign(x[train], weights), assign(x[vi], weights)
            if i == 0:
                for bandwidth in [0.005, 0.01, 0.02, 0.04, 0.08]:
                    for alpha in [0.0, 5.0, 20.0]:
                        sampler, count, fallback = cell_pdfs(
                            tc, data['zspec'][train], weights, bandwidth, alpha
                        )
                        val = float(
                            -np.log(
                                np.maximum(density_at(sampler, data['zspec'][vi], vc), 1e-12)
                            ).mean()
                        )
                        config = dict(
                            side=side, bandwidth=bandwidth, alpha=alpha, validation_nll=val
                        )
                        run['trials'].append(config)
                        if best is None or val < best[0]:
                            best = (val, weights.copy(), config)
                print(f'SOM tuning: side={side} best NLL={best[0]:.4f}', flush=True)
        if i == 0:
            weights, run['selected'] = best[1], best[2]
        tc, sc = assign(x[train], weights), assign(x[test], weights)
        config = run['selected']
        sampler, count, fallback = cell_pdfs(
            tc, data['zspec'][train], weights, config['bandwidth'], config['alpha']
        )
        q = np.array(
            [np.interp([0.025, 0.16, 0.5, 0.84, 0.975], c, sampler.grid) for c in sampler.c]
        )[sc]
        pit = sampler.pit(data['zspec'][test], sc)
        nll = -np.log(np.maximum(density_at(sampler, data['zspec'][test], sc), 1e-12))
        cell_cdf = np.array([np.interp(grid, sampler.grid, c) for c in sampler.c])
        cdf = cell_cdf[sc]
        pred = {k: data[k][test] for k in ['id', 'zspec', 'imag', 'cell', 'flux', 'fluxerr']}
        pred.update(
            grid=grid,
            som_quantiles=q,
            som_pit=pit,
            som_nll=nll,
            som_cdf=cdf,
            som_cell=sc,
            som_training_occupancy=count[sc],
        )
        som_score = score(data['zspec'][test], q, pit, nll, cdf, grid)
        np.savez_compressed(
            args.output / f'som_{seed}.npz',
            weights=weights,
            pdf=sampler.p,
            grid=sampler.grid,
            count=count,
            fallback=fallback,
            feature_mean=mean,
            feature_scale=scale,
        )
        som_seconds = time.monotonic() - start
        fit_args = argparse.Namespace(seed=seed, epochs=200, max_steps=20000, output=args.output)
        model, info = train_specz(fit_data, mean, scale, fit_args)
        # Keep every seed's checkpoint; common helper output names stay unchanged elsewhere.
        (args.output / 'specz.pt').rename(args.output / f'specz_{seed}.pt')
        (args.output / 'specz_history.json').rename(args.output / f'specz_{seed}_history.json')
        ctx = torch.from_numpy(x[test])
        fq, fpit, fnll = posterior_summary(model, ctx, data['zspec'][test])
        fcdf = flow_cdf(model, ctx, grid)
        pred.update(flow_quantiles=fq, flow_pit=fpit, flow_nll=fnll, flow_cdf=fcdf)
        flow_score = score(data['zspec'][test], fq, fpit, fnll, fcdf, grid)
        np.savez_compressed(args.output / f'predictions_{seed}.npz', **pred)
        run['runs'].append(
            dict(
                seed=seed,
                som=som_score,
                flow=flow_score,
                flow_training=info,
                som_seconds=som_seconds,
                empty_cell_test_fraction=float(np.mean(count[sc] == 0)),
            )
        )
        (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        print(
            f'Seed {seed}: SOM NMAD={som_score["sigma_nmad"]:.4f}; flow={flow_score["sigma_nmad"]:.4f}',
            flush=True,
        )


if __name__ == '__main__':
    main()
