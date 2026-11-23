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
import types
import urllib.request
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pzflow.utils import PDFSampler

# Reuse the actual flow implementation, rather than copying its architecture.
from train_flows import context, raw_features, z_to_t, metrics, train_specz, posterior_summary


NOISE_COMMIT = '80d852dc9178075df28929dd575591e6298ef689'
NOISE_SHA256 = '29b3647462c000d1530c50b3bc7ed7edde2a39295b80a4a1f1443d0feb2cc40a'
NOISE_URL = f'https://raw.githubusercontent.com/AndresaCampos/sompz_y6/{NOISE_COMMIT}'


def load_noise_som(path):
    """Load hash-pinned published code; preserve its metric and update equations."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(NOISE_URL + '/sompz/NoiseSOM.py') as response:
            source = response.read()
        if hashlib.sha256(source).hexdigest() != NOISE_SHA256:
            raise ValueError('Unexpected upstream NoiseSOM source hash')
        path.write_bytes(source)
        with urllib.request.urlopen(NOISE_URL + '/LICENSE') as response:
            path.with_name('LICENSE').write_bytes(response.read())
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != NOISE_SHA256:
        raise ValueError('NoiseSOM cache differs from the pinned source')
    # NumPy 2 removed product; the two imports below are used only by upstream
    # plotting helpers, not training. Avoid adding their unused dependencies.
    code = source.decode().replace('np.product(', 'np.prod(')
    code = code.replace('import pandas\n', '').replace('from matplotlib import pyplot as pl\n', '')
    module = types.ModuleType('published_noisesom')
    exec(compile(code, str(path), 'exec'), module.__dict__)
    return module


def assign(flux, errors, weights, metric):
    """Published noise-aware assignment, including negative-flux measurements."""
    cells = []
    # Match the reference classifier's ten-target chunks and supplied errors.
    # The upstream metric caps exp-based weights; suppress harmless overflow
    # warnings from values subsequently capped at 1000, without changing values.
    with np.errstate(over='ignore'):
        for start in range(0, len(flux), 10):
            cells.append(
                metric(weights, flux[start : start + 10], errors[start : start + 10]).argmin(0)
            )
    return np.concatenate(cells)


def train_noise_map(job):
    """Train one seed/map independently; validation labels never enter this step."""
    side, seed, iterations, path, flux, errors, valid_flux, valid_errors = job
    module = load_noise_som(path)
    metric = module.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
    # Keep the previous online update budget, sampling only labeled training rows.
    rows = np.random.default_rng(seed).integers(len(flux), size=iterations)
    np.random.seed(seed)
    start = time.monotonic()
    print(f'NoiseSOM side={side} seed={seed}: starting', flush=True)
    som = module.NoiseSOM(
        metric,
        flux[rows],
        errors[rows],
        learning=module.hFunc(iterations, sigma=(30.0, 1.0)),
        shape=(side, side),
        wrap=False,
        logF=True,
        initialize='sample',
        minError=0.02,
    )
    tc = assign(flux, errors, som.weights, metric)
    vc = assign(valid_flux, valid_errors, som.weights, metric)
    print(
        f'NoiseSOM side={side} seed={seed}: finished in {time.monotonic() - start:.1f}s', flush=True
    )
    return som.weights, tc, vc


def cell_pdfs(cells, z, weights, bandwidth, alpha, metric, prototype_errors):
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
        fallback[empty] = filled[
            assign(
                weights[empty],
                np.broadcast_to(prototype_errors, weights[empty].shape),
                weights[filled],
                metric,
            )
        ]
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
    p.add_argument('--output', type=Path, default=Path('results/noisesom_comparison'))
    p.add_argument(
        '--seeds', type=int, nargs='+', default=[20261024, 20261025, 20261026, 20261027, 20261028]
    )
    p.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 32])
    p.add_argument('--iterations', type=int, default=60000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--noise-som-source', type=Path, default=Path('data/sompz_noise/NoiseSOM.py'))
    p.add_argument('--flow-results', type=Path, default=Path('results/som_comparison'))
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
    # Flow fits are unchanged: verify and reuse their exact saved predictions.
    flow_manifest = args.flow_results / 'run.json'
    flow_run = json.loads(flow_manifest.read_text()) if flow_manifest.exists() else None
    with args.data.open('rb') as f:
        parent_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    if flow_run is not None and flow_run['parent_sha256'] != parent_hash:
        raise ValueError('Cached flow used a different parent')
    module = load_noise_som(args.noise_som_source)
    metric = module.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)
    flux, errors = data['flux'].astype(float), data['fluxerr'].astype(float)
    prototype_errors = np.median(errors[train], axis=0)
    grid = np.linspace(0, 10, 1001)
    run = dict(
        seeds=args.seeds,
        iterations=args.iterations,
        sample_counts=[len(t) for t in [train, valid, test]],
        inputs='identical fixed degraded ugrizy fluxes/errors; noise-aware flux SOM vs unchanged 12-feature flow; no augmentation or deep inputs',
        model='published NoiseSOM/AsinhMetric, cell PDFs; not full deep-to-wide SOMPZ',
        noise_source=dict(
            url=NOISE_URL + '/sompz/NoiseSOM.py', commit=NOISE_COMMIT, sha256=NOISE_SHA256
        ),
        noise_settings=dict(
            lnScaleSigma=0.4, lnScaleStep=0.03, minError=0.02, sigma=[30.0, 1.0], wrap=False
        ),
        compatibility='np.product -> np.prod; omit unused upstream plotting imports',
        flow_results=str(args.flow_results.resolve()),
        selection='Validation NLL; same 4096-object validation subset as seed-0 flow',
        trials=[],
        runs=[],
    )
    with args.data.open('rb') as f:
        run['parent_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    # Reproduce the first flow seed's validation subset; never optimize on test labels.
    vi = np.random.default_rng(args.seeds[0]).choice(valid, min(len(valid), 4096), replace=False)
    best = None

    # Independent maps can run concurrently. All workers receive training
    # photometry plus validation photometry, never validation/test redshifts.
    def job(side, seed):
        return (
            side,
            seed,
            args.iterations,
            args.noise_som_source.resolve(),
            flux[train],
            errors[train],
            flux[vi],
            errors[vi],
        )

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        candidates = list(
            pool.map(train_noise_map, [job(side, args.seeds[0]) for side in args.sizes])
        )
        for side, (weights, tc, vc) in zip(args.sizes, candidates):
            for bandwidth in [0.005, 0.01, 0.02, 0.04, 0.08]:
                for alpha in [0.0, 5.0, 20.0]:
                    sampler, count, fallback = cell_pdfs(
                        tc,
                        data['zspec'][train],
                        weights,
                        bandwidth,
                        alpha,
                        metric,
                        prototype_errors,
                    )
                    val = float(
                        -np.log(
                            np.maximum(density_at(sampler, data['zspec'][vi], vc), 1e-12)
                        ).mean()
                    )
                    config = dict(side=side, bandwidth=bandwidth, alpha=alpha, validation_nll=val)
                    run['trials'].append(config)
                    if best is None or val < best[0]:
                        best = (val, weights.copy(), tc.copy(), config)
            print(f'NoiseSOM tuning: side={side} best NLL={best[0]:.4f}', flush=True)
        run['selected'] = best[3]
        remaining = pool.map(
            train_noise_map, [job(run['selected']['side'], seed) for seed in args.seeds[1:]]
        )
        maps = [(best[1], best[2], None), *remaining]

    for seed, (weights, tc, _) in zip(args.seeds, maps):
        start = time.monotonic()
        sc = assign(flux[test], errors[test], weights, metric)
        config = run['selected']
        sampler, count, fallback = cell_pdfs(
            tc,
            data['zspec'][train],
            weights,
            config['bandwidth'],
            config['alpha'],
            metric,
            prototype_errors,
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
        if flow_run is not None:
            # Check IDs, raw photometry, and preprocessing before reusing a flow.
            cached = args.flow_results / f'predictions_{seed}.npz'
            checkpoint = torch.load(args.flow_results / f'specz_{seed}.pt', weights_only=False)
            if not np.array_equal(checkpoint['feature_mean'], mean) or not np.array_equal(
                checkpoint['feature_scale'], scale
            ):
                raise ValueError('Cached flow preprocessing differs')
            with np.load(cached) as c:
                for key in ['id', 'flux', 'fluxerr', 'zspec', 'grid']:
                    if not np.array_equal(c[key], pred[key]):
                        raise ValueError(f'Cached flow mismatch: {key}')
                for key in ['flow_quantiles', 'flow_pit', 'flow_nll', 'flow_cdf']:
                    pred[key] = c[key]
            fq, fpit, fnll, fcdf = [
                pred[key] for key in ['flow_quantiles', 'flow_pit', 'flow_nll', 'flow_cdf']
            ]
            info = next(row['flow_training'] for row in flow_run['runs'] if row['seed'] == seed)
        else:
            # Fresh runs need no historical cache: reproduce the fixed-photometry
            # flow control, with the same architecture and no noise augmentation.
            fit_data = dict(data, native_flux=data['flux'], sigma_added=np.zeros_like(data['flux']))
            fit_args = argparse.Namespace(
                seed=seed, epochs=200, max_steps=20000, output=args.output
            )
            model, info = train_specz(fit_data, mean, scale, fit_args)
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
                som_evaluation_seconds=som_seconds,
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
