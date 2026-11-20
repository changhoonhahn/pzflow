#!/usr/bin/env python3
'''


Train COSMOS-PDF and spec-z flows from the same catalog constructed using
`match_cosmos.py`.

COMOS-PDF: bootstrap posterior from narrow-band photometry (y) 
    p(z | x) = int p(z | y) p(y | x) dy

spec-z Flow: Traing flow directly on spectroscopic redshifts. 
    p(z | x) 


'''
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from pzflow.utils import PDFSampler, coverage


def make_flow(context_features=12):
    # Two conditional spline transforms map redshift to a standard normal.
    # The context comprises six broadband flux features and six error features.
    transforms = [
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=1,
            hidden_features=64,
            context_features=context_features,
            num_bins=16,
            tails='linear',
            tail_bound=5.0,
            num_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=False,
        )
        for _ in range(2)
    ]
    return Flow(CompositeTransform(transforms), StandardNormal([1]))


def z_to_t(z):
    # Map the finite COSMOS support (0, 10) to the real line with a scaled logit.
    # Clipping avoids divergent logits at the grid endpoints.
    z = torch.clamp(z, 1e-5, 10 - 1e-5)
    return (torch.log(z / (10 - z)) + 2.2) / 1.1


def t_to_z(t):
    # Invert the redshift logit to return predictions in physical redshift units.
    return 10 * torch.sigmoid(t * 1.1 - 2.2)


def raw_features(flux, err):
    # Asinh magnitudes remain defined for negative fluxes and nondetections.
    # Log errors let the flow distinguish measurements with different precision.
    soft = 0.02  # microJy; fixed and independent of labels
    lupt = 23.9 - (2.5 / np.log(10)) * (np.arcsinh(flux / (2 * soft)) + np.log(soft))
    return np.c_[lupt, np.log(np.maximum(err, 1e-7) / soft)].astype(np.float32)


def context(flux, err, mean, scale):
    # Apply training-only feature scaling; clipping limits extreme input values.
    return torch.from_numpy(
        np.clip((raw_features(flux, err) - mean) / scale, -12, 12).astype(np.float32)
    )


@torch.no_grad()
def log_prob_z(model, z, ctx):
    z = torch.as_tensor(z, dtype=torch.float32).reshape(-1, 1)
    w = z_to_t(z)
    # The flow density is in transformed t; add log|dt/dz| to obtain p(z | x).
    jac = np.log(10) - torch.log(z) - torch.log(10 - z) - np.log(1.1)
    return model.log_prob(w, context=ctx) + jac[:, 0]


@torch.no_grad()
def posterior_summary(model, ctx, zspec):
    # Invert standard-normal quantiles rather than estimate intervals from samples.
    probs = torch.tensor([0.025, 0.16, 0.5, 0.84, 0.975])
    noise = torch.distributions.Normal(0.0, 1.0).icdf(probs)
    quant = []
    pit = []
    nll = []
    for start in range(0, len(ctx), 512):
        x = ctx[start : start + 512]
        n = len(x)
        q, _ = model._transform.inverse(
            noise.repeat(n).reshape(-1, 1), context=x.repeat_interleave(5, dim=0)
        )
        quant.append(t_to_z(q).reshape(n, 5).numpy())
        z = torch.tensor(zspec[start : start + 512], dtype=torch.float32).reshape(-1, 1)
        # The base-normal CDF gives the posterior rank (PIT) of the true spec-z.
        u, _ = model._transform(z_to_t(z), context=x)
        pit.append((0.5 * (1 + torch.erf(u[:, 0] / np.sqrt(2)))).numpy())
        nll.append((-log_prob_z(model, z, x)).numpy())
    return np.concatenate(quant), np.concatenate(pit), np.concatenate(nll)


def fit(name, teacher, data, sampler, mean, scale, args):
    '''
    '''


    flux, err, extra = data['native_flux'], data['fluxerr'], data['sigma_added']
    zspec, splits = data['zspec'], data['split']
    epochs, max_steps = args.epochs, args.max_steps
    rng = np.random.default_rng(args.seed)
    train = np.flatnonzero(splits == 0)
    valid = np.flatnonzero(splits == 1)
    # Both models share spatial splits; only the supervised model requires spectra.
    if not teacher:
        train = train[np.isfinite(zspec[train])]
        valid = valid[np.isfinite(zspec[valid])]
    if not len(train) or not len(valid):
        raise ValueError('Empty training or validation sample')

    # initialize the conditional flow
    torch.manual_seed(args.seed)
    model = make_flow()
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    # Fixed parent fluxes and cached labels; no spec-z used for teacher validation.
    vi = rng.choice(valid, min(len(valid), 4096), replace=False)
    vy = data['z_draws'][vi, :4].reshape(-1) if teacher else zspec[vi].astype(np.float32)
    if teacher:
        vi = np.repeat(vi, 4)
    vf = data['flux'][vi]
    vx = context(vf, err[vi], mean, scale)
    vt = z_to_t(torch.from_numpy(vy[:, None]))

    # train with fresh photometric noise and posterior labels
    best = float('inf')
    best_state = None
    bad = 0
    step = 0
    history = []
    start = time.monotonic()
    for epoch in range(epochs):
        model.train()
        rng.shuffle(train)
        losses = []
        for pos in range(0, len(train), 512):
            idx = train[pos : pos + 512]
            # Start from native fluxes, adding only the missing LSST-like variance.
            # Teacher labels are new independent draws from each full COSMOS PDF.
            xflux = flux[idx] + rng.normal(size=(len(idx), 6)).astype(np.float32) * extra[idx]
            x = context(xflux, err[idx], mean, scale)
            z = sampler.draw(idx, rng) if teacher else zspec[idx].astype(np.float32)
            target = z_to_t(torch.from_numpy(z[:, None]))
            opt.zero_grad(set_to_none=True)
            # Monte Carlo maximum likelihood averages over teacher uncertainty.
            # The z-to-t Jacobian is parameter-independent, so it is omitted here.
            loss = -model.log_prob(target, context=x).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss in {name}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
            if step >= max_steps:
                break
        # Fixed validation labels and photometry make epoch losses comparable.
        # Neither model uses the held-out test set for checkpoint selection.
        model.eval()
        with torch.no_grad():
            vals = [
                -model.log_prob(vt[p : p + 1024], context=vx[p : p + 1024]).sum().item()
                for p in range(0, len(vt), 1024)
            ]
            val = sum(vals) / len(vt)
        scheduler.step(val)
        history.append(
            dict(epoch=epoch + 1, step=step, train_loss=float(np.mean(losses)), validation_loss=val)
        )
        print(
            f'{name} epoch={epoch + 1} step={step} train={np.mean(losses):.4f} val={val:.4f}',
            flush=True,
        )
        # Reduce the learning rate on a plateau and stop after eight stale epochs.
        if val < best - 1e-4:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= 8 or step >= max_steps:
            break

    # save the best validation checkpoint
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        dict(
            state_dict=best_state,
            feature_mean=mean,
            feature_scale=scale,
            model='two conditional RQS autoregressive transforms; 16 bins; hidden64; blocks2',
            n_train=len(train),
            n_validation=len(valid),
            seed=args.seed,
        ),
        args.output / f'{name}.pt',
    )
    (args.output / f'{name}_history.json').write_text(json.dumps(history, indent=2) + '\n')
    return model, dict(
        n_train=len(train),
        n_validation=len(valid),
        steps=step,
        epochs=len(history),
        best_validation_loss=best,
        seconds=time.monotonic() - start,
    )


def metrics(z, q, pit, nll):
    # Point estimates use posterior medians; residuals scale as delta-z/(1+z).
    # PIT-based coverage measures equal-tailed intervals, not highest-density sets.
    c68, c95 = coverage(pit)
    r = (q[:, 2] - z) / (1 + z)
    return dict(
        n=len(z),
        bias=float(np.median(r)),
        sigma_nmad=float(1.4826 * np.median(abs(r - np.median(r)))),
        outlier_fraction=float(np.mean(abs(r) > 0.15)),
        rmse_normalized=float(np.sqrt(np.mean(r * r))),
        coverage68=float(c68),
        coverage95=float(c95),
        mean_negative_log_density=float(np.mean(nll)),
        median_width68=float(np.median(q[:, 3] - q[:, 1])),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/parent.npz'))
    p.add_argument('--output', type=Path, default=Path('results/flows'))
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--max-steps', type=int, default=20000)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()

    if args.epochs < 1 or args.max_steps < 1:
        p.error('epochs and max-steps must be positive')
    if (args.output / 'run.json').exists():
        p.error('Output already contains a run; choose a new --output directory')
    torch.set_num_threads(args.threads)

    with np.load(args.data) as c:
        data = {k: c[k] for k in c.files}
    if not json.loads(str(data['metadata'])).get('canonical_parent'):
        raise ValueError('Use the canonical parent from match_cosmos.py')
    if data['grid'][0] != 0 or data['grid'][-1] != 10:
        raise ValueError('This flow transform requires the COSMOS 0 <= z <= 10 grid')
    
    # Estimate preprocessing only on training photometry, shared by both models.
    train = data['split'] == 0
    features = raw_features(data['native_flux'][train], data['fluxerr'][train])
    mean, scale = features.mean(0), np.maximum(features.std(0), 0.1)
    # Score the same secure-spec test subset for both models. Its spectroscopic
    # selection limits conclusions about the full magnitude-limited population.
    test = np.flatnonzero((data['split'] == 2) & data['has_specz'])
    if not len(test):
        raise ValueError('No held-out secure spectroscopy')
    if np.any((data['zspec'][data['has_specz']] <= 0) | (data['zspec'][data['has_specz']] >= 10)):
        raise ValueError('Secure labels must lie inside the model support (0,10)')
    args.output.mkdir(parents=True, exist_ok=True)
    sampler = PDFSampler(data['grid'], data.pop('pdf'))
    predictions = {k: data[k][test] for k in ['id', 'zspec', 'imag', 'flux', 'fluxerr']}
    ctx = context(data['flux'][test], data['fluxerr'][test], mean, scale)
    run = dict(
        parent=str(args.data.resolve()),
        parent_sha256='',
        seed=args.seed,
        epochs_limit=args.epochs,
        steps_limit=args.max_steps,
        comparison='full COSMOS-PDF parent vs secure-spec subset; shared architecture, splits and inference inputs',
        teacher_assumption='unbiased COSMOS PDFs, no prior division',
        photometry='native CFHT/HSC filters with LSST-like noise, not Rubin observations',
        models={},
    )
    with args.data.open('rb') as f:
        run['parent_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    # Fit both models and retain common IDs/inputs for paired comparisons.
    for name, teacher in [('bootstrapped', True), ('specz', False)]:
        model, info = fit(name, teacher, data, sampler, mean, scale, args)
        q, pit, nll = posterior_summary(model, ctx, data['zspec'][test])
        predictions.update({name + '_quantiles': q, name + '_pit': pit, name + '_nll': nll})
        info['metrics'] = metrics(data['zspec'][test], q, pit, nll)
        run['models'][name] = info
    np.savez_compressed(args.output / 'predictions.npz', **predictions)
    (args.output / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    print(json.dumps(run, indent=2), flush=True)


if __name__ == '__main__':
    main()
