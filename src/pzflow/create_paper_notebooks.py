"""Author four separate, readable figure notebooks. Execute with execute_paper_notebooks.py."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]
SETUP=r'''
%matplotlib inline
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'paper/data/predictions.npz').exists())
sys.path.insert(0, str(ROOT / 'src'))
from pzflow.paper_utils import (COLORS, LABELS, set_style, load_predictions,
                         load_metrics, summary, bootstrap, save_figure)
set_style()
'''

def write(name,title,description,cells):
    nb=nbf.v4.new_notebook()
    nb.metadata.update(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                       language_info=dict(name='python',version='3.12'))
    nb.cells=[nbf.v4.new_markdown_cell(f'# {title}\n\n{description}\n\n'
        'Run all cells in a Python environment with `requirements/notebooks.txt` installed. '
        'The plotting style follows SpenderQ methods_paper and requires latex and dvipng. Inputs are frozen, documented snapshots in `paper/data/`; no training or network access is required. '
        'The final cell writes both PDF and PNG versions into `paper/figures/`.'),
        nbf.v4.new_code_cell(textwrap.dedent(SETUP).strip())]
    for kind,source in cells:
        maker=nbf.v4.new_code_cell if kind=='code' else nbf.v4.new_markdown_cell
        nb.cells.append(maker(textwrap.dedent(source).strip()))
    nbf.write(nb,ROOT/'nb'/name)

def main():
    write('figure_01_selection.ipynb','Appendix: Magnitude distributions',
          'The full photometric census is selected independently of spectroscopy. '
          'This notebook shows the selection difference that remains in the scored spectroscopic subset.',[
        ('code',r'''
        with np.load(ROOT / 'paper/data/selection.npz') as f:
            d = {k: f[k] for k in f.files}
        imag, split = d['imag'], d['split']
        has_spec = np.isfinite(d['zspec'])
        assert len(np.unique(d['id'])) == len(imag) == 162239
        for s in range(3):
            for t in range(s):
                assert not np.intersect1d(d['cell'][split == s], d['cell'][split == t]).size
        for label, mask in [('Full training', split == 0),
                            ('Spec training', (split == 0) & has_spec),
                            ('Photometric test', split == 2),
                            ('Spec test', (split == 2) & has_spec)]:
            print(f'{label}: N={mask.sum():,}; median i={np.median(imag[mask]):.3f}')
        '''),
        ('markdown','The magnitude is native HSC $i$ in a 2-arcsec aperture, before adding forecast noise. The spectroscopic samples preferentially contain brighter galaxies.'),
        ('code',r'''
        fig, ax = plt.subplots(figsize=(3.5, 2.8), layout='constrained')
        bins = np.linspace(18, 25, 36)
        groups = [(split == 0, 'Photometric training', COLORS['cosmos_pdf_flow']),
                  ((split == 0) & has_spec, 'Broadband-flow training', COLORS['broadband_spec_flow']),
                  ((split == 2) & has_spec, 'Spectroscopic test', '#333333')]
        for mask, label, color in groups:
            vals = imag[mask]
            ax.hist(vals, bins=bins, weights=np.full(len(vals), 1 / len(vals) / .2),
                    histtype='step', lw=1.35, color=color, label=f'{label} ({len(vals):,})')
        ax.set(xlim=(18, 25), xlabel=r'HSC $i_{\mathrm{2\,arcsec}}$ [AB]', ylabel='Fraction per magnitude')
        ax.legend(loc='upper left', frameon=False, fontsize=6.3)
        save_figure(fig, ROOT, 'fig01_selection')
        plt.show()
        plt.close(fig)
        ''')])
    write('figure_02_accuracy.ipynb','Figure 1: Point-estimate accuracy',
          'Compare the bootstrapped and broadband flows, six-band EAZY with its supplied R prior, '
          'and released full-COSMOS LePhare on exactly the same test IDs. LePhare sees more photometric information.',[
        ('code',r'''
        from matplotlib.colors import LogNorm
        d = load_predictions(ROOT)
        z = d['zspec']
        names = ['cosmos_pdf_flow', 'broadband_spec_flow', 'eazy_rprior', 'lephare_full_cosmos']
        titles = ['Bootstrapped flow: six bands', 'Broadband flow: six bands', 'EAZY: six bands, R prior', 'LePhare: full COSMOS']
        for name in names:
            print(name, summary(z, d[f'{name}_quantiles']))
        '''),
        ('markdown','The scatter panels zoom to $0<z<5$. The annotations and manuscript table include **all** 5,083 objects. All displayed point estimates are posterior medians; the same normalized-residual definition is used for every method.'),
        ('code',r'''
        fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.7), layout='constrained', sharex=True, sharey=True)
        ax = axes.ravel()
        norm = LogNorm(vmin=1, vmax=200)
        for a, name, title, letter in zip(ax, names, titles, ['(a)', '(b)', '(c)', '(d)']):
            q = d[f'{name}_quantiles']
            h = a.hexbin(z, q[:, 2], gridsize=75, extent=(0, 5, 0, 5),
                         mincnt=1, norm=norm, cmap='viridis', linewidths=0, rasterized=True)
            a.plot([0, 5], [0, 5], color='#c34828', lw=.8)
            a.set(xlim=(0, 5), ylim=(0, 5), xlabel=r'$z_{\mathrm{spec}}$', title=title, aspect='equal')
            m = summary(z, q)
            text = (rf'$\sigma_{{\mathrm{{NMAD}}}}={m["sigma_nmad"]:.4f}$' + '\n' +
                    rf'$\eta={100 * m["outlier_fraction"]:.2f}\%$')
            a.text(.05, .95, text, transform=a.transAxes, va='top', fontsize=7,
                   bbox=dict(facecolor='white', edgecolor='none', alpha=.88, pad=2))
            a.text(.95, .05, letter, transform=a.transAxes, ha='right', color='#222222')
        for a in axes[:, 0]:
            a.set_ylabel(r'Posterior median $z_{50}$')
        cb = fig.colorbar(h, ax=ax, fraction=.017, pad=.015, shrink=.83)
        cb.set_label('Galaxies per hexagon', fontsize=7)
        save_figure(fig, ROOT, 'fig02_accuracy')
        plt.show()
        plt.close(fig)
        ''')])
    write('figure_03_calibration.ipynb','Figure 2: Predictive interval coverage',
          'Recompute coverage directly from saved probability integral transforms (PITs). '
          'Coverage is tested on the spectroscopic overlap and does not prove population-wide or conditional calibration.',[
        ('code',r'''
        d = load_predictions(ROOT)
        names = ['cosmos_pdf_flow', 'broadband_spec_flow', 'eazy_rprior', 'lephare_full_cosmos']
        alpha = np.linspace(0, 1, 101)
        curves = {}
        for name in names:
            pit = d[f'{name}_pit']
            ranks = np.abs(2 * pit - 1)
            curve = np.array([(ranks <= a).mean() for a in alpha])
            # Exact endpoints can be affected by clipped or rounded CDF tails.
            curve[0], curve[-1] = 0., 1.
            curves[name] = curve
            print(name, 'C68/C95:', [(ranks <= a).mean() for a in [.68, .95]])
        '''),
        ('markdown','The shaded band is a 95% pointwise galaxy-bootstrap interval for the bootstrapped flow (300 resamples, fixed seed). It excludes training-seed variance, spatial covariance and spectroscopic-selection uncertainty.'),
        ('code',r'''
        rng = np.random.default_rng(20261117)
        ranks = np.abs(2 * d['cosmos_pdf_flow_pit'] - 1)
        order = np.searchsorted(np.sort(ranks), alpha, side='right')
        draws = np.empty((300, len(alpha)))
        for b in range(len(draws)):
            sample = np.sort(ranks[rng.integers(0, len(ranks), len(ranks))])
            draws[b] = np.searchsorted(sample, alpha, side='right') / len(sample)
        lower, upper = np.quantile(draws, [.025, .975], axis=0)
        '''),
        ('code',r'''
        fig, ax = plt.subplots(figsize=(3.5, 2.55), layout='constrained')
        ax.fill_between(alpha, lower, upper, color=COLORS['cosmos_pdf_flow'], alpha=.15, lw=0)
        for name in names:
            ax.plot(alpha, curves[name], color=COLORS[name], label=LABELS[name])
        ax.plot([0, 1], [0, 1], color='black', ls=':', lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'Nominal credibility $\alpha$', ylabel=r'Empirical coverage $C_\alpha$')
        ax.legend(loc='upper left', frameon=False, fontsize=6.5)
        save_figure(fig, ROOT, 'fig03_calibration')
        plt.show()
        plt.close(fig)
        ''')])
    write('figure_04_magnitude_dependence.ipynb','Figure 3: Magnitude-dependent scatter',
          'Compute scatter in bins of native, pre-noise HSC aperture magnitude. '
          'These diagnostics describe the selected spectroscopic test subset.',[
        ('code',r'''
        d = load_predictions(ROOT)
        names = ['cosmos_pdf_flow', 'broadband_spec_flow', 'eazy_rprior', 'lephare_full_cosmos']
        edges = [-np.inf, 22.5, 24., 24.5, 25.]
        masks = [(d['imag'] >= lo) & (d['imag'] < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        counts = [int(m.sum()) for m in masks]
        effective_magnitudes = np.array([np.median(d['imag'][m]) for m in masks])
        print('Spec test counts by magnitude:', counts)
        print('Effective magnitudes (median of each bin):', effective_magnitudes)
        result = {}
        for name in names:
            rows = []
            for b, mask in enumerate(masks):
                z, q = d['zspec'][mask], d[f'{name}_quantiles'][mask]
                rows.append((summary(z, q), bootstrap(z, q, seed=20261117 + b, n_boot=200)))
            result[name] = rows
            print(name, [round(row[0]['sigma_nmad'], 4) for row in rows])
        '''),
        ('markdown','Shaded bands join the 95% pointwise galaxy-bootstrap interval endpoints, using fill_between with alpha=0.5 and the same resampling seed for each method within a bin. Text labels give the number of spectroscopic test galaxies in each bin. The x-axis is numerical HSC i magnitude, with each point at the median magnitude of the test galaxies in its bin. Every method uses exactly the same x-coordinates, with no horizontal offsets. The bins describe only the available spectroscopic test galaxies.'),
        ('code',r'''
        fig, ax = plt.subplots(figsize=(3.5, 2.55), layout='constrained')
        positions = effective_magnitudes
        markers = ['o', 's', '^', 'D']
        for index, name in enumerate(names):
            x = positions
            y = np.array([r[0]['sigma_nmad'] for r in result[name]])
            ci = np.array([r[1]['sigma_nmad'] for r in result[name]])
            ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=.5,
                            color=COLORS[name], linewidth=0)
            ax.plot(x, y, marker=markers[index], ms=3, lw=.9,
                    color=COLORS[name], label=LABELS[name])
        upper_envelope = np.max(np.array([[r[1]['sigma_nmad'][1] for r in result[name]]
                                         for name in names]), axis=0)
        for x, upper, count in zip(positions, upper_envelope, counts):
            ax.annotate(f'$N={count:,}$', xy=(x, upper), xytext=(0, 8),
                        textcoords='offset points', ha='center', va='bottom')
        ax.set_xlim(21.3, 25.)
        ax.set_xticks(np.arange(21.5, 25.01, .5))
        ax.set_xlabel(r'Native HSC $i_{\mathrm{2\,arcsec}}$ [AB]')
        ax.set(ylim=(0, upper_envelope.max()*1.16), ylabel=r'$\sigma_{\mathrm{NMAD}}$')
        ax.legend(frameon=False, fontsize=6.2, loc='upper left')
        save_figure(fig, ROOT, 'fig04_magnitude_dependence')
        plt.show()
        plt.close(fig)
        ''')])
    print('Wrote four figure notebooks with explicit plotting cells.')

if __name__=='__main__':main()
