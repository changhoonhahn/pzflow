"""Create the standalone figure notebook and run its plotting cell as a script."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[2]
CODE = r'''
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p/'paper/data/predictions.npz').exists())
OUT = ROOT / 'results/coverage_prior_correction'
with np.load(OUT/'diagnostics.npz') as f:
    d = {k:f[k] for k in f.files}
report = json.loads((OUT/'summary.json').read_text())
assert len(d['id']) == report['n_test'] == 5083
alpha = np.unique(np.r_[np.linspace(0,1,201), .68, .95])
import sys
sys.path.insert(0,str(ROOT/'src'))
from pzflow.project_plot_style import set_style
set_style()
fig, ax = plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
base = 'h_0.030'
ax[0].plot(d['z_density'],d['parent_'+base]/(1+d['z_density']),
           color='0.45',label='Parent: stacked COSMOS PDFs')
ax[0].plot(d['z_density'],d['spec_'+base]/(1+d['z_density']),
           color='C0',label='Secure spectroscopy')
ax[0].set(xlim=(0,3.5),xlabel='Redshift',ylabel='Normalized density per unit z',
          title='Prior estimates: separate calibration sky cells')
ax[0].legend(fontsize=8.5)
colors = {'original':'C1',base:'C0'}
labels = {'original':'Original LePhare',base:'After n(z) correction'}
for h in report['bandwidth_sensitivity_log1pz']:
    k = f'h_{h:.3f}'
    if k == base: continue
    pit = d['pit_'+k]
    cov = (abs(2*pit[:,None]-1)<=alpha[None,:]).mean(axis=0)
    cov[0]=0;cov[-1]=1
    ax[1].plot(alpha,cov,color='0.6',alpha=.6,lw=1,
               label='Smoothing sensitivity' if h==.015 else None)
for k in ['original',base]:
    pit = d['pit_'+k]
    cov = (abs(2*pit[:,None]-1)<=alpha[None,:]).mean(axis=0)
    cov[0]=0;cov[-1]=1
    ax[1].plot(alpha,cov,color=colors[k],lw=2,label=labels[k],
               ls='--' if k=='original' else '-')
    ax[2].hist(pit,bins=np.linspace(0,1,21),density=True,histtype='step',
               color=colors[k],lw=2,label=labels[k],
               ls='--' if k=='original' else '-')
ax[1].plot([0,1],[0,1],color='black',ls=':',lw=1,label='Nominal coverage')
ax[1].set(xlim=(0,1),ylim=(0,1),xlabel='Nominal interval probability',
          ylabel='Empirical coverage',title='Equal-tailed interval coverage')
ax[1].legend(fontsize=8.5)
ax[2].axhline(1,color='black',ls=':',lw=1)
ax[2].set(xlim=(0,1),xlabel='PIT = posterior CDF at spectroscopic z',
          ylabel='Probability density',title='PIT: uniform density is one')
ax[2].legend(fontsize=8.5)
for a in ax:a.grid(alpha=.15)
fig.suptitle('COSMOS2020 LePhare: redshift-prior correction on 5,083 held-out galaxies',
             fontsize=14,fontweight='bold')
fig.supxlabel('Correction fitted using 135,604 parent PDFs and 27,407 secure redshifts in separate sky cells.\n'
              'Baseline smoothing: 0.03 in log(1+z); sensitivity: 0.015-0.10. Redshift-only selection approximation.',
              fontsize=9)
fig.savefig(OUT/'coverage_prior_comparison.png',dpi=180)
plt.show()
'''

def main():
    nb = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('''# COSMOS2020 coverage after a redshift-prior correction

Compare original LePhare PDFs with PDFs multiplied by an independently estimated
`n_spec(z)/n_parent(z)` ratio and renormalized. Both densities were estimated in
the existing spatial training/validation regions; all 5,083 scored galaxies are
in the disjoint test regions. No test labels were used to fit the correction.

Inputs are saved in `results/coverage_prior_correction/diagnostics.npz` and
`summary.json`. Regenerate them with `bin/coverage_prior_correction.py`.
The parent estimate stacks normalized PDFs; it is not an independently measured
true redshift distribution. Smoothing uses a Gaussian kernel in log(1+z), with
0.1% uniform regularization. The baseline width was fixed at 0.03 before seeing
test results. Other widths are sensitivity checks, not tuned alternatives.

This correction assumes selection changes only the redshift prior. It cannot
restore absent PDF modes or address selection on galaxy properties at fixed z.
Reported bootstrap intervals hold the density ratio fixed and omit its estimation
uncertainty and spatial covariance. The plotting cell exports the PNG directly.
'''),nbformat.v4.new_code_cell(CODE)])
    nb.metadata.kernelspec = dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(nb,ROOT/'nb/figure_prior_correction.ipynb')
    exec(compile(CODE,'figure_prior_correction.ipynb','exec'),{})

if __name__ == '__main__':main()
