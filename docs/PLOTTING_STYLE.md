# Plotting style

All project plots share `src/pzflow/project_plot_style.py`, following the style
in [SpenderQ's methods-paper notebooks](https://github.com/galactic-ai/SpenderQ/tree/main/doc/methods_paper),
specifically `validation.ipynb` and `uncertainty.ipynb` (inspected 2026-11-19).

The reference settings are LaTeX-rendered serif text, 1.5-point axes and major
ticks, 5-point major ticks, large tick labels, frameless legends, and standard
Matplotlib colors. We use the same settings, with text sizes and canvas sizes
adapted to the number of panels. Grids are removed. Data ranges, binning,
statistics, and uncertainty calculations are unchanged by styling.

Consistent model colors: bootstrapped/COSMOS full flow C0, spec-trained broadband
flow C1, EAZY R-prior C2, full-COSMOS LePhare C3, EAZY flat-prior C4, and the
matched-object PDF flow C9 where included. Original-versus-reweighted figures
use C1 dashed and C0 solid to distinguish the two states of the same model.

Install `requirements/notebooks.txt` and a TeX distribution supplying `latex`
and `dvipng` on PATH. The helper also discovers a standard MacTeX installation.
This intentionally uses actual LaTeX rendering, as in the reference notebooks.

Run the four manuscript notebooks with `bin/execute_paper_notebooks.py`.
Run the three standalone prior-correction notebooks with the same command and
`--diagnostics`. Their inputs live in `results/coverage_prior_correction/`.
Regenerate the other diagnostic figures from the full workspace's saved
catalogs and predictions with `bin/restyle_all_figures.py`; it does not
train models. Some earlier, superseded figures are retained in
`results/flow_experiment/` and are explicitly labeled as archived experiments.
