COSMOS2020 POSTERIOR TRANSFER: ApJL-style working draft

Main manuscript: main.tex. Compiled PDF: main.pdf.
Delivery PDF is also copied to output/pdf/cosmos_posterior_transfer_apjl.pdf
at the project root.

Author names and affiliations are intentionally unset, as requested.
Add them to authors.tex when ready. No author names, institutional affiliations,
funding statements or acknowledgments were inherited from the supplied draft.
The new text describes the completed experiments; it does not claim actual
Rubin photometry, LSST selection, globally calibrated posteriors, or optimized
template-code performance.

REFERENCE USE
The user-supplied draft0.zip was used for the AASTeX class, modular section
structure, and compact bibliography style. Its unrelated dust experiment,
results, figures and scientific claims were not reused. The extracted reference
is kept locally under paper/reference/draft0 and is excluded from the delivery
archive. The supplied Makefile was inspected as reference and not executed.
The included aasjournal.bst is from the official AASJournals/AASTeX60 repository.

MANUSCRIPT CONTENTS
  main.tex          title, abstract, data/software note and section includes
  authors.tex       unset author metadata
  intro.tex         motivation and scope
  methods.tex       COSMOS2020, LSST-like photometry, spectroscopy and benchmarks
  bootstrapping.tex probabilistic transfer framework and flow training
  results.tex       accuracy, calibration and magnitude-dependent results
  appendix.tex      magnitude distributions and appendix figure
  discussion.tex    interpretation, limitations and LSST selection comparison
  references.bib    verified bibliographic sources
  numbers.tex       numerical macros generated from saved results
  table_results.tex generated four-method comparison table
  aastex63.cls      class supplied with the reference draft (license in header)
  aasjournal.bst    official AAS bibliography style

The working draft uses two-column AASTeX with line numbers, four main figures, one appendix figure and
one table. It is a draft for scientific review, not a statement of journal
acceptance or completion of all submission metadata.

BUILD THE PAPER
With a TeX installation providing latexmk, pdfLaTeX and BibTeX:
  make -C paper
or, from the paper directory:
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
The supplied main.bbl permits direct pdfLaTeX compilation when BibTeX is not
being rerun. The class and bibliography style are included locally.

FIGURE NOTEBOOKS (one independently executable notebook per figure)
  nb/figure_01_selection.ipynb
    Appendix magnitude distributions (former Figure 1, left panel).
  nb/figure_02_accuracy.ipynb
    Four same-object spec-z versus posterior-median comparisons.
  nb/figure_03_calibration.ipynb
    Equal-tailed coverage and bootstrap band (former Figure 3, left panel).
  nb/figure_04_magnitude_dependence.ipynb
    Magnitude-binned scatter with bootstrap intervals (former Figure 4, left panel).

Each contains the actual plotting cells, explanations, assertions, saved
numerical outputs and an inline figure. It exports matching PDF and PNG files
to paper/figures. It does not call an old plotting script or copy a prebuilt
figure. Shared data/style helpers live in src/pzflow/paper_utils.py.

RUN ALL FIGURES
From the project/package root, create an environment if desired and install:
  python -m pip install -r requirements/notebooks.txt
Then open any notebook in that environment and Run All, or execute all four:
  python bin/execute_paper_notebooks.py
The execution script creates a temporary local kernelspec under tmp/jupyter;
no permanent Jupyter kernel registration is needed. Notebook computation needs
local Jupyter sockets. The execution itself needs no network or model training.
Run make -C paper afterward to incorporate changed figure files.

PORTABLE FIGURE INPUTS
  paper/data/selection.npz     IDs, native aperture i, split, cell, spec-z
  paper/data/predictions.npz   same-test IDs, redshifts, quantiles and PITs
  paper/data/metrics.json      analysis metrics, settings and teacher audit
  paper/data/provenance.json  original relative paths and source SHA256 hashes

All necessary figure inputs are included, totaling only a few MB. The 1.7-GB
teacher-PDF release, raw photometry and model checkpoints are not needed to
regenerate the paper figures. The snapshots represent the revised full census,
not the superseded oversampled experiment.

In the full original workspace only, refresh these snapshots/macros/table from
the previously completed analysis using:
  python bin/build_apjl_assets.py
To re-author unexecuted notebooks from their source generator:
  python bin/create_paper_notebooks.py
This last command intentionally clears saved outputs; execute them afterward.
Ordinary figure edits can be made directly in the notebooks.

VALIDATION
  paper/notebook_validation.json  successful fresh-kernel notebook execution
  paper/paper_validation.json     PDF, numbers, references and data checks
  paper/word_count.txt            texcount output
  paper/package_manifest.json    delivered-file checksums

Scientific limits are part of the paper: selected and brighter spectroscopy,
one training seed, bootstrap omitting spatial covariance/training variance,
teacher dependence and finite PDF grid, untested sufficiency and population
shift, untuned EAZY calibration and approximate R-prior mapping, and native
filters with an analytic noise-degradation kernel.

PLOTTING STYLE (updated 2026-11-19)
All figures follow the SpenderQ methods-paper style. See docs/PLOTTING_STYLE.md.
Actual LaTeX rendering requires latex and dvipng on PATH when regenerating figures.
The additional three executed prior-correction notebooks and compact inputs are
in results/coverage_prior_correction/. Their runner is execute_paper_notebooks.py
with --diagnostics. The four manuscript notebooks remain in nb/.

WORKFLOW DIAGRAM
  python bin/plot_bootstrap_workflow.py
  Reproduces the vector training diagram in Section 3.
