# pzflow

Probabilistically transfer COSMOS2020 LePhare redshift posteriors to a
conditional normalizing flow using six broadband fluxes and their errors.
The experiment uses native CFHT-u/HSC-grizy photometry with an LSST-like noise
kernel, rather than measured Rubin photometry. The selected spectroscopic test
sample does not establish population-wide calibration.

## Layout

```text
pzflow/
  bin/                  executable analysis, training, plotting, and packaging scripts
  nb/                   all eight figure notebooks, with executed outputs
  src/pzflow/           shared models, PDF sampling, metrics, plotting, and workflow functions
  requirements/         recorded dependency versions for each workflow
  docs/                 experiment definitions, caveats, and plotting-style notes
  paper/                manuscript, Makefile, figures, and compact figure inputs
  results/              numerical results, figures, compact predictions, and small checkpoints
  data/                 local downloaded catalogs and full teacher PDFs (not in Git)
  output/               local delivery archives (not in Git)
  tmp/                  local caches, build checks, and pre-migration backups (not in Git)
```

`bin/` contains thin Python entry points; the reusable implementations live in
`src/pzflow/`. Existing script names were retained. Notebook code imports the
same shared package. Scripts resolve paths from the project root and can be
called from any working directory.

## Local checkout and installation

```bash
git clone https://github.com/changhoonhahn/pzflow.git ~/project/pzflow
cd ~/project/pzflow
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebooks]'
```

The import package is `pzflow`; the distribution name is `pzflow-cosmos`.
The original dependency versions are retained under `requirements/`. For
template fitting, install `python -m pip install -e '.[eazy]'` as well.
Plot regeneration uses the SpenderQ methods-paper style and requires a TeX
installation with `latex` and `dvipng` on PATH. Building the manuscript also
requires `latexmk`, pdfLaTeX, and BibTeX.

## Reproduce the figures without retraining

The compact inputs required by the eight notebooks are versioned. No full
catalog downloads or model retraining are needed for these commands:

```bash
python bin/execute_paper_notebooks.py --all
make -C paper
python bin/validate_apjl_package.py
```

Run only the manuscript notebooks with no flags, the three prior-correction
notebooks with `--diagnostics`, or the corruption notebook with `--stress`.
For one manuscript notebook, use e.g.
`--notebook figure_04_magnitude_dependence.ipynb`.

| Notebook | Experiment |
| --- | --- |
| `nb/figure_01_selection.ipynb` | Training/test magnitude distributions |
| `nb/figure_02_accuracy.ipynb` | Same-object redshift accuracy comparison |
| `nb/figure_03_calibration.ipynb` | Credible-interval coverage |
| `nb/figure_04_magnitude_dependence.ipynb` | Magnitude-dependent scatter and bootstrap bands |
| `nb/figure_prior_correction.ipynb` | LePhare redshift-prior sensitivity |
| `nb/figure_all_models.ipynb` | Flow, spec-trained baseline, EAZY, and LePhare coverage/PIT |
| `nb/figure_nz_ratio.ipynb` | Parent/spec redshift distributions and correction ratio |
| `nb/figure_catastrophic_pdf_stress.ipynb` | Random catastrophic teacher-PDF contamination |

## Full workflows

These commands require the local catalog/PDF cache described in
[`data/README.md`](data/README.md). Training commands are explicit and are not
run as part of notebook execution, packaging, or tests.

```bash
python bin/plot_cosmos2020_specz.py
python bin/flow_experiment_data.py
python bin/cosmos_lsst_flow.py
python bin/run_eazy_baseline.py --r-prior
python bin/compare_template_baselines.py
python bin/coverage_prior_correction.py
python bin/coverage_prior_all_models.py
python bin/catastrophic_pdf_stress.py
python bin/analyze_catastrophic_pdf_stress.py
```

The corruption sweep resumes completed cases. Regenerating notebook source with
`create_paper_notebooks.py`, the prior plotting scripts, or
`analyze_catastrophic_pdf_stress.py` clears that notebook's executed outputs;
rerun it afterward. `restyle_all_figures.py` regenerates the older diagnostic
figures from local caches, without training.

Details and scientific limitations:
[`docs/FLOW_EXPERIMENT.txt`](docs/FLOW_EXPERIMENT.txt),
[`docs/TEMPLATE_BASELINES.txt`](docs/TEMPLATE_BASELINES.txt),
[`docs/CATASTROPHIC_PDF_STRESS.txt`](docs/CATASTROPHIC_PDF_STRESS.txt), and
[`docs/PLOTTING_STYLE.md`](docs/PLOTTING_STYLE.md).
Historical provenance files retain the original source names; the path mapping
is recorded in `docs/REORGANIZATION.json`.

## Checks and delivery archives

```bash
python bin/test_flow_experiment.py
python bin/validate_repository.py
python bin/package_apjl_draft.py
python bin/package_tex_source.py
python bin/package_catastrophic_pdf_stress.py
```

Archives are written to `output/`. Raw catalogs, bulk intermediate posterior
grids, third-party EAZY resources, temporary files, and the unrelated supplied
reference draft are excluded from Git. They remain available in the original
local research checkout after relocation. The vendored AASTeX class and AAS
bibliography style retain their respective notices.
