# pzflow
Probabilistically transfer COSMOS2020 LePhare redshift posteriors to LSST-like broadband photometry using normalizing flows.

## Install

Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
and Git first. From a terminal on macOS or Linux:

```bash
git clone https://github.com/changhoonhahn/pzflow.git
cd pzflow
conda env create -f environment.yml
conda activate pzflow
python -m pip check
```

`environment.yml` creates a Python 3.12 Conda environment and installs this checkout
in editable mode. Conda manages Python and pip; pip installs the project's pinned
scientific packages from the single `requirements.txt` through `setup.py`.
There is no second dependency list to keep synchronized. Run these commands from
the repository root so the editable install can find the project.

If you prefer to create the environment manually, or already have a suitable
Conda environment:

```bash
conda create -n pzflow --override-channels -c conda-forge python=3.12 pip
conda activate pzflow
python -m pip install -e .
python -m pip check
```

Skip `conda create` when using an existing Python 3.12 environment. Always use
`python -m pip` from the activated environment, without `--user`. Keep the checkout
and Conda environment on persistent storage, not under `/tmp`. On later logins,
return to the checkout and run `conda activate pzflow`. No `.venv` is required.
After pulling code or dependency updates, run `python -m pip install -e .` again
inside that environment. Recreate a dedicated environment if you need to undo
unrelated package changes.

The distribution is named `pzflow-cosmos`; its Python import is `pzflow`. Installing
an unrelated PyPI package named `pzflow` does not install this project.
The supplied training and comparison scripts use the CPU; no GPU is required.
For a CPU-only PyTorch build on Linux, use the manual Conda setup above and
install the matching pinned version using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/) before
`python -m pip install -e .`.

Check the installation without downloading any catalogs:

```bash
python -c "import pzflow, numpy, scipy, torch, nflows, astropy; print('Imports OK')"
python bin/train_flows.py --help
python bin/compare_som.py --help
```

Installation was verified in a clean Conda 25.7 environment with Python 3.12.16
on macOS ARM64, including editable, source-archive and wheel installs.

Run all commands below from the **repository root**. Scripts stay in `bin/`;
installation does not create system-wide commands with those names.

## Bring the data to the new machine

The Git repository contains code and notebook source, **not** data, trained models,
results, or compiled figures. For flow training and NoiseSOM, the simplest setup
is to copy the existing **`data/parent.npz`** into the same relative path:

```bash
mkdir -p data
scp YOUR_SOURCE_MACHINE:/path/to/pzflow/data/parent.npz data/parent.npz
```

Replace the source host and path with your own. The current parent is about
**60.5 MB** compressed and contains 162,239 galaxies, with degraded broadband
fluxes/errors, COSMOS PDFs, spectroscopic matches and fixed spatial splits. You do
not need the raw catalogs again when using this parent. Check the copy:

```bash
python -c "import numpy as np; d=np.load('data/parent.npz'); print('Galaxies:', len(d['id'])); print('Fields:', d.files); d.close()"
```

For the exact parent used in the current comparisons, SHA256 is
`1c299b737fcd33cb0496eb6758d3cbe469932b5a12b65c451ae5d6511614c647`.
Use `shasum -a 256 data/parent.npz` on macOS or `sha256sum data/parent.npz` on Linux.
A rebuilt or intentionally updated parent may have different bytes; do not silently
mix its predictions with cached fits from another parent.

To rebuild the parent instead, restore the photometry extract, full LePhare PDFs,
and spectroscopic catalog to the locations documented in
[bin/data/README.md](bin/data/README.md), then run:

```bash
python bin/lsst_like.py
python bin/match_cosmos.py
```

The provenance notes include public sources, selection cuts, hashes, and the
limitations of reconstructing the original TAP extract. A clone alone is not a
complete copy of the analysis data.

## Train and compare

Train the COSMOS-PDF and spec-z flows:

```bash
python bin/train_flows.py
```

Outputs go to local `results/flows/`. For a short pipeline check instead of a
scientific fit:

```bash
python bin/train_flows.py --epochs 1 --max-steps 20 --output results/flow_smoke
```

Run the noise-aware broadband SOM versus spec-z flow comparison:

```bash
python bin/compare_som.py
```

This selects SOM settings on validation data and evaluates five seeds. It downloads
the commit- and hash-pinned DES NoiseSOM source and MIT license into
`data/sompz_noise/` on first use, so that first run needs internet access. To run
offline, copy that directory from the original machine too. This is a single-SOM
broadband comparison, not the complete deep-to-wide SOMPZ pipeline.

Outputs go to `results/noisesom_comparison/`. Existing fixed-photometry flow fits
under `results/som_comparison/` are verified and reused if available; otherwise
the script trains them. The augmented fits in `results/flows/` are a different
experiment and should not be supplied as this comparison's flow cache.
Use `--workers 1` to reduce concurrent SOM workers. Choose a new `--output` directory
to repeat an experiment; completed runs are protected against overwriting.

EAZY is optional to run and needs additional inputs: its public template/filter
resources plus the COSMOS reddening extract; R-prior mode also needs the
aperture-to-total correction extract. Restore the files listed in
[the EAZY setup notes](bin/data/README.md#optional-eazy-broadband-baseline), then run:

```bash
python bin/run_eazy.py
python bin/run_eazy.py --r-prior
```

## Notebooks and paper

Register this environment as a notebook kernel:

```bash
python -m ipykernel install --user --name pzflow --display-name "Python (pzflow)"
```

Activate `pzflow` before registering or starting the kernel.
Open `nb/comparison.ipynb` for the COSMOS-PDF/spec-z flow results or
`nb/som_comparison.ipynb` for NoiseSOM, and select **Python (pzflow)**. Use an
existing notebook frontend, or install JupyterLab separately with
`python -m pip install jupyterlab`. These comparison notebooks read saved outputs.
For a step-by-step training and prediction demo, open `nb/noisesom_demo.ipynb`.
It trains one NoiseSOM directly from `data/parent.npz`, builds cell redshift PDFs,
evaluates the test sample, and shows how to reload the saved predictor.
The matching flow demos are `nb/broadband_flow_demo.ipynb` (spec-z labels) and
`nb/bootstrapped_flow_demo.ipynb` (full COSMOS-PDF labels). They show the training
loop, predict from test fluxes/errors, evaluate against spectra, and reload models.
Save executed copies under `results/notebooks_executed/`.

The supplied plotting style uses LaTeX, so figure generation requires a system
LaTeX installation (for example, TeX Live or MacTeX) with `latex` and `dvipng`
available on `PATH`. Model training and inference do not require LaTeX.
Building the paper additionally requires `latexmk`, pdfLaTeX and BibTeX:
`make -C tex`, after restoring/regenerating its figure inputs. The paper's frozen
main-text experiments and the current pipeline are distinct; see the provenance
notes before replacing manuscript figure inputs.
