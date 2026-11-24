# pzflow
Probabilistically transfer COSMOS2020 LePhare redshift posteriors to LSST-like broadband photometry using normalizing flows.

## Install
```bash
python -m pip install -e .
python -m pip check
```

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

For paper-style comparisons of the three demo outputs and the full COSMOS LePhare
PDFs, run [nb/paper_comparison.ipynb](nb/paper_comparison.ipynb). It aligns the same
spectroscopic test galaxies, plots accuracy/coverage and magnitude-dependent NMAD,
and exports new paper-ready PDFs without replacing the manuscript's frozen figures.
