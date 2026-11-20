# Data sources and reproducibility

Data, checkpoints, numerical results, rendered figures, and executed notebooks
are **local only**. Nothing under the project-level `data/`, `results/`, or
`output/` directories is versioned. This directory records what was used.
`SHA256SUMS` identifies the cached files by bytes; `sources.json` adds URLs,
releases, columns, array shapes, sizes, and known provenance gaps for 52 files.
Paths in those files are relative to the project root.

## Primary inputs

| Input | Actual source and version | Local location |
| --- | --- | --- |
| Six-band fluxes/errors and selection columns | [IRSA TAP](https://irsa.ipac.caltech.edu/TAP), table `cosmos2020_classic` | `data/cosmos2020_specz/classic_ugrizy.csv` |
| Aperture magnitudes, total-flux correction, and reported 68% limits | Same Classic TAP table | `classic_magnitude_intervals.csv` in that directory |
| r-band photometry for the spec-z/magnitude diagnostic | Same Classic TAP table | `classic_r_photometry.csv` |
| Galactic reddening for EAZY | Classic `ebv_mw` column | `classic_i25_ebv.csv` |
| Teacher redshift PDFs | [Classic R1 v2.1 LePhare PZ](https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/PZ/COSMOS2020_CLASSIC_R1_v2.1_LEPHARE_PZ_p3.fits.gz) | `classic_lephare_complete.fits.gz` |
| Spectroscopic labels and evaluation redshifts | [COSMOS compilation DR1.1 unique catalog](https://github.com/cosmosastro/speczcompilation/tree/main/specz_compilation) | `specz_compilation_COSMOS_DR1.1_unique.fits` plus `_surveys.list` |
| EAZY templates, filter curves, and priors | [eazy-photoz commit f651e94](https://github.com/gbrammer/eazy-photoz/tree/f651e94d91042a84906090a66d2a3fec75504e21) | `data/eazy-photoz-master/` and cached tar archive |
| LSST-like noise prescription | [Rubin key numbers](https://rubinobservatory.org/for-scientists/rubin-101/key-numbers), frozen numeric forecast values below | Parameters in `bin/lsst_like.py`; no Rubin catalog |

### COSMOS photometry

The flux extract has 711,918 clean galaxy rows and the columns listed in
`sources.json`. Fluxes and errors are in microJy, magnitudes are AB, positions
are J2000 degrees, and `APER2` means a **2-arcsec diameter** aperture.
The six bands are **CFHT u and HSC grizy**, not Rubin passbands. Magnitude/interval
and r-band extracts also have 711,918 rows. The reddening extract has 170,172
rows; it is ID-matched where needed and does not define the training selection.

Use the [Classic column definitions](https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos2020c_colDescriptions.html)
and cite [Weaver et al. (2022)](https://doi.org/10.3847/1538-4365/ac3078).
The original TAP request payload, retrieval timestamp, and catalog server
version were not preserved. The CSV hashes are authoritative for this analysis.
`catalog_queries.sql` gives **reconstructed compatible column selections**, not
an asserted recovery of the original transactions or row serialization.
In particular, today's main Classic FITS download is named v2.2 while the
released PZ file is named v2.1; do not silently label the saved TAP CSVs as an
exact v2.1 or v2.2 full-catalog snapshot. Match IDs and check the actual inputs.

### Full LePhare PDFs

The compressed image has shape `(1720701, 1002)` with big-endian float32 data.
Its first row contains the 1,001-point redshift grid, 0 through 10 with spacing
0.01. Subsequent rows contain an explicit Classic ID followed by PDF values.
**Join on that ID; do not assume a row number is the ID.** Normalize each PDF
using trapezoidal integration and use the piecewise-linear density/CDF in
`src/pzflow/utils.py`.
These are full deep multiband posteriors, not fits using only narrow bands.
No Farmer catalog or released full-COSMOS EAZY CZ archive was used.

### Spectroscopy

The actual DR1.1 file used contains **261,975 rows**, not the row count of a
newer compilation. Its SHA256 is recorded. The original upstream Git commit is
unknown; a mutable `main` download URL alone does not guarantee identical data.
See [Khostovan et al.](https://arxiv.org/abs/2503.00120), the compilation's survey
list and README, and `results/cosmos2020_specz/summary.json` for program-level
references and counts. Cite contributing survey papers where appropriate.

The broad audit keeps exactly:

```python
public_or_private == 1
flag in [3, 4, 13, 14]
photoz_type == 0
Id_COS20_Classic > 0
isfinite(specz) & (specz > 0)
isfinite(photoz) & (photoz >= 0)
```

This gives 34,792 unique matched galaxies. No photo-z/spec-z agreement cut is
applied. The unique catalog has already selected the preferred spectrum using
its quality/recency rules. The code uses the supplied Classic ID match, not a
new positional nearest-neighbor match. This heterogeneous collection of
successful spectra is not a magnitude-complete or representative parent sample.

### Historical EAZY resources and preprocessing

The original cached GitHub tarball has SHA256
`111fad2328e974ad601c220a8d3bf0d3562606039f9161f4335c44576024e965`.
Its global PAX header pins commit
`f651e94d91042a84906090a66d2a3fec75504e21`. A commit-named tarball can have a
different archive hash because its top-level directory name differs; compare
the per-resource hashes in `sources.json` as well.

EAZY-py 0.8.7 uses `FILTER.RES.latest`, filter IDs **409, 314, 315, 316, 317,
318** for u,g,r,i,z,y; the 12 `tweak_fsps_QSF_12_v3` templates; the UVISTA
`template_error_10.def` and `lambda.def` files; and, for the prior run,
`prior_R_zmax7.dat`. The F160W prior is a configuration default, not an applied
prior in the flat run. The historical environment is archived in `tmp/three_step_before`; EAZY is not a dependency of the current workflow.

EAZY applies per-object `ebv_mw` with coefficients
`[4.807, 3.690, 2.715, 2.000, 1.515, 1.298]`, multiplying **both** fluxes and
errors by `10**(0.4*EBV*coefficient)`. The R-prior run additionally multiplies
both by `10**(-0.4*total_off2)` to define a total r magnitude. See the
[official aperture-preparation notes](https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/misc/COSMOS2020_prepare_apertures.txt)
and the saved EAZY manifests. The HSC-r/R-prior mapping is approximate. Neither
EAZY setup was tuned on the spectroscopic test labels.

## Selection, splits, and simulated observations

`bin/lsst_like.py` retains the full eligible census: combined footprint flag 0,
galaxy template `lp_type=0`, native HSC-i aperture magnitude `<25`, native i-band
SNR `>=5`, finite six-band fluxes, positive finite native/effective errors,
`abs(flux)<1e8` sentinel rejection, and nonnegative finite nonempty PDFs.
Negative fluxes/nondetections in other bands are retained. The magnitude cut
precedes added noise. There are 162,239 galaxies, with no spec-z oversampling.

Sky cells are `floor(ra/0.05)*100000 + floor(dec/0.05)`, assigned using seed
20261024 and nominal probabilities 70/15/15 before the magnitude cut. The actual
counts are 111,952 train, 23,652 validation, and 26,635 photometric test. The
secure-spec subsets have 22,604, 4,803, and **5,083** objects respectively.
All augmentations of an object remain in its split. Normalization uses training
photometry only. IDs, split labels, and cell labels are preserved in the caches.

Frozen five-sigma forecast depths are `[25.6,26.9,26.9,26.4,25.6,24.8]` AB in
ugrizy. In microJy:

```python
sigma_target = 10**((23.9 - depth)/2.5) / 5
sigma_effective = maximum(sigma_native, sigma_target)
sigma_added = sqrt(maximum(sigma_target**2 - sigma_native**2, 0))
```

Noise is added to the **fluxes**, and effective errors are supplied as inputs.
Training resamples independent Gaussian noise each pass. Validation and test use
the fixed realization generated for **all** selected parent objects with seed
20261025, saved in `data/parent.npz` as `flux` and `fluxerr`. Both flows use the
same test realization. Historical analyses generated noise for the secure-test
subset alone; the new all-parent realization differs, even with the same seed.
These are approximate point-source forecast depths applied to aperture errors,
not a Rubin selection or full instrument simulation.

## Three-step workflow and one parent dataset

From the repository root:

```bash
python -m pip install -e .
python bin/lsst_like.py
python bin/match_cosmos.py
python bin/train_flows.py
```

1. `lsst_like.py` selects the full magnitude-limited population and writes the
   intermediate `data/lsst_like.npz`, with native and degraded fluxes/errors,
   added-noise scales, IDs, positions, native i magnitude, and spatial splits.
2. `match_cosmos.py` joins full COSMOS PDFs and secure spectroscopy by Classic
   ID. It retains galaxies without spectra (NaN spec-z), validates every PDF,
   and generates 32 continuous redshift draws per galaxy. The **single analysis
   parent** is `data/parent.npz`. All intermediate arrays are carried into it;
   additional fields are `grid`, `pdf`, `z_draws`, `zspec`, `has_specz`,
   `spec_flag`, and `spec_survey`. Missing or invalid PDFs stop the script rather
   than silently selecting a different parent. `--pdfs path/to/cache.npz` can
   read an ID-checked `id,grid,pdf` cache instead of the full FITS image.
3. `train_flows.py` reads only that parent and trains exactly two models:
   **bootstrapped**, using all training galaxies and fresh draws from their full
   PDFs each batch; **specz**, using the secure-spec subset of those galaxies.
   Cached draws provide fixed teacher validation labels. Both use the same
   architecture, train/validation/test cells, training-only feature scaling,
   and held-out broadband inputs. COSMOS PDFs are assumed unbiased; no prior
   correction is applied. Evaluation uses the same 5,083 secure test objects.

`results/flows/` contains two checkpoints, histories, `run.json`, and one
`predictions.npz` keyed by the common test IDs. Quantile columns are 0.025,
0.16, 0.50, 0.84, 0.975; PIT is the conditional CDF at spec-z. Source hashes,
selection, seeds, and counts are saved in local JSON manifests. Choose another
`--output` directory to retain additional runs; an existing completed run is
not overwritten. Use `--epochs 1 --max-steps 20 --output results/smoke` for a
pipeline check, not a scientific result.

Execute `nb/comparison.ipynb` to generate accuracy, PIT/coverage, and magnitude
dependence plots from the new run. Plot uncertainty is a 95% interval from 200
paired galaxy-bootstrap resamples; it excludes retraining variance and cosmic
variance. Magnitude points are bin medians, counts are printed on the plot, and
bands use alpha=0.5. Save executed copies under `results/notebooks_executed/`.
Source notebooks remain without outputs. Spectroscopic selection limits all
reported metrics; they cannot establish coverage of the full parent population.

## Historical products

The earlier pipeline and eight source notebooks are retained locally under
`tmp/three_step_before/`, and in Git history. Existing `results/` and `tex/`
products remain intact. The 52-file manifest and checksums include historical
EAZY resources, prior-correction results, figure snapshots, and old prepared
caches; these are provenance records, not alternate active analysis parents.
Reproducing those earlier paper figures requires the archived workflow.
The `flow_experiment/` caches (153,641 objects; 5,279 secure test objects) describe
an older selection and must not be mixed with the current parent. The previous
162,239-object magnitude-limited results also use the older test-noise draw.

## Restoring and verifying data

Install with `python -m pip install -e .`. Restore the local caches to the paths
in `sources.json`, or retrieve the primary public inputs and run the preparation
and training steps above. This repository deliberately does not distribute the
large catalogs or derived results. A fresh clone alone cannot regenerate the
numerical results until those inputs are restored or recomputed.

Verify the recorded snapshots from the project root:

```bash
shasum -a 256 -c bin/data/SHA256SUMS
make -C tex
```

Hash mismatches must be investigated; do not substitute a newer catalog silently.
CSV serialization/order can differ when re-querying TAP, so byte differences
are not alone proof that numeric values differ. Compare IDs and values when
reconstructing an extract. `.partial` files, `classic_pdf_part_*`, `.prefix`
files, and `classic_ebv_failed_query.xml` are transport/error artifacts, not
analysis inputs. The supplied unrelated TeX reference draft is not a dataset.
