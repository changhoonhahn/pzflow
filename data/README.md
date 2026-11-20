# Local research data cache

Downloaded catalogs, the full COSMOS2020 PDF archive, and EAZY resources are
kept locally and excluded from Git. The repository includes compact snapshots
under `paper/data/` and `results/` sufficient to run all eight figure notebooks.

The full experiment uses these public sources:

- [COSMOS2020 Classic photometry and LePhare PDFs at IRSA](https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/).
- [COSMOS spectroscopic compilation](https://github.com/cosmosastro/speczcompilation),
  specifically the public DR1.1 unique catalog used in the saved analysis.
- [EAZY templates, filters, and priors](https://github.com/gbrammer/eazy-photoz).

Expected cached inputs include:

```text
data/cosmos2020_specz/
  specz_compilation_COSMOS_DR1.1_unique.fits
  specz_compilation_COSMOS_DR1.1_surveys.list
  classic_lephare_complete.fits.gz
  classic_ugrizy.csv
  classic_magnitude_intervals.csv
  classic_r_photometry.csv
  classic_i25_ebv.csv
  classic_matched_pdfs.npz
data/flow_magnitude_limited/
  prepared.npz
  teacher_pdfs.npz
data/eazy-photoz-master/
```

The CSV files are COSMOS Classic catalog extracts keyed by Classic `id`.
Their required columns are explicit in `src/pzflow/flow_experiment_data.py`,
`run_eazy_baseline.py`, and the plotting modules. The full PDF archive is
about 1.8 GB compressed. Its first image row is the redshift grid; subsequent
rows contain an explicit Classic ID and PDF values, so matching must use IDs.

`bin/flow_experiment_data.py` prepares the spatial splits and training cache
once the source catalogs are present. `bin/download_cosmos_pdf_ranges.py` is a
recovery utility for the original partially downloaded archive; it expects an
existing 600-MB prefix and is not a general fresh-download command.

Source URLs, selection rules, and available file hashes are recorded in the
result manifests and experiment documentation. Older `flow_experiment/`
outputs are an archived spectroscopically enriched experiment, superseded by
the full magnitude-limited census in `flow_magnitude_limited/`.

