# Manuscript

Run `make -C tex` from the project root. Figures and compiled outputs stay local.
Figure notebooks in `nb/` read `results/figure_data/` and write `tex/figures/`.
Run `python bin/notebooks.py --all` to regenerate figures from the local result
cache; executed notebooks are saved under `results/notebooks_executed/`.
Run `python bin/export.py` to rebuild the frozen figure inputs and numerical TeX
macros after preparing the datasets and running the model comparisons.
TeX compilation requires latexmk, pdfLaTeX, and BibTeX. Source-data details and
reproduction order are documented in `bin/data/README.md`.
