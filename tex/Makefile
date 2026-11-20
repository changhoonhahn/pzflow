LATEXMK ?= latexmk

.PHONY: all
all: main.pdf

main.pdf: main.tex authors.tex intro.tex methods.tex bootstrapping.tex appendix.tex results.tex discussion.tex references.bib numbers.tex table_results.tex figures/fig01_selection.pdf figures/fig02_accuracy.pdf figures/fig03_calibration.pdf figures/fig04_magnitude_dependence.pdf figures/fig05_bootstrap_workflow.pdf
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex
