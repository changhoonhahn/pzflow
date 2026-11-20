PYTHON ?= python

.PHONY: test notebooks paper validate packages
test:
	$(PYTHON) bin/test_flow_experiment.py
	$(PYTHON) bin/validate_repository.py

notebooks:
	$(PYTHON) bin/execute_paper_notebooks.py --all

paper:
	$(MAKE) -C paper

validate:
	$(PYTHON) bin/validate_repository.py
	$(PYTHON) bin/validate_apjl_package.py

packages:
	$(PYTHON) bin/package_apjl_draft.py
	$(PYTHON) bin/package_tex_source.py
	$(PYTHON) bin/package_catastrophic_pdf_stress.py
