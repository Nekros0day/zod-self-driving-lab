PYTHON ?= python

.PHONY: setup test lint notebooks

setup:
	$(PYTHON) -m pip install -e ".[all]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m mypy src/zod_driveformer/dynamics src/zod_driveformer/segmentation

notebooks:
	$(PYTHON) scripts/build_notebooks.py
