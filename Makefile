export PYTHONPATH := src
RAW ?= data/raw/FreddieMac.csv

test:
	pytest -q

train:
	python -m credit_risk.train --raw $(RAW)

train-ever:
	python -m credit_risk.train --raw $(RAW) --target secondary --cv 0

simulate:
	python scripts/simulate_batches.py --real $(RAW)

.PHONY: test train train-ever simulate
