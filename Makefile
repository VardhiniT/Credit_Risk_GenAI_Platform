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

# dbt reads Snowflake settings from .env (never committed)
DBT = set -a; [ -f .env ] && . ./.env; set +a; dbt
DBTARGS = --project-dir dbt --profiles-dir dbt

dbt-run:
	$(DBT) run $(DBTARGS)

dbt-test:
	$(DBT) test $(DBTARGS)

parity:
	python scripts/parity_check.py

.PHONY: test train train-ever simulate dbt-run dbt-test parity
