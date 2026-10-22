export PYTHONPATH := src
RAW ?= data/raw/FreddieMac.csv

test:
	pytest -q

train:
	python -m credit_risk.train --raw $(RAW)

train-sp:
	python -m credit_risk.train --raw $(RAW) --engine scorecardpy

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

load-sim:
	python scripts/load_sim.py

.PHONY: test train train-sp train-ever simulate dbt-run dbt-test parity load-sim

baseline:
	python scripts/build_baseline.py --raw $(RAW)

monitor:
	python scripts/run_monitoring.py

report:
	python -m credit_risk.genai.chat --report

Q ?= Why is this loan risky?
ask:
	python -m credit_risk.genai.chat --loan examples/loan_example.json --ask "$(Q)"

chat:
	python -m credit_risk.genai.chat --loan examples/loan_example.json --loan-b examples/loan_example_b.json

loan-note:
	python -m credit_risk.genai.chat --loan examples/loan_example.json --loan-note

.PHONY: baseline monitor report ask chat loan-note

chat-data:
	python -m credit_risk.genai.chat

.PHONY: chat-data

llm-check:
	python -m credit_risk.genai.chat --check-llm

.PHONY: llm-check
