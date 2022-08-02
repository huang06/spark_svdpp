SHELL=/bin/bash

.PHONY: all
all:

.PHONY: python
python:
	@rm -rf .venv
	@python3 -m venv .venv
	@source .venv/bin/activate && pip install -U pip setuptools wheel
	@source .venv/bin/activate && pip install -r requirements.txt
	@source .venv/bin/activate && pip install -r requirements-dev.txt

.PHONY: test
test:
	@source .venv/bin/activate && pytest -m pytest spark_svdpp

.PHONY: example
example:
	@source .venv/bin/activate && ./yarn-client.sh
