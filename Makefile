PY ?= .venv/bin/python

.PHONY: setup run test eval record trace-demo

setup:
	python3 -m venv .venv && $(PY) -m pip install -q -e ".[dev]"

run:
	$(PY) -m uvicorn errand.web.app:app --reload --port 8000

test:
	$(PY) -m pytest tests/unit -q

eval:
	$(PY) -m pytest tests -q

record:
	$(PY) -m pytest tests/scenarios -q --record

trace-demo:
	$(PY) scripts/trace_demo.py
