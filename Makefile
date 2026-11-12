PY ?= .venv/bin/python

.PHONY: setup run test eval smoke trace-demo

setup:
	python3 -m venv .venv && $(PY) -m pip install -q -e ".[dev]"

run:
	$(PY) -m uvicorn --factory errand.web.app:create_app --port 8000

test:
	$(PY) -m pytest tests/unit -q

eval:
	$(PY) -m pytest tests -q

trace-demo:
	$(PY) scripts/trace_demo.py

smoke:
	$(PY) scripts/smoke_llm.py
