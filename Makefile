.PHONY: setup lint test run

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

lint:
	ruff check . && black --check .

test:
	pytest -q

run:
	streamlit run app/app.py