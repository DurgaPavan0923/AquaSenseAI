.PHONY: install test lint format clean train evaluate run-api run-dashboard run-sim docker-build docker-up

PYTHON := python
PIP := pip

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache dist build *.egg-info

data:
	$(PYTHON) -m src.data.loader

eda:
	$(PYTHON) -m src.data.eda

train:
	$(PYTHON) -m src.ml.train

evaluate:
	$(PYTHON) -m src.ml.evaluate

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

run-sim:
	$(PYTHON) -m src.simulation.generator

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d
