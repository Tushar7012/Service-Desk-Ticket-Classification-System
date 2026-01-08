.PHONY: install dev-install test lint format clean train evaluate serve

# Installation
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code Quality
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	ruff check src/ tests/ --fix

# Training & Evaluation
train:
	python scripts/train.py --config configs/training_config.yaml

evaluate:
	python scripts/evaluate.py --model-path models/production/model.pt

# API
serve:
	uvicorn src.api.inference_api:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api.inference_api:app --host 0.0.0.0 --port 8000 --workers 4

# Model Export
export-onnx:
	python scripts/export_model.py --format onnx

# Cleanup
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
