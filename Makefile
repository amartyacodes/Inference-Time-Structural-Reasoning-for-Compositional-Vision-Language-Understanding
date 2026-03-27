.PHONY: help install dev test lint format clean run-quick run-full run-main run-qwen

help:
	@echo "LLaVA Evaluation Framework - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make dev           Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make lint          Run flake8 linter"
	@echo "  make format        Format code with black"
	@echo "  make test          Run tests (if available)"
	@echo ""
	@echo "Execution:"
	@echo "  make run-quick     Global: Quick test (CLIP + BLIP, 10 samples)"
	@echo "  make run-full      Global: Full multi-model evaluation"
	@echo "  make run-main      LLaVA: Basic LLaVA evaluation"
	@echo "  make run-qwen      Qwen3: Basic Qwen3 evaluation"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         Remove cached files and artifacts"
	@echo ""

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

dev:
	pip install -r requirements.txt
	pip install black flake8 mypy pytest
	python -m spacy download en_core_web_sm

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100

format:
	black . --line-length 100

test:
	@echo "Running quick sanity check with 5 samples..."
	python final.py --methods clip --max_samples 5

run-quick:
	@echo "Quick multi-model test (CLIP + BLIP)..."
	python final.py --methods clip blip --max_samples 10

run-full:
	@echo "Full evaluation on all methods..."
	python final.py --methods all --max_samples 100

run-main:
	@echo "LLaVA evaluation..."
	python main.py --max_samples 50

run-qwen:
	@echo "Qwen3-VL evaluation..."
	python qwen_3_gen.py --max_samples 50

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf build dist *.egg-info
	@echo "Cleaned up cache files"
