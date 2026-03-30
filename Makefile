.PHONY: help test test-unit test-integration test-eval test-cov lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Testing ────────────────────────────────────────────────────────

test:  ## Run all tests
	python -m pytest tests/ -v --tb=short

test-unit:  ## Run unit tests only
	python -m pytest tests/unit/ -v --tb=short

test-integration:  ## Run integration tests only
	python -m pytest tests/integration/ -v --tb=short

test-eval:  ## Run evaluation tests only
	python -m pytest tests/training/ -v --tb=short

test-cov:  ## Run tests with coverage report
	python -m pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# ── Code Quality ───────────────────────────────────────────────────

lint:  ## Run linters (flake8)
	python -m flake8 src/ --max-line-length=120 --ignore=E501,W503,E203

format:  ## Format code with black + isort
	python -m black src/ tests/
	python -m isort src/ tests/

# ── Project ────────────────────────────────────────────────────────

evaluate:  ## Run model evaluation (requires trained checkpoint)
	python -m tests.training.run_training

serve:  ## Start web UI on port 8080 (foreground)
	python -m src.api.app

train:  ## Start model training
	python -m src.training.train

# ── Multi-Environment Management ──────────────────────────────

start-dev:  ## Start dev environment (background, port 8080)
	@./scripts/env/start.sh dev

start-qa:  ## Start qa environment (background, port 8081)
	@./scripts/env/start.sh qa

start-preprod:  ## Start preprod environment (background, port 8082)
	@./scripts/env/start.sh preprod

start-prod:  ## Start prod environment (background, port 8083, gunicorn)
	@./scripts/env/start.sh prod --gunicorn

start-all:  ## Start dev + qa simultaneously
	@./scripts/env/start.sh dev
	@./scripts/env/start.sh qa

stop:  ## Stop a specific env (usage: make stop ENV=dev)
	@./scripts/env/stop.sh $(ENV)

stop-all:  ## Stop ALL running environments
	@./scripts/env/stop.sh --all

restart:  ## Restart a specific env (usage: make restart ENV=dev)
	@./scripts/env/restart.sh $(ENV)

restart-all:  ## Restart ALL running environments
	@./scripts/env/restart.sh --all

status:  ## Show status of all environments
	@./scripts/env/status.sh

control-panel:  ## Launch interactive env control panel TUI
	@python3 scripts/env/control_panel.py

clean:  ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage .mypy_cache/ build/ dist/ *.egg-info/
