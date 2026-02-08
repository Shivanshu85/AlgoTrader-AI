.PHONY: help setup install clean test test-unit test-integration test-performance test-cov lint format type-check security docs docker-build docker-up docker-down docker-logs run-api run-training run-scheduler health-check pre-commit

# Variables
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
PYLINT := pylint

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m # No Color

help:  ## Show this help message
	@echo "$(BLUE)Stock Predictor Production - Development Tasks$(NC)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-25s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

setup: install pre-commit  ## Set up development environment
	@echo "$(GREEN)✓ Development environment ready$(NC)"

install:  ## Install project dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

clean:  ## Clean up generated files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name .coverage -delete
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# ============================================================================
# CODE QUALITY
# ============================================================================

lint: format type-check pylint flake8  ## Run all linters (format, type-check, pylint, flake8)
	@echo "$(GREEN)✓ All linting checks passed$(NC)"

format:  ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(ISORT) production tests scripts
	$(BLACK) production tests scripts
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check:  ## Type check with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	$(MYPY) production
	@echo "$(GREEN)✓ Type checks passed$(NC)"

flake8:  ## Run flake8 linting
	@echo "$(BLUE)Running flake8...$(NC)"
	$(FLAKE8) production tests scripts --max-line-length=100 --extend-ignore=E203,W503
	@echo "$(GREEN)✓ Flake8 checks passed$(NC)"

pylint:  ## Run pylint linting
	@echo "$(BLUE)Running pylint...$(NC)"
	$(PYLINT) production --disable=C0111,R0913 --max-line-length=100 || true
	@echo "$(GREEN)✓ Pylint checks completed$(NC)"

security:  ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(PIP) install bandit safety
	bandit -r production -ll
	safety check
	@echo "$(GREEN)✓ Security checks passed$(NC)"

# ============================================================================
# TESTING
# ============================================================================

test: test-unit test-integration  ## Run all tests (unit + integration)
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-unit:  ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) tests/unit -v --tb=short
	@echo "$(GREEN)✓ Unit tests passed$(NC)"

test-integration:  ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/integration -v --tb=short
	@echo "$(GREEN)✓ Integration tests passed$(NC)"

test-performance:  ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTEST) tests/performance -v --tb=short -m "performance"
	@echo "$(GREEN)✓ Performance tests passed$(NC)"

test-cov:  ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=production --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated: htmlcov/index.html$(NC)"

# ============================================================================
# DOCUMENTATION
# ============================================================================

docs:  ## Build documentation with Sphinx
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && make html
	@echo "$(GREEN)✓ Documentation built: docs/_build/html/index.html$(NC)"

docs-serve:  ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# ============================================================================
# DOCKER
# ============================================================================

docker-build:  ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)✓ Docker images built$(NC)"

docker-up:  ## Start Docker containers
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Docker containers started$(NC)"
	@sleep 3
	@make health-check

docker-down:  ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Docker containers stopped$(NC)"

docker-logs:  ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-clean:  ## Remove Docker containers and volumes
	@echo "$(BLUE)Removing Docker containers and volumes...$(NC)"
	$(DOCKER_COMPOSE) down -v
	@echo "$(GREEN)✓ Docker cleanup complete$(NC)"

# ============================================================================
# RUNNING SERVICES
# ============================================================================

run-api:  ## Run FastAPI development server
	@echo "$(BLUE)Starting API server at http://localhost:8000...$(NC)"
	uvicorn production.serving.api:app --reload --host 0.0.0.0 --port 8000

run-api-prod:  ## Run FastAPI production server with gunicorn
	@echo "$(BLUE)Starting production API server...$(NC)"
	gunicorn \
		-w 4 \
		-b 0.0.0.0:8000 \
		--timeout 120 \
		--access-logfile - \
		production.serving.api:app

run-training:  ## Run model training pipeline
	@echo "$(BLUE)Starting training pipeline...$(NC)"
	$(PYTHON) -m production.training.train

run-scheduler:  ## Run Airflow scheduler
	@echo "$(BLUE)Starting Airflow scheduler...$(NC)"
	airflow scheduler

run-webui:  ## Run Airflow WebUI
	@echo "$(BLUE)Starting Airflow WebUI at http://localhost:8080...$(NC)"
	airflow webserver

health-check:  ## Run health check
	@echo "$(BLUE)Running health checks...$(NC)"
	$(PYTHON) production/health_check.py
	@echo "$(GREEN)✓ All services healthy$(NC)"

# ============================================================================
# DEVELOPMENT TOOLS
# ============================================================================

pre-commit:  ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	pre-commit install
	pre-commit run --all-files || true
	@echo "$(GREEN)✓ Pre-commit hooks installed$(NC)"

notebook:  ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab at http://localhost:8888...$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

shell:  ## Start Python interactive shell with project imports
	@echo "$(BLUE)Starting Python shell...$(NC)"
	PYTHONPATH=. $(PYTHON)

# ============================================================================
# DATABASE
# ============================================================================

db-migrate:  ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)✓ Database migrated$(NC)"

db-downgrade:  ## Rollback database migrations
	@echo "$(BLUE)Downgrading database...$(NC)"
	alembic downgrade -1
	@echo "$(GREEN)✓ Database downgraded$(NC)"

db-init:  ## Initialize database
	@echo "$(BLUE)Initializing database...$(NC)"
	$(PYTHON) -m production.data_ingestion.database init
	@echo "$(GREEN)✓ Database initialized$(NC)"

# ============================================================================
# PROJECT INFO
# ============================================================================

info:  ## Show project information
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "  Name: Stock Predictor Production"
	@echo "  Version: 0.1.0"
	@echo "  Python: $$($(PYTHON) --version)"
	@echo "  Pip: $$($(PIP) --version)"
	@echo ""
	@echo "$(BLUE)Installed Packages:$(NC)"
	@$(PIP) list --format=columns

version:  ## Show version
	@echo "Stock Predictor Production v0.1.0"

# ============================================================================
# CI/CD
# ============================================================================

ci: clean lint test type-check  ## Run CI pipeline locally
	@echo "$(GREEN)✓ CI pipeline passed$(NC)"

ci-full: ci test-cov security docs  ## Run full CI pipeline with coverage and security
	@echo "$(GREEN)✓ Full CI pipeline passed$(NC)"

# ============================================================================
# DEFAULT
# ============================================================================

.DEFAULT_GOAL := help
