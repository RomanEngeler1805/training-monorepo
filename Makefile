.PHONY: run format lint test check all help

# Default target
.DEFAULT_GOAL := help

# Run the main file
run:
	PYTHONPATH=. uv run python scripts/run_training.py

# Format code with ruff
format:
	uv run ruff format .

# Lint code with ruff (check only, no fixes)
lint:
	uv run ruff check .

# Lint and auto-fix issues
lint-fix:
	uv run ruff check --fix .

# Run tests
test:
	uv run pytest

# Run tests with verbose output
test-verbose:
	uv run pytest -v

# Run tests with coverage
test-cov:
	uv run pytest --cov=data --cov=models --cov-report=term-missing

# Format, lint, and test all in one
check: format lint-fix test
	@echo "✅ All checks passed!"

# Format, lint, and test (alias for check)
all: check

# Show help message
help:
	@echo "Available commands:"
	@echo "  make run         - Run the main.py file"
	@echo "  make format      - Format code with ruff"
	@echo "  make lint        - Lint code (check only)"
	@echo "  make lint-fix    - Lint and auto-fix issues"
	@echo "  make test        - Run tests"
	@echo "  make test-verbose - Run tests with verbose output"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make check       - Format, lint-fix, and test (recommended)"
	@echo "  make all         - Alias for 'check'"
	@echo "  make help        - Show this help message"