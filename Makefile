.PHONY: install run debug clean lint lint-strict

MYPY_FLAGS = --warn-return-any --warn-unused-ignores \
	  --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
