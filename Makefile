.PHONY: lint fix typecheck test check

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=splitkit --cov-report=term-missing

check: lint typecheck test
