.PHONY: lint fix format

lint:
	uv run ruff check .

fix:
	uv run ruff check --fix .

format:
	uv run ruff format .
