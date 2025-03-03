build:
	.venv/bin/python -m build --no-isolation

test:
	.venv/bin/python -m pytest -q

format-check:
	.venv/bin/ruff format --check src tests examples

lint:
	.venv/bin/ruff check src tests examples

typecheck:
	.venv/bin/mypy src
