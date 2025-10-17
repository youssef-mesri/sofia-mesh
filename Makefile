.PHONY: test lint type

test:
	pytest -q

lint:
	flake8

type:
	mypy
