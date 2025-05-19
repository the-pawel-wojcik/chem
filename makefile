.PHONY: check
check:
	-isort --check .
	-black --check .
	-mypy .
	-pytest -v --color=yes --doctest-modules tests/ src/chem


.PHONY: build
build:
	python -m build


.PHONY: release
release: build
	pip install --upgrade twine
	python -m twine upload --repository pypi dist/*
