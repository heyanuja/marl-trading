.PHONY: test lint train eval wow all

test:
	python -m unittest discover -s tests

lint:
	python -m ruff check .

train:
	python train.py --overwrite

eval:
	python evaluate.py --run-dir results/wow --episodes 100

wow:
	python train.py --preset wow --overwrite

all: lint test train eval
