.PHONY: install test data numeric tsp fast full clean

install:
	pip install -r requirements.txt

test:
	pytest

data:
	python -m src.tsp.data_loader

fast:
	python run_all.py --mode fast

full:
	python run_all.py --mode full

numeric:
	python run_all.py --only numeric --mode full

tsp:
	python run_all.py --only tsp --mode full

clean:
	rm -f data/results/*.csv assets/figures/*.png assets/gifs/*.gif
