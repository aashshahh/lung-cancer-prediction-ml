.PHONY: setup train explain app mlflow clean

setup:
	pip install -r requirements.txt

train:
	python src/train.py

explain:
	python src/explain_shap.py
	python src/explain_lime.py
	python src/subgrp.py
	python src/robustness.py

app:
	streamlit run app/streamlit_app.py

mlflow:
	mlflow ui

clean:
	rm -rf artifacts/models/*.pkl
	rm -rf artifacts/plots/*.png
	rm -rf artifacts/results/*.csv
	rm -rf mlruns/