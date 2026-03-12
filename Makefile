PYTHON ?= python

.PHONY: train evaluate run test mlflow docker-up benchmark

train:
	$(PYTHON) scripts/run_training.py

evaluate:
	$(PYTHON) scripts/run_evaluation.py

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns

docker-up:
	docker compose up --build

benchmark:
	$(PYTHON) scripts/benchmark_inference.py