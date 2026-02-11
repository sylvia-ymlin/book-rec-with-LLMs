# Environment
env-create:
	conda env create -f environment.yml

env-update:
	conda env update -f environment.yml --prune

# Development
run:
	uvicorn src.main:app --reload --port 6006

# Data Pipeline
data-pipeline:
	python scripts/run_pipeline.py

data-validate:
	python scripts/data/validate_data.py

data-prep:
	python scripts/run_pipeline.py --skip-models

train-models:
	python scripts/run_pipeline.py --stage models

# Quality
test:
	pytest tests/

lint:
	ruff check src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up

