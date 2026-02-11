.PHONY: setup run test lint docker-build docker-up

setup:
	pip install -r requirements.txt

run:
	uvicorn src.main:app --reload --port 6006

test:
	pytest tests/

lint:
	pip install ruff
	ruff check src/

docker-build:
	docker-compose build

docker-up:
	docker-compose up
