.PHONY: install build down up run test

install:
	pip install -r requirements.txt

build:
	docker compose build --no-cache

down:
	docker compose down -v

up:
	docker compose up -d

run:
	uvicorn src.deploying.app.main:app --reload

test:
	pytest src/deploying/tests