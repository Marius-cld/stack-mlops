.PHONY: install run test build up down

PYTHON ?= python3
API_COMPOSE = docker compose -f services/api/docker-compose.yml

install:
	$(PYTHON) -m pip install -r requirements.txt

# API en local, rechargement à chaud -> http://localhost:8001/docs
run:
	$(PYTHON) -m src.deploying.app.main

test:
	$(PYTHON) -m pytest src/deploying/tests

# API conteneurisée (services/api) -> http://localhost:8001/docs
build:
	$(API_COMPOSE) build --no-cache

up:
	$(API_COMPOSE) up -d --build

down:
	$(API_COMPOSE) down
