# Multimodal RAG – developer shortcuts (requires GNU make, bash, docker)
SHELL := /bin/bash
.DEFAULT_GOAL := help

BACKEND_DIR := backend
FRONTEND_DIR := frontend
PYTHON ?= python

.PHONY: help env up up-ollama down clean logs ps test test-backend test-frontend lint fmt sample-pdf demo dev-api dev-worker dev-infra

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env – set OPENAI_API_KEY or LLM_PROVIDER=ollama")

up: env ## Build and start the full stack (api, worker, frontend, qdrant, redis)
	docker compose up --build -d
	@echo "UI: http://localhost:3000  API docs: http://localhost:8000/docs  Qdrant: http://localhost:6333/dashboard"

up-ollama: env ## Start the stack plus a local Ollama server (profile "ollama")
	docker compose --profile ollama up --build -d
	docker compose exec ollama ollama pull $${OLLAMA_CHAT_MODEL:-llama3.2-vision}
	docker compose exec ollama ollama pull $${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}

down: ## Stop the stack
	docker compose --profile ollama down

clean: ## Stop the stack and delete volumes (uploaded data, vectors, jobs)
	docker compose --profile ollama down -v

logs: ## Follow logs of api and worker
	docker compose logs -f api worker

ps: ## Show service status
	docker compose ps

test: test-backend test-frontend ## Run all tests

test-backend: ## Run backend unit tests with coverage (offline, mocks only)
	cd $(BACKEND_DIR) && $(PYTHON) -m pytest --cov=app --cov-report=term-missing

test-frontend: ## Run frontend tests
	cd $(FRONTEND_DIR) && npm test -- --run

lint: ## Lint backend (ruff) and typecheck frontend
	cd $(BACKEND_DIR) && $(PYTHON) -m ruff check .
	cd $(FRONTEND_DIR) && npm run typecheck

fmt: ## Auto-format backend code
	cd $(BACKEND_DIR) && $(PYTHON) -m ruff check --fix . && $(PYTHON) -m ruff format .

sample-pdf: ## Regenerate docs/sample/Manual_Motor.pdf
	$(PYTHON) scripts/generate_sample_pdf.py

demo: ## Upload the sample PDF, wait for ingestion and ask a question
	bash scripts/demo.sh

dev-infra: ## Start only qdrant + redis (for running api/worker locally)
	docker compose up -d qdrant redis

dev-api: ## Run the API locally with autoreload
	cd $(BACKEND_DIR) && $(PYTHON) -m uvicorn app.api.main:app --reload --port 8000

dev-worker: ## Run the Celery worker locally (use --pool=solo on Windows)
	cd $(BACKEND_DIR) && $(PYTHON) -m celery -A app.infrastructure.queue.celery_app worker -Q ingestion --loglevel=INFO --pool=solo
