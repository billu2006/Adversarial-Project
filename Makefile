# Everything you need for a local run. `make help` lists the targets.
#
# The stack is Docker Compose only: postgres, redis, the API and the worker.
# Nothing else needs installing to run it.

COMPOSE ?= docker compose

.DEFAULT_GOAL := help

.PHONY: help
help: ## List the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: up
up: ## Build and start the stack (api, worker, postgres, redis)
	$(COMPOSE) up --build

.PHONY: start
start: ## Same as `up`, but in the background
	$(COMPOSE) up --build -d

.PHONY: down
down: ## Stop the stack and remove its volumes
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail the API and worker logs
	$(COMPOSE) logs -f api worker

.PHONY: demo
demo: ## Run a benchmark end to end against the running stack
	./scripts/demo.sh

.PHONY: scale
scale: ## Run three workers instead of one
	$(COMPOSE) up --build -d --scale worker=3

.PHONY: psql
psql: ## Open a psql shell against the running database
	$(COMPOSE) exec postgres psql -U benchmark -d benchmark

# --- Development ---------------------------------------------------------- #
# These run on the host, not in a container. `pip install -e ".[dev]"` first.

.PHONY: test
test: ## Run the test suite (SQLite; no services required)
	pytest

.PHONY: test-fast
test-fast: ## Run the test suite without the slow PyTorch tests
	pytest -m "not engine"

.PHONY: lint
lint: ## Check formatting and lint rules
	ruff check .
	ruff format --check .

.PHONY: format
format: ## Apply formatting and safe lint fixes
	ruff check . --fix
	ruff format .
