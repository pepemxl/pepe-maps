# Pepe Maps — development tasks.
# Default target prints help.

IMAGE       ?= pepe-maps-dev
CONTAINER   ?= pepe-maps-dev
PORT        ?= 8000
HOST_BIND   ?= 127.0.0.1
DOCKER      ?= docker

.DEFAULT_GOAL := help

.PHONY: help build up down restart logs shell ps serve lint clean

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Pepe Maps dev targets:\n\n"} \
	      /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build the dev container image.
	$(DOCKER) build -t $(IMAGE) .

up: build ## Start the dev server in a container at http://$(HOST_BIND):$(PORT).
	@$(DOCKER) rm -f $(CONTAINER) >/dev/null 2>&1 || true
	$(DOCKER) run -d --name $(CONTAINER) \
		-p $(HOST_BIND):$(PORT):8000 \
		-v $(CURDIR):/app \
		$(IMAGE)
	@echo "→ http://$(HOST_BIND):$(PORT)/"

down: ## Stop and remove the dev container.
	@$(DOCKER) rm -f $(CONTAINER) >/dev/null 2>&1 || true

restart: down up ## Restart the dev container.

logs: ## Tail container logs (Ctrl-C to exit).
	$(DOCKER) logs -f $(CONTAINER)

shell: ## Open a shell inside the running container.
	$(DOCKER) exec -it $(CONTAINER) sh

ps: ## Show container status.
	@$(DOCKER) ps --filter name=$(CONTAINER)

serve: ## Run the PHP built-in server on the host (no Docker).
	php -S $(HOST_BIND):$(PORT) -t mexico

lint: ## php -l every .php file under mexico/.
	@find mexico -name '*.php' -print0 | xargs -0 -n1 php -l | grep -v '^No syntax errors' || echo "All PHP files OK."

clean: down ## Stop the container and remove the image.
	@$(DOCKER) image rm $(IMAGE) >/dev/null 2>&1 || true
