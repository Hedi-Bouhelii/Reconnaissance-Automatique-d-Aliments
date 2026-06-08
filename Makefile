.PHONY: help build up down logs shell-api shell-frontend clean test lint format

# Color output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help:
	@echo "$(CYAN)Food Recognition AI - Docker Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Available Commands:$(NC)"
	@echo ""
	@echo "$(YELLOW)Build$(NC)"
	@echo "  make build              Build all Docker images"
	@echo "  make build-api          Build backend API image only"
	@echo "  make build-frontend     Build frontend image only"
	@echo "  make build-clean        Clean build all images (no cache)"
	@echo ""
	@echo "$(YELLOW)Run$(NC)"
	@echo "  make up                 Start all services (production)"
	@echo "  make up-dev             Start services with hot reload (development)"
	@echo "  make down               Stop all services"
	@echo "  make restart            Restart all services"
	@echo "  make pause              Pause all services"
	@echo "  make unpause            Unpause all services"
	@echo ""
	@echo "$(YELLOW)Logs & Status$(NC)"
	@echo "  make logs               Show logs from all services"
	@echo "  make logs-api           Show logs from API service"
	@echo "  make logs-frontend      Show logs from Frontend service"
	@echo "  make ps                 Show running containers"
	@echo "  make stats              Show container resource usage"
	@echo "  make health             Check health of services"
	@echo ""
	@echo "$(YELLOW)Shell Access$(NC)"
	@echo "  make shell-api          Access backend API container shell"
	@echo "  make shell-frontend     Access frontend container shell"
	@echo "  make shell-bash         Access backend shell with bash"
	@echo ""
	@echo "$(YELLOW)Database & Cleanup$(NC)"
	@echo "  make clean              Stop and remove all containers"
	@echo "  make clean-volumes      Remove all volumes (data loss!)"
	@echo "  make clean-images       Remove all project images"
	@echo "  make prune              Remove unused Docker resources"
	@echo ""
	@echo "$(YELLOW)Development$(NC)"
	@echo "  make lint               Lint Python code"
	@echo "  make format             Format code"
	@echo "  make test               Run tests"
	@echo ""
	@echo "$(YELLOW)Push & Deploy$(NC)"
	@echo "  make push               Push images to registry"
	@echo "  make deploy             Deploy to production"
	@echo ""

# Build targets
build:
	@echo "$(CYAN)Building all Docker images...$(NC)"
	docker-compose build

build-api:
	@echo "$(CYAN)Building backend API image...$(NC)"
	docker-compose build api

build-frontend:
	@echo "$(CYAN)Building frontend image...$(NC)"
	docker-compose build frontend

build-clean:
	@echo "$(CYAN)Clean building all Docker images (no cache)...$(NC)"
	docker-compose build --no-cache

# Run targets
up:
	@echo "$(GREEN)Starting services (production mode)...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Services started!$(NC)"
	@echo ""
	@echo "$(CYAN)Access the application:$(NC)"
	@echo "  Frontend: http://localhost:3000"
	@echo "  Backend:  http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  Web UI:   http://localhost:8000/web"

up-dev:
	@echo "$(GREEN)Starting services (development mode with hot reload)...$(NC)"
	docker-compose -f docker-compose.dev.yml up -d
	@echo "$(GREEN)✓ Development services started!$(NC)"
	@echo ""
	@echo "$(CYAN)Access the application:$(NC)"
	@echo "  Frontend: http://localhost:5173 (with hot reload)"
	@echo "  Backend:  http://localhost:8000 (with auto-restart)"
	@echo "  API Docs: http://localhost:8000/docs"

down:
	@echo "$(YELLOW)Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Services stopped!$(NC)"

restart:
	@echo "$(YELLOW)Restarting services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✓ Services restarted!$(NC)"

pause:
	@echo "$(YELLOW)Pausing services...$(NC)"
	docker-compose pause
	@echo "$(GREEN)✓ Services paused!$(NC)"

unpause:
	@echo "$(YELLOW)Unpausing services...$(NC)"
	docker-compose unpause
	@echo "$(GREEN)✓ Services unpaused!$(NC)"

# Logs targets
logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-frontend:
	docker-compose logs -f frontend

# Status targets
ps:
	docker-compose ps

stats:
	docker stats

health:
	@echo "$(CYAN)Checking service health...$(NC)"
	@echo "API Health:"
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"
	@echo ""
	@echo "Docker containers:"
	@docker-compose ps

# Shell access
shell-api:
	docker-compose exec api sh

shell-frontend:
	docker-compose exec frontend sh

shell-bash:
	docker-compose exec api bash

# Cleanup targets
clean:
	@echo "$(YELLOW)Stopping and removing all containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Containers removed!$(NC)"

clean-volumes:
	@echo "$(YELLOW)WARNING: Removing all volumes (data loss!)...$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		echo "$(GREEN)✓ Volumes removed!$(NC)"; \
	else \
		echo "Cancelled"; \
	fi

clean-images:
	@echo "$(YELLOW)Removing project images...$(NC)"
	docker rmi food-ai-api:latest food-ai-frontend:latest 2>/dev/null || true
	@echo "$(GREEN)✓ Images removed!$(NC)"

prune:
	@echo "$(YELLOW)Pruning unused Docker resources...$(NC)"
	docker system prune -f
	@echo "$(GREEN)✓ Pruned!$(NC)"

# Development targets
lint:
	@echo "$(CYAN)Linting Python code...$(NC)"
	docker-compose exec api bash -c "pip install flake8 && flake8 app.py" || echo "No flake8 installed"

format:
	@echo "$(CYAN)Formatting code...$(NC)"
	docker-compose exec api bash -c "pip install black && black app.py" || echo "No black installed"

test:
	@echo "$(CYAN)Running tests...$(NC)"
	docker-compose exec api python -m pytest

# Push and deploy
push:
	@echo "$(CYAN)Pushing images to registry...$(NC)"
	@echo "Please set REGISTRY variable: make push REGISTRY=yourusername"

deploy:
	@echo "$(CYAN)Deploying to production...$(NC)"
	docker-compose -f docker-compose.prod.yml up -d

# Utility
info:
	@echo "$(CYAN)Docker Information:$(NC)"
	@docker version --format '{{.Server.Version}}'
	@echo ""
	@echo "$(CYAN)Docker Compose Version:$(NC)"
	@docker-compose version --short || docker compose version
	@echo ""
	@echo "$(CYAN)Images:$(NC)"
	@docker images | grep food-ai || echo "No images built yet"
	@echo ""
	@echo "$(CYAN)Containers:$(NC)"
	@docker-compose ps

# Default target
.DEFAULT_GOAL := help
