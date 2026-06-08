# 🐳 Docker Quick Reference - Food Recognition AI

Fast commands for common Docker tasks in this project.

## Quick Start

### 🚀 Start Application

```bash
# Production mode
make up

# Development mode (with hot reload)
make up-dev

# Or manually
docker-compose up -d
docker-compose -f docker-compose.dev.yml up -d
```

### ⛔ Stop Application

```bash
make down
docker-compose down
```

### 🔄 Restart Services

```bash
make restart
docker-compose restart api
docker-compose restart frontend
```

## Build Commands

| Command | Effect |
|---------|--------|
| `make build` | Build all images |
| `make build-api` | Build backend only |
| `make build-frontend` | Build frontend only |
| `make build-clean` | Clean build (no cache) |
| `docker-compose build --progress=plain` | Show build details |

## View Logs

```bash
make logs                  # All services
make logs-api              # Backend API only
make logs-frontend         # Frontend only
docker-compose logs -f api # Live logs

# Search logs
docker-compose logs | grep ERROR
docker-compose logs | grep -i warning
```

## View Status

```bash
make ps                    # List containers
make stats                 # Resource usage
docker-compose ps -a       # All containers
docker inspect food-ai-api # Detailed info
```

## Access Container Shell

```bash
make shell-api             # Backend shell (sh)
make shell-bash            # Backend shell (bash)
make shell-frontend        # Frontend shell

# Or directly
docker-compose exec api bash
docker-compose exec frontend sh
```

## Run Commands in Containers

```bash
# Python commands
docker-compose exec api python -c "print('hello')"

# Check Python packages
docker-compose exec api pip list | grep torch

# Run Python scripts
docker-compose exec api python script.py

# Check model
docker-compose exec api ls -la /app/models/

# Test API
docker-compose exec api curl http://localhost:8000/health

# NPM commands
docker-compose exec frontend npm list
docker-compose exec frontend npm install new-package
```

## Cleanup Commands

| Command | Effect |
|---------|--------|
| `make clean` | Stop & remove containers |
| `make clean-volumes` | Remove data (⚠️ data loss) |
| `make clean-images` | Remove images |
| `make prune` | Clean up unused resources |

## Copy Files

```bash
# Copy from container to host
docker-compose cp api:/app/models/best_model.pth ./models/

# Copy from host to container
docker cp models/best_model.pth food-ai-api:/app/models/

# Copy directory
docker-compose cp api:/app/src ./src
```

## View File in Container

```bash
docker-compose exec api cat /app/app.py
docker-compose exec api ls -la /app/models/
docker-compose exec frontend ls -la /app
```

## Rebuild After Code Changes

```bash
# Backend changes
docker-compose build api
docker-compose up -d api

# Frontend changes
docker-compose build frontend
docker-compose up -d frontend

# Both
docker-compose up --build
```

## Debug Container Issues

```bash
# View full logs with timestamps
docker-compose logs -t -f

# Check container processes
docker-compose exec api ps aux

# Check network
docker-compose exec api ping frontend
docker network inspect <network-name>

# Memory/CPU stats
docker stats food-ai-api

# Check environment variables
docker-compose exec api env

# Inspect volumes
docker inspect food-ai-api | grep -A 20 Mounts
```

## Common Issues

### Container won't start

```bash
# Check logs
docker-compose logs api

# Try rebuilding
docker-compose build --no-cache api

# Check if port is in use
lsof -i :8000
```

### Model not found

```bash
# Check volume mount
docker-compose exec api ls -la /app/models/

# Manually mount
docker run -v /path/to/models:/app/models food-ai-api
```

### Slow performance

```bash
# Check resource limits
docker stats

# Check disk usage
docker system df

# Clean up
docker system prune -a
```

## Docker Images

```bash
# List images
docker images | grep food-ai

# View image details
docker inspect food-ai-api:latest

# View image layers
docker history food-ai-api:latest

# Remove image
docker rmi food-ai-api:latest
```

## Volumes

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect <volume-name>

# Remove volume
docker volume rm <volume-name>

# Backup volume
docker run --rm -v <volume>:/data -v $(pwd):/backup alpine tar czf /backup/volume.tar.gz -C /data .

# Restore volume
docker run --rm -v <volume>:/data -v $(pwd):/backup alpine tar xzf /backup/volume.tar.gz -C /data
```

## Networks

```bash
# List networks
docker network ls

# Inspect network
docker network inspect <network-name>

# Create custom network
docker network create my-network

# Connect container to network
docker network connect my-network food-ai-api
```

## Port Management

```bash
# Find processes using ports
lsof -i :8000
lsof -i :3000

# Kill process
kill -9 <PID>

# Use different ports
# In docker-compose.yml:
# ports:
#   - "8001:8000"  # Use 8001 instead of 8000
```

## Push to Registry

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag food-ai-api:latest yourusername/food-ai-api:v1.0

# Push
docker push yourusername/food-ai-api:v1.0

# Pull from registry
docker pull yourusername/food-ai-api:v1.0
```

## Compose File Management

```bash
# Use different compose file
docker-compose -f docker-compose.prod.yml up

# Multiple files
docker-compose -f docker-compose.yml -f docker-compose.override.yml up

# Validate compose file
docker-compose config

# Get services
docker-compose config --services
```

## Environment Variables

```bash
# Set variables
export COMPOSE_PROJECT_NAME=food-ai
docker-compose up

# From .env file
# Create .env file with:
# API_PORT=8000
# FRONTEND_PORT=3000

# Use in docker-compose.yml:
# ports:
#   - "${API_PORT}:8000"
```

## Advanced Debugging

```bash
# Execute with specific user
docker-compose exec -u root api bash

# Execute as detached (background)
docker-compose exec -d api python script.py

# Set environment for command
docker-compose exec -e VAR=value api bash

# Get only output (no interactive)
docker-compose exec api curl http://localhost:8000/health
```

## Performance Optimization

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker build -f Dockerfile.backend -t food-ai-api .

# Use .dockerignore to exclude files
# See .dockerignore file

# Prune regularly
docker system prune -a -f

# Check image size
docker images --format "table {{.Repository}}\t{{.Size}}"
```

## Useful Shortcuts

```bash
# Clear all Docker data
docker system prune -a --volumes -f

# Stop all containers
docker stop $(docker ps -q)

# Remove all containers
docker rm $(docker ps -aq)

# View all running processes
docker ps

# Full Docker info
docker info

# Docker version
docker --version
docker-compose --version
```

## Using Makefile (Easier!)

```bash
make help               # Show all commands
make up                 # Start services
make down               # Stop services
make logs               # View logs
make ps                 # List containers
make shell-api          # Access API shell
make clean              # Clean up
make build              # Build images
```

---

**Tip:** Use `make` commands - they're easier to remember! 🎯

For detailed information, see `DOCKER_GUIDE.md`
