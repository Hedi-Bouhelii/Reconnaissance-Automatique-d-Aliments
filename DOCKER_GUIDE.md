# 🐳 Docker Setup Guide - Food Recognition AI

Complete guide for containerizing and running the Food Recognition AI project using Docker.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Understanding Docker in This Project](#understanding-docker-in-this-project)
4. [Quick Start](#quick-start)
5. [Detailed Commands](#detailed-commands)
6. [Docker Compose Configuration](#docker-compose-configuration)
7. [Building Images](#building-images)
8. [Running Containers](#running-containers)
9. [Troubleshooting](#troubleshooting)
10. [Production Deployment](#production-deployment)
11. [Best Practices](#best-practices)

---

## Prerequisites

Before starting, ensure you have:

- **Docker** installed: [Download Docker](https://www.docker.com/products/docker-desktop)
- **Docker Compose** installed (usually comes with Docker Desktop)
- Verify installation:

```bash
docker --version
docker-compose --version
```

---

## Project Structure

```
Reconnaissance-Automatique-d-Aliments/
├── Dockerfile.backend          # Backend API container definition
├── Dockerfile.frontend         # Frontend React container definition
├── docker-compose.yml          # Orchestrates multiple containers
├── .dockerignore               # Files/folders to exclude from Docker build
├── app.py                      # FastAPI application
├── requirements.txt            # Python dependencies
├── models/
│   └── best_model.pth         # Pre-trained PyTorch model (43MB)
├── frontend/
│   ├── package.json           # Node.js dependencies
│   ├── index.html
│   └── src/
└── nginx.conf                 # (optional) Reverse proxy configuration
```

---

## Understanding Docker in This Project

### What is Docker?

Docker packages your entire application (code, dependencies, runtime) into a container that runs identically on any machine.

### Benefits for This Project

| Benefit | Description |
|---------|-------------|
| **Consistency** | Runs same way on dev, testing, and production |
| **Isolation** | Python 3.11, Node.js 20, PyTorch - all isolated |
| **Easy Deployment** | Deploy to cloud with single command |
| **Scalability** | Run multiple instances for load balancing |
| **No Version Conflicts** | Each container has its own dependencies |

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│        Docker Environment                   │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────────────┐                   │
│  │  Frontend Container  │                   │
│  │  - Node.js 20        │                   │
│  │  - React App         │                   │
│  │  - Port 3000         │                   │
│  └──────────────────────┘                   │
│           ↕                                  │
│  ┌──────────────────────┐                   │
│  │  Backend Container   │                   │
│  │  - Python 3.11       │                   │
│  │  - FastAPI           │                   │
│  │  - PyTorch           │                   │
│  │  - Port 8000         │                   │
│  └──────────────────────┘                   │
│           ↕                                  │
│  ┌──────────────────────┐                   │
│  │  Shared Network      │                   │
│  └──────────────────────┘                   │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: Start Everything with Docker Compose (Recommended)

```bash
# Navigate to project directory
cd "/home/hedi/Desktop/Project/Projet AI/Reconnaissance-Automatique-d-Aliments"

# Start all services
docker-compose up

# Access the application
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Web Interface: http://localhost:8000/web
```

### Option 2: Start in Background

```bash
# Start services in detached mode (background)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 3: Rebuild After Code Changes

```bash
# Rebuild images and start
docker-compose up --build

# Rebuild without cache (clean build)
docker-compose up --build --no-cache
```

---

## Detailed Commands

### Building Docker Images

#### Build Backend API Image Only

```bash
# Build the backend API image
docker build -f Dockerfile.backend -t food-ai-api:latest .

# Build with specific tag/version
docker build -f Dockerfile.backend -t food-ai-api:v1.0 .

# View build process in detail
docker build -f Dockerfile.backend -t food-ai-api:latest . --progress=plain
```

#### Build Frontend Image Only

```bash
# Build the frontend image
docker build -f Dockerfile.frontend -t food-ai-frontend:latest .

# Build with specific tag
docker build -f Dockerfile.frontend -t food-ai-frontend:v1.0 .
```

#### Build Both Using Docker Compose

```bash
# Build all services
docker-compose build

# Build specific service only
docker-compose build api
docker-compose build frontend

# Build without cache
docker-compose build --no-cache

# Build and push to registry
docker-compose build --push
```

### Viewing Images

```bash
# List all local Docker images
docker images

# List images related to this project
docker images | grep food-ai

# View image details
docker inspect food-ai-api:latest

# View image size
docker images --format "{{.Repository}}\t{{.Size}}" | grep food-ai
```

---

## Docker Compose Configuration

### Understanding docker-compose.yml

```yaml
version: '3.8'  # Compose file version

services:
  api:                          # Service name
    build:                      # Build configuration
      context: .               # Build context (project root)
      dockerfile: Dockerfile.backend  # Which Dockerfile to use
    container_name: food-ai-api      # Readable container name
    ports:
      - "8000:8000"            # Map port 8000 (host) → 8000 (container)
    volumes:                    # Mount directories
      - ./models:/app/models   # Mount models directory
      - ./data:/app/data       # Mount data directory
    networks:                   # Connect to network
      - food-ai-network
    depends_on:                 # Service dependencies
      - api                    # Frontend depends on API
    restart: unless-stopped    # Restart policy
```

### Service Details

| Service | Purpose | Port | Base Image |
|---------|---------|------|-----------|
| `api` | FastAPI backend | 8000 | python:3.11-slim |
| `frontend` | React frontend | 3000 | node:20-alpine |
| `nginx` | Reverse proxy (optional) | 80, 443 | nginx:alpine |

---

## Running Containers

### Start Services

```bash
# Start all services
docker-compose up

# Start specific service
docker-compose up api
docker-compose up frontend

# Start in background
docker-compose up -d

# Start and rebuild if needed
docker-compose up --build
```

### Stop Services

```bash
# Stop all services (containers remain)
docker-compose stop

# Stop specific service
docker-compose stop api

# Remove all containers and networks
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v
```

### View Logs

```bash
# View all service logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs -f api
docker-compose logs -f frontend

# Show last 100 lines
docker-compose logs --tail 100

# Show logs with timestamps
docker-compose logs -t
```

### Execute Commands in Running Containers

```bash
# Access backend container shell
docker-compose exec api bash

# Access frontend container shell
docker-compose exec frontend sh

# Run Python command in backend
docker-compose exec api python -c "import torch; print(torch.__version__)"

# Check if API is healthy
docker-compose exec api curl http://localhost:8000/health
```

### View Service Status

```bash
# Show running containers
docker-compose ps

# Show all containers (including stopped)
docker-compose ps -a

# Show detailed info
docker-compose stats

# Check service health
docker-compose ps
```

---

## Building Images

### Backend API Image

**What Dockerfile.backend does:**

1. Starts with `python:3.11-slim` (slim Python image)
2. Sets working directory to `/app`
3. Installs system dependencies (build-essential)
4. Copies and installs Python requirements
5. Installs PyTorch CPU version (to save space)
6. Copies project files and model
7. Exposes port 8000
8. Adds health check
9. Runs the FastAPI app

**Build command:**

```bash
docker build -f Dockerfile.backend -t food-ai-api:latest .
```

**Result:**
- Image size: ~3-4 GB (includes PyTorch)
- Built on: Linux minimal environment
- Includes: Python 3.11, PyTorch, FastAPI, model weights

### Frontend Image

**What Dockerfile.frontend does:**

1. **Builder stage**: 
   - Starts with `node:20-alpine`
   - Installs npm dependencies
   - Builds React app (`npm run build`)
   - Creates optimized `dist` folder

2. **Serve stage**:
   - Fresh Node.js Alpine image
   - Copies built app from builder
   - Uses `serve` to run production build
   - Exposes port 3000

**Build command:**

```bash
docker build -f Dockerfile.frontend -t food-ai-frontend:latest .
```

**Result:**
- Image size: ~150-200 MB
- Uses multi-stage build (optimized)
- Includes: Node.js 20, React app, serve

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Port Already in Use

```bash
# Problem: Error "bind: address already in use"

# Solution: Change port in docker-compose.yml
# Change: "8000:8000" to "8001:8000"

# Or kill existing process
lsof -i :8000              # Find process using port 8000
kill -9 <PID>              # Kill the process

# Or use a different compose file
docker-compose -f docker-compose.dev.yml up
```

#### 2. Model File Not Found

```bash
# Problem: "Model not found at /app/models/best_model.pth"

# Solution: Check volume mounts
docker-compose exec api ls -la /app/models

# Verify volume is mounted correctly
docker inspect food-ai-api | grep -A 20 Mounts
```

#### 3. Container Exits Immediately

```bash
# Check logs
docker-compose logs api

# Rebuild without cache
docker-compose build --no-cache api

# Check if requirements are installed
docker-compose exec api pip list | grep torch
```

#### 4. API Health Check Failing

```bash
# Check API status
docker-compose exec api curl http://localhost:8000/health

# View detailed logs
docker-compose logs -f api

# Test API manually
curl http://localhost:8000/
```

#### 5. Slow Build Times

```bash
# Solution: Docker uses caching, but clean build needed sometimes

# Check image layers
docker history food-ai-api:latest

# Build with progress output
docker build -f Dockerfile.backend -t food-ai-api:latest . --progress=plain

# Use BuildKit for faster builds (optional)
DOCKER_BUILDKIT=1 docker build -f Dockerfile.backend -t food-ai-api:latest .
```

#### 6. Permission Denied Errors

```bash
# On Linux, you might need to use sudo
sudo docker-compose up

# Or add user to docker group
sudo usermod -aG docker $USER
newgrp docker  # Apply group changes

# Then restart Docker daemon
```

#### 7. Network Issues Between Containers

```bash
# Check if containers can reach each other
docker-compose exec frontend ping api

# Verify network
docker network ls
docker network inspect <network_name>

# Check DNS resolution
docker-compose exec frontend nslookup api
```

---

## Production Deployment

### Creating Production Compose File

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    image: your-registry/food-ai-api:v1.0
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    restart: always
    networks:
      - food-ai-network

  frontend:
    image: your-registry/food-ai-frontend:v1.0
    ports:
      - "3000:3000"
    restart: always
    networks:
      - food-ai-network

networks:
  food-ai-network:
    driver: bridge
```

### Deploy with Production Compose

```bash
# Start production containers
docker-compose -f docker-compose.prod.yml up -d

# Monitor
docker-compose -f docker-compose.prod.yml logs -f

# Update services
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### Push to Registry (Docker Hub)

```bash
# Login to Docker Hub
docker login

# Tag image for Docker Hub
docker tag food-ai-api:latest yourusername/food-ai-api:v1.0
docker tag food-ai-frontend:latest yourusername/food-ai-frontend:v1.0

# Push images
docker push yourusername/food-ai-api:v1.0
docker push yourusername/food-ai-frontend:v1.0

# Pull and run from any machine
docker run -p 8000:8000 yourusername/food-ai-api:v1.0
```

### Deploy to Cloud Platforms

#### AWS EC2

```bash
# SSH into EC2 instance
ssh -i your-key.pem ec2-user@your-instance

# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker

# Clone project
git clone your-repo
cd Reconnaissance-Automatique-d-Aliments

# Run docker-compose
docker-compose up -d
```

#### Google Cloud Run

```bash
# Build and push to Google Container Registry
docker build -f Dockerfile.backend -t gcr.io/your-project/food-ai-api:latest .
docker push gcr.io/your-project/food-ai-api:latest

# Deploy
gcloud run deploy food-ai-api \
  --image gcr.io/your-project/food-ai-api:latest \
  --platform managed \
  --region us-central1 \
  --port 8000
```

#### Docker Swarm

```bash
# Initialize Docker Swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml food-ai

# View services
docker service ls

# Scale service
docker service scale food-ai_api=3
```

#### Kubernetes (K8s)

```bash
# Create deployment manifest (deployment.yaml)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: food-ai-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: food-ai-api
  template:
    metadata:
      labels:
        app: food-ai-api
    spec:
      containers:
      - name: api
        image: yourusername/food-ai-api:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"

# Deploy to K8s
kubectl apply -f deployment.yaml
kubectl get pods
kubectl logs <pod-name>
```

---

## Best Practices

### 1. Use .dockerignore

Exclude unnecessary files to reduce build time and image size:

```
.git
.gitignore
__pycache__
*.pyc
node_modules/
.env
venv/
```

### 2. Use Specific Base Image Versions

❌ Don't:
```dockerfile
FROM python:latest
FROM node:latest
```

✅ Do:
```dockerfile
FROM python:3.11-slim
FROM node:20-alpine
```

### 3. Minimize Layer Count

```dockerfile
# ❌ Many layers (bad)
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git

# ✅ Single layer (good)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*
```

### 4. Order Dockerfile Instructions Efficiently

```dockerfile
# ❌ Bad order - rebuilds on any change
COPY . .
RUN pip install -r requirements.txt

# ✅ Good order - only rebuilds if requirements change
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### 5. Use Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
```

### 6. Set Environment Variables

```dockerfile
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production
```

### 7. Use Volumes for Persistence

```yaml
volumes:
  - ./models:/app/models        # Model files
  - ./data:/app/data            # Data directory
  - model-cache:/root/.cache    # Cache directory
```

### 8. Set Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 9. Use Multi-Stage Builds

```dockerfile
# Stage 1: Build
FROM node:20 AS builder
COPY . .
RUN npm install && npm run build

# Stage 2: Runtime (smaller)
FROM node:20-alpine
COPY --from=builder /app/dist ./dist
CMD ["serve", "-s", "dist"]
```

### 10. Monitor Container Health

```bash
# Check container status
docker ps --format "table {{.Names}}\t{{.Status}}"

# View resource usage
docker stats

# Check logs for errors
docker logs food-ai-api | grep ERROR
```

---

## Useful Commands Reference

| Command | Purpose |
|---------|---------|
| `docker-compose up` | Start all services |
| `docker-compose up -d` | Start in background |
| `docker-compose down` | Stop and remove containers |
| `docker-compose build` | Build all images |
| `docker-compose logs -f` | View live logs |
| `docker-compose ps` | Show running services |
| `docker-compose exec api bash` | Access container shell |
| `docker images` | List all images |
| `docker ps` | List running containers |
| `docker stop <container>` | Stop a container |
| `docker rm <container>` | Remove a container |
| `docker rmi <image>` | Remove an image |
| `docker inspect <container>` | Detailed container info |

---

## Next Steps

1. **Build the images**:
   ```bash
   docker-compose build
   ```

2. **Start the services**:
   ```bash
   docker-compose up
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

4. **Monitor logs**:
   ```bash
   docker-compose logs -f
   ```

5. **Deploy to production** (see Production Deployment section)

---

## Additional Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Docker Hub](https://hub.docker.com/)
- [Play with Docker](https://labs.play-with-docker.com/)

---

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. View container logs: `docker-compose logs`
3. Check Docker documentation
4. Consult project README.md

