# 🐳 Docker Setup - Food Recognition AI

## Quick Start (2 minutes)

```bash
# Navigate to project directory
cd "/home/hedi/Desktop/Project/Projet AI/Reconnaissance-Automatique-d-Aliments"

# Build and start services
docker compose up --build

# Access the application
# Frontend:  http://localhost:3000
# Backend:   http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

That's it! 🎉

---

## Files Created

Here's what was set up for Docker:

| File | Purpose |
|------|---------|
| `Dockerfile.backend` | Builds the FastAPI backend image |
| `Dockerfile.frontend` | Builds the React frontend image |
| `docker-compose.yml` | Orchestrates both services (production) |
| `docker-compose.dev.yml` | Development setup with hot reload |
| `.dockerignore` | Excludes unnecessary files from build |
| `DOCKER_GUIDE.md` | **Comprehensive guide** (read this!) |
| `DOCKER_QUICK_REFERENCE.md` | Quick command reference |
| `Makefile` | Convenient commands (e.g., `make up`) |
| `nginx.conf` | Optional reverse proxy configuration |
| `docker-entrypoint.sh` | Startup script for the API |

---

## Commands

### Using Docker Compose (Recommended)

```bash
# Start services
docker compose up

# Start in background
docker compose up -d

# Stop services
docker compose down

# Rebuild images
docker compose build

# View logs
docker compose logs -f

# Access container shell
docker compose exec api bash
docker compose exec frontend sh

# See full list of commands
docker compose help
```

### Using Makefile (Easiest!)

```bash
make up              # Start services
make down            # Stop services
make logs            # View logs
make shell-api       # Access API shell
make build           # Build images
make clean           # Cleanup

make help            # Show all commands
```

---

## Directory Structure

```
Reconnaissance-Automatique-d-Aliments/
├── 📄 Dockerfile.backend          ← Backend API image
├── 📄 Dockerfile.frontend         ← Frontend image
├── 📄 docker-compose.yml          ← Production setup
├── 📄 docker-compose.dev.yml      ← Development setup
├── 📄 .dockerignore               ← Files to exclude
├── 📄 DOCKER_GUIDE.md             ← Full documentation
├── 📄 DOCKER_QUICK_REFERENCE.md   ← Command reference
├── 📄 Makefile                    ← Easy commands
├── 📄 nginx.conf                  ← Reverse proxy
├── 📄 docker-entrypoint.sh        ← Startup script
│
├── 📂 app.py                      ← FastAPI app
├── 📂 requirements.txt            ← Python dependencies
│
├── 📂 frontend/                   ← React app
│   ├── package.json
│   ├── src/
│   └── ...
│
├── 📂 models/
│   └── best_model.pth             ← AI model
│
└── 📂 data/                       ← Data directory
```

---

## What Each Docker File Does

### **Dockerfile.backend**
- **Base Image**: Python 3.11 (slim)
- **Installs**: PyTorch, FastAPI, Uvicorn
- **Copies**: App code, models, source files
- **Exposes**: Port 8000
- **Runs**: `python app.py`
- **Size**: ~3-4 GB (includes PyTorch)

### **Dockerfile.frontend**
- **Stage 1 (Build)**:
  - Base Image: Node.js 20
  - Installs dependencies with npm
  - Builds React app
  
- **Stage 2 (Runtime)**:
  - Base Image: Node.js 20-alpine (small)
  - Copies built app
  - Uses `serve` to run production build
  - Exposes: Port 3000
  - Size: ~150-200 MB (optimized)

---

## Common Workflows

### 🚀 Get Started Immediately

```bash
docker compose up --build
```

Then open:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 🔧 Development (with Hot Reload)

```bash
# Use development compose file
docker compose -f docker-compose.dev.yml up

# Or with Makefile
make up-dev
```

- **Frontend hot reload**: http://localhost:5173
- **Backend auto-restart**: http://localhost:8000
- Code changes are reflected immediately!

### 🔍 Debug Issues

```bash
# View logs
docker compose logs -f

# Access container shell
docker compose exec api bash

# Check if services are healthy
docker compose ps

# View resource usage
docker stats
```

### 🧹 Clean Up

```bash
# Stop and remove containers
docker compose down

# Also remove volumes (data loss!)
docker compose down -v

# Remove images
docker image rm food-ai-api food-ai-frontend

# Clean everything
docker system prune -a
```

### 📤 Deploy to Production

1. **Push to registry**:
   ```bash
   docker tag food-ai-api yourusername/food-ai-api:v1.0
   docker push yourusername/food-ai-api:v1.0
   ```

2. **Use production compose file**:
   ```bash
   docker compose -f docker-compose.prod.yml up -d
   ```

3. **Or deploy to cloud** (see DOCKER_GUIDE.md)

---

## Ports Used

| Service | Port | Access URL |
|---------|------|------------|
| Backend API | 8000 | http://localhost:8000 |
| Frontend | 3000 | http://localhost:3000 |
| Frontend (dev) | 5173 | http://localhost:5173 |
| Nginx (optional) | 80, 443 | http://localhost |

---

## Environment Setup

### Production Environment
- Uses `docker-compose.yml`
- Backend on port 8000
- Frontend on port 3000
- Optimal for deployment

### Development Environment
- Uses `docker-compose.dev.yml`
- Frontend on port 5173 (with hot reload)
- Backend on port 8000 (with auto-restart)
- Volumes mounted for live code editing

---

## Docker Images

Once built, you'll have these images:

```bash
$ docker images | grep food-ai

food-ai-api          latest    3-4 GB    ← Backend (PyTorch heavy)
food-ai-frontend     latest    200 MB    ← Frontend (optimized)
```

To rebuild from scratch:
```bash
docker compose build --no-cache
```

---

## Performance Tips

1. **Use `.dockerignore`**: Excludes unnecessary files from build
2. **Multi-stage builds**: Frontend uses 2 stages for optimization
3. **Cache layers**: Order dependencies wisely in Dockerfile
4. **Alpine images**: Frontend uses lightweight Alpine Linux
5. **Volume mounts**: Development reloads without rebuilding

---

## Troubleshooting

### Port Already in Use
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead of 8000
```

### Model Not Found
```bash
# Check volume mount
docker compose exec api ls -la /app/models/
```

### Container Won't Start
```bash
# View logs
docker compose logs api

# Try rebuilding
docker compose build --no-cache api
```

### Out of Disk Space
```bash
# Clean up Docker
docker system prune -a -f

# Check sizes
docker system df
```

For more detailed troubleshooting, see **DOCKER_GUIDE.md**

---

## Next Steps

1. **Start the application**:
   ```bash
   docker compose up --build
   ```

2. **Open in browser**:
   - http://localhost:3000 (Frontend)
   - http://localhost:8000/docs (API Documentation)

3. **Learn more**:
   - Read `DOCKER_GUIDE.md` for comprehensive guide
   - Read `DOCKER_QUICK_REFERENCE.md` for commands
   - Use `make help` for Makefile commands

4. **Deploy when ready**:
   - See "Production Deployment" in DOCKER_GUIDE.md
   - Push images to Docker Hub
   - Deploy to cloud platforms

---

## Key Commands to Remember

| Task | Command |
|------|---------|
| Start | `docker compose up` |
| Stop | `docker compose down` |
| Logs | `docker compose logs -f` |
| Shell | `docker compose exec api bash` |
| Build | `docker compose build` |
| Status | `docker compose ps` |

---

## Important Notes

⚠️ **Remember**:
- First build takes 10-15 minutes (includes PyTorch)
- Subsequent builds are faster (Docker caching)
- Model file (43 MB) is required in `models/` folder
- Use `.env` file for sensitive configuration
- Never commit passwords or API keys to git

---

## Support & Resources

- **Full Guide**: See `DOCKER_GUIDE.md`
- **Quick Commands**: See `DOCKER_QUICK_REFERENCE.md`
- **Easy Commands**: Use `make help`
- **Docker Docs**: https://docs.docker.com/
- **Docker Compose**: https://docs.docker.com/compose/

---

**Happy containerizing! 🐳** 🚀
