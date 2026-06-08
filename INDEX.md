# 📑 Docker Documentation Index

Quick guide to which file to read based on what you want to do.

## 🚀 Just Want to Get Started?

**Read**: `DOCKER_README.md`  
**Time**: 5 minutes  
**Content**: Quick start, basic commands, common workflows

## 📖 Want to Learn Docker?

**Read**: `DOCKER_GUIDE.md`  
**Time**: 30-60 minutes  
**Content**: Complete guide, best practices, production deployment, troubleshooting

## ⚡ Need Quick Commands?

**Read**: `DOCKER_QUICK_REFERENCE.md`  
**Time**: 2-5 minutes  
**Content**: Command reference, common tasks, one-liners

## 📋 Project Summary

**Read**: `DOCKER_SETUP_SUMMARY.md`  
**Time**: 10 minutes  
**Content**: What was created, architecture, next steps

## 🛠️ Prefer Easy Commands?

**Use**: `Makefile`  
```bash
make help        # Show all commands
make up          # Start services
make down        # Stop services
make logs        # View logs
```

---

## File Quick Reference

| File | Best For | Time | Content |
|------|----------|------|---------|
| `DOCKER_README.md` | Getting started | 5 min | Quick start, workflows |
| `DOCKER_GUIDE.md` | Learning Docker | 30-60 min | Complete reference |
| `DOCKER_QUICK_REFERENCE.md` | Quick lookup | 2-5 min | Commands & tips |
| `DOCKER_SETUP_SUMMARY.md` | Overview | 10 min | What was created |
| `Makefile` | Easy commands | 1 min | Simple commands |

---

## Common Scenarios

### "I just want to run it"
```bash
docker compose up --build
```
→ Read: `DOCKER_README.md`

### "It's not working, help!"
```bash
docker compose logs -f
```
→ Read: `DOCKER_GUIDE.md` (Troubleshooting section)

### "What command do I use for...?"
→ Read: `DOCKER_QUICK_REFERENCE.md`

### "I want to deploy to the cloud"
→ Read: `DOCKER_GUIDE.md` (Production Deployment section)

### "Tell me what was set up"
→ Read: `DOCKER_SETUP_SUMMARY.md`

### "I'm learning Docker from scratch"
→ Read: `DOCKER_GUIDE.md` (Understanding Docker section)

---

## Directory Structure

```
Documentation Files:
├── INDEX.md                     ← You are here
├── DOCKER_README.md             ← Start here! (5 min read)
├── DOCKER_GUIDE.md              ← Comprehensive guide (80 sections)
├── DOCKER_QUICK_REFERENCE.md    ← Commands cheatsheet
└── DOCKER_SETUP_SUMMARY.md      ← What was created

Configuration Files:
├── Dockerfile.backend           ← Backend container
├── Dockerfile.frontend          ← Frontend container
├── docker-compose.yml           ← Production setup
├── docker-compose.dev.yml       ← Development setup
├── .dockerignore                ← Build exclusions
├── nginx.conf                   ← Reverse proxy
├── docker-entrypoint.sh         ← Startup script
└── Makefile                     ← Easy commands
```

---

## Quick Start (Pick One)

### Option 1: Just Run It
```bash
cd "/home/hedi/Desktop/Project/Projet AI/Reconnaissance-Automatique-d-Aliments"
docker compose up --build
```

### Option 2: Using Makefile
```bash
cd "/home/hedi/Desktop/Project/Projet AI/Reconnaissance-Automatique-d-Aliments"
make up
```

### Option 3: Development Mode
```bash
cd "/home/hedi/Desktop/Project/Projet AI/Reconnaissance-Automatique-d-Aliments"
docker compose -f docker-compose.dev.yml up
```

Then access:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Reading Order (Recommended)

1. **This file** (2 min) - Overview
2. **DOCKER_README.md** (5 min) - Quick start
3. **DOCKER_SETUP_SUMMARY.md** (10 min) - What was created
4. **DOCKER_QUICK_REFERENCE.md** (As needed) - Commands
5. **DOCKER_GUIDE.md** (30-60 min) - Deep dive

---

## What Each File Does

### Dockerfiles
- `Dockerfile.backend` → Builds FastAPI backend image
- `Dockerfile.frontend` → Builds React frontend image

### Compose Files
- `docker-compose.yml` → Production setup (optimized)
- `docker-compose.dev.yml` → Development setup (with hot reload)

### Documentation
- `DOCKER_README.md` → Quick start guide
- `DOCKER_GUIDE.md` → Comprehensive guide (3000+ lines!)
- `DOCKER_QUICK_REFERENCE.md` → Command reference
- `DOCKER_SETUP_SUMMARY.md` → Setup overview
- `INDEX.md` → This file

### Other
- `Makefile` → Easy commands (make up, make down, etc.)
- `nginx.conf` → Optional reverse proxy
- `docker-entrypoint.sh` → Startup script
- `.dockerignore` → Files to exclude from build

---

## Key Commands

```bash
# Start services
docker compose up
docker compose up -d                    # Background
docker compose -f docker-compose.dev.yml up  # Development

# Stop services
docker compose down

# View logs
docker compose logs -f
docker compose logs -f api
docker compose logs -f frontend

# Build images
docker compose build
docker compose build --no-cache

# Access containers
docker compose exec api bash
docker compose exec frontend sh

# Check status
docker compose ps
docker stats

# Using Makefile (easier!)
make up
make down
make logs
make build
make help
```

---

## Next Steps

1. **Start here**: Read `DOCKER_README.md`
2. **Build & run**: `docker compose up --build`
3. **Access app**: http://localhost:3000
4. **Learn more**: Read `DOCKER_GUIDE.md`
5. **For quick help**: Use `DOCKER_QUICK_REFERENCE.md`

---

## Support

- 📚 **Full documentation**: `DOCKER_GUIDE.md`
- ⚡ **Quick commands**: `DOCKER_QUICK_REFERENCE.md`
- 🔍 **What was created**: `DOCKER_SETUP_SUMMARY.md`
- 🛠️ **Easy commands**: `make help`
- 🌐 **Docker docs**: https://docs.docker.com

---

**Start with `DOCKER_README.md` → 5 minute quick start! 🚀**
