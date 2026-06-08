#!/bin/bash

# 🐳 Food Recognition AI - Docker Starter Script
# This script makes it easy to run the Docker application

set -e

PROJECT_DIR="/home/hedi/Desktop/Project/Projet AI/Reconnaissance-Automatique-d-Aliments"

cd "$PROJECT_DIR"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🐳 Food Recognition AI - Docker Startup                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

echo "✅ Docker found: $(docker --version)"

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available."
    exit 1
fi

echo "✅ Docker Compose found: $(docker compose version --short)"
echo ""

# Show menu
echo "🎯 What would you like to do?"
echo ""
echo "  1. Start application (production mode)"
echo "  2. Start with hot reload (development mode)"
echo "  3. Build images only"
echo "  4. Stop all services"
echo "  5. View logs"
echo "  6. Show help"
echo ""
read -p "Select an option (1-6): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting application in production mode..."
        echo ""
        echo "⏳ Building images (this may take 10-15 minutes on first run)..."
        docker compose build
        echo ""
        echo "▶️  Starting services..."
        docker compose up -d
        echo ""
        echo "✅ Services started!"
        echo ""
        echo "🌐 Access the application:"
        echo "   Frontend: http://localhost:3000"
        echo "   Backend:  http://localhost:8000"
        echo "   API Docs: http://localhost:8000/docs"
        echo ""
        echo "📊 View logs: docker compose logs -f"
        echo "⛔ Stop:      docker compose down"
        ;;
    2)
        echo ""
        echo "🚀 Starting application in development mode (with hot reload)..."
        echo ""
        echo "⏳ Building images..."
        docker compose -f docker-compose.dev.yml build
        echo ""
        echo "▶️  Starting services with hot reload..."
        docker compose -f docker-compose.dev.yml up
        ;;
    3)
        echo ""
        echo "🔨 Building Docker images..."
        docker compose build
        echo ""
        echo "✅ Build complete!"
        echo "   Images created: food-ai-api, food-ai-frontend"
        echo ""
        echo "To start: docker compose up"
        ;;
    4)
        echo ""
        echo "⛔ Stopping all services..."
        docker compose down
        echo "✅ Services stopped!"
        ;;
    5)
        echo ""
        echo "📊 Showing logs from all services..."
        echo "(Press Ctrl+C to stop)"
        docker compose logs -f
        ;;
    6)
        echo ""
        echo "📚 Documentation Files:"
        echo ""
        echo "  📖 INDEX.md"
        echo "     Quick navigation guide (START HERE)"
        echo ""
        echo "  📖 DOCKER_README.md"
        echo "     Quick start guide (5-minute read)"
        echo ""
        echo "  📖 DOCKER_QUICK_REFERENCE.md"
        echo "     Commands cheatsheet"
        echo ""
        echo "  📖 DOCKER_GUIDE.md"
        echo "     Comprehensive guide (80+ sections)"
        echo ""
        echo "  📖 DOCKER_SETUP_SUMMARY.md"
        echo "     Setup overview and next steps"
        echo ""
        echo "🛠️  Makefile Commands:"
        echo ""
        docker make help 2>/dev/null || make help
        ;;
    *)
        echo "❌ Invalid option. Please select 1-6."
        exit 1
        ;;
esac
