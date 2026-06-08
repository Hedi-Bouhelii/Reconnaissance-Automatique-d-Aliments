#!/bin/bash

echo "🚀 Starting Food Recognition AI Application"
echo "=========================================="

# Activate venv
source venv/bin/activate

# Check if model exists
if [ ! -f "models/best_model.pth" ]; then
    echo "❌ Model file not found!"
    exit 1
fi

echo "✅ Model found"
echo "✅ Dependencies ready"
echo ""
echo "📊 Starting Backend API Server on http://localhost:8000"
echo "🌐 Web Interface: http://localhost:8000/web"
echo "📖 API Docs: http://localhost:8000/docs"
echo ""

# Start API in background
python app.py &
API_PID=$!

sleep 3

echo ""
echo "🎨 Starting Frontend Dev Server on http://localhost:5173"
echo ""

# Start frontend
cd frontend && npm run dev &
FRONTEND_PID=$!

# Keep running
wait
