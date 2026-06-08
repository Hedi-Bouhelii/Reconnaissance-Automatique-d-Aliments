#!/bin/sh
# Docker entrypoint script for the API service

set -e

echo "🚀 Starting Food Classification API..."
echo "========================================"

# Wait for any services to be ready (if using wait-for-it)
if [ -z "$SKIP_STARTUP_CHECKS" ]; then
    echo "✓ Startup checks passed"
fi

# Print environment info
echo "✓ Python version: $(python --version)"
echo "✓ PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check if model exists
if [ -f "/app/models/best_model.pth" ]; then
    echo "✓ Model found: /app/models/best_model.pth"
else
    echo "⚠ Warning: Model not found at /app/models/best_model.pth"
    echo "  The API will start but predictions won't work"
fi

echo "========================================"
echo "✅ Startup complete! Starting application..."
echo ""

# Execute the main command
exec "$@"
