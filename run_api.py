#!/usr/bin/env python3
"""
Script to run the Food Classification API with proper setup
"""

import sys
import subprocess
from pathlib import Path

def check_model():
    """Check if model exists"""
    model_path = Path('./models/best_model.pth')
    if not model_path.exists():
        print("❌ Model not found!")
        print("Please run the model training first:")
        print("1. Open 03_Model_Training.ipynb")
        print("2. Run all cells to train the model")
        print("3. Ensure best_model.pth is saved in ./models/")
        return False
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing API dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_api.txt"
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def main():
    print("🚀 Food Classification API Setup")
    print("=" * 40)
    
    # Check model
    if not check_model():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    print("\n✅ Setup complete!")
    print("\n🌐 Starting API server...")
    print("📊 Available endpoints:")
    print("  - http://localhost:8000/web     : Interactive web interface")
    print("  - http://localhost:8000/docs    : API documentation")
    print("  - http://localhost:8000/health  : Health check")
    print("\n💡 Usage:")
    print("1. Open http://localhost:8000/web in your browser")
    print("2. Upload a food image to get predictions")
    print("3. Or use the API endpoints programmatically")
    
    # Start the API
    try:
        import uvicorn
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    except ImportError:
        print("❌ uvicorn not found. Please install dependencies first.")
    except Exception as e:
        print(f"❌ Error starting API: {e}")

if __name__ == "__main__":
    main()
