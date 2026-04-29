#!/usr/bin/env python3
"""
Setup MLflow tracking server for Food Classification project
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_mlflow_installed():
    """Check if MLflow is installed"""
    try:
        import mlflow
        print(f"✅ MLflow is installed (version: {mlflow.__version__})")
        return True
    except ImportError:
        print("❌ MLflow not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mlflow"])
            print("✅ MLflow installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install MLflow")
            return False

def create_mlflow_directories():
    """Create necessary directories for MLflow"""
    directories = [
        "./mlruns",
        "./mlflow/artifacts",
        "./mlflow/models",
        "./mlflow/data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def start_mlflow_server():
    """Start MLflow tracking server"""
    print("🚀 Starting MLflow tracking server...")
    
    # Check if server is already running
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow server is already running!")
            return True
    except requests.exceptions.ConnectionError:
        pass
    
    # Start server in background
    try:
        # Start MLflow server
        process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "server", 
            "--host", "0.0.0.0", 
            "--port", "5000",
            "--backend-store-uri", "sqlite:///mlflow.db",
            "--default-artifact-root", "./mlruns"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("⏳ Waiting for MLflow server to start...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:5000", timeout=2)
                if response.status_code == 200:
                    print("✅ MLflow server started successfully!")
                    print(f"📊 MLflow UI: http://localhost:5000")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            
            time.sleep(1)
            print(f"   Checking... ({i+1}/30)")
        
        print("❌ MLflow server failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start MLflow server: {e}")
        return False

def setup_mlflow_experiment():
    """Setup initial MLflow experiment"""
    try:
        import mlflow
        
        # Set tracking URI
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Create experiment
        experiment_name = "food-classification"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags={
                    "project": "food-classification",
                    "version": "1.0",
                    "framework": "pytorch",
                    "dataset": "food-101"
                }
            )
            print(f"✅ Created MLflow experiment: {experiment_name}")
        else:
            print(f"✅ Using existing MLflow experiment: {experiment_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to setup MLflow experiment: {e}")
        return False

def test_mlflow_connection():
    """Test MLflow connection and logging"""
    try:
        import mlflow
        
        # Test logging
        with mlflow.start_run(experiment_id="food-classification", run_name="test-run") as run:
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.95)
            mlflow.set_tag("test", True)
        
        print("✅ MLflow connection and logging test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MLflow test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 MLflow Setup for Food Classification")
    print("=" * 50)
    
    # Step 1: Check MLflow installation
    if not check_mlflow_installed():
        return False
    
    # Step 2: Create directories
    create_mlflow_directories()
    
    # Step 3: Start MLflow server
    if not start_mlflow_server():
        print("⚠️ Please start MLflow server manually:")
        print("   mlflow server --host 0.0.0.0 --port 5000")
        return False
    
    # Step 4: Setup experiment
    if not setup_mlflow_experiment():
        return False
    
    # Step 5: Test connection
    if not test_mlflow_connection():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 MLflow setup complete!")
    print("\n📊 MLflow UI: http://localhost:5000")
    print("📖 API Docs: http://localhost:5000/docs")
    print("\n📝 Next steps:")
    print("1. Open MLflow UI in browser")
    print("2. Run training notebooks with MLflow tracking")
    print("3. Monitor experiments and model performance")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
