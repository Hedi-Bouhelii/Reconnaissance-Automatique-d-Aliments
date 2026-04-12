import torch
import torchvision
import fastapi
import mlflow
import numpy as np
import matplotlib
import sklearn

print("=" * 40)
print("SETUP VERIFICATION")
print("=" * 40)
print(f"Python:       OK")
print(f"PyTorch:      {torch.__version__}")
print(f"Torchvision:  {torchvision.__version__}")
print(f"FastAPI:      {fastapi.__version__}")
print(f"MLflow:       {mlflow.__version__}")
print(f"NumPy:        {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print("=" * 40)

# GPU Check
if torch.cuda.is_available():
    print(f"GPU:          ✅ {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU:          ❌ Not detected (will use CPU)")

print("=" * 40)
print("All good! You can start the project.")