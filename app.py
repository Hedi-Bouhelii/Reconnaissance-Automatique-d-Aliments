#!/usr/bin/env python3
"""
Food Classification FastAPI Web Service
Provides REST API for food image classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import json
from pathlib import Path
import uvicorn
import numpy as np
from typing import List, Dict, Any

# Initialize FastAPI app
app = FastAPI(
    title="Food Classification API",
    description="AI-powered food classification using PyTorch",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
DATA_DIR = Path('./data')
FOOD10_DIR = DATA_DIR / 'food10'
MODELS_DIR = Path('./models')

SELECTED_CLASSES = [
    'pizza', 'hamburger', 'hot_dog', 'french_fries',
    'ice_cream', 'omelette', 'pancakes', 'ramen', 'steak',
    'fried_rice'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

IMG_SIZE = 224

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model architecture
class FoodClassifier(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(FoodClassifier, self).__init__()
        
        import torchvision.models as models
        self.backbone = models.resnet18(pretrained=pretrained)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Global model instance
model = None
transform = None

def load_model():
    """Load the trained model"""
    global model, transform
    
    try:
        # Model path
        model_path = MODELS_DIR / 'best_model.pth'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Initialize model
        model = FoodClassifier(num_classes=len(SELECTED_CLASSES), pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Setup transforms
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image_bytes):
    """Preprocess image for inference"""
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        input_tensor = transform(image).unsqueeze(0)
        return input_tensor.to(device)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def predict_image(input_tensor, top_k=3):
    """Make prediction on preprocessed image"""
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            results = []
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                class_name = IDX_TO_CLASS[class_idx]
                confidence = top_probs[0][i].item()
                
                results.append({
                    'class': class_name,
                    'confidence': float(confidence),
                    'rank': i + 1
                })
            
            return results
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    success = load_model()
    if not success:
        print("⚠️ Warning: Model not loaded. API will not work properly.")

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Food Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "classes": "/classes",
            "web_interface": "/web"
        },
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": len(SELECTED_CLASSES)
    }

@app.get("/classes")
async def get_classes():
    """Get available food classes"""
    return {
        "classes": SELECTED_CLASSES,
        "total": len(SELECTED_CLASSES),
        "class_to_idx": CLASS_TO_IDX
    }

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    """Predict food class from uploaded image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file content
        image_bytes = await file.read()
        
        # Preprocess
        input_tensor = preprocess_image(image_bytes)
        
        # Predict
        predictions = predict_image(input_tensor, top_k=3)
        
        return {
            "success": True,
            "filename": file.filename,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/web", response_class=HTMLResponse)
async def web_interface():
    """Interactive web interface for testing"""
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>🍕 Food Classification Demo</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .upload-area {
            border: 3px dashed #cbd5e1;
            transition: all 0.3s ease;
        }
        .upload-area:hover {
            border-color: #3b82f6;
            background-color: #f0f9ff;
        }
        .upload-area.dragover {
            border-color: #3b82f6;
            background-color: #dbeafe;
        }
        .confidence-bar {
            transition: width 0.5s ease;
        }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-2xl">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">🍕 Food Classification AI</h1>
            <p class="text-gray-600">Upload an image to classify the food type using our AI model</p>
        </div>

        <!-- Upload Area -->
        <div id="uploadArea" class="upload-area bg-white rounded-xl p-8 mb-6 cursor-pointer">
            <div class="text-center">
                <div class="mb-4">
                    <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                </div>
                <p class="text-lg font-medium text-gray-700 mb-2">Drop your image here or click to browse</p>
                <p class="text-sm text-gray-500">Supports JPG, PNG, JPEG (Max 10MB)</p>
            </div>
        </div>

        <input type="file" id="fileInput" accept="image/*" class="hidden">

        <!-- Preview -->
        <div id="previewContainer" class="hidden mb-6">
            <div class="bg-white rounded-xl p-4 shadow-sm">
                <h3 class="text-lg font-semibold mb-3">📸 Preview</h3>
                <img id="preview" class="w-full max-h-64 object-contain rounded-lg">
            </div>
        </div>

        <!-- Loading -->
        <div id="loading" class="hidden mb-6">
            <div class="bg-blue-50 border border-blue-200 rounded-xl p-4 text-center">
                <div class="inline-flex items-center">
                    <svg class="animate-spin h-5 w-5 mr-3 text-blue-600" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span class="text-blue-700 font-medium">🤖 AI is analyzing your image...</span>
                </div>
            </div>
        </div>

        <!-- Error -->
        <div id="error" class="hidden mb-6">
            <div class="bg-red-50 border border-red-200 rounded-xl p-4">
                <div class="flex items-center">
                    <svg class="h-5 w-5 text-red-600 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span id="errorMessage" class="text-red-700"></span>
                </div>
            </div>
        </div>

        <!-- Results -->
        <div id="results" class="hidden mb-6">
            <div class="bg-white rounded-xl p-6 shadow-sm">
                <h3 class="text-xl font-bold mb-4">🎯 Prediction Results</h3>
                <div id="predictions"></div>
            </div>
        </div>

        <!-- Info -->
        <div class="bg-gray-100 rounded-xl p-4 text-center text-sm text-gray-600">
            <p>🤖 This AI can identify 10 food types: Pizza, Hamburger, Hot Dog, French Fries, Ice Cream, Omelette, Pancakes, Ramen, Steak, Fried Rice</p>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const previewContainer = document.getElementById('previewContainer');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const errorMessage = document.getElementById('errorMessage');
        const results = document.getElementById('results');
        const predictions = document.getElementById('predictions');

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please upload an image file.');
                return;
            }

            if (file.size > 10 * 1024 * 1024) {
                showError('File size must be less than 10MB.');
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                predictImage(file);
            };
            reader.readAsDataURL(file);
        }

        async function predictImage(file) {
            hideError();
            hideResults();
            showLoading();

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || 'Prediction failed');
                }

                hideLoading();
                showResults(data.predictions);

            } catch (err) {
                hideLoading();
                showError(err.message);
            }
        }

        function showResults(predictionData) {
            predictions.innerHTML = '';
            
            predictionData.forEach((pred, index) => {
                const confidencePercent = Math.round(pred.confidence * 100);
                
                const item = document.createElement('div');
                item.className = 'mb-4';
                item.innerHTML = `
                    <div class="flex items-center justify-between mb-2">
                        <div class="flex items-center">
                            <span class="text-lg font-bold text-gray-700 mr-2">${index + 1}.</span>
                            <span class="text-lg font-medium capitalize">${pred.class.replace('_', ' ')}</span>
                        </div>
                        <span class="text-lg font-bold text-blue-600">${confidencePercent}%</span>
                    </div>
                    <div class="w-full bg-gray-200 rounded-full h-3">
                        <div class="confidence-bar bg-gradient-to-r from-blue-500 to-blue-600 h-3 rounded-full" 
                             style="width: ${confidencePercent}%"></div>
                    </div>
                `;
                
                predictions.appendChild(item);
            });
            
            results.classList.remove('hidden');
        }

        function showLoading() {
            loading.classList.remove('hidden');
        }

        function hideLoading() {
            loading.classList.add('hidden');
        }

        function showError(message) {
            errorMessage.textContent = message;
            error.classList.remove('hidden');
        }

        function hideError() {
            error.classList.add('hidden');
        }

        function hideResults() {
            results.classList.add('hidden');
        }
    </script>
</body>
</html>
    """
    
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("🚀 Starting Food Classification API...")
    print("📊 Available endpoints:")
    print("  - GET  /        : API info")
    print("  - GET  /health  : Health check")
    print("  - GET  /classes : Available classes")
    print("  - POST /predict : Image prediction")
    print("  - GET  /web     : Web interface")
    print(f"🌐 Web interface: http://localhost:8000/web")
    print(f"📖 API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
