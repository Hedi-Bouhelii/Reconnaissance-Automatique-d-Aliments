# 🍕 Food Classification API

A FastAPI-based web service for food image classification using PyTorch.

## 🚀 Quick Start

### 1. Train the Model First
Before using the API, you need to train the model:

```bash
# Open and run the training notebook
jupyter notebook 03_Model_Training.ipynb
```

### 2. Install Dependencies
```bash
pip install -r requirements_api.txt
```

### 3. Start the API Server
```bash
python run_api.py
```

### 4. Test the API
```bash
# Automated testing
python test_api.py

# Or open in browser
http://localhost:8000/web
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/classes` | Available food classes |
| POST | `/predict` | Predict food from image |
| GET | `/web` | Interactive web interface |

## 🌐 Web Interface

Visit `http://localhost:8000/web` for an interactive demo where you can:
- Upload food images
- Get instant AI predictions
- See confidence scores
- View top 3 predictions

## 📱 Usage Examples

### Python Client
```python
import requests

# Upload image for prediction
with open('pizza.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    
result = response.json()
print(f"Prediction: {result['top_prediction']['class']}")
print(f"Confidence: {result['top_prediction']['confidence']:.2%}")
```

### curl Commands
```bash
# Health check
curl -X GET http://localhost:8000/health

# Get classes
curl -X GET http://localhost:8000/classes

# Predict food
curl -X POST -F "file=@pizza.jpg" http://localhost:8000/predict
```

## 🤖 Model Information

- **Architecture**: ResNet18-based CNN
- **Classes**: 10 food types
  - Pizza, Hamburger, Hot Dog, French Fries
  - Ice Cream, Omelette, Pancakes, Ramen, Steak, Fried Rice
- **Input**: 224x224 RGB images
- **Output**: Class probabilities with confidence scores

## 📁 Project Structure

```
project ai/
├── app.py                 # FastAPI application
├── run_api.py            # API startup script
├── test_api.py           # API testing script
├── requirements_api.txt  # API dependencies
├── models/
│   └── best_model.pth    # Trained model
├── data/
│   └── food10/          # Dataset
└── notebooks/           # Jupyter notebooks
    ├── 03_Model_Training.ipynb
    └── 05_Inference_Fixed.ipynb
```

## 🔧 Configuration

- **Port**: 8000 (default)
- **Host**: 0.0.0.0 (accessible from network)
- **Model Path**: `./models/best_model.pth`
- **Device**: Auto-detects CUDA/CPU

## 📝 Development

### Running in Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## 🚨 Troubleshooting

### Model Not Found
```
❌ Model not found!
```
**Solution**: Run the training notebook first to create `best_model.pth`

### Connection Error
```
❌ Cannot connect to API
```
**Solution**: Make sure the API server is running on port 8000

### Prediction Error
```
❌ Prediction failed
```
**Solution**: Ensure you're uploading a valid image file (JPG/PNG)

## 📈 Performance

- **Inference Time**: ~50ms per image (CPU)
- **Memory Usage**: ~500MB
- **Supported Formats**: JPG, PNG, JPEG
- **Max File Size**: 10MB

## 🤝 Contributing

1. Train the model using `03_Model_Training.ipynb`
2. Test the API with `test_api.py`
3. Improve the web interface in `app.py`
4. Add new food classes to the training data

## 📄 License

This project uses the Food-101 dataset for research and educational purposes.
