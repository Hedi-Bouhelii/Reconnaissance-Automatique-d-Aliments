#!/usr/bin/env python3
"""
Test script for Food Classification API
"""

import requests
import json
from pathlib import Path
import time

def test_api():
    """Test all API endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Food Classification API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return False
    
    # Test 2: Get classes
    print("\n2. Testing classes endpoint...")
    try:
        response = requests.get(f"{base_url}/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classes endpoint working")
            print(f"   Total classes: {data['total']}")
            print(f"   Classes: {', '.join(data['classes'])}")
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Classes endpoint error: {e}")
    
    # Test 3: Root endpoint
    print("\n3. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working")
            print(f"   API: {data['message']}")
            print(f"   Version: {data['version']}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 4: Prediction with sample image (if available)
    print("\n4. Testing prediction endpoint...")
    
    # Look for a sample image in the dataset
    sample_image = None
    food10_path = Path('./data/food10')
    
    if food10_path.exists():
        # Find any image in the test set
        for split in ['test', 'train']:
            split_path = food10_path / split
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        images = list(class_dir.glob('*.jpg'))
                        if images:
                            sample_image = images[0]
                            break
                if sample_image:
                    break
    
    if sample_image:
        print(f"   Using sample image: {sample_image.name}")
        try:
            with open(sample_image, 'rb') as f:
                files = {'file': (sample_image.name, f, 'image/jpeg')}
                response = requests.post(f"{base_url}/predict", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Prediction successful")
                print(f"   Filename: {data['filename']}")
                print(f"   Top prediction: {data['top_prediction']['class']} ({data['top_prediction']['confidence']:.2%})")
                print(f"   All predictions:")
                for pred in data['predictions']:
                    print(f"     {pred['rank']}. {pred['class']}: {pred['confidence']:.2%}")
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text}")
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    else:
        print("⚠️ No sample image found for testing prediction")
        print("   You can test manually by uploading an image to http://localhost:8000/web")
    
    print("\n" + "=" * 50)
    print("🎉 API testing complete!")
    print("\n📚 Additional testing options:")
    print("1. Open http://localhost:8000/web for interactive testing")
    print("2. Open http://localhost:8000/docs for API documentation")
    print("3. Use curl or Postman for API testing")
    
    return True

def test_with_curl():
    """Show curl commands for manual testing"""
    print("\n📋 Manual testing with curl:")
    print("=" * 30)
    
    commands = [
        "# Health check",
        "curl -X GET http://localhost:8000/health",
        "",
        "# Get available classes", 
        "curl -X GET http://localhost:8000/classes",
        "",
        "# Predict with image (replace path to your image)",
        "curl -X POST -F \"file=@/path/to/your/image.jpg\" http://localhost:8000/predict",
        "",
        "# Get API info",
        "curl -X GET http://localhost:8000/"
    ]
    
    for cmd in commands:
        print(cmd)

if __name__ == "__main__":
    # Wait a moment for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(2)
    
    # Run tests
    success = test_api()
    
    # Show manual testing commands
    test_with_curl()
    
    if success:
        print(f"\n🌐 Open your browser and go to: http://localhost:8000/web")
        print("📸 Upload a food image to test the classification!")
