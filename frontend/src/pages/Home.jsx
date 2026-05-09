import React, { useState, useRef } from 'react'
import AnalysisResults from '../components/AnalysisResults'
import '../styles/Home.css'

function Home({ onScan }) {
  const [selectedImage, setSelectedImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const [showCamera, setShowCamera] = useState(false)

  const handleImageSelect = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      processImage(file)
    }
  }

  const processImage = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file')
      return
    }

    setSelectedImage(file)
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target.result)
    }
    reader.readAsDataURL(file)
    setError(null)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.currentTarget.classList.add('dragover')
  }

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('dragover')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.currentTarget.classList.remove('dragover')
    const file = e.dataTransfer.files?.[0]
    if (file) {
      processImage(file)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to analyze image')
      }

      const data = await response.json()
      
      const scanData = {
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        image: preview,
        predictions: data.predictions,
        topPrediction: data.top_prediction
      }

      setResults(scanData)
      onScan(scanData)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setSelectedImage(null)
    setPreview(null)
    setResults(null)
    setError(null)
    setShowCamera(false)
  }

  if (results) {
    return <AnalysisResults data={results} onNewScan={handleClear} />
  }

  return (
    <div className="home-container">
      <div className="home-header">
        <h1>See Food, Understand It</h1>
        <p className="tagline">Fuel Your Day</p>
        <p className="description">
          Snap or upload a photo of your meal, and let FoodieLens identify it in seconds.
        </p>
      </div>

      <div className="home-content">
        {!preview ? (
          <div
            className="upload-area"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-content">
              <div className="upload-icon">📸</div>
              <h2>Drop your image here or click to browse</h2>
              <p>Supports JPG, PNG, JPEG (Max 10MB)</p>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={{ display: 'none' }}
            />
          </div>
        ) : (
          <div className="preview-section">
            <div className="preview-container">
              <img src={preview} alt="Preview" className="preview-image" />
              <div className="preview-overlay">
                {loading && (
                  <div className="loader">
                    <div className="spinner"></div>
                    <p>🤖 AI is analyzing your food...</p>
                  </div>
                )}
              </div>
            </div>

            {error && (
              <div className="error-message">
                <span>⚠️ {error}</span>
              </div>
            )}

            <div className="action-buttons">
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? 'Analyzing...' : '🔍 Analyze'}
              </button>
              <button className="btn btn-secondary" onClick={handleClear}>
                ✕ Clear
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="info-box">
        <p>🤖 This AI can identify 10 food types: Pizza, Sushi, Hamburger, Hot Dog, French Fries, Ice Cream, Omelette, Pancakes, Ramen, Steak</p>
      </div>
    </div>
  )
}

export default Home
