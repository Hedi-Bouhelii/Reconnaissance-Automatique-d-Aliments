import React from 'react'
import '../styles/AnalysisResults.css'

function AnalysisResults({ data, onNewScan }) {
  const topPred = data.topPrediction

  return (
    <div className="analysis-container">
      <button className="back-btn" onClick={onNewScan}>
        ← New Scan
      </button>

      <div className="analysis-content">
        <div className="analysis-image-section">
          <img src={data.image} alt="Analyzed food" className="analysis-image" />
          <div className="timestamp">{data.timestamp}</div>
        </div>

        <div className="analysis-results-section">
          <div className="analysis-header">
            <h2>✅ Analysis Complete</h2>
            <p>Looks like</p>
            <h1 className="food-name">
              {topPred.class.replace('_', ' ').toUpperCase()}
            </h1>
            <div className="confidence-badge">
              {Math.round(topPred.confidence * 100)}% Confidence
            </div>
          </div>

          <div className="predictions-list">
            <h3>Top Predictions</h3>
            <div className="predictions">
              {data.predictions.map((pred, index) => {
                const confidence = Math.round(pred.confidence * 100)
                return (
                  <div key={index} className="prediction-item">
                    <div className="prediction-rank">{pred.rank}</div>
                    <div className="prediction-info">
                      <span className="prediction-name">
                        {pred.class.replace('_', ' ')}
                      </span>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{ width: `${confidence}%` }}
                        ></div>
                      </div>
                    </div>
                    <span className="prediction-percentage">{confidence}%</span>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="nutrition-info">
            <h3>📊 Typical Nutrition (Per 100g)</h3>
            <div className="nutrition-grid">
              <div className="nutrition-item">
                <span className="nutrition-label">Calories</span>
                <span className="nutrition-value">~150</span>
              </div>
              <div className="nutrition-item">
                <span className="nutrition-label">Protein</span>
                <span className="nutrition-value">~12g</span>
              </div>
              <div className="nutrition-item">
                <span className="nutrition-label">Carbs</span>
                <span className="nutrition-value">~20g</span>
              </div>
              <div className="nutrition-item">
                <span className="nutrition-label">Fats</span>
                <span className="nutrition-value">~5g</span>
              </div>
            </div>
          </div>

          <div className="action-buttons">
            <button className="btn btn-save">❤️ Save to Favorites</button>
            <button className="btn btn-share">📤 Share</button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AnalysisResults
