import React from 'react'
import '../styles/History.css'

function History({ scans }) {
  return (
    <div className="history-container">
      <div className="history-header">
        <h1>📜 Scan History</h1>
        <p>Your recent food analysis scans</p>
      </div>

      {scans.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">📭</div>
          <h2>No scans yet</h2>
          <p>Start analyzing food to build your history</p>
        </div>
      ) : (
        <div className="scans-grid">
          {scans.map(scan => (
            <div key={scan.id} className="scan-card">
              <div className="scan-image">
                <img src={scan.image} alt="Scanned food" />
              </div>
              <div className="scan-info">
                <h3 className="scan-title">
                  {scan.topPrediction.class.replace('_', ' ')}
                </h3>
                <p className="scan-confidence">
                  {Math.round(scan.topPrediction.confidence * 100)}% confidence
                </p>
                <p className="scan-date">{scan.timestamp}</p>
                <div className="scan-details">
                  {scan.predictions.slice(0, 3).map((pred, idx) => (
                    <span key={idx} className="detail-tag">
                      {pred.class.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default History
