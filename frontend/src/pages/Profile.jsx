import React from 'react'
import '../styles/Profile.css'

function Profile() {
  const userStats = {
    totalScans: 24,
    favoriteFood: 'Pizza',
    streakDays: 7,
    avgConfidence: 94
  }

  return (
    <div className="profile-container">
      <div className="profile-header">
        <h1>👤 Your Profile</h1>
      </div>

      <div className="profile-content">
        <div className="profile-card">
          <div className="profile-avatar">
            <span>🧑</span>
          </div>
          <div className="profile-info">
            <h2>Food Enthusiast</h2>
            <p>Member since 2024</p>
          </div>
        </div>

        <div className="stats-grid">
          <div className="stat-item">
            <div className="stat-icon">📊</div>
            <div className="stat-details">
              <span className="stat-value">{userStats.totalScans}</span>
              <span className="stat-label">Total Scans</span>
            </div>
          </div>
          <div className="stat-item">
            <div className="stat-icon">⭐</div>
            <div className="stat-details">
              <span className="stat-value">{userStats.avgConfidence}%</span>
              <span className="stat-label">Avg Confidence</span>
            </div>
          </div>
          <div className="stat-item">
            <div className="stat-icon">🔥</div>
            <div className="stat-details">
              <span className="stat-value">{userStats.streakDays}</span>
              <span className="stat-label">Day Streak</span>
            </div>
          </div>
          <div className="stat-item">
            <div className="stat-icon">🍕</div>
            <div className="stat-details">
              <span className="stat-value">{userStats.favoriteFood}</span>
              <span className="stat-label">Favorite Food</span>
            </div>
          </div>
        </div>

        <div className="preferences-section">
          <h3>⚙️ Preferences</h3>
          <div className="preference-item">
            <label>
              <input type="checkbox" defaultChecked />
              <span>Enable Notifications</span>
            </label>
          </div>
          <div className="preference-item">
            <label>
              <input type="checkbox" defaultChecked />
              <span>Save Analysis History</span>
            </label>
          </div>
          <div className="preference-item">
            <label>
              <input type="checkbox" />
              <span>Dark Mode</span>
            </label>
          </div>
        </div>

        <div className="achievements-section">
          <h3>🏆 Achievements</h3>
          <div className="achievements-grid">
            <div className="achievement">
              <span className="achievement-icon">🥇</span>
              <span className="achievement-title">First Scan</span>
            </div>
            <div className="achievement">
              <span className="achievement-icon">📸</span>
              <span className="achievement-title">10 Scans</span>
            </div>
            <div className="achievement">
              <span className="achievement-icon">⭐</span>
              <span className="achievement-title">Accuracy Master</span>
            </div>
            <div className="achievement locked">
              <span className="achievement-icon">🌟</span>
              <span className="achievement-title">Coming Soon</span>
            </div>
          </div>
        </div>

        <button className="logout-btn">🚪 Logout</button>
      </div>
    </div>
  )
}

export default Profile
