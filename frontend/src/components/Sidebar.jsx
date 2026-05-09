import React from 'react'
import '../styles/Sidebar.css'

function Sidebar({ currentPage, setCurrentPage }) {
  const menuItems = [
    { id: 'home', icon: '🏠', label: 'Home' },
    { id: 'history', icon: '📜', label: 'History' },
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="logo">
          <span className="logo-icon">🥗</span>
          <div className="logo-text">
            <h1>FoodieLens</h1>
            <p>Food Intelligence</p>
          </div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {menuItems.map(item => (
          <button
            key={item.id}
            className={`nav-item ${currentPage === item.id ? 'active' : ''}`}
            onClick={() => setCurrentPage(item.id)}
          >
            <span className="nav-icon">{item.icon}</span>
            <span className="nav-label">{item.label}</span>
          </button>
        ))}
      </nav>

      
    </aside>
  )
}

export default Sidebar
