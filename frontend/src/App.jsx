import React, { useState } from 'react'
import Sidebar from './components/Sidebar'
import Home from './pages/Home'
import History from './pages/History'
import Profile from './pages/Profile'
import './styles/App.css'

function App() {
  const [currentPage, setCurrentPage] = useState('home')
  const [scanHistory, setScanHistory] = useState([])

  const handleNewScan = (scanData) => {
    setScanHistory([scanData, ...scanHistory])
  }

  return (
    <div className="app-container">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <main className="main-content">
        {currentPage === 'home' && <Home onScan={handleNewScan} />}
        {currentPage === 'history' && <History scans={scanHistory} />}
        {currentPage === 'profile' && <Profile />}
      </main>
    </div>
  )
}

export default App
