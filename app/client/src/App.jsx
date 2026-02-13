import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [data, setData] = useState({
    grid: Array(144).fill(0),
    g_count: 0,
    ng_count: 0,
    empty_count: 0,
    total: 0,
    defect_rate: 0,
    active: false
  })

  // Automatic IP Detection: Connects to the Flask backend
  const PI_SERVER_URL = `http://${window.location.hostname}:5000`;

  // --- DATA FETCHING LOOP ---
  // This asks the Python Backend "What do you see?" every 200ms
  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${PI_SERVER_URL}/api/data`)
        .then(res => res.json())
        .then(jsonData => {
          // Only update if the data is valid
          if (jsonData) {
            setData(jsonData);
          }
        })
        .catch(err => {
          // Optional: Log errors if backend is offline
          // console.error("Waiting for Backend...", err)
        })
    }, 200) // Fast refresh rate for smooth updates
    return () => clearInterval(interval)
  }, [PI_SERVER_URL])

  // --- BUTTON CONTROL ---
  const sendAction = (actionName) => {
    fetch(`${PI_SERVER_URL}/api/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: actionName })
    })
    .then(res => res.json())
    .then(data => {
      console.log(`Action ${actionName} sent!`, data);
      // Immediate UI update for "Processing" badge
      if (actionName === 'start') setData(prev => ({ ...prev, active: true }));
      if (actionName === 'stop') setData(prev => ({ ...prev, active: false }));
      if (actionName === 'reset') setData(prev => ({ ...prev, active: false, g_count: 0, ng_count: 0, empty_count: 0, grid: Array(144).fill(0) }));
    })
    .catch(err => console.error("Failed to send action:", err));
  }

  return (
    <div className="container">
      
      {/* HEADER */}
      <div className="header">
        <h1>SERA</h1>
        <div className="header-status">
          <p className="sub-text">SYSTEM STATUS</p>
          <span className={`badge ${data.active ? 'badge-active' : 'badge-idle'}`}>
            {data.active ? '● ONLINE' : '● IDLE'}
          </span>
        </div>
      </div>
      
      <hr />

      {/* METRICS */}
      <div className="metric-card card-green">
        <div className="metric-label">PASSED (G)</div>
        <div className="metric-value text-green">{data.g_count}</div>
      </div>

      <div className="metric-card card-red">
        <div className="metric-label">REJECTED (NG)</div>
        <div className="metric-value text-red">{data.ng_count}</div>
      </div>

      {/* STATS */}
      <div className="stats-row">
        <div className="stat-box">
          <span className="label">Empty</span>
          <span className="value">{data.empty_count}</span>
        </div>
        <div className="stat-box">
          <span className="label">Total</span>
          <span className="value">{data.total}/144</span>
        </div>
        <div className="stat-box">
          <span className="label">Defect %</span>
          <span className="value">{data.defect_rate}%</span>
        </div>
      </div>

      {/* CONTROLS */}
      <div className="controls">
        <div className="btn-group">
          <button className="btn btn-start" onClick={() => sendAction('start')}>START</button>
          <button className="btn btn-stop" onClick={() => sendAction('stop')}>STOP</button>
        </div>
        <button className="btn btn-reset" onClick={() => sendAction('reset')}>RESET BATCH</button>
      </div>

      {/* GRID */}
      <div className="mountage-panel">
        <h3>Mountage Grid</h3>
        <div className="heatmap-container">
          {data.grid.map((status, index) => (
            <div 
              key={index} 
              className={`cocoon-cell cell-${status}`}
              title={`Slot ${index + 1}`}
            />
          ))}
        </div>
        <div className="legend">
          <div><span className="dot cell-0"></span>Pending</div>
          <div><span className="dot cell-1"></span>Pass</div>
          <div><span className="dot cell-2"></span>Reject</div>
          <div><span className="dot cell-3"></span>Empty</div>
        </div>
      </div>

    </div>
  )
}

export default App