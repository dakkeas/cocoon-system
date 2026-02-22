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
    active: false,
    last_scan_time: null,
    logs: [] // New Logs State
  })

  // Automatic IP Detection
  const PI_SERVER_URL = `http://${window.location.hostname}:5000`;

  // --- DATA FETCHING LOOP ---
  useEffect(() => {
    const interval = setInterval(() => {
      fetch('/api/data')
        .then(res => res.json())
        .then(jsonData => {
          if (jsonData) {
            setData(jsonData);
          }
        })
        .catch(err => {
           // console.error("Waiting for Backend...", err)
        })
    }, 500) 
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
    .then(resData => {
      console.log(`Action ${actionName} sent!`);
      // Optimistic Updates
      if (actionName === 'start') setData(prev => ({ ...prev, active: true }));
      if (actionName === 'stop') setData(prev => ({ ...prev, active: false }));
      if (actionName === 'reset') setData(prev => ({ 
        ...prev, active: false, g_count: 0, ng_count: 0, empty_count: 0, 
        grid: Array(144).fill(0), last_scan_time: null, logs: [] 
      }));
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
          <button className="btn btn-start" onClick={() => sendAction('start')}>START SCAN</button>
          <button className="btn btn-stop" onClick={() => sendAction('stop')}>STOP</button>
        </div>
        <button className="btn btn-reset" onClick={() => sendAction('reset')}>RESET BATCH</button>
      </div>

      {/* --- SYSTEM LOG (NEW) --- */}
      <div className="log-container">
        <h3>System Process Log</h3>
        <div className="log-box">
            {data.logs && data.logs.length > 0 ? (
                data.logs.map((log, index) => (
                    <div key={index} className="log-entry">{log}</div>
                ))
            ) : (
                <div className="log-placeholder">System Ready...</div>
            )}
        </div>
      </div>

      {/* --- LIVE CAMERA VIEW (NEW) --- */}
      <div className="image-panel">
        <div className="panel-header">
            <h3>Live Camera View</h3>
            <div className="live-indicator"> LIVE</div>
        </div>
        
        {/* We use a timestamp query param to prevent caching if the stream restarts */}
        <img 
            src={`${PI_SERVER_URL}/api/video_feed`} 
            className="captured-image" 
            alt="Live Camera Feed" 
            onError={(e) => {e.target.onerror = null; e.target.src="https://placehold.co/600x400?text=Camera+Offline"}}
        />
      </div>

      {/* --- INFERENCE VIEW --- */}
      <div className="image-panel">
        <div className="panel-header">
            <h3>AI Classification Result</h3>
        </div>
        {data.last_scan_time ? (
          <img 
            src={`${PI_SERVER_URL}/api/latest_image?t=${data.last_scan_time}`} 
            className="captured-image" 
            alt="AI Inference Result" 
          />
        ) : (
          <div className="placeholder-box">
            <span>Waiting for Scan...</span>
          </div>
        )}
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