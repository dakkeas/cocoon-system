import { useState, useEffect, useRef } from 'react'
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
    last_scan_id: null
  })

  const [logs, setLogs] = useState([])
  const lastLogRef = useRef("")

  const PI_SERVER_URL = `http://${window.location.hostname}:5000`
  const [latestImageSrc, setLatestImageSrc] = useState(null)

  // 1. Update Image Source when a new Scan ID is received
  useEffect(() => {
    if (data.last_scan_id) {
      console.log('Updating image source with scan ID:', data.last_scan_id)
      // Force browser to fetch fresh image using cache-busting timestamp
      setLatestImageSrc(
        `${PI_SERVER_URL}/api/latest_image?t=${encodeURIComponent(data.last_scan_id)}&cache=${Date.now()}`
      )
    }
  }, [data.last_scan_id, PI_SERVER_URL])

  // 2. Main Polling Interval (Every 500ms)
  useEffect(() => {
    const interval = setInterval(() => {

      // --- FETCH LATEST JSON (Grid + Stats) ---
      fetch(`${PI_SERVER_URL}/api/latest_json`)
        .then(res => res.json())
        .then(jsonData => {
          if (jsonData) {
            const g = jsonData.stats?.g_count || 0
            const ng = jsonData.stats?.ng_count || 0
            const empty = jsonData.stats?.empty_count || 0
            const total_processed = g + ng + empty
            const rate = (g + ng) > 0 ? ((ng / (g + ng)) * 100).toFixed(1) : 0

            setData(prev => ({
              ...prev,
              grid: jsonData.cocoon_grid || Array(144).fill(0),
              g_count: g,
              ng_count: ng,
              empty_count: empty,
              total: total_processed,
              defect_rate: rate,
              last_scan_id: jsonData.latest_image_path || prev.last_scan_id,
              active: jsonData.sorting_active
            }))
          }
        })
        .catch(err => console.warn("Failed to fetch latest JSON:", err))

      // --- FETCH LATEST LOG ---
      fetch(`${PI_SERVER_URL}/api/latest_log`)
        .then(res => res.json())
        .then(logData => {
          const incomingLog = logData.log
          // Only update if the log is new and not empty
          if (incomingLog && incomingLog !== lastLogRef.current) {
            const time = new Date().toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit'
            });
            const formattedLog = `[${time}] ${incomingLog}`;

            setLogs(prev => [formattedLog, ...prev].slice(0, 50))
            lastLogRef.current = incomingLog
          }
        })
        .catch(err => console.warn("Failed to fetch logs:", err))

    }, 500)

    return () => clearInterval(interval)
  }, [PI_SERVER_URL])

  // 3. Send Actions to Raspberry Pi
  const sendAction = (actionName) => {
    fetch(`${PI_SERVER_URL}/api/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: actionName })
    })
      .then(res => res.json())
      .then(() => {
        console.log(`Action "${actionName}" sent successfully.`)
        if (actionName === 'start') setData(prev => ({ ...prev, active: true }))
        if (actionName === 'stop') setData(prev => ({ ...prev, active: false }))
        if (actionName === 'reset') {
          setData({
            grid: Array(144).fill(0),
            g_count: 0,
            ng_count: 0,
            empty_count: 0,
            total: 0,
            defect_rate: 0,
            active: false,
            last_scan_id: null
          })
          setLogs([])
          lastLogRef.current = ""
        }
      })
      .catch(err => console.error("Failed to send action:", err))
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

      {/* SYSTEM LOG */}
      <div className="log-container">
        <h3>System Process Log</h3>
        <div className="log-box">
          {logs.length > 0 ? (
            logs.map((log, index) => (
              <div key={index} className="log-entry">{log}</div>
            ))
          ) : (
            <div className="log-placeholder">System Ready...</div>
          )}
        </div>
      </div>

      {/* AI INFERENCE IMAGE */}
      <div className="image-panel">
        <div className="panel-header">
          <h3>AI Classification Result</h3>
        </div>
        {latestImageSrc ? (
          <img
            src={latestImageSrc}
            className="captured-image"
            alt="AI Inference Result"
          />
        ) : (
          <div className="placeholder-box">
            <span>Waiting for Scan...</span>
          </div>
        )}
      </div>

      {/* MOUNTAGE GRID */}
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