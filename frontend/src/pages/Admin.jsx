import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

function StatCard({ label, value, unit = '' }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value ?? '—'}{unit && value != null ? unit : ''}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

export default function Admin() {
  const { token, username, logout } = useAuth()
  const navigate = useNavigate()

  const [stats, setStats] = useState(null)
  const [statsError, setStatsError] = useState(null)

  const [uploads, setUploads] = useState([])
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef(null)

  useEffect(() => {
    fetchStats()
  }, [token])

  async function fetchStats() {
    try {
      const res = await fetch('/api/stats', {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) { setStatsError(`Erreur ${res.status}`); return }
      setStats(await res.json())
    } catch (e) {
      setStatsError(e.message)
    }
  }

  async function uploadFile(file) {
    const entry = { name: file.name, status: 'uploading', chunks: null, error: null, id: Date.now() + Math.random() }
    setUploads(prev => [entry, ...prev])

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      })

      if (res.status === 401) { logout(); navigate('/login', { replace: true }); return }

      const data = await res.json()
      if (!res.ok) {
        setUploads(prev => prev.map(u => u.id === entry.id ? { ...u, status: 'error', error: data.detail } : u))
      } else {
        setUploads(prev => prev.map(u => u.id === entry.id ? { ...u, status: 'done', chunks: data.chunks_indexed } : u))
        fetchStats()
      }
    } catch (e) {
      setUploads(prev => prev.map(u => u.id === entry.id ? { ...u, status: 'error', error: e.message } : u))
    }
  }

  function handleFiles(files) {
    Array.from(files).forEach(uploadFile)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  function formatDate(iso) {
    if (!iso) return null
    return new Date(iso).toLocaleString('fr-FR', { dateStyle: 'short', timeStyle: 'short' })
  }

  return (
    <div className="admin-shell">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">D</span>
          <span className="logo-text">DocAssist</span>
        </div>

        <div className="sidebar-section-label">Administration</div>
        <div className="history-list">
          <button className="history-item history-item-active">⬆ Ajouter des documents</button>
          <button className="history-item" onClick={() => navigate('/chat')}>💬 Aller au chat</button>
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-user">{username} <span className="role-badge">admin</span></div>
          <button className="sidebar-btn sidebar-btn-logout" onClick={logout}>
            ↩ Déconnexion
          </button>
        </div>
      </aside>

      <div className="main">
        <header className="topbar">
          <span className="topbar-title">Administration — Index documentaire</span>
        </header>

        <div className="admin-area">
          {/* Stats */}
          <section className="admin-section">
            <h2 className="admin-section-title">Statistiques</h2>
            {statsError
              ? <p className="admin-error">⚠ {statsError}</p>
              : (
                <div className="stats-row">
                  <StatCard label="Documents" value={stats?.document_count} />
                  <StatCard label="Chunks indexés" value={stats?.chunk_count} />
                  <StatCard
                    label="Dernière ingestion"
                    value={stats ? formatDate(stats.last_ingestion) ?? 'Jamais' : null}
                  />
                </div>
              )
            }
          </section>

          {/* Upload */}
          <section className="admin-section">
            <h2 className="admin-section-title">Ajouter des documents</h2>
            <p className="admin-hint">Formats acceptés : PDF, HTML, TXT</p>

            <div
              className={`drop-zone ${dragging ? 'drop-zone-active' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <span className="drop-icon">⬆</span>
              <p>Glissez vos fichiers ici ou <span className="drop-link">parcourir</span></p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.html,.txt"
                style={{ display: 'none' }}
                onChange={e => handleFiles(e.target.files)}
              />
            </div>

            {uploads.length > 0 && (
              <div className="upload-list">
                {uploads.map(u => (
                  <div key={u.id} className={`upload-item upload-item-${u.status}`}>
                    <span className="upload-name">{u.name}</span>
                    {u.status === 'uploading' && <span className="upload-status">En cours…</span>}
                    {u.status === 'done' && <span className="upload-status">{u.chunks} chunks indexés ✓</span>}
                    {u.status === 'error' && <span className="upload-status">⚠ {u.error}</span>}
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}
