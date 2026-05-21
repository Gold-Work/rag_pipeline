import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_USERNAME = import.meta.env.VITE_API_USERNAME || 'admin'
const API_PASSWORD = import.meta.env.VITE_API_PASSWORD || 'changeme123'

function fmt(n) {
  return typeof n === 'number' ? n.toFixed(3) + 's' : '—'
}

function LatencyBar({ label, value, max }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0
  return (
    <div className="lat-row">
      <span className="lat-label">{label}</span>
      <div className="lat-track">
        <div className="lat-fill" style={{ width: pct + '%' }} />
      </div>
      <span className="lat-value">{fmt(value)}</span>
    </div>
  )
}

function BotMessage({ result }) {
  const { answer, sources = [], latencies = {}, chunks_used = 0, cached = false } = result
  const noContext = answer.includes("n'est pas présente dans le contexte") || answer.includes("Je suis DocAssist")
  const r = latencies.retrieval || 0
  const rk = latencies.rerank || 0
  const llm = latencies.llm || 0
  const total = r + rk + llm
  const maxLat = Math.max(r, rk, llm)

  return (
    <div className="msg-bot">
      <div className="bot-avatar">D</div>
      <div className="bot-card">
        {cached && !noContext && <span className="cache-badge">CACHE HIT</span>}
        <p className="answer-text">{answer}</p>
        {!noContext && sources.length > 0 && (
          <div className="sources-row">
            {sources.map((s, i) => (
              <span key={i} className="source-pill">
                {s.split('/').pop().split('\\').pop()}
              </span>
            ))}
          </div>
        )}
        {!noContext && (
          <div className="latencies">
            <LatencyBar label="Retrieval" value={r} max={maxLat} />
            <LatencyBar label="Rerank" value={rk} max={maxLat} />
            <LatencyBar label="LLM" value={llm} max={maxLat} />
            <div className="lat-total">
              Total : {fmt(total)} · {chunks_used} chunk{chunks_used !== 1 ? 's' : ''}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function UserMessage({ text }) {
  return (
    <div className="msg-user">
      <div className="user-bubble">{text}</div>
    </div>
  )
}

export default function App() {
  const [token, setToken] = useState(null)
  const [authError, setAuthError] = useState(null)
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [docCount, setDocCount] = useState(null)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    login()
  }, [])

  async function login() {
    try {
      const form = new URLSearchParams()
      form.append('username', API_USERNAME)
      form.append('password', API_PASSWORD)
      const res = await fetch('/api/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: form.toString(),
      })
      if (!res.ok) {
        setAuthError('Authentification échouée — vérifiez API_USERNAME / API_PASSWORD')
        return null
      }
      const data = await res.json()
      setToken(data.access_token)
      fetchDocCount(data.access_token)
      return data.access_token
    } catch (e) {
      setAuthError(`Impossible de joindre l'API : ${e.message}`)
      return null
    }
  }

  function fetchDocCount(t) {
    fetch('/api/stats', { headers: { Authorization: `Bearer ${t}` } })
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d?.document_count != null) setDocCount(d.document_count) })
      .catch(() => {})
  }

  async function handleSubmit(e) {
    e?.preventDefault()
    const q = question.trim()
    if (!q || loading) return

    setQuestion('')
    setMessages(prev => [...prev, { type: 'user', text: q }])
    setLoading(true)

    try {
      let t = token
      if (!t) t = await login()
      if (!t) { setLoading(false); return }

      let res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${t}` },
        body: JSON.stringify({ question: q, top_k: 5 }),
      })

      if (res.status === 401) {
        t = await login()
        if (!t) { setLoading(false); return }
        res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${t}` },
          body: JSON.stringify({ question: q, top_k: 5 }),
        })
      }

      const data = await res.json()
      if (!res.ok) {
        setMessages(prev => [...prev, { type: 'error', text: data.detail || `Erreur ${res.status}` }])
      } else {
        setMessages(prev => [...prev, { type: 'bot', result: data }])
      }
    } catch (e) {
      setMessages(prev => [...prev, { type: 'error', text: `Erreur réseau : ${e.message}` }])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleSubmit()
  }

  const history = messages.filter(m => m.type === 'user')

  if (authError) {
    return (
      <div className="auth-error-screen">
        <div className="auth-error-box">
          <span className="auth-error-icon">⚠</span>
          <p>{authError}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">D</span>
          <span className="logo-text">DocAssist</span>
        </div>

        <div className="sidebar-section-label">Historique</div>
        <div className="history-list">
          {history.length === 0
            ? <p className="history-empty">Aucune question</p>
            : history.map((m, i) => (
                <button key={i} className="history-item" onClick={() => setQuestion(m.text)}>
                  {m.text.length > 46 ? m.text.slice(0, 46) + '…' : m.text}
                </button>
              ))
          }
        </div>

        <div className="sidebar-bottom">
          <button className="sidebar-btn">＋ Ajouter des documents</button>
          <button className="sidebar-btn">⚙ Paramètres</button>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="main">
        <header className="topbar">
          <span className="topbar-title">Assistant documentaire</span>
          <span className="topbar-docs">
            {docCount != null ? `${docCount} documents indexés` : 'Documents indexés : —'}
          </span>
        </header>

        <div className="chat-area">
          {messages.length === 0 && (
            <div className="chat-empty">
              <div className="chat-empty-icon">D</div>
              <p>Posez une question sur vos documents</p>
            </div>
          )}

          {messages.map((m, i) => {
            if (m.type === 'user') return <UserMessage key={i} text={m.text} />
            if (m.type === 'bot') return <BotMessage key={i} result={m.result} />
            if (m.type === 'error') return (
              <div key={i} className="msg-error">⚠ {m.text}</div>
            )
            return null
          })}

          {loading && (
            <div className="msg-bot">
              <div className="bot-avatar">D</div>
              <div className="bot-card typing">
                <span /><span /><span />
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <form className="input-area" onSubmit={handleSubmit}>
          <textarea
            className="input-box"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Posez une question… (Ctrl+Entrée pour envoyer)"
            rows={2}
            disabled={loading || !token}
          />
          <button
            type="submit"
            className="send-btn"
            disabled={loading || !token || !question.trim()}
          >
            {loading ? '…' : '↑'}
          </button>
        </form>
      </div>
    </div>
  )
}
