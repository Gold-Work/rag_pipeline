import { useState, useEffect } from 'react'

const API_USERNAME = import.meta.env.VITE_API_USERNAME || 'admin'
const API_PASSWORD = import.meta.env.VITE_API_PASSWORD || 'changeme123'

function LatencyBar({ label, value, total }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0
  return (
    <div className="latency-row">
      <span className="latency-label">{label}</span>
      <div className="latency-track">
        <div className="latency-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="latency-value">{value.toFixed(3)}s</span>
    </div>
  )
}

function ResultPanel({ result }) {
  const { answer, sources, latencies, chunks_used, cached } = result
  const stages = ['retrieval', 'rerank', 'llm']

  return (
    <div className="result-panel">
      <div className="result-header">
        <span className="chunks-info">{chunks_used} chunk{chunks_used !== 1 ? 's' : ''} utilisé{chunks_used !== 1 ? 's' : ''}</span>
        <span className={`cache-badge ${cached ? 'hit' : 'miss'}`}>
          {cached ? 'CACHE HIT' : 'CACHE MISS'}
        </span>
      </div>

      <div className="answer">{answer}</div>

      {sources.length > 0 && (
        <div className="sources">
          <h3>Sources</h3>
          <ul>
            {sources.map((src, i) => (
              <li key={i}>{src}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="latencies">
        <h3>Latences</h3>
        {stages.map((stage) =>
          latencies[stage] != null ? (
            <LatencyBar
              key={stage}
              label={stage}
              value={latencies[stage]}
              total={latencies.total}
            />
          ) : null
        )}
        <div className="latency-total">Total : {latencies.total?.toFixed(3)}s</div>
      </div>
    </div>
  )
}

export default function App() {
  const [token, setToken] = useState(null)
  const [authError, setAuthError] = useState(null)
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

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
        return
      }

      const data = await res.json()
      setToken(data.access_token)
    } catch (e) {
      setAuthError(`Impossible de joindre l'API : ${e.message}`)
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!question.trim() || !token) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ question: question.trim(), top_k: 5 }),
      })

      if (res.status === 401) {
        setToken(null)
        setError('Session expirée — reconnexion en cours...')
        await login()
        return
      }

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        setError(data.detail || `Erreur ${res.status}`)
        return
      }

      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(`Erreur réseau : ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>RAG Pipeline</h1>

      {authError && <div className="error auth-error">{authError}</div>}

      {!authError && !token && (
        <div className="connecting">Connexion à l'API...</div>
      )}

      <form onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleSubmit(e)
          }}
          placeholder="Votre question... (Ctrl+Entrée pour envoyer)"
          rows={3}
          disabled={loading}
        />
        <button type="submit" disabled={loading || !token || !question.trim()}>
          {loading ? 'Recherche en cours...' : 'Rechercher'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}
      {result && <ResultPanel result={result} />}
    </div>
  )
}
