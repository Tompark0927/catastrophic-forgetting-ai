import { useEffect, useState, useRef } from 'react'
import type { FormEvent } from 'react'
import './index.css'

type Message = { id: string; role: 'user' | 'engine'; content: string }
type ActiveNode = { id: string; weight: number }

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMsg, setInputMsg] = useState('')
  const [streamingToken, setStreamingToken] = useState('')

  const [rootFacts, setRootFacts] = useState<Record<string, string>>({})
  const [leafMemories, setLeafMemories] = useState<string[]>([])
  const [adaptersCount, setAdaptersCount] = useState(0)

  const [reflexActive, setReflexActive] = useState<'jellyfish' | 'micro' | null>(null)
  const [pageRankNodes, setPageRankNodes] = useState<ActiveNode[]>([])
  const [sleepSpindles, setSleepSpindles] = useState<number | null>(null)
  const [connected, setConnected] = useState(false)

  const ws = useRef<WebSocket | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingToken])

  useEffect(() => {
    const connect = () => {
      const socket = new WebSocket('ws://localhost:8000/ws')
      ws.current = socket

      socket.onopen = () => setConnected(true)
      socket.onclose = () => {
        setConnected(false)
        // Reconnect after 3 seconds
        setTimeout(connect, 3000)
      }

      socket.onmessage = (event) => {
        const payload = JSON.parse(event.data)

        switch (payload.type) {
          case 'engine_state':
            setRootFacts(payload.data.root_facts || {})
            setLeafMemories(payload.data.leaf_memories || [])
            setAdaptersCount(payload.data.adapters || 0)
            break
          case 'user_message':
            setMessages(prev => [...prev, { id: `${Date.now()}-${Math.random()}`, role: 'user', content: payload.data }])
            setStreamingToken('')
            break
          case 'token':
            setStreamingToken(prev => prev + payload.data.text)
            break
          case 'engine_reply':
            setMessages(prev => [...prev, { id: `${Date.now()}-${Math.random()}`, role: 'engine', content: payload.data }])
            setStreamingToken('')
            break
          case 'reflex_intercept':
            setReflexActive('jellyfish')
            setTimeout(() => setReflexActive(null), 3000)
            break
          case 'micro_reflex':
            setReflexActive('micro')
            setTimeout(() => setReflexActive(null), 3000)
            break
          case 'pagerank_morph': {
            const nodes: ActiveNode[] = (payload.data.nodes as string[]).map((n, i) => ({
              id: n,
              weight: payload.data.weights[i] as number,
            }))
            setPageRankNodes(nodes)
            setTimeout(() => setPageRankNodes([]), 5000)
            break
          }
          case 'sleep_spindles':
            setSleepSpindles(payload.data.expanded_count)
            setTimeout(() => setSleepSpindles(null), 6000)
            break
        }
      }
    }

    connect()
    return () => ws.current?.close()
  }, [])

  const handleSend = (e: FormEvent) => {
    e.preventDefault()
    if (!inputMsg.trim() || !ws.current || ws.current.readyState !== WebSocket.OPEN) return
    ws.current.send(JSON.stringify({ action: 'chat', message: inputMsg }))
    setInputMsg('')
  }

  const rootEntries = Object.entries(rootFacts)

  return (
    <div className="app-container">

      {/* ─── LEFT: The Vault ─── */}
      <div className="panel vault-panel">
        <div className="panel-header">
          <span className="header-icon">❖</span>
          The Vault
        </div>
        <div className="panel-content">
          <div className="section-title">Root Profile ({rootEntries.length})</div>
          {rootEntries.length === 0 && (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              No permanent facts yet. Try "평생 기억해" after sharing something.
            </div>
          )}
          {rootEntries.map(([k, v]) => (
            <div key={k} className="vault-item">
              <span className="key">{k}</span>{String(v)}
            </div>
          ))}

          <div className="section-title" style={{ marginTop: '2rem' }}>
            Leaf Memories ({leafMemories.length})
          </div>
          {leafMemories.length === 0 && (
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
              No short-term memories yet.
            </div>
          )}
          {leafMemories.map((m, i) => (
            <div key={i} className="vault-item">{String(m)}</div>
          ))}
        </div>
      </div>

      {/* ─── CENTER: Chat Arena ─── */}
      <div className="panel chat-panel">
        <div className="panel-header">
          <span className="header-icon">✦</span>
          Liquid Sensorium Console
          <span style={{
            marginLeft: 'auto',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: connected ? 'var(--accent-emerald)' : 'var(--accent-rose)',
            boxShadow: connected ? '0 0 8px var(--accent-emerald)' : '0 0 8px var(--accent-rose)',
            display: 'inline-block',
          }} />
        </div>

        <div className="panel-content">
          {messages.length === 0 && !streamingToken && (
            <div style={{
              margin: 'auto',
              textAlign: 'center',
              color: 'var(--text-muted)',
              padding: '3rem 2rem',
            }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '1rem', filter: 'drop-shadow(0 0 16px var(--accent-purple))' }}>༄</div>
              <div style={{ fontSize: '1rem', fontWeight: 500 }}>The Sensorium is listening.</div>
              <div style={{ fontSize: '0.85rem', marginTop: '0.5rem', opacity: 0.5 }}>
                Send a message to begin. Speak to remember.
              </div>
            </div>
          )}

          {messages.map((m) => (
            <div key={m.id} className={`message ${m.role}`}>
              <div className="message-bubble">{m.content}</div>
            </div>
          ))}

          {streamingToken && (
            <div className="message engine">
              <div className="message-bubble">
                {streamingToken}
                <span style={{
                  display: 'inline-block',
                  width: '2px',
                  height: '1em',
                  background: 'var(--accent-purple)',
                  marginLeft: '2px',
                  animation: 'blink 0.8s step-end infinite',
                  verticalAlign: 'text-bottom',
                }} />
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        <form onSubmit={handleSend} className="chat-input-area">
          <div className="chat-input-form">
            <input
              type="text"
              className="chat-input"
              placeholder={connected ? 'Ask the engine or whisper a memory...' : 'Connecting to engine...'}
              value={inputMsg}
              disabled={!connected}
              onChange={e => setInputMsg(e.target.value)}
            />
            <button type="submit" className="send-btn" disabled={!connected}>➔</button>
          </div>
        </form>
      </div>

      {/* ─── RIGHT: The Liquid Sensorium ─── */}
      <div className="panel sensorium-panel">
        <div className="panel-header">
          <span className="header-icon">༄</span>
          System Analytics
        </div>
        <div className="panel-content">

          <div className="section-title">Neural Activity</div>

          <div className={`indicator-pill ${reflexActive === 'micro' ? 'active cyan' : ''}`}>
            <span>⚡ Micro Reflex (0ms)</span>
            <div className={`status-dot cyan`} />
          </div>

          <div className={`indicator-pill ${reflexActive === 'jellyfish' ? 'active emerald' : ''}`}>
            <span>🪼 Jellyfish Node (v6.0)</span>
            <div className={`status-dot emerald`} />
          </div>

          <div className={`indicator-pill ${pageRankNodes.length > 0 ? 'active purple' : ''}`}>
            <span>🧬 PageRank Matrix</span>
            <div className={`status-dot purple`} />
          </div>

          <div className={`indicator-pill ${sleepSpindles ? 'active emerald' : ''}`}>
            <span>🌙 Sleep Spindles</span>
            <div className={`status-dot emerald`} />
          </div>

          {/* PageRank Node Weights */}
          {pageRankNodes.length > 0 && (
            <div style={{
              marginTop: '1.5rem',
              background: 'rgba(0,0,0,0.3)',
              padding: '1rem',
              borderRadius: '12px',
              animation: 'fade-in-up 0.5s ease',
            }}>
              <div className="section-title" style={{ marginTop: 0 }}>Active LoRA Graph</div>
              {pageRankNodes.map((n, i) => (
                <div key={i} style={{ marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#a1a1aa', marginBottom: '4px' }}>
                    <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>{n.id.substring(0, 10)}…</span>
                    <span style={{ color: 'var(--accent-purple)' }}>{(n.weight * 100).toFixed(1)}%</span>
                  </div>
                  <div className="node-weight-bar">
                    <div className="node-fill" style={{ width: `${n.weight * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Sleep Spindles Card */}
          {sleepSpindles && (
            <div style={{
              marginTop: '1.5rem',
              background: 'rgba(52, 211, 153, 0.05)',
              border: '1px solid rgba(52, 211, 153, 0.2)',
              padding: '1rem',
              borderRadius: '12px',
              animation: 'fade-in-up 0.5s ease',
            }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--accent-emerald)', marginBottom: '0.5rem' }}>
                🌙 Sleep Cycle Active
              </div>
              <div style={{ fontSize: '0.8rem', color: '#a1a1aa', lineHeight: 1.6 }}>
                Hyper-distillation generated <strong style={{ color: 'var(--text-main)' }}>{sleepSpindles}</strong> semantic
                variations to prevent catastrophic forgetting.
              </div>
            </div>
          )}

          <div style={{ flex: 1 }} />
          <div style={{ textAlign: 'center', opacity: 0.4, fontSize: '0.75rem', marginTop: '2rem' }}>
            {adaptersCount} Graph Adapters Active
          </div>
        </div>
      </div>

    </div>
  )
}

export default App
