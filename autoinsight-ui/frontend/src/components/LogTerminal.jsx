import { useEffect, useRef } from 'react'
import { Terminal } from 'lucide-react'

const STAGE_COLOR = {
  '[START]':  '#00e5ff',
  '[STAGE]':  '#ffb300',
  '[DATA]':   '#d500f9',
  '[DONE]':   '#00e676',
  '[FAILED]': '#ff1744',
  '[OK]':     '#00e676',
  '[WARN]':   '#ffb300',
}

function colorLine(line) {
  for (const [key, color] of Object.entries(STAGE_COLOR)) {
    if (line.includes(key)) return color
  }
  if (line.includes('[ERROR]') || line.includes('ERROR')) return '#ff1744'
  if (line.includes('[WARNING]') || line.includes('WARNING')) return '#ffb300'
  if (line.includes('[INFO]')) return '#9ca3af'
  return '#6b7280'
}

export default function LogTerminal({ logs, status }) {
  const bottomRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  const statusColor = {
    running: '#ffb300',
    done:    '#00e676',
    failed:  '#ff1744',
    pending: '#6b7280',
  }[status] || '#6b7280'

  return (
    <div style={styles.wrap}>
      <div style={styles.bar}>
        <div style={styles.barLeft}>
          <Terminal size={13} color="var(--cyan)" />
          <span style={styles.barLabel}>EXECUTION LOG</span>
        </div>
        <div style={styles.statusDot}>
          <span style={{ ...styles.dot, background: statusColor }} />
          <span style={{ color: statusColor, fontSize: 11, fontWeight: 700 }}>
            {status.toUpperCase()}
          </span>
        </div>
      </div>

      <div style={styles.terminal}>
        {logs.length === 0 ? (
          <span style={styles.empty}>Awaiting execution...</span>
        ) : (
          logs.map((line, i) => (
            <div key={i} style={{ ...styles.line, color: colorLine(line) }}>
              <span style={styles.lineNum}>{String(i + 1).padStart(4, ' ')}</span>
              {line}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

const styles = {
  wrap: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  bar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '10px 16px',
    background: 'var(--surface2)',
    borderBottom: '1px solid var(--border)',
  },
  barLeft: { display: 'flex', alignItems: 'center', gap: 8 },
  barLabel: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: 'var(--cyan)',
    letterSpacing: '0.12em',
    fontWeight: 700,
  },
  statusDot: { display: 'flex', alignItems: 'center', gap: 6 },
  dot: { width: 7, height: 7, borderRadius: '50%' },
  terminal: {
    flex: 1,
    overflowY: 'auto',
    padding: '14px 16px',
    maxHeight: 340,
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    lineHeight: 1.8,
    background: 'var(--terminal-bg)',
  },
  line: { display: 'flex', gap: 12, wordBreak: 'break-all' },
  lineNum: { color: 'var(--terminal-line)', minWidth: 36, userSelect: 'none', flexShrink: 0 },
  empty: { color: 'var(--text-muted)', fontStyle: 'italic' },
}
