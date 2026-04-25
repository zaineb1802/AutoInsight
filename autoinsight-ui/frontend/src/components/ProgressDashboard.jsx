import { useState, useEffect, useRef } from 'react'
import { Check, Loader, AlertCircle, TrendingUp } from 'lucide-react'

const STAGES = [
  'Parsing goal',
  'Exploratory data analysis',
  'Planning strategy',
  'Validating data',
  'Cleaning data',
  'Engineering features',
  'Training models',
  'Generating report',
]

export default function ProgressDashboard({ activeJob, status, onStreamLogs }) {
  const [stageResults, setStageResults] = useState({})
  const [jobStats, setJobStats] = useState(null)
  const bottomRef = useRef()

  useEffect(() => {
    if (!activeJob?.job_id) return
    
    // Connect to events stream
    const stopStream = onStreamLogs
      ? onStreamLogs(activeJob.job_id, (event) => {
          try {
            const data = JSON.parse(event)
            handleEvent(data)
          } catch (e) {
            // Silently ignore parse errors
          }
        })
      : null

    return () => stopStream?.()
  }, [activeJob?.job_id, onStreamLogs])

  function handleEvent(data) {
    if (data.type === 'stage_result') {
      setStageResults(prev => ({
        ...prev,
        [data.stage]: {
          ...(prev[data.stage] || {}),
          [Object.keys(data.data || {})[0]]: Object.values(data.data || {})[0],
        },
      }))
    } else if (data.type === 'job_complete') {
      setJobStats(data.data)
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [stageResults])

  const completedStages = activeJob?.stages_done || []
  const isRunning = status === 'running'
  const isFailed = status === 'failed'

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <h3 style={styles.title}>Pipeline Progress</h3>
          {isRunning && <Loader size={16} style={{ animation: 'spin 1s linear infinite' }} />}
        </div>
        <div style={{
          ...styles.statusBadge,
          background: isRunning ? '#ffb300' : isFailed ? '#ff1744' : '#00e676',
        }}>
          {status.toUpperCase()}
        </div>
      </div>

      {/* Stages */}
      <div style={styles.stagesContainer}>
        {STAGES.map((stage, idx) => {
          const isDone = completedStages.includes(stage)
          const isActive = activeJob?.current_stage === stage && isRunning
          const results = stageResults[stage]

          return (
            <div key={stage} style={styles.stageCard}>
              <div style={styles.stageHeader}>
                <div style={styles.stageBadge}>
                  {isDone ? (
                    <Check size={14} color="white" strokeWidth={3} />
                  ) : isActive ? (
                    <Loader
                      size={14}
                      color="white"
                      style={{ animation: 'spin 1s linear infinite' }}
                    />
                  ) : (
                    <span style={styles.stageNum}>{idx + 1}</span>
                  )}
                </div>
                <h4 style={{
                  ...styles.stageName,
                  color: isDone ? '#00e5ff' : isActive ? '#ffb300' : '#9ca3af',
                }}>
                  {stage}
                </h4>
              </div>

              {/* Stage Results */}
              {results && Object.keys(results).length > 0 && (
                <div style={styles.results}>
                  {Object.entries(results).map(([key, val]) => (
                    <div key={key} style={styles.result}>
                      <span style={styles.resultLabel}>{key}:</span>
                      <span style={styles.resultValue}>
                        {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>

      {/* Final Stats */}
      {jobStats && (
        <div style={styles.stats}>
          <div style={styles.statsGrid}>
            {jobStats.best_model && (
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Best Model</span>
                <span style={styles.statValue}>{jobStats.best_model}</span>
              </div>
            )}
            {jobStats.best_score && (
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Score</span>
                <span style={styles.statValue}>{jobStats.best_score.toFixed(4)}</span>
              </div>
            )}
            {jobStats.duration_seconds && (
              <div style={styles.statItem}>
                <span style={styles.statLabel}>Duration</span>
                <span style={styles.statValue}>{jobStats.duration_seconds}s</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {isFailed && activeJob?.error && (
        <div style={styles.error}>
          <AlertCircle size={16} color="#ff1744" />
          <span>{activeJob.error}</span>
        </div>
      )}
    </div>
  )
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    padding: '16px',
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    minHeight: '600px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '12px',
    borderBottom: '1px solid var(--border)',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  title: {
    margin: 0,
    fontSize: '14px',
    fontWeight: 700,
    color: 'var(--text)',
  },
  statusBadge: {
    padding: '4px 12px',
    borderRadius: '4px',
    fontSize: '10px',
    fontWeight: 700,
    color: 'white',
    fontFamily: 'var(--font-mono)',
  },
  stagesContainer: {
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    paddingRight: '4px',
  },
  stageCard: {
    padding: '12px',
    background: 'var(--surface2)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  stageHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '8px',
  },
  stageBadge: {
    width: '24px',
    height: '24px',
    borderRadius: '50%',
    background: 'var(--surface3)',
    border: '1px solid var(--border)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    fontSize: '10px',
    fontWeight: 700,
    color: 'var(--text-muted)',
  },
  stageName: {
    margin: 0,
    fontSize: '12px',
    fontWeight: 600,
  },
  results: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    paddingLeft: '34px',
  },
  result: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '11px',
  },
  resultLabel: {
    color: 'var(--cyan)',
    fontFamily: 'var(--font-mono)',
    fontWeight: 600,
    minWidth: '100px',
  },
  resultValue: {
    color: 'var(--text)',
    fontFamily: 'var(--font-mono)',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  stats: {
    padding: '12px',
    background: 'var(--surface2)',
    border: '1px solid var(--border)',
    borderRadius: '6px',
  },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
    gap: '12px',
  },
  statItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  statLabel: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  statValue: {
    fontSize: '14px',
    fontWeight: 700,
    color: 'var(--green)',
    fontFamily: 'var(--font-mono)',
  },
  error: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '12px',
    background: 'rgba(255, 23, 68, 0.1)',
    border: '1px solid #ff1744',
    borderRadius: '6px',
    fontSize: '12px',
    color: '#ff1744',
  },
}
