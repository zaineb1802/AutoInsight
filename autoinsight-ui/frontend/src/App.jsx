import { useState, useEffect, useRef } from 'react'
import { Play, RotateCcw, BrainCircuit, Sun, Moon } from 'lucide-react'
import DataSourcePanel from './components/DataSourcePanel.jsx'
import LogTerminal from './components/LogTerminal.jsx'
import StageProgress from './components/StageProgress.jsx'
import ResultsDashboard from './components/ResultsDashboard.jsx'
import ReportViewer from './components/ReportViewer.jsx'
import JobHistory from './components/JobHistory.jsx'
import { runJob, getJob, listJobs, streamLogs } from './api.js'

const LLM_OPTIONS = ['auto', 'groq', 'gemini']

export default function App() {
  // Theme state
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem('theme')
    return saved || 'dark'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  // Form state
  const [source, setSource] = useState(null)
  const [goal, setGoal] = useState('')
  const [llm, setLlm] = useState('auto')

  // Job state
  const [jobs, setJobs] = useState([])
  const [activeJobId, setActiveJobId] = useState(null)
  const [activeJob, setActiveJob] = useState(null)
  const [logs, setLogs] = useState([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const stopStreamRef = useRef(null)
  const pollRef = useRef(null)

  // Load history on mount
  useEffect(() => {
    listJobs().then(setJobs).catch(() => {})
  }, [])

  // Poll active job
  useEffect(() => {
    if (!activeJobId) return
    clearInterval(pollRef.current)
    pollRef.current = setInterval(async () => {
      try {
        const job = await getJob(activeJobId)
        setActiveJob(job)
        setJobs(prev => prev.map(j => j.job_id === job.job_id ? job : j))
        if (job.status === 'done' || job.status === 'failed') {
          clearInterval(pollRef.current)
        }
      } catch {}
    }, 1000)
    return () => clearInterval(pollRef.current)
  }, [activeJobId])

  async function handleRun() {
    if (!source) return setError('Please select a data source.')
    if (!goal.trim()) return setError('Please describe your ML goal.')
    setError(null)
    setSubmitting(true)
    setLogs([])

    try {
      const config = {
        goal: goal.trim(),
        llm,
        source_type: source.source_type,
        file_path: source.file_path || null,
        url: source.url || null,
        gsheet_url: source.gsheet_url || null,
      }

      const job = await runJob(config)
      setActiveJobId(job.job_id)
      setActiveJob(job)
      setJobs(prev => [job, ...prev])

      // Stop previous SSE stream
      if (stopStreamRef.current) stopStreamRef.current()

      // Stream logs
      stopStreamRef.current = streamLogs(
        job.job_id,
        (line) => setLogs(prev => [...prev, line]),
        (finalStatus) => { /* polling handles final state */ },
      )
    } catch (e) {
      setError(e.message)
    } finally {
      setSubmitting(false)
    }
  }

  function handleSelectJob(jobId) {
    setActiveJobId(jobId)
    const job = jobs.find(j => j.job_id === jobId)
    if (job) {
      setActiveJob(job)
      setLogs(job.logs || [])
    }
  }

  function handleDeleteJob(jobId) {
    setJobs(prev => prev.filter(j => j.job_id !== jobId))
    if (activeJobId === jobId) {
      setActiveJobId(null)
      setActiveJob(null)
      setLogs([])
    }
  }

  function handleReset() {
    if (stopStreamRef.current) stopStreamRef.current()
    clearInterval(pollRef.current)
    setActiveJobId(null)
    setActiveJob(null)
    setLogs([])
    setError(null)
  }

  const isRunning = activeJob?.status === 'running' || submitting
  const status = activeJob?.status || 'pending'

  return (
    <div style={styles.root}>
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <BrainCircuit size={22} color="var(--cyan)" />
          <div>
            <div style={styles.logo}>AUTOINSIGHT</div>
            <div style={styles.tagline}>LLM-Driven AutoML Pipeline</div>
          </div>
        </div>
        <div style={styles.headerRight}>
          <button style={styles.themeBtn} onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
          </button>
          {activeJob && (
            <button style={styles.resetBtn} onClick={handleReset}>
              <RotateCcw size={13} /> New Run
            </button>
          )}
          <div style={styles.versionBadge}>v1.0</div>
        </div>
      </header>

      {/* ── Main layout ────────────────────────────────────────────── */}
      <div style={styles.layout}>

        {/* Left sidebar — history */}
        <aside style={styles.sidebar}>
          <div style={styles.sidebarHeader}>
            <span style={styles.sidebarLabel}>RUN HISTORY</span>
            <span style={styles.sidebarCount}>{jobs.length}</span>
          </div>
          <JobHistory
            jobs={jobs}
            activeJobId={activeJobId}
            onSelect={handleSelectJob}
            onDelete={handleDeleteJob}
          />
        </aside>

        {/* Center — form + run + results */}
        <main style={styles.main}>

          {/* Config form */}
          <section style={styles.configSection}>
            <DataSourcePanel onChange={setSource} />

            {/* Goal + LLM row */}
            <div style={styles.goalRow}>
              <div style={styles.goalWrap}>
                <label style={styles.fieldLabel}>02 / ML GOAL</label>
                <textarea
                  style={styles.textarea}
                  placeholder='e.g. "Predict whether a passenger survived"'
                  value={goal}
                  onChange={e => setGoal(e.target.value)}
                  rows={2}
                />
              </div>
              <div style={styles.llmWrap}>
                <label style={styles.fieldLabel}>03 / LLM BACKEND</label>
                <div style={styles.llmBtns}>
                  {LLM_OPTIONS.map(opt => (
                    <button
                      key={opt}
                      style={{ ...styles.llmBtn, ...(llm === opt ? styles.llmBtnActive : {}) }}
                      onClick={() => setLlm(opt)}
                    >
                      {opt.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {error && <div style={styles.errorBanner}>{error}</div>}

            {/* Run button */}
            <button
              style={{ ...styles.runBtn, ...(isRunning ? styles.runBtnDisabled : {}) }}
              onClick={handleRun}
              disabled={isRunning}
            >
              {isRunning ? (
                <>
                  <div style={styles.spinner} />
                  Running Analysis...
                </>
              ) : (
                <>
                  <Play size={15} fill="currentColor" />
                  Run AutoInsight
                </>
              )}
            </button>
          </section>

          {/* Results */}
          {activeJob && (
            <>
              <ResultsDashboard job={activeJob} />
              <LogTerminal logs={logs} status={status} />
              <ReportViewer jobId={activeJobId} status={status} />
            </>
          )}
        </main>

        {/* Right — stage progress */}
        {activeJob && (
          <aside style={styles.rightSidebar}>
            <StageProgress
              currentStage={activeJob.current_stage}
              stagesDone={activeJob.stages_done || []}
              status={status}
            />

            {/* Target + metric info */}
            {activeJob.target_column && (
              <div style={styles.infoCard}>
                <InfoRow label="Target" value={activeJob.target_column} color="var(--amber)" />
                <InfoRow label="Task" value={activeJob.task_type} color="var(--cyan)" />
                <InfoRow label="Metric" value={activeJob.metric?.toUpperCase()} color="var(--green)" />
              </div>
            )}
          </aside>
        )}
      </div>
    </div>
  )
}

function InfoRow({ label, value, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '6px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.08em' }}>{label}</span>
      <span style={{ fontSize: 12, color, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{value || '—'}</span>
    </div>
  )
}

const styles = {
  root: { display: 'flex', flexDirection: 'column', minHeight: '100vh' },

  header: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '14px 28px',
    background: 'var(--surface)',
    borderBottom: '1px solid var(--border)',
    position: 'sticky', top: 0, zIndex: 100,
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 12 },
  logo: { fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 800, letterSpacing: '0.08em', color: 'var(--text)' },
  tagline: { fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.06em' },
  headerRight: { display: 'flex', alignItems: 'center', gap: 12 },
  resetBtn: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '6px 12px', borderRadius: 'var(--radius)',
    background: 'var(--surface3)', border: '1px solid var(--border2)',
    color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 11,
    cursor: 'pointer',
  },
  versionBadge: {
    padding: '3px 8px', borderRadius: 4,
    background: 'rgba(0,229,255,0.1)', border: '1px solid rgba(0,229,255,0.2)',
    color: 'var(--cyan)', fontFamily: 'var(--font-mono)', fontSize: 10, fontWeight: 700,
  },
  themeBtn: {
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: '6px', borderRadius: 'var(--radius)',
    background: 'var(--surface3)', border: '1px solid var(--border2)',
    color: 'var(--text-dim)', cursor: 'pointer',
  },

  layout: { display: 'flex', flex: 1, overflow: 'hidden' },

  sidebar: {
    width: 240, flexShrink: 0,
    borderRight: '1px solid var(--border)',
    background: 'var(--surface)',
    padding: '16px 12px',
    overflowY: 'auto',
    display: 'flex', flexDirection: 'column', gap: 12,
  },
  sidebarHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 },
  sidebarLabel: { fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--cyan)', letterSpacing: '0.12em', fontWeight: 700 },
  sidebarCount: {
    fontSize: 10, padding: '2px 6px', borderRadius: 10,
    background: 'var(--surface3)', color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
  },

  main: { flex: 1, overflowY: 'auto', padding: '24px 28px', display: 'flex', flexDirection: 'column', gap: 20 },

  configSection: { display: 'flex', flexDirection: 'column', gap: 14 },

  goalRow: { display: 'flex', gap: 14 },
  goalWrap: { flex: 1, display: 'flex', flexDirection: 'column', gap: 8 },
  llmWrap: { width: 200, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 8 },
  fieldLabel: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--cyan)', letterSpacing: '0.12em', fontWeight: 700 },
  textarea: {
    width: '100%',
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    color: 'var(--text)',
    fontFamily: 'var(--font-mono)',
    fontSize: 13,
    padding: '10px 14px',
    resize: 'vertical',
    outline: 'none',
    lineHeight: 1.5,
  },
  llmBtns: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 },
  llmBtn: {
    padding: '8px 6px', borderRadius: 'var(--radius)',
    background: 'var(--surface3)', border: '1px solid var(--border2)',
    color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
    fontSize: 10, fontWeight: 700, cursor: 'pointer', letterSpacing: '0.06em',
    transition: 'all 0.15s',
  },
  llmBtnActive: {
    background: 'rgba(0,229,255,0.1)',
    border: '1px solid rgba(0,229,255,0.5)',
    color: 'var(--cyan)',
  },

  errorBanner: {
    padding: '10px 14px',
    background: 'rgba(255,23,68,0.08)', border: '1px solid rgba(255,23,68,0.3)',
    borderRadius: 'var(--radius)', color: 'var(--red)',
    fontFamily: 'var(--font-mono)', fontSize: 12,
  },

  runBtn: {
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
    padding: '14px 28px', width: '100%',
    background: 'var(--cyan)', color: 'var(--bg)',
    border: 'none', borderRadius: 'var(--radius)',
    fontFamily: 'var(--font-display)', fontSize: 15, fontWeight: 800,
    letterSpacing: '0.04em', cursor: 'pointer', transition: 'all 0.2s',
  },
  runBtnDisabled: { background: 'var(--surface3)', color: 'var(--text-muted)', cursor: 'not-allowed' },
  spinner: {
    width: 16, height: 16,
    border: '2px solid rgba(0,0,0,0.2)',
    borderTopColor: 'var(--bg)',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },

  rightSidebar: {
    width: 220, flexShrink: 0,
    borderLeft: '1px solid var(--border)',
    background: 'var(--surface)',
    padding: '16px 12px',
    overflowY: 'auto',
    display: 'flex', flexDirection: 'column', gap: 12,
  },
  infoCard: {
    background: 'var(--surface2)', border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)', padding: '12px 14px',
    display: 'flex', flexDirection: 'column',
  },
}