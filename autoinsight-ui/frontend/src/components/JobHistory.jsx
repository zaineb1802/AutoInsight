import { Trash2, ChevronRight } from 'lucide-react'
import { deleteJob } from '../api.js'

const STATUS_COLOR = {
  pending: '#6b7280',
  running: '#ffb300',
  done:    '#00e676',
  failed:  '#ff1744',
}

export default function JobHistory({ jobs, activeJobId, onSelect, onDelete }) {
  async function handleDelete(e, jobId) {
    e.stopPropagation()
    await deleteJob(jobId)
    onDelete(jobId)
  }

  if (jobs.length === 0) {
    return (
      <div style={styles.empty}>
        <span style={styles.emptyText}>No jobs yet.</span>
        <span style={styles.emptyText}>Run your first analysis.</span>
      </div>
    )
  }

  return (
    <div style={styles.wrap}>
      {jobs.map(job => (
        <div
          key={job.job_id}
          style={{ ...styles.item, ...(job.job_id === activeJobId ? styles.itemActive : {}) }}
          onClick={() => onSelect(job.job_id)}
        >
          <div style={styles.itemHeader}>
            <span style={{ ...styles.statusDot, background: STATUS_COLOR[job.status] }} />
            <span style={styles.jobId}>#{job.job_id.slice(0, 8)}</span>
            <button style={styles.deleteBtn} onClick={e => handleDelete(e, job.job_id)} title="Delete">
              <Trash2 size={11} />
            </button>
          </div>
          <div style={styles.goal}>{job.goal.slice(0, 52)}{job.goal.length > 52 ? '...' : ''}</div>
          <div style={styles.meta}>
            <span style={{ color: STATUS_COLOR[job.status], fontSize: 10, fontWeight: 700 }}>
              {job.status.toUpperCase()}
            </span>
            {job.best_model && (
              <span style={styles.metaText}>{job.best_model}</span>
            )}
            {job.best_score != null && (
              <span style={styles.metaText}>{job.best_score.toFixed(4)}</span>
            )}
          </div>
          {job.job_id === activeJobId && (
            <ChevronRight size={12} color="var(--cyan)" style={styles.arrow} />
          )}
        </div>
      ))}
    </div>
  )
}

const styles = {
  wrap: { display: 'flex', flexDirection: 'column', gap: 6 },
  item: {
    padding: '12px 14px', borderRadius: 'var(--radius)',
    border: '1px solid var(--border)',
    background: 'var(--surface2)',
    cursor: 'pointer', position: 'relative',
    transition: 'all 0.15s',
  },
  itemActive: { border: '1px solid rgba(0,229,255,0.4)', background: 'rgba(0,229,255,0.04)' },
  itemHeader: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 },
  statusDot: { width: 7, height: 7, borderRadius: '50%', flexShrink: 0 },
  jobId: { fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', flex: 1 },
  deleteBtn: {
    background: 'transparent', border: 'none', color: 'var(--text-muted)',
    cursor: 'pointer', padding: 2, borderRadius: 3,
    display: 'flex', alignItems: 'center',
    opacity: 0.5,
  },
  goal: { fontSize: 12, color: 'var(--text-dim)', lineHeight: 1.4, marginBottom: 6 },
  meta: { display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' },
  metaText: { fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' },
  arrow: { position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)' },
  empty: { display: 'flex', flexDirection: 'column', gap: 4, padding: '12px 0' },
  emptyText: { fontSize: 12, color: 'var(--text-muted)', fontStyle: 'italic' },
}
