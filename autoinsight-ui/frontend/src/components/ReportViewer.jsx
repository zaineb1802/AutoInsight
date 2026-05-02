import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { FileText, Download, Copy, Check } from 'lucide-react'
import { getModelDownloadUrl, getReport } from '../api.js'

export default function ReportViewer({ jobId, status }) {
  const [markdown, setMarkdown] = useState(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (status === 'done' && jobId) {
      setLoading(true)
      getReport(jobId)
        .then(d => setMarkdown(d.markdown))
        .catch(() => setMarkdown('_Report unavailable_'))
        .finally(() => setLoading(false))
    }
  }, [status, jobId])

  function handleCopy() {
    navigator.clipboard.writeText(markdown || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function handleDownload() {
    const blob = new Blob([markdown || ''], { type: 'text/markdown' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `autoinsight-report-${jobId?.slice(0, 8)}.md`
    a.click()
  }

  if (status !== 'done') return null

  return (
    <div style={styles.wrap}>
      <div style={styles.bar}>
        <div style={styles.barLeft}>
          <FileText size={13} color="var(--amber)" />
          <span style={styles.barLabel}>REPORT</span>
        </div>
        <div style={styles.actions}>
          <button style={styles.actionBtn} onClick={handleCopy} title="Copy markdown">
            {copied ? <Check size={13} color="var(--green)" /> : <Copy size={13} />}
            {copied ? 'Copied' : 'Copy'}
          </button>
          <button style={styles.actionBtn} onClick={handleDownload} title="Download .md">
            <Download size={13} />
            Download
          </button>
          <a
            style={{ ...styles.actionBtn, textDecoration: 'none' }}
            href={getModelDownloadUrl(jobId)}
            title="Download best model .pkl"
          >
            <Download size={13} />
            Model PKL
          </a>
        </div>
      </div>

      <div style={styles.body}>
        {loading && <div style={styles.loading}>Loading report...</div>}
        {markdown && (
          <div style={styles.markdown}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => <h1 style={styles.h1}>{children}</h1>,
                h2: ({ children }) => <h2 style={styles.h2}>{children}</h2>,
                h3: ({ children }) => <h3 style={styles.h3}>{children}</h3>,
                p:  ({ children }) => <p style={styles.p}>{children}</p>,
                table: ({ children }) => <div style={styles.tableWrap}><table style={styles.table}>{children}</table></div>,
                th: ({ children }) => <th style={styles.th}>{children}</th>,
                td: ({ children }) => <td style={styles.td}>{children}</td>,
                code: ({ inline, children }) => inline
                  ? <code style={styles.inlineCode}>{children}</code>
                  : <pre style={styles.codeBlock}><code>{children}</code></pre>,
                blockquote: ({ children }) => <blockquote style={styles.blockquote}>{children}</blockquote>,
                li: ({ children }) => <li style={styles.li}>{children}</li>,
                strong: ({ children }) => <strong style={styles.strong}>{children}</strong>,
              }}
            >
              {markdown}
            </ReactMarkdown>
          </div>
        )}
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
  },
  bar: {
    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
    padding: '10px 16px',
    background: 'var(--surface2)',
    borderBottom: '1px solid var(--border)',
  },
  barLeft: { display: 'flex', alignItems: 'center', gap: 8 },
  barLabel: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--amber)', letterSpacing: '0.12em', fontWeight: 700 },
  actions: { display: 'flex', gap: 8 },
  actionBtn: {
    display: 'flex', alignItems: 'center', gap: 5,
    padding: '5px 10px', borderRadius: 'var(--radius)',
    background: 'var(--surface3)', border: '1px solid var(--border2)',
    color: 'var(--text-dim)', fontFamily: 'var(--font-mono)', fontSize: 11,
    cursor: 'pointer', transition: 'all 0.15s',
  },
  body: { padding: '24px 32px', maxHeight: 600, overflowY: 'auto' },
  loading: { color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 13, fontStyle: 'italic' },
  markdown: {},
  h1: { fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 800, color: 'var(--text)', marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid var(--border)' },
  h2: { fontFamily: 'var(--font-display)', fontSize: 16, fontWeight: 700, color: 'var(--cyan)', marginTop: 28, marginBottom: 12, letterSpacing: '0.02em' },
  h3: { fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 600, color: 'var(--amber)', marginTop: 20, marginBottom: 8 },
  p:  { fontSize: 13, color: 'var(--text-dim)', lineHeight: 1.75, marginBottom: 12 },
  tableWrap: { overflowX: 'auto', marginBottom: 16, border: '1px solid var(--border)', borderRadius: 'var(--radius)' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: 'var(--font-mono)' },
  th: { padding: '7px 12px', background: 'var(--surface3)', color: 'var(--cyan)', fontWeight: 700, textAlign: 'left', borderBottom: '1px solid var(--border)' },
  td: { padding: '6px 12px', color: 'var(--text-dim)', borderBottom: '1px solid var(--border)' },
  inlineCode: { fontFamily: 'var(--font-mono)', fontSize: 12, background: 'rgba(0,229,255,0.08)', color: 'var(--cyan)', padding: '1px 5px', borderRadius: 3 },
  codeBlock: { background: 'var(--terminal-bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '14px 16px', fontSize: 11, color: 'var(--text-dim)', overflowX: 'auto', marginBottom: 16 },
  blockquote: { borderLeft: '3px solid var(--amber)', paddingLeft: 14, margin: '12px 0', color: 'var(--text-muted)', fontStyle: 'italic', fontSize: 13 },
  li: { fontSize: 13, color: 'var(--text-dim)', marginBottom: 4, lineHeight: 1.65 },
  strong: { color: 'var(--text)', fontWeight: 700 },
}
