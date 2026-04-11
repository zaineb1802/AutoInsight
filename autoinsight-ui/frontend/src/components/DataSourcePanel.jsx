import { useState, useRef } from 'react'
import { Upload, Link, FileSpreadsheet, X, CheckCircle } from 'lucide-react'
import { uploadFile } from '../api.js'

const TABS = [
  { id: 'file',   icon: Upload,          label: 'Upload File' },
  { id: 'url',    icon: Link,            label: 'URL' },
  { id: 'gsheet', icon: FileSpreadsheet, label: 'Google Sheets' },
]

export default function DataSourcePanel({ onChange }) {
  const [tab, setTab] = useState('file')
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState(null)
  const [url, setUrl] = useState('')
  const [gsheet, setGsheet] = useState('')
  const inputRef = useRef()

  async function handleFile(file) {
    if (!file) return
    setUploading(true)
    setError(null)
    setPreview(null)
    try {
      const result = await uploadFile(file)
      setPreview(result)
      onChange({ source_type: 'file', file_path: result.path, preview: result.preview })
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  function handleDrop(e) {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  function handleUrlSubmit() {
    if (!url.trim()) return
    onChange({ source_type: 'url', url: url.trim(), preview: null })
    setPreview({ filename: url.split('/').pop() })
  }

  function handleGsheetSubmit() {
    if (!gsheet.trim()) return
    onChange({ source_type: 'gsheet', gsheet_url: gsheet.trim(), preview: null })
    setPreview({ filename: 'Google Sheet' })
  }

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <span style={styles.label}>01 / DATA SOURCE</span>
      </div>

      {/* Tabs */}
      <div style={styles.tabs}>
        {TABS.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            style={{ ...styles.tab, ...(tab === id ? styles.tabActive : {}) }}
            onClick={() => { setTab(id); setPreview(null); setError(null) }}
          >
            <Icon size={13} />
            {label}
          </button>
        ))}
      </div>

      {/* File tab */}
      {tab === 'file' && (
        <div
          style={{ ...styles.dropzone, ...(dragging ? styles.dropzoneActive : {}) }}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef} type="file"
            accept=".csv,.xlsx,.xls,.json,.parquet"
            style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])}
          />
          {uploading ? (
            <div style={styles.dropContent}>
              <div style={styles.spinner} />
              <span style={styles.dropText}>Uploading...</span>
            </div>
          ) : preview ? (
            <div style={styles.dropContent}>
              <CheckCircle size={24} color="var(--green)" />
              <span style={styles.dropText}>{preview.filename}</span>
              {preview.preview && (
                <span style={styles.dropMeta}>
                  {preview.preview.rows?.toLocaleString()} rows · {preview.preview.columns?.length} columns
                </span>
              )}
            </div>
          ) : (
            <div style={styles.dropContent}>
              <Upload size={24} color="var(--cyan)" />
              <span style={styles.dropText}>Drop file or click to browse</span>
              <span style={styles.dropMeta}>CSV · Excel · JSON · Parquet</span>
            </div>
          )}
        </div>
      )}

      {/* URL tab */}
      {tab === 'url' && (
        <div style={styles.inputRow}>
          <input
            style={styles.input}
            placeholder="https://raw.githubusercontent.com/.../dataset.csv"
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleUrlSubmit()}
          />
          <button style={styles.smallBtn} onClick={handleUrlSubmit}>Load</button>
        </div>
      )}

      {/* Google Sheets tab */}
      {tab === 'gsheet' && (
        <div style={styles.inputRow}>
          <input
            style={styles.input}
            placeholder="https://docs.google.com/spreadsheets/d/..."
            value={gsheet}
            onChange={e => setGsheet(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleGsheetSubmit()}
          />
          <button style={styles.smallBtn} onClick={handleGsheetSubmit}>Load</button>
        </div>
      )}

      {/* Preview table */}
      {preview?.preview?.head && (
        <div style={styles.tableWrap}>
          <table style={styles.table}>
            <thead>
              <tr>
                {preview.preview.columns.slice(0, 7).map(c => (
                  <th key={c} style={styles.th}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.preview.head.map((row, i) => (
                <tr key={i}>
                  {preview.preview.columns.slice(0, 7).map(c => (
                    <td key={c} style={styles.td}>
                      {String(row[c] ?? '—').slice(0, 20)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {error && <div style={styles.error}>{error}</div>}
    </div>
  )
}

const styles = {
  card: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    overflow: 'hidden',
  },
  header: {
    padding: '12px 20px',
    borderBottom: '1px solid var(--border)',
    background: 'var(--surface2)',
  },
  label: {
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    color: 'var(--cyan)',
    letterSpacing: '0.12em',
    fontWeight: 700,
  },
  tabs: {
    display: 'flex',
    borderBottom: '1px solid var(--border)',
  },
  tab: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '10px 16px',
    background: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  tabActive: {
    color: 'var(--cyan)',
    borderBottomColor: 'var(--cyan)',
    background: 'rgba(0,229,255,0.04)',
  },
  dropzone: {
    margin: 20,
    border: '1px dashed var(--border2)',
    borderRadius: 'var(--radius)',
    padding: '32px 24px',
    cursor: 'pointer',
    transition: 'all 0.15s',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
  },
  dropzoneActive: {
    borderColor: 'var(--cyan)',
    background: 'rgba(0,229,255,0.04)',
  },
  dropContent: {
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
  },
  dropText: { fontSize: 14, color: 'var(--text-dim)' },
  dropMeta: { fontSize: 11, color: 'var(--text-muted)' },
  inputRow: {
    display: 'flex', gap: 8, padding: '16px 20px',
  },
  input: {
    flex: 1,
    background: 'var(--surface3)',
    border: '1px solid var(--border2)',
    borderRadius: 'var(--radius)',
    color: 'var(--text)',
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    padding: '8px 12px',
    outline: 'none',
  },
  smallBtn: {
    background: 'var(--cyan)',
    color: 'var(--bg)',
    border: 'none',
    borderRadius: 'var(--radius)',
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
    fontSize: 12,
    padding: '8px 16px',
    cursor: 'pointer',
  },
  tableWrap: {
    overflowX: 'auto',
    margin: '0 20px 20px',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
  },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 11 },
  th: {
    padding: '6px 10px',
    background: 'var(--surface3)',
    color: 'var(--cyan)',
    textAlign: 'left',
    fontWeight: 700,
    borderBottom: '1px solid var(--border)',
    whiteSpace: 'nowrap',
  },
  td: {
    padding: '5px 10px',
    borderBottom: '1px solid var(--border)',
    color: 'var(--text-dim)',
    whiteSpace: 'nowrap',
  },
  error: {
    margin: '0 20px 20px',
    padding: '10px 14px',
    background: 'rgba(255,23,68,0.08)',
    border: '1px solid rgba(255,23,68,0.3)',
    borderRadius: 'var(--radius)',
    color: 'var(--red)',
    fontSize: 12,
  },
  spinner: {
    width: 24, height: 24,
    border: '2px solid var(--border2)',
    borderTopColor: 'var(--cyan)',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
  },
}
