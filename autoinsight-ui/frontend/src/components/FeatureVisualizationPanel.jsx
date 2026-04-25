import { useState, useEffect } from 'react'
import { Download, ChevronDown } from 'lucide-react'
import { getFeatures, getVisualizationUrl } from '../api.js'

export default function FeatureVisualizationPanel({ jobId, status }) {
  const [features, setFeatures] = useState([])
  const [selectedFeature, setSelectedFeature] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (status === 'done' && jobId) {
      setLoading(true)
      getFeatures(jobId)
        .then(data => {
          setFeatures(data.features || [])
          if (data.features && data.features.length > 0) {
            setSelectedFeature(data.features[0])
          }
        })
        .catch(e => setError(e.message))
        .finally(() => setLoading(false))
    }
  }, [status, jobId])

  function handleDownload() {
    if (!selectedFeature) return
    const link = document.createElement('a')
    link.href = getVisualizationUrl(jobId, selectedFeature)
    link.download = `${selectedFeature}_visualization.png`
    link.click()
  }

  if (status !== 'done' || features.length === 0) return null

  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <span style={styles.label}>FEATURE VISUALIZATION</span>
      </div>

      <div style={styles.body}>
        {/* Feature selector */}
        <div style={styles.selectorWrap}>
          <label style={styles.fieldLabel}>SELECT FEATURE</label>
          <div style={styles.selectWrapper}>
            <select
              value={selectedFeature || ''}
              onChange={e => setSelectedFeature(e.target.value)}
              style={styles.select}
            >
              {features.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
            <ChevronDown size={14} style={styles.selectIcon} />
          </div>
        </div>

        {/* Visualization display */}
        {selectedFeature && (
          <div style={styles.vizContainer}>
            <img
              src={getVisualizationUrl(jobId, selectedFeature)}
              alt={`${selectedFeature} visualization`}
              style={styles.vizImage}
              onError={() => setError('Failed to load visualization')}
            />
            <button
              style={styles.downloadBtn}
              onClick={handleDownload}
              title="Download as PNG"
            >
              <Download size={13} />
              Download PNG
            </button>
          </div>
        )}

        {error && <div style={styles.errorMsg}>{error}</div>}
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
    marginBottom: 16,
  },
  header: {
    padding: '12px 16px',
    borderBottom: '1px solid var(--border)',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  label: {
    fontFamily: 'var(--font-mono)',
    fontSize: 10,
    fontWeight: 700,
    color: 'var(--cyan)',
    letterSpacing: '0.12em',
  },
  body: {
    padding: 16,
    display: 'flex',
    flexDirection: 'column',
    gap: 14,
  },
  selectorWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: 8,
  },
  fieldLabel: {
    fontSize: 10,
    fontWeight: 600,
    color: 'var(--text-muted)',
    letterSpacing: '0.08em',
  },
  selectWrapper: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
  },
  select: {
    width: '100%',
    padding: '8px 32px 8px 12px',
    borderRadius: 'var(--radius)',
    border: '1px solid var(--border)',
    background: 'var(--surface2)',
    color: 'var(--text)',
    fontFamily: 'var(--font-mono)',
    fontSize: 12,
    cursor: 'pointer',
    appearance: 'none',
    outline: 'none',
  },
  selectIcon: {
    position: 'absolute',
    right: 10,
    pointerEvents: 'none',
    color: 'var(--text-dim)',
  },
  vizContainer: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 12,
    padding: 12,
    background: 'var(--surface2)',
    borderRadius: 'var(--radius)',
  },
  vizImage: {
    maxWidth: '100%',
    maxHeight: 400,
    borderRadius: 'var(--radius)',
    border: '1px solid var(--border)',
  },
  downloadBtn: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 12px',
    borderRadius: 'var(--radius)',
    border: '1px solid var(--border)',
    background: 'var(--surface3)',
    color: 'var(--text)',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
    fontWeight: 600,
    cursor: 'pointer',
    transition: 'all 0.2s',
  },
  errorMsg: {
    padding: 12,
    borderRadius: 'var(--radius)',
    background: 'rgba(255, 23, 68, 0.1)',
    color: '#ff1744',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
  },
}
