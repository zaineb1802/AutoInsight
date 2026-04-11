import { Check, Circle, Loader } from 'lucide-react'

const ALL_STAGES = [
  'Parsing goal',
  'Exploratory data analysis',
  'Planning strategy',
  'Validating data',
  'Cleaning data',
  'Engineering features',
  'Training models',
  'Generating report',
]

export default function StageProgress({ currentStage, stagesDone, status }) {
  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <span style={styles.label}>PIPELINE STAGES</span>
      </div>
      <div style={styles.stages}>
        {ALL_STAGES.map((stage, i) => {
          const done = stagesDone.includes(stage)
          const active = currentStage === stage && status === 'running'
          const pending = !done && !active

          return (
            <div key={stage} style={styles.stage}>
              <div style={styles.iconCol}>
                <div style={{
                  ...styles.icon,
                  ...(done ? styles.iconDone : active ? styles.iconActive : styles.iconPending)
                }}>
                  {done
                    ? <Check size={11} color="var(--bg)" strokeWidth={3} />
                    : active
                    ? <Loader size={11} style={{ animation: 'spin 1s linear infinite' }} />
                    : <span style={styles.stageNum}>{i + 1}</span>
                  }
                </div>
                {i < ALL_STAGES.length - 1 && (
                  <div style={{ ...styles.connector, ...(done ? styles.connectorDone : {}) }} />
                )}
              </div>
              <span style={{
                ...styles.stageName,
                color: done ? 'var(--text)' : active ? 'var(--cyan)' : 'var(--text-muted)',
                fontWeight: done ? 600 : active ? 700 : 400,
              }}>
                {stage}
              </span>
            </div>
          )
        })}
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
  header: {
    padding: '10px 16px',
    borderBottom: '1px solid var(--border)',
    background: 'var(--surface2)',
  },
  label: { fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--cyan)', letterSpacing: '0.12em', fontWeight: 700 },
  stages: { padding: '12px 16px', display: 'flex', flexDirection: 'column' },
  stage: { display: 'flex', alignItems: 'flex-start', gap: 12, minHeight: 36 },
  iconCol: { display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 },
  icon: {
    width: 22, height: 22, borderRadius: '50%',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 10, fontWeight: 700, fontFamily: 'var(--font-mono)',
    flexShrink: 0,
  },
  iconDone:    { background: 'var(--green)' },
  iconActive:  { background: 'transparent', border: '2px solid var(--cyan)', color: 'var(--cyan)' },
  iconPending: { background: 'var(--surface3)', border: '1px solid var(--border2)', color: 'var(--text-muted)' },
  stageNum: { fontSize: 9 },
  connector: {
    width: 1, flex: 1, minHeight: 14,
    background: 'var(--border)', margin: '2px 0',
  },
  connectorDone: { background: 'var(--green)' },
  stageName: { fontSize: 12, paddingTop: 3, transition: 'color 0.2s', lineHeight: 1.4 },
}
