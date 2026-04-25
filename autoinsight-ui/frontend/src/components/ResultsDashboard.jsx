import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'
import { Trophy, Zap, Target, Clock } from 'lucide-react'

const COLORS = ['#00e5ff', '#ffb300', '#00e676', '#d500f9', '#ff1744']

export default function ResultsDashboard({ job }) {
  if (!job || job.status !== 'done') return null

  const models = [...(job.model_results || [])].sort((a, b) => {
    const isError = ['rmse', 'mae', 'mape'].includes(job.metric?.toLowerCase())
    return isError ? a.score - b.score : b.score - a.score
  })

  const topFeatures = Object.entries(job.feature_importance || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([name, val]) => ({ name: name.length > 20 ? name.slice(0, 18) + '…' : name, value: val }))

  const isError = ['rmse', 'mae', 'mape'].includes(job.metric?.toLowerCase())
  const isRegression = job.task_type?.toLowerCase() === 'regression'

  return (
    <div style={styles.wrap}>
      {/* Stat cards */}
      <div style={styles.stats}>
        <StatCard icon={Trophy} color="var(--amber)" label="Best Model" value={job.best_model || '—'} />
        <StatCard icon={Target} color="var(--cyan)" label={job.metric?.toUpperCase() || 'Score'}
          value={job.best_score != null ? job.best_score.toFixed(4) : '—'} />
        {isRegression && (
          <StatCard icon={Target} color="var(--green)" label="R2"
            value={job.best_r2 != null ? job.best_r2.toFixed(4) : '—'} />
        )}
        <StatCard icon={Zap} color="var(--green)" label="Task Type"
          value={job.task_type ? job.task_type.charAt(0).toUpperCase() + job.task_type.slice(1) : '—'} />
        <StatCard icon={Clock} color="var(--purple)" label="Elapsed"
          value={job.elapsed_seconds != null ? `${job.elapsed_seconds}s` : '—'} />
      </div>

      <div style={styles.charts}>
        {/* Model comparison */}
        <div style={styles.chartCard}>
          <div style={styles.chartHeader}>
            <span style={styles.chartLabel}>MODEL COMPARISON</span>
            <span style={styles.metricBadge}>
              {isRegression ? `${job.metric?.toUpperCase()} + R2` : job.metric?.toUpperCase()}
            </span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={models} layout="vertical" margin={{ left: 0, right: 24, top: 8, bottom: 8 }}>
              <XAxis
                type="number" dataKey="score"
                tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'Space Mono' }}
                axisLine={false} tickLine={false}
              />
              <YAxis
                type="category" dataKey="model_name" width={170}
                tick={{ fill: '#9ca3af', fontSize: 10, fontFamily: 'Space Mono' }}
                axisLine={false} tickLine={false}
              />
              <Tooltip
                contentStyle={{ background: '#0f1117', border: '1px solid #2a2d3a', borderRadius: 6,
                  fontFamily: 'Space Mono', fontSize: 11 }}
                labelStyle={{ color: '#e8eaf0' }}
                itemStyle={{ color: '#00e5ff' }}
                formatter={(v, _name, item) => {
                  const payload = item?.payload || {}
                  const extra = isRegression && payload.r2_score != null ? ` | R2 ${payload.r2_score.toFixed(4)}` : ''
                  return [`${v.toFixed(4)}${extra}`, job.metric?.toUpperCase()]
                }}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]} maxBarSize={22}>
                {models.map((_, i) => (
                  <Cell
                    key={i}
                    fill={COLORS[i % COLORS.length]}
                    opacity={i === 0 ? 1 : 0.82}
                    stroke={i === 0 ? '#e8f9ff' : 'none'}
                    strokeWidth={i === 0 ? 1 : 0}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Feature importance */}
        {topFeatures.length > 0 && (
          <div style={styles.chartCard}>
            <div style={styles.chartHeader}>
              <span style={styles.chartLabel}>FEATURE IMPORTANCE</span>
              <span style={styles.metricBadge}>TOP 10</span>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={topFeatures} layout="vertical" margin={{ left: 0, right: 24, top: 8, bottom: 8 }}>
                <XAxis
                  type="number"
                  tick={{ fill: '#6b7280', fontSize: 10, fontFamily: 'Space Mono' }}
                  axisLine={false} tickLine={false}
                />
                <YAxis
                  type="category" dataKey="name" width={160}
                  tick={{ fill: '#9ca3af', fontSize: 10, fontFamily: 'Space Mono' }}
                  axisLine={false} tickLine={false}
                />
                <Tooltip
                  contentStyle={{ background: '#0f1117', border: '1px solid #2a2d3a', borderRadius: 6,
                    fontFamily: 'Space Mono', fontSize: 11 }}
                  labelStyle={{ color: '#e8eaf0' }}
                  itemStyle={{ color: '#ffb300' }}
                  formatter={(v) => [v.toFixed(4), 'Importance']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18}>
                  {topFeatures.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} opacity={0.85 - i * 0.06} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

function StatCard({ icon: Icon, color, label, value }) {
  return (
    <div style={styles.statCard}>
      <Icon size={16} color={color} style={{ flexShrink: 0 }} />
      <div>
        <div style={{ ...styles.statLabel, color }}>{label}</div>
        <div style={styles.statValue}>{value}</div>
      </div>
    </div>
  )
}

const styles = {
  wrap: { display: 'flex', flexDirection: 'column', gap: 16 },
  stats: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 },
  statCard: {
    display: 'flex', alignItems: 'flex-start', gap: 12,
    padding: '14px 16px',
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
  },
  statLabel: { fontSize: 10, letterSpacing: '0.1em', fontWeight: 700, marginBottom: 3 },
  statValue: { fontSize: 15, fontWeight: 700, color: 'var(--text)', fontFamily: 'var(--font-display)' },
  charts: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
  chartCard: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-lg)',
    padding: '16px 20px',
  },
  chartHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  chartLabel: { fontSize: 11, fontWeight: 700, color: 'var(--cyan)', letterSpacing: '0.12em' },
  metricBadge: {
    fontSize: 10, padding: '2px 8px',
    background: 'rgba(0,229,255,0.1)', color: 'var(--cyan)',
    border: '1px solid rgba(0,229,255,0.25)', borderRadius: 4, fontWeight: 700,
  },
}
