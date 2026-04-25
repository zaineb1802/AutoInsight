const BASE = '/api'

export async function uploadFile(file) {
  const fd = new FormData()
  fd.append('file', file)
  const r = await fetch(`${BASE}/upload`, { method: 'POST', body: fd })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function runJob(config) {
  const r = await fetch(`${BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getJob(jobId) {
  const r = await fetch(`${BASE}/jobs/${jobId}`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function listJobs() {
  const r = await fetch(`${BASE}/jobs`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function deleteJob(jobId) {
  const r = await fetch(`${BASE}/jobs/${jobId}`, { method: 'DELETE' })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getReport(jobId) {
  const r = await fetch(`${BASE}/jobs/${jobId}/report`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export function getModelDownloadUrl(jobId) {
  return `${BASE}/jobs/${jobId}/model`
}

export function streamLogs(jobId, onLine, onDone) {
  const es = new EventSource(`${BASE}/jobs/${jobId}/logs`)
  es.onmessage = (e) => {
    const data = e.data
    if (data.startsWith('[STATUS:')) {
      onDone(data.includes('done') ? 'done' : 'failed')
      es.close()
    } else {
      onLine(data)
    }
  }
  es.onerror = () => { es.close(); onDone('failed') }
  return () => es.close()
}
