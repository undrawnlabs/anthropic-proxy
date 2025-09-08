export const metrics = {
  started_at: new Date().toISOString(),
  calls_total: 0,
  errors_total: 0,
  timeouts_total: 0,
  p95_ms: 0,
  last_ms: 0,
}
const last: number[] = []
export function recordLatency(ms: number) {
  metrics.last_ms = ms
  last.push(ms)
  if (last.length > 200) last.shift()
  const s = [...last].sort((a,b)=>a-b)
  const i = Math.floor(0.95 * (s.length - 1))
  metrics.p95_ms = s[i] || ms
}
