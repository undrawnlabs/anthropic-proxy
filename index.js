// index.js — Render + Upstash Redis + Anthropic Messages API
// Single-user persistent memory (core_id + session_id) — Redis LISTS (append-only)

import Fastify from 'fastify'
import cors from '@fastify/cors'
import fetch from 'node-fetch'
import { Redis } from '@upstash/redis'

// ===== Config =====
const PORT = process.env.PORT || 10000
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '400', 10)
// TTL: if set (seconds) → key auto-expires; if unset → persistent
const TTL_SECONDS = process.env.TTL_SECONDS ? parseInt(process.env.TTL_SECONDS, 10) : null
const CORE = (process.env.CORE_SYSTEM_PROMPT || '').trim()
const DEBUG = !!process.env.DEBUG

// ===== Redis (with RAM fallback) =====
const redis = (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN)
  ? new Redis({ url: process.env.UPSTASH_REDIS_REST_URL, token: process.env.UPSTASH_REDIS_REST_TOKEN })
  : null

const RAM = new Map()
const keyOf = (coreId, sessionId) => `hist:${coreId || 'exec'}:${sessionId}`

// --- Read full history (LIST -> array of {role,content})
async function readHistory(coreId, sessionId) {
  const key = keyOf(coreId, sessionId)
  if (redis) {
    const arr = await redis.lrange(key, 0, -1) // array of strings
    const parsed = arr.map(s => {
      try { return JSON.parse(s) } catch { return null }
    }).filter(Boolean)
    if (DEBUG) console.log('[READ]', key, 'len=', parsed.length)
    return parsed
  } else {
    const v = RAM.get(key) || []
    if (DEBUG) console.log('[READ:RAM]', key, 'len=', v.length)
    return v
  }
}

// --- Append messages & trim to HISTORY_MAX (RPUSH + LTRIM)
async function appendHistory(coreId, sessionId, newMessages) {
  const key = keyOf(coreId, sessionId)
  if (redis) {
    const payloads = newMessages.map(m => JSON.stringify(m))
    // push both messages atomically (server side)
    const newLen = await redis.rpush(key, ...payloads) // returns new length
    if (HISTORY_MAX && newLen > HISTORY_MAX) {
      // keep last HISTORY_MAX items
      const start = newLen - HISTORY_MAX
      await redis.ltrim(key, start, -1)
    }
    if (TTL_SECONDS && TTL_SECONDS > 0) {
      await redis.expire(key, TTL_SECONDS) // refresh TTL on write
    }
    if (DEBUG) console.log('[APPEND]', key, 'added=', newMessages.length, 'len≈', Math.min(newLen, HISTORY_MAX))
    return true
  } else {
    const cur = RAM.get(key) || []
    const next = [...cur, ...newMessages]
    const trimmed = HISTORY_MAX ? next.slice(-HISTORY_MAX) : next
    RAM.set(key, trimmed)
    if (DEBUG) console.log('[APPEND:RAM]', key, 'len=', trimmed.length)
    return true
  }
}

// ===== Server =====
const app = Fastify({ logger: true })
await app.register(cors, { origin: '*' })

app.get('/health', async () => ({
  ok: true,
  redis: !!redis,
  history_cap: HISTORY_MAX,
  ttl_seconds: TTL_SECONDS ?? null
}))

/**
 * POST /v1/complete
 * {
 *   "core_id": "exec",
 *   "session_id": "perm-1",
 *   "model": "claude-3-7-sonnet-20250219",
 *   "prompt": "text",
 *   "locale": "uk",
 *   "max_tokens": 500
 * }
 */
app.post('/v1/complete', async (req, reply) => {
  try {
    const {
      core_id = 'exec',
      session_id,
      model,
      prompt,
      locale = 'uk',
      max_tokens = 500
    } = req.body || {}

    if (!model)  return reply.code(400).send({ ok:false, error:'model_required' })
    if (!prompt) return reply.code(400).send({ ok:false, error:'prompt_required' })
    if (!session_id) return reply.code(400).send({ ok:false, error:'session_id_required' })

    // 1) read current history
    const history = await readHistory(core_id, session_id)

    // 2) system prompt (core + language discipline)
    const languageDiscipline = [
      'Language Discipline:',
      `• Respond only in the user's language: ${locale}.`,
      '• Do not translate unless explicitly asked.',
      '• Do not mix languages in a single reply.',
      '• No hallucinations — if unknown: "Unknown with current data."'
    ].join('\n')
    const system = [CORE, languageDiscipline].filter(Boolean).join('\n\n')

    // 3) Anthropic messages: history + current user
    const anthroHistory = history.map(m => ({
      role: m.role,
      content: [{ type: 'text', text: m.content }]
    }))
    const userMsg = { role: 'user', content: [{ type: 'text', text: String(prompt) }] }
    const messages = [...anthroHistory, userMsg]
    // для безпеки: если история велика — Anthropic все одно має ліміт контексту

    // 4) call Anthropic
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({ model, max_tokens, system, messages })
    })

    if (!r.ok) {
      const detail = await r.text().catch(() => '')
      if (DEBUG) console.error('[ANTHROPIC]', r.status, detail)
      return reply.code(r.status).send({ ok:false, error:'anthropic_error', detail })
    }

    const data = await r.json()
    const text =
      Array.isArray(data?.content)
        ? data.content.map(b => b?.text ?? '').join('')
        : (data?.content?.text ?? '')
    const assistantReply = (text || '').trim()

    // 5) append both messages to Redis LIST
    await appendHistory(core_id, session_id, [
      { role: 'user',      content: String(prompt) },
      { role: 'assistant', content: assistantReply }
    ])

    // 6) (optional) return current length — читаємо швидко (без JSON парсу)
    let messagesLen = null
    if (redis) {
      messagesLen = await redis.llen(keyOf(core_id, session_id))
    } else {
      messagesLen = (RAM.get(keyOf(core_id, session_id)) || []).length
    }

    return reply.send({
      ok: true,
      content: assistantReply,
      meta: {
        core_id,
        session_id,
        history_messages: messagesLen,
        history_cap: HISTORY_MAX
      }
    })
  } catch (err) {
    if (DEBUG) console.error('[SERVER]', err)
    req.log.error(err)
    return reply.code(500).send({ ok:false, error:'server_error' })
  }
})

app.get('/v1/history/len', async (req, reply) => {
  const { core_id = 'exec', session_id } = req.query || {}
  if (!session_id) return reply.code(400).send({ ok:false, error:'session_id_required' })
  if (redis) {
    const n = await redis.llen(keyOf(core_id, session_id))
    return reply.send({ ok:true, messages: n, cap: HISTORY_MAX })
  } else {
    const n = (RAM.get(keyOf(core_id, session_id)) || []).length
    return reply.send({ ok:true, messages: n, cap: HISTORY_MAX })
  }
})

app.delete('/v1/history', async (req, reply) => {
  const { core_id = 'exec', session_id } = req.query || {}
  if (!session_id) return reply.code(400).send({ ok:false, error:'session_id_required' })
  const key = keyOf(core_id, session_id)
  if (redis) await redis.del(key); else RAM.delete(key)
  return reply.send({ ok:true, cleared:true })
})

app.listen({ port: PORT, host: '0.0.0.0' })
  .catch(err => { console.error(err); process.exit(1) })
