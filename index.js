// index.js — Render + Upstash Redis + Anthropic Messages API
// Persistent memory via Redis LIST (append-only). Reads full context from Redis.

import Fastify from 'fastify'
import cors from '@fastify/cors'
import fetch from 'node-fetch'
import { Redis } from '@upstash/redis'

// ===== Config =====
const PORT = process.env.PORT || 10000
const CORE = (process.env.CORE_SYSTEM_PROMPT || '').trim()
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '400', 10)
// TTL (seconds). If set -> auto-expire keys. If unset -> persistent.
const TTL_SECONDS = process.env.TTL_SECONDS ? parseInt(process.env.TTL_SECONDS, 10) : null
const DEBUG = !!process.env.DEBUG

// ===== Redis =====
const redis = (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN)
  ? new Redis({ url: process.env.UPSTASH_REDIS_REST_URL, token: process.env.UPSTASH_REDIS_REST_TOKEN })
  : null

const keyOf = (coreId, sessionId) => `histlist:${coreId || 'exec'}:${sessionId}`

// Read full history (LIST -> [{role,content}, ...])
async function readHistory(coreId, sessionId) {
  const key = keyOf(coreId, sessionId)
  if (!redis) return []
  const arr = await redis.lrange(key, 0, -1) // strings
  const items = arr.map(s => { try { return JSON.parse(s) } catch { return null } }).filter(Boolean)
  if (DEBUG) console.log('[READ]', key, 'len=', items.length)
  return items
}

// Append messages & trim (RPUSH + LTRIM)
async function appendHistory(coreId, sessionId, newMessages) {
  const key = keyOf(coreId, sessionId)
  if (!redis) return
  const payloads = newMessages.map(m => JSON.stringify(m))
  const newLen = await redis.rpush(key, ...payloads)
  if (HISTORY_MAX && newLen > HISTORY_MAX) {
    await redis.ltrim(key, newLen - HISTORY_MAX, -1)
  }
  if (TTL_SECONDS && TTL_SECONDS > 0) await redis.expire(key, TTL_SECONDS)
  if (DEBUG) console.log('[APPEND]', key, 'added=', newMessages.length)
}

// Convert our history to Anthropic messages format
function toAnthropicMessages(history) {
  return (history || []).map(m => ({
    role: m.role, // 'user' | 'assistant'
    content: [{ type: 'text', text: String(m.content ?? '') }]
  }))
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
 * Body:
 * {
 *   "core_id": "exec",
 *   "session_id": "perm-1",
 *   "model": "claude-3-7-sonnet-20250219",
 *   "prompt": "text",
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
      max_tokens = 500
    } = req.body || {}

    if (!model)  return reply.code(400).send({ ok:false, error:'model_required' })
    if (!prompt) return reply.code(400).send({ ok:false, error:'prompt_required' })
    if (!session_id) return reply.code(400).send({ ok:false, error:'session_id_required' })

    // 1) read entire history from Redis
    const history = await readHistory(core_id, session_id)

    // 2) build system
    const system = CORE || ''

    // 3) build Anthropic messages: full history + current user
    const anthroHistory = toAnthropicMessages(history)
    const userMsg = { role: 'user', content: [{ type: 'text', text: String(prompt) }] }
    const messages = [...anthroHistory, userMsg].slice(-HISTORY_MAX)

    if (DEBUG) app.log.info({ sendLen: messages.length }, 'anthropic_payload')

    // 4) call Anthropic
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model,
        max_tokens,
        temperature: 0,
        system,
        messages
      })
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

    // 5) append both messages to Redis
    await appendHistory(core_id, session_id, [
      { role: 'user',      content: String(prompt) },
      { role: 'assistant', content: assistantReply }
    ])

    // 6) length hint
    let messagesLen = null
    if (redis) messagesLen = await redis.llen(keyOf(core_id, session_id))

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
  const n = redis ? await redis.llen(keyOf(core_id, session_id)) : 0
  return reply.send({ ok:true, messages: n, cap: HISTORY_MAX })
})

app.delete('/v1/history', async (req, reply) => {
  const { core_id = 'exec', session_id } = req.query || {}
  if (!session_id) return reply.code(400).send({ ok:false, error:'session_id_required' })
  const key = keyOf(core_id, session_id)
  if (redis) await redis.del(key)
  return reply.send({ ok:true, cleared:true })
})

app.listen({ port: PORT, host: '0.0.0.0' })
  .catch(err => { console.error(err); process.exit(1) })
