// index.js — Render + Upstash Redis + Anthropic Messages API
// Persistent memory per session (Redis LIST), full-context to Claude, correct Anthropic format

import Fastify from 'fastify'
import cors from '@fastify/cors'
import fetch from 'node-fetch'
import { Redis } from '@upstash/redis'

// ===== Config =====
const PORT = process.env.PORT || 10000
const TTL_SECONDS = 60 * 60 * 24 * 30 // 30 days
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '400', 10)
const CORE = (process.env.CORE_SYSTEM_PROMPT || '').trim()
const DEBUG = !!process.env.DEBUG

// ===== Redis =====
const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN
})

const histKey = (coreId, sessionId) => `hist:${coreId || 'exec'}:${sessionId}`

// Read entire history as [{role, content}, ...]
async function getHistory(coreId, sessionId) {
  const key = histKey(coreId, sessionId)
  const arr = await redis.lrange(key, 0, -1) // strings
  const items = arr.map(s => { try { return JSON.parse(s) } catch { return null } }).filter(Boolean)
  if (DEBUG) console.log('[READ]', key, 'len=', items.length)
  return items
}

// Append one message and trim to HISTORY_MAX
async function addToHistory(coreId, sessionId, role, content) {
  const key = histKey(coreId, sessionId)
  await redis.rpush(key, JSON.stringify({ role, content }))
  // trim to last HISTORY_MAX items using positive indexes (Upstash-safe)
  const len = await redis.llen(key)
  if (len > HISTORY_MAX) {
    await redis.ltrim(key, len - HISTORY_MAX, -1)
  }
  await redis.expire(key, TTL_SECONDS)
  if (DEBUG) console.log('[APPEND]', key, role)
}

function uuid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random()*16|0, v = c === 'x' ? r : (r&0x3|0x8)
    return v.toString(16)
  })
}

// Convert our history to Anthropic Messages API format
function toAnthropicMessages(history, userPrompt) {
  const mapItem = (m) => ({
    role: m.role, // 'user' | 'assistant'
    content: Array.isArray(m.content)
      ? m.content // already blocks (rare)
      : [{ type: 'text', text: String(m.content ?? '') }]
  })
  const blocks = (history || []).map(mapItem)
  blocks.push({
    role: 'user',
    content: [{ type: 'text', text: String(userPrompt) }]
  })
  return blocks
}

// Extract plain text from Anthropic response
function extractText(data) {
  if (!data) return ''
  if (Array.isArray(data.content)) {
    return data.content
      .map(b => (typeof b?.text === 'string' ? b.text : ''))
      .join('')
      .trim()
  }
  if (data.content && typeof data.content.text === 'string') {
    return data.content.text.trim()
  }
  return ''
}

// ===== Server =====
const app = Fastify({ logger: true })
await app.register(cors, { origin: '*' })

app.get('/health', async () => ({ ok:true, redis:true, history_cap:HISTORY_MAX }))

app.post('/v1/complete', async (req, reply) => {
  try {
    const {
      core_id = 'exec',
      session_id,
      model,
      prompt,
      locale = 'uk', // kept for your CORE rules; not injected into content
      max_tokens = 500
    } = req.body || {}

    if (!model)  return reply.code(400).send({ ok:false, error:'model_required' })
    if (!prompt) return reply.code(400).send({ ok:false, error:'prompt_required' })

    const sid = session_id || uuid()

    // 1) load history from Redis
    const history = await getHistory(core_id, sid)

    // 2) build system (your CORE + optional language discipline text kept compact)
    const lang = [
      'Language Discipline:',
      `• Respond only in the user\'s language: ${locale}.`,
      '• Do not translate unless asked.',
      '• If unknown: "Unknown with current data."'
    ].join('\n')
    const system = [CORE, lang].filter(Boolean).join('\n\n')

    // 3) build Anthropic messages (full history + current user) in correct format
    const messages = toAnthropicMessages(history, prompt).slice(-HISTORY_MAX)

    if (DEBUG) app.log.info({ sendLen: messages.length }, 'anthropic_payload')

    // 4) call Anthropic Messages API
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
    const assistantReply = extractText(data)

    // 5) persist user + assistant into Redis (append; no overwrite)
    await addToHistory(core_id, sid, 'user', prompt)
    await addToHistory(core_id, sid, 'assistant', assistantReply)

    // 6) respond
    const newLen = await redis.llen(histKey(core_id, sid))
    return reply.send({
      ok: true,
      content: assistantReply,
      meta: {
        core_id,
        session_id: sid,
        history_messages: newLen,
        history_cap: HISTORY_MAX
      }
    })
  } catch (err) {
    req.log.error(err)
    return reply.code(500).send({ ok:false, error:'server_error' })
  }
})

app.listen({ port: PORT, host: '0.0.0.0' })
  .catch(err => { app.log.error(err); process.exit(1) })
