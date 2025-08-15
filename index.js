// index.js — Render + Upstash Redis + Anthropic Messages API
// Native persistent context (full history) + optional system prompt + locale discipline

import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';

// ===== 0) Config =====
const PORT = process.env.PORT || 10000;
const TTL_SECONDS = 60 * 60 * 24 * 30;           // 30 days
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '30', 10);
const CORE = (process.env.CORE_SYSTEM_PROMPT || '').trim();

// ===== 1) Redis (with RAM fallback) =====
const redis = (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN)
  ? new Redis({ url: process.env.UPSTASH_REDIS_REST_URL, token: process.env.UPSTASH_REDIS_REST_TOKEN })
  : null;

const RAM = new Map(); // fallback for history only

function histKey(coreId, sessionId) {
  return `hist:${coreId || 'exec'}:${sessionId}`;
}

async function getHistory(coreId, sessionId) {
  const key = histKey(coreId, sessionId);
  if (redis) {
    const raw = await redis.get(key);
    if (!raw) return [];
    try { return JSON.parse(raw); } catch { return []; }
  }
  return RAM.get(key) || [];
}

async function setHistory(coreId, sessionId, history) {
  const key = histKey(coreId, sessionId);
  const trimmed = Array.isArray(history) ? history.slice(-HISTORY_MAX) : [];
  const payload = JSON.stringify(trimmed);
  if (redis) {
    await redis.set(key, payload, { ex: TTL_SECONDS });
  } else {
    RAM.set(key, trimmed);
  }
}

function uuid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random()*16|0, v = c === 'x' ? r : (r&0x3|0x8);
    return v.toString(16);
  });
}

// ===== 2) Fastify =====
const app = Fastify({ logger: true });
await app.register(cors, { origin: '*' });

// Health
app.get('/health', async () => ({ ok:true, service:'undrawn-core', redis: !!redis }));

/**
 * POST /v1/complete
 * Body:
 * {
 *   "model": "claude-3-7-sonnet-20250219",
 *   "prompt": "user text",
 *   "session_id": "optional",          // if missing, server generates and returns it
 *   "core_id": "exec | other",         // optional, for isolating multiple cores
 *   "locale": "uk | en | ru",          // default "uk"
 *   "max_tokens": 500                  // optional
 * }
 */
app.post('/v1/complete', async (req, reply) => {
  try {
    const {
      model,
      prompt,
      session_id,
      core_id = 'exec',
      locale = 'uk',
      max_tokens = 500
    } = req.body || {};

    if (!model || !prompt) {
      return reply.code(400).send({ ok:false, error:'model and prompt are required' });
    }

    // 1) load history
    const sid = session_id || uuid();
    const history = await getHistory(core_id, sid); // [{role,content}, ...]

    // 2) build system
    const languageDiscipline = [
      `Language Discipline:`,
      `• Respond only in the user's language: ${locale}.`,
      `• Do not translate unless asked.`,
      `• Do not mix languages in one reply.`
    ].join('\n');

    const system = [CORE, languageDiscipline].filter(Boolean).join('\n\n');

    // 3) compose messages
    const messages = [
      ...history,
      { role: 'user', content: prompt }
    ].slice(-HISTORY_MAX);

    // 4) call Anthropic Messages API
    const r = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({ model, max_tokens, system, messages })
    });

    if (!r.ok) {
      const detail = await r.text().catch(()=> '');
      return reply.code(r.status).send({ ok:false, error:'anthropic_error', detail });
    }

    const data = await r.json();
    const text = Array.isArray(data?.content)
      ? data.content.map(c => c?.text ?? '').join('')
      : (data?.content?.text ?? '');

    const assistantReply = (text || '').trim();

    // 5) update history and persist
    const newHistory = [
      ...history,
      { role: 'user', content: prompt },
      { role: 'assistant', content: assistantReply }
    ];
    await setHistory(core_id, sid, newHistory);

    return reply.send({
      ok: true,
      content: assistantReply,
      meta: { session_id: sid, core_id, history_messages: newHistory.length, history_cap: HISTORY_MAX }
    });

  } catch (err) {
    req.log.error(err);
    return reply.code(500).send({ ok:false, error:'server_error' });
  }
});

// ===== 3) Start =====
app.listen({ port: PORT, host: '0.0.0.0' })
  .catch(err => { app.log.error(err); process.exit(1); });
