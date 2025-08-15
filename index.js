// index.js — Render + Upstash Redis + Anthropic Messages API
// Memory per user/core/session. NO AUTH (open), for testing only.

import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';

// ===== Config =====
const PORT = process.env.PORT || 10000;
const TTL_SECONDS = 60 * 60 * 24 * 30;                    // 30 days
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '40', 10);
const CORE = (process.env.CORE_SYSTEM_PROMPT || '').trim(); // optional system/core prompt

// ===== Redis (with RAM fallback) =====
const redis = (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN)
  ? new Redis({ url: process.env.UPSTASH_REDIS_REST_URL, token: process.env.UPSTASH_REDIS_REST_TOKEN })
  : null;

const RAM = new Map(); // fallback only if Redis absent
const histKey = (userId, coreId, sessionId) => `hist:${userId}:${coreId || 'exec'}:${sessionId}`;

async function getHistory(userId, coreId, sessionId) {
  const key = histKey(userId, coreId, sessionId);
  if (redis) {
    const raw = await redis.get(key);
    if (!raw) return [];
    try { return JSON.parse(raw); } catch { return []; }
  }
  return RAM.get(key) || [];
}

async function setHistory(userId, coreId, sessionId, history) {
  const key = histKey(userId, coreId, sessionId);
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

// ===== Server =====
const app = Fastify({ logger: true });
await app.register(cors, { origin: '*' });

// ---- Health (public)
app.get('/health', async () => ({ ok:true, service:'undrawn-core', redis: !!redis }));

/**
 * POST /v1/complete
 * Headers (optional):
 *   X-User-Id: <string>     // для розділення пам'яті між користувачами; якщо нема — "anon"
 *
 * Body:
 * {
 *   "core_id": "exec",                         // опц.; ізолює пам'ять по інструментам
 *   "session_id": "s1",                        // опц.; якщо нема — згенеруємо
 *   "model": "claude-3-7-sonnet-20250219",
 *   "prompt": "text",
 *   "locale": "uk",                            // опц.; дисципліна мови
 *   "max_tokens": 500                          // опц.
 * }
 */
app.post('/v1/complete', async (req, reply) => {
  try {
    const userId = req.headers['x-user-id'] ? String(req.headers['x-user-id']) : 'anon';

    const {
      core_id = 'exec',
      session_id,
      model,
      prompt,
      locale = 'uk',
      max_tokens = 500
    } = req.body || {};

    if (!model)  return reply.code(400).send({ ok:false, error:'model_required' });
    if (!prompt) return reply.code(400).send({ ok:false, error:'prompt_required' });

    const sid = session_id || uuid();

    // 1) load history
    const history = await getHistory(userId, core_id, sid);

    // 2) system prompt (core + language discipline)
    const languageDiscipline = [
      `Language Discipline:`,
      `• Respond only in the user's language: ${locale}.`,
      `• Do not translate unless explicitly asked.`,
      `• Do not mix languages in a single reply.`,
      `• No hallucinations — if unknown: "Unknown with current data."`
    ].join('\n');

    const system = [CORE, languageDiscipline].filter(Boolean).join('\n\n');

    // 3) compose messages (trim to cap)
    const messages = [...history, { role:'user', content: prompt }].slice(-HISTORY_MAX);

    // 4) call Anthropic
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
      const detail = await r.text().catch(() => '');
      return reply.code(r.status).send({ ok:false, error:'anthropic_error', detail });
    }

    const data = await r.json();
    const text = Array.isArray(data?.content)
      ? data.content.map(c => c?.text ?? '').join('')
      : (data?.content?.text ?? '');
    const assistantReply = (text || '').trim();

    // 5) persist history
    const newHistory = [
      ...history,
      { role:'user',      content: prompt },
      { role:'assistant', content: assistantReply }
    ];
    await setHistory(userId, core_id, sid, newHistory);

    return reply.send({
      ok: true,
      content: assistantReply,
      meta: {
        user_id: userId,
        core_id,
        session_id: sid,
        history_messages: newHistory.length,
        history_cap: HISTORY_MAX
      }
    });

  } catch (err) {
    req.log.error(err);
    return reply.code(500).send({ ok:false, error:'server_error' });
  }
});

// Start
app.listen({ port: PORT, host: '0.0.0.0' })
  .catch(err => { app.log.error(err); process.exit(1); });
