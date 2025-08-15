// index.js — Render + Upstash Redis + Anthropic Messages API
// Multi-user memory, multi-core isolation, JWT or simple-key auth

import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';
import jwt from 'jsonwebtoken';

// ===== Config =====
const PORT = process.env.PORT || 10000;
const TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '40', 10);
const CORE = (process.env.CORE_SYSTEM_PROMPT || '').trim();

const JWT_SECRET = process.env.JWT_SECRET || '';           // optional (prod)
const BACKEND_API_KEY = process.env.BACKEND_API_KEY || ''; // optional (dev)

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

// ---- Auth hook: prefer JWT; else simple key; else open (danger, for local only)
app.addHook('preHandler', async (req, reply) => {
  // Health is public
  if (req.routerPath === '/health') return;

  const auth = req.headers.authorization || '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : '';

  if (JWT_SECRET) {
    if (!token) return reply.code(401).send({ ok:false, error:'unauthorized' });
    try {
      const payload = jwt.verify(token, JWT_SECRET); // { sub, scope?, exp? }
      req.user = { id: String(payload.sub || ''), scope: Array.isArray(payload.scope) ? payload.scope : [] };
      if (!req.user.id) return reply.code(401).send({ ok:false, error:'invalid_token_no_sub' });
      return;
    } catch {
      return reply.code(401).send({ ok:false, error:'invalid_token' });
    }
  }

  if (BACKEND_API_KEY) {
    if (token !== BACKEND_API_KEY) return reply.code(401).send({ ok:false, error:'unauthorized' });
    // Dev mode user id (single-user). For multi-user with key auth, pass X-User-Id.
    req.user = { id: req.headers['x-user-id'] ? String(req.headers['x-user-id']) : 'dev' , scope: [] };
    return;
  }

  // No auth set — open mode (not recommended). Assign anonymous user.
  req.user = { id: 'anon', scope: [] };
});

// ---- Health
app.get('/health', async () => ({ ok:true, service:'undrawn-core', redis: !!redis }));

/**
 * POST /v1/complete
 * Auth:
 *   - JWT:  Authorization: Bearer <JWT(sub=user_id, scope=[...])>
 *   - or Simple key: Authorization: Bearer <BACKEND_API_KEY> (+ optional X-User-Id)
 *
 * Body:
 * {
 *   "core_id": "exec",                         // optional; isolates memory per tool
 *   "session_id": "s1",                        // optional; generated if missing
 *   "model": "claude-3-7-sonnet-20250219",
 *   "prompt": "text",
 *   "locale": "uk",                            // optional; language discipline hint
 *   "max_tokens": 500                          // optional
 * }
 */
app.post('/v1/complete', async (req, reply) => {
  try {
    const userId = req.user?.id || 'anon';
    const allowed = req.user?.scope || [];

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

    if (allowed.length && !allowed.includes(core_id)) {
      return reply.code(403).send({ ok:false, error:'forbidden_core' });
    }

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

    // 3) compose messages
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

// Debug helpers (auth applies)
app.get('/v1/history/len', async (req, reply) => {
  const userId = req.user?.id || 'anon';
  const { core_id = 'exec', session_id } = req.query || {};
  if (!session_id) return reply.code(400).send({ ok:false, error:'session_id_required' });
  const h = await getHistory(userId, core_id, session_id);
  return reply.send({ ok:true, messages: Array.isArray(h) ? h.length : 0, cap: HISTORY_MAX });
});

app.delete('/v1/history', async (req, reply) => {
  const userId = req.user?.id || 'anon';
  const { core_id = 'exec', session_id } = req.query || {};
  if (!session_id) return reply.code(400).send({ ok:false, error:'session_id_required' });
  const key = histKey(userId, core_id, session_id);
  if (redis) await redis.del(key); else RAM.delete(key);
  return reply.send({ ok:true, cleared:true });
});

// Start
app.listen({ port: PORT, host: '0.0.0.0' })
  .catch(err => { app.log.error(err); process.exit(1); });
