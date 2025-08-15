// index.js — Render + Upstash Redis + Anthropic Messages API
// Native context memory: full message history + optional structured STATE

import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';

// 0) Redis + RAM fallback
const redis = (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN)
  ? new Redis({
      url: process.env.UPSTASH_REDIS_REST_URL,
      token: process.env.UPSTASH_REDIS_REST_TOKEN
    })
  : null;

const RAM_STATE = new Map();
const RAM_HISTORY = new Map();
const TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '30', 10);

// 1) STATE helpers (optional structured memory)
function extractState(text) {
  const m = text?.match?.(/STATE:\s*({[\s\S]+?})\s*$/);
  return m ? m[1] : null; // JSON string or null
}
function stripState(text) {
  return text?.replace?.(/STATE:\s*({[\s\S]+?})\s*$/, '').trim() ?? '';
}
function stateKey(coreId, sessionId) {
  return `state:${coreId || 'exec'}:${sessionId}`;
}
async function getState(coreId, sessionId) {
  const key = stateKey(coreId, sessionId);
  if (redis) {
    const v = await redis.get(key);
    return v || null;
  }
  return RAM_STATE.get(key) || null;
}
async function setState(coreId, sessionId, stateJSON) {
  const key = stateKey(coreId, sessionId);
  if (redis) {
    await redis.set(key, stateJSON, { ex: TTL_SECONDS });
  } else {
    RAM_STATE.set(key, stateJSON);
  }
}

// 2) History helpers (native context)
function historyKey(coreId, sessionId) {
  return `hist:${coreId || 'exec'}:${sessionId}`;
}
async function getHistory(coreId, sessionId) {
  const key = historyKey(coreId, sessionId);
  if (redis) {
    const v = await redis.get(key);
    if (!v) return [];
    try { return JSON.parse(v); } catch { return []; }
  }
  const v = RAM_HISTORY.get(key);
  return Array.isArray(v) ? v : [];
}
async function setHistory(coreId, sessionId, history) {
  const key = historyKey(coreId, sessionId);
  const trimmed = Array.isArray(history) ? history.slice(-HISTORY_MAX) : [];
  const payload = JSON.stringify(trimmed);
  if (redis) {
    await redis.set(key, payload, { ex: TTL_SECONDS });
  } else {
    RAM_HISTORY.set(key, trimmed);
  }
}
function cryptoRandom() {
  // UUID v4 without deps
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// 3) Default core (if CORE_SYSTEM_PROMPT not provided)
const DEFAULT_CORE = `
You are undrawn labs Executive Core.
- No hallucinations. If unknown: "Unknown with current data."
- Analyst/operator tone. No generic marketing phrases.
- Respond only in the user's language.
- If appropriate, append one-line STATE: {"product": "...", "phase": "..."} at the very end.
`.trim();

// 4) Server
const fastify = Fastify({ logger: true });
await fastify.register(cors, { origin: '*' });

/**
 * POST /v1/complete
 * Body:
 * {
 *   "model": "claude-3-7-sonnet-20250219",
 *   "prompt": "user text",
 *   "max_tokens": 400,
 *   "session_id": "optional",
 *   "core_id": "exec | other",
 *   "locale": "uk | en | ru"
 * }
 */
fastify.post('/v1/complete', async (request, reply) => {
  try {
    const { model, prompt, max_tokens = 400, session_id, core_id = 'exec', locale = 'en' } = request.body || {};
    if (!model || !prompt) {
      return reply.code(400).send({ ok: false, error: 'model and prompt are required' });
    }

    const sid = session_id || cryptoRandom();

    // 4.1 Load memory
    const lastStateJSON = await getState(core_id, sid);
    const history = await getHistory(core_id, sid); // array of {role, content}

    // 4.2 Build system prompt
    const core = process.env.CORE_SYSTEM_PROMPT || DEFAULT_CORE;
    const system = [
      core,
      lastStateJSON ? `LAST_STATE: ${lastStateJSON}` : '',
      `Language Discipline:
• Respond only in the user's language: ${locale}.
• Do not translate unless asked.
• Do not mix languages in one reply.`
    ].filter(Boolean).join('\n\n');

    // 4.3 Compose messages with native context
    const messages = [
      ...history, // previous {role:'user'|'assistant', content:string}
      { role: 'user', content: prompt }
    ].slice(-HISTORY_MAX);

    // 4.4 Call Anthropic
    const resp = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model,
        max_tokens,
        system,
        messages
      })
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      return reply.code(resp.status).send({ ok: false, error: 'anthropic_error', detail: errText });
    }

    const data = await resp.json();

    // 4.5 Extract assistant text
    const fullText = Array.isArray(data?.content)
      ? data.content.map(c => c?.text ?? '').join('')
      : (data?.content?.text ?? '');

    const stateJSON = extractState(fullText);
    if (stateJSON) await setState(core_id, sid, stateJSON);

    const content = stripState(fullText);

    // 4.6 Update history (native memory)
    const newHistory = [
      ...history,
      { role: 'user', content: prompt },
      { role: 'assistant', content } // store clean text (no STATE)
    ];
    await setHistory(core_id, sid, newHistory);

    return reply.send({
      ok: true,
      content,
      meta: { session_id: sid, core_id }
    });

  } catch (e) {
    request.log.error(e);
    return reply.code(500).send({ ok: false, error: 'server_error' });
  }
});

// Optional: read raw STATE
fastify.get('/v1/state', async (request, reply) => {
  const { session_id, core_id = 'exec' } = request.query || {};
  if (!session_id) return reply.code(400).send({ ok: false, error: 'session_id required' });
  const s = await getState(core_id, session_id);
  return reply.send({ ok: true, state: s ? JSON.parse(s) : null });
});

// Optional: read history length
fastify.get('/v1/history/len', async (request, reply) => {
  const { session_id, core_id = 'exec' } = request.query || {};
  if (!session_id) return reply.code(400).send({ ok: false, error: 'session_id required' });
  const h = await getHistory(core_id, session_id);
  return reply.send({ ok: true, messages: Array.isArray(h) ? h.length : 0, max: HISTORY_MAX });
});

// 5) Start server
const PORT = process.env.PORT || 10000;
fastify.listen({ port: PORT, host: '0.0.0.0' }).catch((err) => {
  fastify.log.error(err);
  process.exit(1);
});
