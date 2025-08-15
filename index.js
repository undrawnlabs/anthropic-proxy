// index.js — Render + Upstash Redis + Anthropic Messages API
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
const TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days

// 1) STATE helpers
function extractState(text) {
  const m = text?.match?.(/STATE:\s*({[\s\S]+?})\s*$/);
  return m ? m[1] : null; // returns JSON string or null
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

// 2) Fastify server
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

    // Load previous STATE
    const sid = session_id || cryptoRandom();
    const lastState = await getState(core_id, sid);

    // Build system prompt
    const core = process.env.CORE_SYSTEM_PROMPT || '';
    const system = [
      core,
      lastState ? `LAST_STATE: ${lastState}` : '',
      `Language Discipline:
• Respond only in the user's language: ${locale}.
• Do not translate unless asked.
• Do not mix languages in one reply.`
    ].filter(Boolean).join('\n\n');

    // Call Anthropic Messages API
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
        messages: [{ role: 'user', content: prompt }]
      })
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      return reply.code(resp.status).send({ ok: false, error: 'anthropic_error', detail: errText });
    }

    const data = await resp.json();

    // Extract text
    const text = Array.isArray(data?.content)
      ? data.content.map(c => c?.text ?? '').join('')
      : (data?.content?.text ?? '');

    // Save new STATE and return clean content
    const newState = extractState(text);
    if (newState) await setState(core_id, sid, newState);
    const content = stripState(text);

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

// 3) Start server
const PORT = process.env.PORT || 10000;
fastify.listen({ port: PORT, host: '0.0.0.0' }).catch((err) => {
  fastify.log.error(err);
  process.exit(1);
});

// Utils
function cryptoRandom() {
  // UUID v4 generator without extra deps
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}
