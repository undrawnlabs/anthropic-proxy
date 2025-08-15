// index.js — Render + Upstash Redis + Anthropic Messages API with persistent STATE
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

// 2) Minimal default core (used if CORE_SYSTEM_PROMPT is not set)
const DEFAULT_CORE = `
You are undrawn labs Executive Core.
- Never invent facts. If unknown: "Unknown with current data."
- Respond in the user's language only.
- At the end of each reply, append a single line STATE: {"product": <string|null>, "phase": <string|null>, "notes": <string|null>}
  - Only include fields you are confident about.
  - If not sure, use null. Do not guess.
  - Keep STATE on one line, valid JSON, no trailing text after it.
`.trim();

// 3) Light parser to capture explicit state from user prompt (UA/EN/RU)
function parseExplicitStateFromPrompt(prompt) {
  if (!prompt) return null;
  const norm = prompt.replace(/\s+/g, ' ').trim();

  // UA examples: "Запам'ятай: продукт — X; фаза — Y"
  // EN examples: "Remember: product — X; phase — Y"
  // Variants: "product:", "продукт:", "-", "—"
  const prodMatch = norm.match(/(?:продукт|product)\s*[:\-—]\s*([^;,.]+)\s*/i);
  const phaseMatch = norm.match(/(?:фаза|phase)\s*[:\-—]\s*([^;,.]+)\s*/i);

  if (!prodMatch && !phaseMatch) return null;

  const product = prodMatch ? prodMatch[1].trim() : null;
  const phase = phaseMatch ? phaseMatch[1].trim() : null;

  return { product, phase };
}

// 4) Fastify server
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

    // 4.1 Load previous STATE (JSON string or null)
    const lastStateJSON = await getState(core_id, sid);
    let lastState = null;
    try { lastState = lastStateJSON ? JSON.parse(lastStateJSON) : null; } catch {}

    // 4.2 If user explicitly set product/phase, merge and save immediately
    const explicit = parseExplicitStateFromPrompt(prompt);
    if (explicit) {
      const merged = {
        product: explicit.product ?? lastState?.product ?? null,
        phase: explicit.phase ?? lastState?.phase ?? null,
        notes: lastState?.notes ?? null
      };
      await setState(core_id, sid, JSON.stringify(merged));
    }

    // 4.3 Build system prompt (env or default) and include LAST_STATE
    const core = process.env.CORE_SYSTEM_PROMPT || DEFAULT_CORE;
    const effectiveLastJSON = await getState(core_id, sid); // may have just been updated
    const system = [
      core,
      effectiveLastJSON ? `LAST_STATE: ${effectiveLastJSON}` : '',
      `Language Discipline:
• Respond only in the user's language: ${locale}.
• Do not translate unless asked.
• Do not mix languages in one reply.`
    ].filter(Boolean).join('\n\n');

    // 4.4 Call Anthropic Messages API
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

    // 4.5 Extract text
    const text = Array.isArray(data?.content)
      ? data.content.map(c => c?.text ?? '').join('')
      : (data?.content?.text ?? '');

    // 4.6 Save new STATE if model appended it
    const newState = extractState(text);
    if (newState) await setState(core_id, sid, newState);

    // 4.7 Return visible content without STATE
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

// Optional: quick debug endpoint to read raw STATE
fastify.get('/v1/state', async (request, reply) => {
  const { session_id, core_id = 'exec' } = request.query || {};
  if (!session_id) return reply.code(400).send({ ok: false, error: 'session_id required' });
  const s = await getState(core_id, session_id);
  return reply.send({ ok: true, state: s ? JSON.parse(s) : null });
});

// 5) Start server
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
