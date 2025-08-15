// index.js — Render + Upstash Redis + Anthropic Messages API
// Persistent memory per session, no overwrite, always full context

import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';

// ===== Config =====
const PORT = process.env.PORT || 10000;
const TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days
const HISTORY_MAX = parseInt(process.env.HISTORY_MAX_MESSAGES || '400', 10);
const CORE = (process.env.CORE_SYSTEM_PROMPT || '').trim();

// ===== Redis =====
const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN
});

function histKey(coreId, sessionId) {
  return `hist:${coreId || 'exec'}:${sessionId}`;
}

async function getHistory(coreId, sessionId) {
  const key = histKey(coreId, sessionId);
  const items = await redis.lrange(key, 0, -1);
  return items.map(i => JSON.parse(i));
}

async function addToHistory(coreId, sessionId, role, content) {
  const key = histKey(coreId, sessionId);
  await redis.rpush(key, JSON.stringify({ role, content }));
  await redis.ltrim(key, -HISTORY_MAX, -1);
  await redis.expire(key, TTL_SECONDS);
}

function uuid() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

// ===== Server =====
const app = Fastify({ logger: true });
await app.register(cors, { origin: '*' });

// Health
app.get('/health', async () => ({ ok: true, redis: true, history_cap: HISTORY_MAX }));

// Complete endpoint
app.post('/v1/complete', async (req, reply) => {
  try {
    const {
      core_id = 'exec',
      session_id,
      model,
      prompt,
      locale = 'uk',
      max_tokens = 500
    } = req.body || {};

    if (!model) return reply.code(400).send({ ok: false, error: 'model_required' });
    if (!prompt) return reply.code(400).send({ ok: false, error: 'prompt_required' });

    const sid = session_id || uuid();

    // Load history
    const history = await getHistory(core_id, sid);

    // System prompt
    const languageDiscipline = [
      `Language Discipline:`,
      `• Respond only in the user's language: ${locale}.`,
      `• Do not translate unless explicitly asked.`,
      `• Do not mix languages in a single reply.`,
      `• No hallucinations — if unknown: "Unknown with current data."`
    ].join('\n');

    const system = [CORE, languageDiscipline].filter(Boolean).join('\n\n');

    // Prepare messages for Claude
    const messages = [...history, { role: 'user', content: prompt }].slice(-HISTORY_MAX);

    // Call Anthropic
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
      return reply.code(r.status).send({ ok: false, error: 'anthropic_error', detail });
    }

    const data = await r.json();
    const text = Array.isArray(data?.content)
      ? data.content.map(c => c?.text ?? '').join('')
      : (data?.content?.text ?? '');
    const assistantReply = (text || '').trim();

    // Save user + assistant messages
    await addToHistory(core_id, sid, 'user', prompt);
    await addToHistory(core_id, sid, 'assistant', assistantReply);

    return reply.send({
      ok: true,
      content: assistantReply,
      meta: {
        core_id,
        session_id: sid,
        history_messages: history.length + 2,
        history_cap: HISTORY_MAX
      }
    });

  } catch (err) {
    req.log.error(err);
    return reply.code(500).send({ ok: false, error: 'server_error' });
  }
});

// Start
app.listen({ port: PORT, host: '0.0.0.0' })
  .catch(err => { app.log.error(err); process.exit(1); });
