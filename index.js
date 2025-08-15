// index.js
// Fastify + Upstash Redis + Anthropic Messages API
// Cost-efficient memory: Short-term buffer (STB) + Summary + Targeted semantic recall
// Node 18.x
// ENV (required): ANTHROPIC_API_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, CORE_SYSTEM_PROMPT (or SYSTEM_PROMPT_BASE)
// ENV (optional): OPENAI_API_KEY, PORT, STB_MAX_ITEMS, RECALL_TOP_K, LTM_SCAN_LIMIT, TOKEN_BUDGET, MAX_OUTPUT_TOKENS, ANTHROPIC_MODEL

import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';

// ---------------------- ENV ----------------------
const {
  ANTHROPIC_API_KEY,
  UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN,
  CORE_SYSTEM_PROMPT,
  SYSTEM_PROMPT_BASE: SP_FALLBACK,
  OPENAI_API_KEY,
  PORT = 3000,
  ANTHROPIC_MODEL = 'claude-3-5-sonnet-20240620',
} = process.env;

const SYSTEM_PROMPT_BASE = CORE_SYSTEM_PROMPT || SP_FALLBACK;

if (!ANTHROPIC_API_KEY || !UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN || !SYSTEM_PROMPT_BASE) {
  throw new Error('Missing ENV: ANTHROPIC_API_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, CORE_SYSTEM_PROMPT (or SYSTEM_PROMPT_BASE)');
}

// ---------------------- App & Clients ----------------------
const app = Fastify({ logger: true });
await app.register(cors, { origin: true });

const redis = new Redis({
  url: UPSTASH_REDIS_REST_URL,
  token: UPSTASH_REDIS_REST_TOKEN,
});

// ---------------------- Config ----------------------
const HISTORY_TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days
const STB_MAX_ITEMS = Number(process.env.STB_MAX_ITEMS || 20); // ~10 turns (user+assistant counted separately)
const RECALL_TOP_K = Number(process.env.RECALL_TOP_K || 6);
const LTM_SCAN_LIMIT = Number(process.env.LTM_SCAN_LIMIT || 1500);
const TOKEN_BUDGET = Number(process.env.TOKEN_BUDGET || 170_000); // headroom under 200k
const MAX_OUTPUT_TOKENS = Number(process.env.MAX_OUTPUT_TOKENS || 1024);

// ---------------------- Redis Keys ----------------------
const keySTB = (coreId, sessionId) => `stb:exec:${coreId}:${sessionId}`;  // LIST
const keyLTM = (coreId, sessionId) => `ltm:exec:${coreId}:${sessionId}`;  // LIST
const keySUM = (coreId, sessionId) => `sum:exec:${coreId}:${sessionId}`;  // STRING

// ---------------------- Utils ----------------------
const toAnthropicMessage = (item) => ({
  role: item.role === 'assistant' ? 'assistant' : 'user',
  content: [{ type: 'text', text: String(item.content ?? '') }],
});

async function ensureListKey(client, key) {
  const t = await client.type(key);
  if (t && t !== 'none' && t !== 'list') await client.del(key);
}

async function appendList(client, key, ...entries) {
  if (!entries.length) return;
  await ensureListKey(client, key);
  await client.rpush(key, entries.map((e) => JSON.stringify(e)));
  await client.expire(key, HISTORY_TTL_SECONDS);
}

async function lrangeJSON(client, key, start, stop) {
  await ensureListKey(client, key);
  const raw = await client.lrange(key, start, stop);
  return raw.map((s) => { try { return JSON.parse(s); } catch { return null; } }).filter(Boolean);
}

async function ltrimKeepLast(client, key, maxItems) {
  await ensureListKey(client, key);
  await client.ltrim(key, -maxItems, -1);
  await client.expire(key, HISTORY_TTL_SECONDS);
}

async function readSummary(client, key) {
  const v = await client.get(key);
  return typeof v === 'string' ? v : '';
}

async function writeSummary(client, key, text) {
  await client.set(key, text || '');
  await client.expire(key, HISTORY_TTL_SECONDS);
}

// ---------------------- Embeddings (optional) ----------------------
async function embedText(text) {
  if (!OPENAI_API_KEY) return null;
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      authorization: `Bearer ${OPENAI_API_KEY}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({ model: 'text-embedding-3-small', input: text }),
  });
  if (!res.ok) {
    const err = await res.text().catch(() => '');
    throw new Error(`OpenAI embeddings error: ${res.status} ${err}`);
  }
  const data = await res.json();
  return data.data?.[0]?.embedding || null;
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function keywordScore(q, doc) {
  const qTokens = new Set(String(q).toLowerCase().split(/\W+/).filter(Boolean));
  const dTokens = new Set(String(doc).toLowerCase().split(/\W+/).filter(Boolean));
  let inter = 0; for (const t of qTokens) if (dTokens.has(t)) inter++;
  return inter / Math.sqrt((qTokens.size || 1) * (dTokens.size || 1));
}

// ---------------------- Token Counting ----------------------
async function countTokens({ system, messages, model }) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: {
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({ model, system, messages }),
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`count_tokens failed: ${res.status} ${txt}`);
  }
  const data = await res.json();
  return data.input_tokens ?? 0;
}

// ---------------------- Summary Builder (heuristic) ----------------------
function updateSummaryHeuristic(prevSummary, entries) {
  const facts = [];
  const add = (line) => {
    const t = String(line || '').trim();
    if (t && t.length <= 220) facts.push(`- ${t}`);
  };
  for (const e of entries) {
    if (e.role === 'user') {
      const t = String(e.content || '');
      const candidates = t.split(/\n+/).slice(0, 3);
      for (const c of candidates) {
        if (/(^i\s(am|work|live|prefer|use)\b)|(\bmy\s[a-z]+)/i.test(c) || /\b(is|=|:)\b/.test(c)) add(c);
      }
    }
  }
  const head = `Memory state (concise facts from earlier context):`;
  const merged = [
    prevSummary ? prevSummary.trim() : `${head}\n- (none yet)`,
    ...(facts.length ? [`\n# New observations:\n${facts.join('\n')}`] : []),
  ].join('\n');
  return merged.slice(0, 1600);
}

// ---------------------- Recall from LTM ----------------------
async function recallFromLTM(coreId, sessionId, userMsg) {
  const all = await lrangeJSON(redis, keyLTM(coreId, sessionId), -LTM_SCAN_LIMIT, -1);
  if (!all.length) return [];

  let qEmb = null;
  if (OPENAI_API_KEY) { try { qEmb = await embedText(userMsg); } catch (_) {} }

  const scored = all.map((e) => {
    const content = String(e.content || '');
    let score = 0;
    if (qEmb && Array.isArray(e.emb)) score = cosineSim(qEmb, e.emb);
    else score = keywordScore(userMsg, content);
    return { score, entry: e };
  });

  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, RECALL_TOP_K).map(s => s.entry);

  const seen = new Set();
  const dedup = [];
  for (const e of top) {
    const k = `${e.role}:${e.content.slice(0, 120)}`;
    if (!seen.has(k)) { seen.add(k); dedup.push(e); }
  }
  return dedup;
}

// ---------------------- Build Prompt ----------------------
async function buildPrompt({ coreId, sessionId, locale, userEntry }) {
  const stb = await lrangeJSON(redis, keySTB(coreId, sessionId), 0, -1);
  const summary = await readSummary(redis, keySUM(coreId, sessionId));
  const recalled = await recallFromLTM(coreId, sessionId, userEntry.content);

  const summaryBlock = summary ? `\n\n${summary}` : '';
  const recalledBlock = recalled.length
    ? `\n\nRelevant context (retrieved):\n${recalled.map((e) => `- ${e.role}: ${e.content}`).join('\n')}`
    : '';

  let system = `${SYSTEM_PROMPT_BASE}\nLocale: ${locale}${summaryBlock}${recalledBlock}`;
  let messages = [...stb.map(toAnthropicMessage), toAnthropicMessage(userEntry)];

  let tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
  if (tokens + MAX_OUTPUT_TOKENS <= TOKEN_BUDGET) return { system, messages };

  // Trim STB from the oldest side
  const work = [...stb];
  while (work.length > 0) {
    work.shift();
    messages = [...work.map(toAnthropicMessage), toAnthropicMessage(userEntry)];
    tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
    if (tokens + MAX_OUTPUT_TOKENS <= TOKEN_BUDGET) break;
  }

  // Drop recalled, then summary if still too large
  if (tokens + MAX_OUTPUT_TOKENS > TOKEN_BUDGET && recalledBlock) {
    system = `${SYSTEM_PROMPT_BASE}\nLocale: ${locale}${summaryBlock}`;
    tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
  }
  if (tokens + MAX_OUTPUT_TOKENS > TOKEN_BUDGET && summaryBlock) {
    system = `${SYSTEM_PROMPT_BASE}\nLocale: ${locale}`;
  }
  return { system, messages };
}

// ---------------------- STB → LTM rollover ----------------------
async function rolloverSTBIfNeeded(coreId, sessionId) {
  const k = keySTB(coreId, sessionId);
  await ensureListKey(redis, k);
  const len = await redis.llen(k);
  if (len <= STB_MAX_ITEMS) return;

  const excess = len - STB_MAX_ITEMS;
  const toMove = await lrangeJSON(redis, k, 0, excess - 1);

  await ltrimKeepLast(redis, k, STB_MAX_ITEMS);

  const enriched = [];
  for (const e of toMove) {
    const item = { ...e };
    if (OPENAI_API_KEY) {
      try {
        const emb = await embedText(String(e.content || ''));
        if (emb) item.emb = emb;
      } catch (_) {}
    }
    enriched.push(item);
  }
  await appendList(redis, keyLTM(coreId, sessionId), ...enriched);
}

// ---------------------- Summary maintenance ----------------------
async function maybeRefreshSummary(coreId, sessionId, recentEntries) {
  const prev = await readSummary(redis, keySUM(coreId, sessionId));
  const next = updateSummaryHeuristic(prev, recentEntries);
  if (next !== prev) await writeSummary(redis, keySUM(coreId, sessionId), next);
}

// ---------------------- Routes ----------------------
app.post('/chat', {
  schema: {
    body: {
      type: 'object',
      required: ['core_id', 'session_id', 'user_message'],
      properties: {
        core_id: { type: 'string', minLength: 1 },
        session_id: { type: 'string', minLength: 1 },
        user_message: { type: 'string', minLength: 1 },
        locale: { type: 'string' },
      },
      additionalProperties: false,
    },
  },
}, async (req, reply) => {
  const { core_id, session_id, user_message, locale = 'uk' } = req.body;

  const userEntry = { role: 'user', content: user_message, ts: Date.now(), locale };
  const { system, messages } = await buildPrompt({ coreId: core_id, sessionId: session_id, locale, userEntry });

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: ANTHROPIC_MODEL,
      system,
      messages,
      max_tokens: MAX_OUTPUT_TOKENS,
      temperature: 0.2,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    req.log.error({ status: res.status, body: text }, 'Anthropic API error');
    return reply.code(502).send({ error: 'upstream_error', detail: text });
  }

  const data = await res.json();
  const assistantText =
    Array.isArray(data.content) && data.content[0]?.type === 'text'
      ? data.content[0].text
      : '';

  const assistantEntry = { role: 'assistant', content: assistantText, ts: Date.now(), locale };

  await appendList(redis, keySTB(core_id, session_id), userEntry, assistantEntry);
  await ltrimKeepLast(redis, keySTB(core_id, session_id), STB_MAX_ITEMS);
  await rolloverSTBIfNeeded(core_id, session_id);
  await maybeRefreshSummary(core_id, session_id, [userEntry, assistantEntry]);

  return reply.send({ core_id, session_id, locale, reply: assistantText });
});

// Debug: estimate tokens without real Claude call
app.get('/debug/tokens', async (req) => {
  const { core_id, session_id, locale = 'uk', prompt = 'ping' } = req.query;
  const userEntry = { role: 'user', content: String(prompt || ''), ts: Date.now(), locale };
  const { system, messages } = await buildPrompt({ coreId: core_id, sessionId: session_id, locale, userEntry });
  const tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
  return { input_tokens: tokens, system_chars: system.length, messages_count: messages.length };
});

// Health
app.get('/health', async () => ({ ok: true }));

// Start
app.listen({ port: Number(PORT), host: '0.0.0.0' })
  .then((addr) => app.log.info(`listening on ${addr}`))
  .catch((err) => { app.log.error(err); process.exit(1); });
