// index.js — final, aligned with updated CORE_SYSTEM_PROMPT
// Fastify + Upstash Redis + Anthropic Messages API
// Memory model: STB (short-term buffer) + LTM (archive) + Summary + Targeted recall
// Economy: token budget + safe trimming (no full reread of 400+ turns)
// Node 18.x

import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';

// ---------- ENV ----------
const {
  ANTHROPIC_API_KEY,
  UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN,
  CORE_SYSTEM_PROMPT,
  SYSTEM_PROMPT_BASE: SP_FALLBACK, // optional fallback
  OPENAI_API_KEY,                  // optional (improves recall quality)
  PORT = 3000,
  ANTHROPIC_MODEL = 'claude-3-5-sonnet-20240620',
  ROUTE_PREFIX = '',
  ANTHROPIC_TEMPERATURE = '0.2',
  STB_MAX_ITEMS: STB_ENV,
  RECALL_TOP_K: RECALL_ENV,
  LTM_SCAN_LIMIT: LTM_ENV,
  TOKEN_BUDGET: BUDGET_ENV,
  MAX_OUTPUT_TOKENS: OUT_ENV,
} = process.env;

const SYSTEM_PROMPT_BASE = CORE_SYSTEM_PROMPT || SP_FALLBACK;
if (!ANTHROPIC_API_KEY || !UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN || !SYSTEM_PROMPT_BASE) {
  throw new Error('Missing ENV: ANTHROPIC_API_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, CORE_SYSTEM_PROMPT (or SYSTEM_PROMPT_BASE)');
}

const app = Fastify({ logger: true });
await app.register(cors, { origin: true });
const redis = new Redis({ url: UPSTASH_REDIS_REST_URL, token: UPSTASH_REDIS_REST_TOKEN });

// ---------- Tunables ----------
const HISTORY_TTL_SECONDS = 60 * 60 * 24 * 30;                     // 30 days
const STB_MAX_ITEMS = Number(STB_ENV || 20);                        // ~10 turns (user+assistant counted separately)
const RECALL_TOP_K = Number(RECALL_ENV || 6);                       // retrieved LTM items
const LTM_SCAN_LIMIT = Number(LTM_ENV || 1500);                     // tail window for recall scan
const TOKEN_BUDGET = Number(BUDGET_ENV || 170_000);                 // headroom under ~200k context
const MAX_OUTPUT_TOKENS = Number(OUT_ENV || 1024);
const TEMPERATURE = Number(ANTHROPIC_TEMPERATURE);

// ---------- Redis Keys ----------
const keySTB = (coreId, sessionId) => `stb:${coreId}:${sessionId}`; // LIST
const keyLTM = (coreId, sessionId) => `ltm:${coreId}:${sessionId}`; // LIST
const keySUM = (coreId, sessionId) => `sum:${coreId}:${sessionId}`; // STRING

// ---------- Helpers ----------
const toAnthropicMessage = (item) => ({
  role: item.role === 'assistant' ? 'assistant' : 'user',
  content: [{ type: 'text', text: String(item.content ?? '') }],
});

async function ensureListKey(client, key) {
  const t = await client.type(key);
  if (t && t !== 'none' && t !== 'list') await client.del(key);
}

// IMPORTANT: push each entry as a separate Redis list item (no arrays-in-one-slot)
async function appendList(client, key, ...entries) {
  if (!entries.length) return;
  await ensureListKey(client, key);
  const payloads = entries.map((e) => JSON.stringify(e));
  await client.rpush(key, ...payloads);
  await client.expire(key, HISTORY_TTL_SECONDS);
}

// Backward-compatible reader: flattens old broken shape if any
async function lrangeJSON(client, key, start, stop) {
  await ensureListKey(client, key);
  const raw = await client.lrange(key, start, stop);
  const out = [];
  for (const s of raw) {
    let v; try { v = JSON.parse(s); } catch { v = null; }
    if (v == null) continue;
    if (Array.isArray(v)) {
      for (const inner of v) {
        try {
          const obj = typeof inner === 'string' ? JSON.parse(inner) : inner;
          if (obj && typeof obj === 'object') out.push(obj);
        } catch {}
      }
    } else if (typeof v === 'object') {
      out.push(v);
    }
  }
  return out;
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

// ---------- Embeddings (optional) ----------
async function embedText(text) {
  if (!OPENAI_API_KEY) return null;
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: { authorization: `Bearer ${OPENAI_API_KEY}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'text-embedding-3-small', input: text }),
  });
  if (!res.ok) { const err = await res.text().catch(() => ''); throw new Error(`OpenAI embeddings error: ${res.status} ${err}`); }
  const data = await res.json();
  return data.data?.[0]?.embedding || null;
}
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0; const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}
function keywordScore(q, doc) {
  const qT = new Set(String(q).toLowerCase().split(/\W+/).filter(Boolean));
  const dT = new Set(String(doc).toLowerCase().split(/\W+/).filter(Boolean));
  let inter = 0; for (const t of qT) if (dT.has(t)) inter++;
  return inter / Math.sqrt((qT.size || 1) * (dT.size || 1));
}

// ---------- Token counting ----------
async function countTokens({ system, messages, model }) {
  const res = await fetch('https://api.anthropic.com/v1/messages/count_tokens', {
    method: 'POST',
    headers: { 'x-api-key': ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
    body: JSON.stringify({ model, system, messages }),
  });
  if (!res.ok) { const txt = await res.text().catch(() => ''); throw new Error(`count_tokens failed: ${res.status} ${txt}`); }
  const data = await res.json();
  return data.input_tokens ?? 0;
}

// ---- Summary builder (heuristic) ----
function updateSummaryHeuristic(prevSummary, entries) {
  const facts = [];
  const add = (line) => {
    const t = String(line || '').trim();
    if (!t) return;
    const item = `- ${t}`;
    if (!facts.includes(item)) facts.push(item);
  };

  // Simple multilingual patterns: EN + UA + RU (first-person, preferences, location)
  const tests = [
    // EN
    (s) => /(^i\s*(am|work|live|prefer|use)\b)|(\bmy\s+[a-z]+)/i.test(s),
    // UA
    (s) => /(^я\s*(живу|переїхав|працюю|користуюсь|використовую|люблю|віддаю\s*перевагу)\b)|\b(мій|моя|моє|мої)\b/i.test(s),
    // RU
    (s) => /(^я\s*(живу|переехал|работаю|пользуюсь|использую|люблю|предпочитаю)\b)|\b(мой|моя|моё|мои)\b/i.test(s),
  ];

  for (const e of entries) {
    if (e.role !== 'user') continue;
    const text = String(e.content || '');
    // split by sentence-ish boundaries
    const parts = text.split(/[.\n!?]+/).map(x => x.trim()).filter(Boolean).slice(0, 6);
    for (const p of parts) {
      if (tests.some(fn => fn(p))) add(p);
    }
  }

  const head = `Memory state (concise facts from earlier context):`;
  const base = prevSummary ? prevSummary.trim() : `${head}\n- (none yet)`;
  const merged = facts.length ? `${base}\n# New observations:\n${facts.join('\n')}` : base;
  return merged.slice(0, 1600);
}
}

// ---------- Recall from LTM ----------
async function recallFromLTM(coreId, sessionId, userMsg) {
  const all = await lrangeJSON(redis, keyLTM(coreId, sessionId), -LTM_SCAN_LIMIT, -1);
  if (!all.length) return [];
  let qEmb = null;
  if (OPENAI_API_KEY) { try { qEmb = await embedText(userMsg); } catch {} }
  const scored = all.map((e) => {
    const content = String(e.content || '');
    const score = (qEmb && Array.isArray(e.emb)) ? cosineSim(qEmb, e.emb) : keywordScore(userMsg, content);
    return { score, entry: e };
  }).sort((a, b) => b.score - a.score);
  const top = scored.slice(0, RECALL_TOP_K).map(s => s.entry);
  const seen = new Set(); const dedup = [];
  for (const e of top) { const k = `${e.role}:${e.content.slice(0, 120)}`; if (!seen.has(k)) { seen.add(k); dedup.push(e); } }
  return dedup;
}

// ---------- Build prompt under budget ----------
async function buildPrompt({ coreId, sessionId, locale, userEntry }) {
  const stb = await lrangeJSON(redis, keySTB(coreId, sessionId), 0, -1);
  const summary = await readSummary(redis, keySUM(coreId, sessionId));
  const recalled = await recallFromLTM(coreId, sessionId, userEntry.content);

  const summaryBlock = summary ? `\n\n${summary}` : '';
  const recalledBlock = recalled.length ? `\n\nRelevant context (retrieved):\n${recalled.map(e => `- ${e.role}: ${e.content}`).join('\n')}` : '';
  let system = `${SYSTEM_PROMPT_BASE}\nLocale: ${locale}${summaryBlock}${recalledBlock}`;
  let messages = [...stb.map(toAnthropicMessage), toAnthropicMessage(userEntry)];

  let tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
  if (tokens + MAX_OUTPUT_TOKENS <= TOKEN_BUDGET) return { system, messages };

  // Trim STB oldest-first
  const work = [...stb];
  while (work.length > 0) {
    work.shift();
    messages = [...work.map(toAnthropicMessage), toAnthropicMessage(userEntry)];
    tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
    if (tokens + MAX_OUTPUT_TOKENS <= TOKEN_BUDGET) break;
  }
  // Drop recalled, then summary if still too big
  if (tokens + MAX_OUTPUT_TOKENS > TOKEN_BUDGET && recalledBlock) {
    system = `${SYSTEM_PROMPT_BASE}\nLocale: ${locale}${summaryBlock}`;
    tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
  }
  if (tokens + MAX_OUTPUT_TOKENS > TOKEN_BUDGET && summaryBlock) {
    system = `${SYSTEM_PROMPT_BASE}\nLocale: ${locale}`;
  }
  return { system, messages };
}

// ---------- STB → LTM rollover ----------
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
      try { const emb = await embedText(String(e.content || '')); if (emb) item.emb = emb; } catch {}
    }
    enriched.push(item);
  }
  await appendList(redis, keyLTM(coreId, sessionId), ...enriched);
}

// ---------- Summary maintenance ----------
async function maybeRefreshSummary(coreId, sessionId, recentEntries) {
  const prev = await readSummary(redis, keySUM(coreId, sessionId));
  const next = updateSummaryHeuristic(prev, recentEntries);
  if (next !== prev) await writeSummary(redis, keySUM(coreId, sessionId), next);
}

// ---------- Core handler ----------
const chatHandler = async (req, reply) => {
  const b = req.body || {};
  const { core_id, session_id, user_message, prompt, locale = 'uk' } = b;
  const content = user_message ?? prompt;
  if (!core_id || !session_id || !content) {
    return reply.code(400).send({ error: 'bad_request', detail: 'core_id, session_id, and user_message/prompt are required' });
  }

  const userEntry = { role: 'user', content, ts: Date.now(), locale };
  const { system, messages } = await buildPrompt({ coreId: core_id, sessionId: session_id, locale, userEntry });

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'x-api-key': ANTHROPIC_API_KEY, 'anthropic-version': '2023-06-01', 'content-type': 'application/json' },
    body: JSON.stringify({ model: ANTHROPIC_MODEL, system, messages, max_tokens: MAX_OUTPUT_TOKENS, temperature: TEMPERATURE }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    req.log.error({ status: res.status, body: text }, 'Anthropic API error');
    return reply.code(502).send({ error: 'upstream_error', detail: text });
  }

  const data = await res.json();
  const assistantText = Array.isArray(data.content) && data.content[0]?.type === 'text' ? data.content[0].text : '';
  const assistantEntry = { role: 'assistant', content: assistantText, ts: Date.now(), locale };

  // Append BOTH user and assistant to STB (two separate list items)
  await appendList(redis, keySTB(core_id, session_id), userEntry, assistantEntry);
  await ltrimKeepLast(redis, keySTB(core_id, session_id), STB_MAX_ITEMS);
  await rolloverSTBIfNeeded(core_id, session_id);
  await maybeRefreshSummary(core_id, session_id, [userEntry, assistantEntry]);

  return reply.send({ core_id, session_id, locale, reply: assistantText });
};

// ---------- Routes ----------
const r = ROUTE_PREFIX;
app.get(`${r}/health`, async () => ({ ok: true }));
app.get(`${r}/debug/tokens`, async (req) => {
  const { core_id, session_id, locale = 'uk', prompt = 'ping' } = req.query;
  const ue = { role: 'user', content: String(prompt || ''), ts: Date.now(), locale };
  const { system, messages } = await buildPrompt({ coreId: core_id, sessionId: session_id, locale, userEntry: ue });
  const tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
  return { input_tokens: tokens, system_chars: system.length, messages_count: messages.length };
});
app.post(`${r}/chat`, chatHandler);

// Legacy aliases (to keep old clients working)
app.post(`${r}/v1/chat`, chatHandler);
app.post(`${r}/complete`, chatHandler);
app.post(`${r}/v1/complete`, chatHandler);
app.post(`${r}/api/chat`, chatHandler);

// Admin: purge a session (for clean tests)
app.post(`${r}/purge`, async (req, reply) => {
  const { core_id, session_id } = req.body || {};
  if (!core_id || !session_id) return reply.code(400).send({ error: 'bad_request', detail: 'core_id and session_id required' });
  await redis.del(keySTB(core_id, session_id));
  await redis.del(keyLTM(core_id, session_id));
  await redis.del(keySUM(core_id, session_id));
  return reply.send({ ok: true });
});

// Startup logs
app.ready().then(() => app.log.info(app.printRoutes()));
app.listen({ port: Number(PORT), host: '0.0.0.0' })
  .then(addr => app.log.info(`listening on ${addr}`))
  .catch(err => { app.log.error(err); process.exit(1); });
