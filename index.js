// index.js — FAST profile (hardened: json-only, stable reply, metrics, ratelimit)
import Fastify from "fastify";
import cors from "@fastify/cors";
import { Redis } from "@upstash/redis";
import fetch from "node-fetch";
import crypto from "crypto";

const {
  UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN,
  ANTHROPIC_API_KEY,

  ANTHROPIC_MODEL = "claude-3-haiku-20240307",
  SUMMARIZER_MODEL = "claude-3-haiku-20240307",
  MAX_OUTPUT_TOKENS = "300",
  TEMPERATURE = "0.2",
  UI_PROFILE = "",
  TIMEOUT_MS = (UI_PROFILE === "builder" ? "9000" : "30000"),
  HISTORY_KEEP = "10",
  SUMMARIZE_THRESHOLD = "120",
  SUMMARY_MAX_TOKENS = "250",
  TTL_SECONDS = String(60 * 60 * 24 * 7),

  CORE_SYSTEM_PROMPT = "",
  BODY_LIMIT_BYTES = String(5 * 1024 * 1024),
  MAX_INPUT_CHARS = "0",

  // rate-limit
  RL_PER_MIN = "60"
} = process.env;

if (!UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN) throw new Error("Missing Upstash env");
if (!ANTHROPIC_API_KEY) throw new Error("Missing ANTHROPIC_API_KEY");

const app = Fastify({ logger: true, bodyLimit: Number(BODY_LIMIT_BYTES) });
app.register(cors, { origin: true });

const redis = new Redis({ url: UPSTASH_REDIS_REST_URL, token: UPSTASH_REDIS_REST_TOKEN });

const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
const hash = (s) => crypto.createHash("sha256").update(String(s)).digest("hex").slice(0, 16);
const sanitize = (s) =>
  String(s || "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\u0000/g, "")
    .replace(/[^\S\r\n]+/g, " ")
    .trim()
    .slice(0, 8000);

const baseRules =
  "Use Memory Summary (if present) and recent history as ground truth. " +
  "Do not re-ask facts unless conflicting. Be concise.";

const buildSystem = (locale, memorySummary) =>
  [
    "You are undrawn Core.",
    baseRules,
    memorySummary ? `Memory Summary: ${memorySummary}` : "",
    `Reply in the user's language (locale hint: ${locale || "auto"}).`,
    sanitize(CORE_SYSTEM_PROMPT)
  ].filter(Boolean).join(" ");

const metrics = {
  started_at: new Date().toISOString(),
  calls_total: 0,
  errors_total: 0,
  timeouts_total: 0,
  p95_ms: 0,
  last_ms: 0,
};
const lastLatencies = [];

// ---------- OpenWebUI/OpenAI-like response wrapper ----------
function jsonReply(replyObj) {
  const text = String(replyObj ?? "");
  return {
    id: crypto.randomUUID(),
    object: "chat.completion",
    model: "hanna-core",
    created: Math.floor(Date.now() / 1000),
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: [
            { type: "text", text }
          ]
        },
        finish_reason: "stop"
      }
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: text.length,
      total_tokens: text.length
    }
  };
}

function recordLatency(ms) {
  metrics.last_ms = ms;
  lastLatencies.push(ms);
  if (lastLatencies.length > 200) lastLatencies.shift();
  const sorted = [...lastLatencies].sort((a,b)=>a-b);
  const idx = Math.floor(0.95 * (sorted.length - 1));
  metrics.p95_ms = sorted[idx] || ms;
}

async function callAnthropic({ model, system, messages, maxTokens, temperature, timeoutMs }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
      },
      body: JSON.stringify({ model, system, messages, max_tokens: maxTokens, temperature })
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Anthropic ${res.status}: ${detail}`);
    }
    return await res.json();
  } catch (e) {
    if (String(e?.message || "").includes("aborted")) metrics.timeouts_total++;
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

// ---------- health/version/metrics ----------
app.get("/health", async () => ({ ok: true }));
app.get("/version", async () => ({ version: "core-fast-1.1.0" }));
app.get("/metrics", async () => metrics);

// ---------- main handler ----------
app.addHook("onRequest", (req, res, done) => {
  req.id ||= crypto.randomUUID();
  done();
});

app.post("/v1/complete", async (req, reply) => {
  const t0 = Date.now();
  metrics.calls_total++;

  // JSON-only gate (мʼяка помилка у форматі reply)
  const ct = String(req.headers["content-type"] || "").toLowerCase();
  if (!ct.includes("application/json")) {
    const ms = Date.now() - t0; recordLatency(ms);
    reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, reason: "non-json" }));
    return reply.code(200).send(jsonReply("CORE error: JSON only (use content-type: application/json)"));
  }

  try {
    // rate limit по IP
    const ipKey = `rl:${hash(req.ip || "unknown")}:${Math.floor(Date.now()/60000)}`; // 1-хв вікно
    const count = await redis.incr(ipKey);
    if (count === 1) await redis.expire(ipKey, 65);
    if (count > num(RL_PER_MIN, 60)) {
      const ms = Date.now() - t0; recordLatency(ms);
      reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, rate_limited: true }));
      return reply.code(200).send(jsonReply("CORE error: rate limit exceeded, try again later"));
    }

    // body parse
    let body = req.body;
    if (typeof body === "string") {
      try { body = JSON.parse(body); } catch { body = null; }
    }
    if (!body || typeof body !== "object") {
      const ms = Date.now() - t0; recordLatency(ms);
      reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, reason: "empty body" }));
      return reply.code(200).send(jsonReply("CORE error: invalid or empty JSON body"));
    }

    const { core_id, session_id, prompt, locale } = body;
    if (!core_id || !session_id || !prompt) {
      const ms = Date.now() - t0; recordLatency(ms);
      reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, reason: "missing fields" }));
      return reply.code(200).send(jsonReply("CORE error: missing core_id, session_id or prompt"));
    }

    const histKey = `hist:${core_id}:${session_id}`;
    const sumKey  = `sum:${core_id}:${session_id}`;

    const safeParse = (v) => {
      if (typeof v === "string") { try { return JSON.parse(v); } catch { return null; } }
      return v && typeof v === "object" ? v : null;
    };

    const keepN = num(HISTORY_KEEP, 10);
    const recent = await redis.lrange(histKey, -keepN, -1);
    const recentMsgs = (recent || []).map(safeParse).filter(Boolean);

    const summary = await redis.get(sumKey);

    const cap = num(MAX_INPUT_CHARS, 0);
    let cleanPrompt = String(prompt).replace(/\r/g, "").replace(/\t/g, " ").replace(/ {2,}/g, " ").trim();
    if (cap > 0 && cleanPrompt.length > cap) cleanPrompt = cleanPrompt.slice(0, cap);

    const messages = [];
    if (recentMsgs.length) messages.push(...recentMsgs);
    messages.push({ role: "user", content: cleanPrompt });

    const res = await callAnthropic({
      model: ANTHROPIC_MODEL,
      system: buildSystem(locale, summary),
      messages,
      maxTokens: num(MAX_OUTPUT_TOKENS, 300),
      temperature: num(TEMPERATURE, 0.2),
      timeoutMs: num(TIMEOUT_MS, 30000)
    });

    const modelReply = res?.content?.[0]?.text || "";
    if (!modelReply) throw new Error("Invalid Claude response");

    await redis.rpush(histKey, JSON.stringify({ role: "user", content: cleanPrompt }));
    await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: modelReply }));
    await redis.expire(histKey, num(TTL_SECONDS, 604800));
    await redis.expire(sumKey, num(TTL_SECONDS, 604800));

    const total = await redis.llen(histKey);
    if (total > num(SUMMARIZE_THRESHOLD, 120)) {
      const older = await redis.lrange(histKey, 0, -keepN - 1);
      if (older.length) {
        const olderPlain = (older || []).map(safeParse).filter(Boolean);
        const sum = await callAnthropic({
          model: SUMMARIZER_MODEL,
          system: "Summarize into compact, durable facts: identities, decisions, preferences, constraints, dates.",
          messages: [{ role: "user", content: JSON.stringify(olderPlain) }],
          maxTokens: num(SUMMARY_MAX_TOKENS, 250),
          temperature: 0,
          timeoutMs: 8000
        });
        const text = sum?.content?.[0]?.text || "";
        if (text) {
          await redis.set(sumKey, text);
          const lastN = await redis.lrange(histKey, -keepN, -1);
          await redis.del(histKey);
          if (lastN.length) await redis.rpush(histKey, ...lastN);
        }
      }
    }

    const ms = Date.now() - t0; recordLatency(ms);
    reply.header(
      "X-Core-Diag",
      JSON.stringify({
        ms, req_id: req.id, has_summary: Boolean(summary), history_count: recentMsgs.length
      }).slice(0, 256)
    );

    return reply.code(200).send({
  id: "chatcmpl_" + crypto.randomUUID(),
  object: "chat.completion",
  model: ANTHROPIC_MODEL,
  created: Math.floor(Date.now() / 1000),
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: modelReply },
      finish_reason: "stop"
    }
  ],
  usage: {
    prompt_tokens: prompt.length,  // можна краще рахувати
    completion_tokens: modelReply.length,
    total_tokens: prompt.length + modelReply.length
  }
});
  } catch (e) {
    metrics.errors_total++;
    const ms = Date.now() - t0; recordLatency(ms);
    const msg = String(e?.message || e);
    req.log.error({ err: msg }, "core-fail");
    reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, fail: true }).slice(0,256));
    // мʼяка помилка — щоб UI не падав
    return reply.code(200).send(jsonReply(`CORE error: ${msg}`));
  }
});

app.listen({ port: Number(process.env.PORT || 3000), host: "0.0.0.0" }, (err, address) => {
  if (err) { app.log.error(err); process.exit(1); }
  app.log.info(`undrawn Core (FAST) at ${address}`);
});
