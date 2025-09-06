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
  BODY_LIMIT_BYTES = String(25 * 1024 * 1024), // 25MB
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

// OpenAI-compatible helpers for multimodal & streaming ===
const TEXT_DECODER = new TextDecoder();
const { FormData, Blob } = globalThis;

const isDataUrl = (u) => typeof u === "string" && u.startsWith("data:");
function parseDataUrl(u) {
  const m = /^data:([^;]+);base64,(.+)$/i.exec(u || "");
  return m ? { mediaType: m[1], b64: m[2] } : null;
}
async function fetchToBase64(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${r.status}`);
  const buf = Buffer.from(await r.arrayBuffer());
  const ct = r.headers.get("content-type") || "application/octet-stream";
  return { mediaType: ct, b64: buf.toString("base64") };
}

// OpenAI content[] -> Anthropic parts[]
async function toAnthropicParts(content) {
  const out = [];
  const pushText = (t) => { const s = String(t || "").trim(); if (s) out.push({ type: "text", text: s }); };

  if (typeof content === "string") {
    pushText(content);
    return out;
  }
  if (!Array.isArray(content)) return out;

  for (const c of content) {
    if (c?.type === "text") pushText(c.text);
    if (c?.type === "image_url") {
      const u = c.image_url?.url || c.image_url;
      if (!u) continue;
      let mediaType, b64;
      if (isDataUrl(u)) {
        const p = parseDataUrl(u);
        if (p) { mediaType = p.mediaType; b64 = p.b64; }
      } else {
        const r = await fetchToBase64(u);
        mediaType = r.mediaType; b64 = r.b64;
      }
      if (b64) out.push({ type: "image", source: { type: "base64", media_type: mediaType || "image/png", data: b64 } });
    }
  }
  return out;
}

// OpenAI messages[] -> { system, anthropicMessages[] }
async function mapOpenAIToAnthropic(openaiMessages) {
  let systemAll = [];
  const msgs = [];

  for (const m of openaiMessages || []) {
    if (m.role === "system") {
      const s = typeof m.content === "string"
        ? m.content
        : (m.content || []).filter(x => x?.type === "text").map(x => x.text).join("\n");
      if (s) systemAll.push(s);
      continue;
    }
    if (m.role === "user") {
      const parts = await toAnthropicParts(m.content ?? m.text ?? "");
      msgs.push({
        role: "user",
        content: parts.length ? parts : [{ type: "text", text: String(m.content ?? m.text ?? "") }]
      });
      continue;
    }
    if (m.role === "assistant") {
      const t = typeof m.content === "string"
        ? m.content
        : (m.content || []).filter(x => x?.type === "text").map(x => x.text).join("");
      msgs.push({ role: "assistant", content: [{ type: "text", text: String(t || "") }] });
      continue;
    }
  }

  return { system: systemAll.join("\n\n").trim() || null, messages: msgs };
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

// /v1/models For UI
app.get("/v1/models", async (req, reply) => {
  return reply.send({
    object: "list",
    data: [
      { id: ANTHROPIC_MODEL, object: "model", owned_by: "core" },
      { id: "hanna-core", object: "model", owned_by: "core" }
    ]
  });
});

// /v1/chat/completions OpenAI-compatible
app.post("/v1/chat/completions", async (req, reply) => {
  try {
    const { model = ANTHROPIC_MODEL, messages = [], temperature, max_tokens, stream } = req.body || {};
    const session_id = req.headers["x-session-id"] || hash(req.headers.authorization || req.ip || "anon");
    const core_id = "webui";
    const histKey = `hist:${core_id}:${session_id}`;
    const sumKey  = `sum:${core_id}:${session_id}`;

    const { system: systemExtra, messages: mapped } = await mapOpenAIToAnthropic(messages);
    const summary = await redis.get(sumKey);
    const system = [buildSystem(req.headers["x-locale"] || "auto", summary), systemExtra].filter(Boolean).join("\n\n");

    if (stream) {
      reply.raw.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });

      const r = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json"
        },
        body: JSON.stringify({
          model,
          system,
          messages: mapped.length ? mapped : [{ role: "user", content: [{ type: "text", text: "" }] }],
          max_tokens: num(max_tokens, num(MAX_OUTPUT_TOKENS, 300)),
          temperature: num(temperature, num(TEMPERATURE, 0.2)),
          stream: true
        })
      });

      if (!r.ok || !r.body) {
        const detail = r && !r.ok ? await r.text().catch(() => "") : "no body";
        throw new Error(`Anthropic stream error: ${detail}`);
      }

      const reader = r.body.getReader();
      let buffer = "", accText = "";

      const send = (obj) => reply.raw.write(`data: ${JSON.stringify(obj)}\n\n`);
      const base = () => ({
        id: "chatcmpl_" + crypto.randomUUID(),
        object: "chat.completion.chunk",
        model,
        created: Math.floor(Date.now()/1000),
        choices: [{ index: 0, delta: {}, finish_reason: null }]
      });

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += TEXT_DECODER.decode(value, { stream: true });
        const parts = buffer.split("\n\n"); buffer = parts.pop() || "";

        for (const p of parts) {
          const line = p.trim();
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (!data || data === "[DONE]") continue;
          try {
            const evt = JSON.parse(data);
            if (evt.type === "content_block_delta" && evt.delta?.text) {
              accText += evt.delta.text;
              const chunk = base();
              chunk.choices[0].delta = { content: evt.delta.text };
              send(chunk);
            }
            if (evt.type === "message_stop") {
              // сохранить историю (только текстовую часть запроса)
              const lastUser = messages.filter(m => m.role === "user").pop();
              const userText = typeof lastUser?.content === "string"
                ? lastUser.content
                : (Array.isArray(lastUser?.content) ? (lastUser.content.find(x => x.type === "text")?.text || "") : "");
              if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }));
              if (accText) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: accText }));
              await redis.expire(histKey, num(TTL_SECONDS, 604800));
              await redis.expire(sumKey,  num(TTL_SECONDS, 604800));

              reply.raw.write("data: [DONE]\n\n");
              reply.raw.end();
            }
          } catch {}
        }
      }
      return;
    }

    // non-stream
    const res = await callAnthropic({
      model,
      system,
      messages: mapped.length ? mapped : [{ role: "user", content: [{ type: "text", text: "" }] }],
      maxTokens: num(max_tokens, num(MAX_OUTPUT_TOKENS, 300)),
      temperature: num(temperature, num(TEMPERATURE, 0.2)),
      timeoutMs: num(TIMEOUT_MS, 30000)
    });

    const text = res?.content?.[0]?.text || "";

    const lastUser = messages.filter(m => m.role === "user").pop();
    const userText = typeof lastUser?.content === "string"
      ? lastUser.content
      : (Array.isArray(lastUser?.content) ? (lastUser.content.find(x => x.type === "text")?.text || "") : "");

    if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }));
    if (text) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: text }));
    await redis.expire(histKey, num(TTL_SECONDS, 604800));
    await redis.expire(sumKey,  num(TTL_SECONDS, 604800));

    return reply.send({
      id: "chatcmpl_" + crypto.randomUUID(),
      object: "chat.completion",
      model,
      created: Math.floor(Date.now() / 1000),
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
      usage: {
        prompt_tokens: userText?.length || 0,
        completion_tokens: text.length,
        total_tokens: (userText?.length || 0) + text.length
      }
    });
  } catch (e) {
    req.log.error({ err: String(e?.message || e) }, "chat-completions-fail");
    return reply.code(500).send({ error: { message: String(e?.message || e) } });
  }
});

app.listen({ port: Number(process.env.PORT || 3000), host: "0.0.0.0" }, (err, address) => {
  if (err) { app.log.error(err); process.exit(1); }
  app.log.info(`undrawn Core (FAST) at ${address}`);
});
