// index.js — FAST profile (fixed for long inputs & clear errors)
import Fastify from "fastify";
import cors from "@fastify/cors";
import { Redis } from "@upstash/redis";
import fetch from "node-fetch";

const {
  UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN,
  ANTHROPIC_API_KEY,

  // runtime knobs
  ANTHROPIC_MODEL = "claude-3-haiku-20240307",
  SUMMARIZER_MODEL = "claude-3-haiku-20240307",
  MAX_OUTPUT_TOKENS = "300",
  TEMPERATURE = "0.2",
  // if UI_PROFILE=builder -> 9000ms, otherwise 30000ms
  UI_PROFILE = "",
  TIMEOUT_MS = (UI_PROFILE === "builder" ? "9000" : "30000"),
  HISTORY_KEEP = "10",
  SUMMARIZE_THRESHOLD = "120",
  SUMMARY_MAX_TOKENS = "250",
  TTL_SECONDS = String(60 * 60 * 24 * 7), // 7 days
  CORE_SYSTEM_PROMPT = "",
  // optional safety caps
  BODY_LIMIT_BYTES = String(5 * 1024 * 1024), // 5MB default
  MAX_INPUT_CHARS = "0" // 0 = no cap
} = process.env;

if (!UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN) throw new Error("Missing Upstash env");
if (!ANTHROPIC_API_KEY) throw new Error("Missing ANTHROPIC_API_KEY");

const app = Fastify({
  logger: true,
  bodyLimit: Number(BODY_LIMIT_BYTES) // allow large request bodies
});
app.register(cors, { origin: true });

const redis = new Redis({ url: UPSTASH_REDIS_REST_URL, token: UPSTASH_REDIS_REST_TOKEN });

const num = (v, d) => (Number.isFinite(Number(v)) ? Number(v) : d);
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
  ]
    .filter(Boolean)
    .join(" ");

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
      body: JSON.stringify({
        model,
        system,                   // system prompt here
        messages,                 // only user/assistant
        max_tokens: maxTokens,
        temperature
      })
    });

    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`Anthropic ${res.status}: ${detail}`);
    }

    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

app.get("/health", async () => ({ ok: true }));

// -------------------- MAIN HANDLER --------------------
app.post("/v1/complete", async (req, reply) => {
  const t0 = Date.now();
  try {
    // parse body (no double parsing)
    let body = req.body;
    if (typeof body === "string") {
      try { body = JSON.parse(body); }
      catch { return reply.code(400).send({ error: "Invalid JSON body" }); }
    }
    if (!body || typeof body !== "object") {
      return reply.code(400).send({ error: "Empty body" });
    }

    const { core_id, session_id, prompt, locale } = body;
    if (!core_id || !session_id || !prompt) {
      return reply.code(400).send({ error: "Missing core_id, session_id or prompt" });
    }

    const histKey = `hist:${core_id}:${session_id}`;
    const sumKey  = `sum:${core_id}:${session_id}`;

    // safe JSON parse helper
    const safeParse = (v) => {
      if (typeof v === "string") { try { return JSON.parse(v); } catch { return null; } }
      return v && typeof v === "object" ? v : null;
    };

    // recent history
    const keepN = num(HISTORY_KEEP, 10);
    const recent = await redis.lrange(histKey, -keepN, -1);
    const recentMsgs = (recent || []).map(safeParse).filter(Boolean);

    // memory summary
    const summary = await redis.get(sumKey);

    // normalize + optional cap
    const cap = num(MAX_INPUT_CHARS, 0);
    let cleanPrompt = String(prompt).replace(/\r/g, "").replace(/\t/g, " ").replace(/ {2,}/g, " ").trim();
    if (cap > 0 && cleanPrompt.length > cap) cleanPrompt = cleanPrompt.slice(0, cap);

    // assemble messages (user/assistant only)
    const messages = [];
    if (recentMsgs.length) messages.push(...recentMsgs);
    messages.push({ role: "user", content: cleanPrompt });

    // main answer
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

    // persist
    await redis.rpush(histKey, JSON.stringify({ role: "user", content: cleanPrompt }));
    await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: modelReply }));
    await redis.expire(histKey, num(TTL_SECONDS, 604800));
    await redis.expire(sumKey, num(TTL_SECONDS, 604800));

    // autosummary (fast path)
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

    reply.header(
      "X-Core-Diag",
      JSON.stringify({
        ms: Date.now() - t0,
        has_summary: Boolean(summary),
        history_count: recentMsgs.length
      }).slice(0, 256)
    );
    return reply.send({ reply: modelReply });
  } catch (e) {
    req.log.error(e);
    // propagate real upstream error to client
    return reply.code(502).send({ error: String(e?.message || e) });
  }
});
// ------------------ END HANDLER -------------------

app.listen({ port: Number(process.env.PORT || 3000), host: "0.0.0.0" }, (err, address) => {
  if (err) { app.log.error(err); process.exit(1); }
  app.log.info(`undrawn Core (FAST) at ${address}`);
});
