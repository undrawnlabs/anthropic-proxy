// index.js  — FAST profile for Builder (<9s)
import Fastify from "fastify";
import cors from "@fastify/cors";
import { Redis } from "@upstash/redis";
import fetch from "node-fetch";

const {
  UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN,
  ANTHROPIC_API_KEY,

  // FAST defaults for Builder
  ANTHROPIC_MODEL = "claude-3-haiku-20240307",
  SUMMARIZER_MODEL = "claude-3-haiku-20240307",
  MAX_OUTPUT_TOKENS = "300",
  TEMPERATURE = "0.2",
  TIMEOUT_MS = "9000",            // <—— щоб влізти у Builder ~10s
  HISTORY_KEEP = "10",            // менше повідомлень → швидше
  SUMMARIZE_THRESHOLD = "120",    // коли >120 меседжів — стискаємо
  SUMMARY_MAX_TOKENS = "250",
  TTL_SECONDS = String(60 * 60 * 24 * 7), // 7 днів
  CORE_SYSTEM_PROMPT = ""
} = process.env;

if (!UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN) throw new Error("Missing Upstash env");
if (!ANTHROPIC_API_KEY) throw new Error("Missing ANTHROPIC_API_KEY");

const app = Fastify({ logger: true });
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
        system,                   // <-- system prompt тут (НЕ в messages)
        messages,                 // <-- тільки user/assistant
        max_tokens: maxTokens,
        temperature
      })
    });
    if (!res.ok) throw new Error(`Anthropic ${res.status}: ${await res.text().catch(() => "")}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

app.get("/health", async () => ({ ok: true }));

// -------------------- FIXED HANDLER --------------------
app.post("/v1/complete", async (req, reply) => {
  const t0 = Date.now();
  try {
    // НОРМАЛІЗАЦІЯ ТІЛА (без дубль-парсингу)
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

    // SAFE PARSE для Redis
    const safeParse = (v) => {
      if (typeof v === "string") { try { return JSON.parse(v); } catch { return null; } }
      return v && typeof v === "object" ? v : null;
    };

    // recent
    const keepN = num(HISTORY_KEEP, 10);
    const recent = await redis.lrange(histKey, -keepN, -1);
    const recentMsgs = (recent || []).map(safeParse).filter(Boolean);

    // summary
    const summary = await redis.get(sumKey);

    // assemble messages (ЛИШЕ user/assistant)
    const messages = [];
    if (recentMsgs.length) messages.push(...recentMsgs);
    messages.push({ role: "user", content: String(prompt) });

    // main answer (fast)
    const res = await callAnthropic({
      model: ANTHROPIC_MODEL,
      system: buildSystem(locale, summary),   // <-- summary в system prompt
      messages,
      maxTokens: num(MAX_OUTPUT_TOKENS, 300),
      temperature: num(TEMPERATURE, 0.2),
      timeoutMs: num(TIMEOUT_MS, 9000)
    });

    const modelReply = res?.content?.[0]?.text || "";
    if (!modelReply) throw new Error("Invalid Claude response");

    // persist
    await redis.rpush(histKey, JSON.stringify({ role: "user", content: String(prompt) }));
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
    return reply.code(500).send({ error: "Internal Server Error" });
  }
});
// ------------------ END FIXED HANDLER -------------------

app.listen({ port: Number(process.env.PORT || 3000), host: "0.0.0.0" }, (err, address) => {
  if (err) { app.log.error(err); process.exit(1); }
  app.log.info(`undrawn Core (FAST) at ${address}`);
});
