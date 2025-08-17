// index.js
import Fastify from "fastify";
import cors from "@fastify/cors";
import { Redis } from "@upstash/redis";
import fetch from "node-fetch";

// ---------- ENV ----------
const {
  UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN,
  ANTHROPIC_API_KEY,
  PORT = 3000,
  HOST = "0.0.0.0",

  // Tunables
  ANTHROPIC_MODEL = "claude-3-opus-20240229",
  SUMMARIZER_MODEL = "claude-3-haiku-20240307",
  MAX_OUTPUT_TOKENS = "800",
  TEMPERATURE = "0.2",
  TIMEOUT_MS = "20000",
  HISTORY_KEEP = "20",            // recent messages sent to Claude
  SUMMARIZE_THRESHOLD = "200",    // when total > threshold -> summarize older part
  SUMMARY_MAX_TOKENS = "400",
  TTL_SECONDS = String(60 * 60 * 24 * 7) // 7 days
} = process.env;

if (!UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN) {
  throw new Error("Missing UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN");
}
if (!ANTHROPIC_API_KEY) throw new Error("Missing ANTHROPIC_API_KEY");

// ---------- SERVER ----------
const app = Fastify({ logger: true });
await app.register(cors, { origin: true });

// ---------- REDIS (Upstash REST) ----------
const redis = new Redis({
  url: UPSTASH_REDIS_REST_URL,
  token: UPSTASH_REDIS_REST_TOKEN
});

// ---------- HELPERS ----------
const n = (v, def) => (Number.isFinite(Number(v)) ? Number(v) : def);

async function callAnthropic({ model, system, messages, maxTokens, temperature, timeoutMs }) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  const body = JSON.stringify({ model, system, messages, max_tokens: maxTokens, temperature });

  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
      },
      body
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => "");
      throw new Error(`Anthropic ${res.status}: ${txt}`);
    }
    return await res.json();
  } finally {
    clearTimeout(id);
  }
}

const buildSystem = (locale) =>
  [
    "You are undrawn Core.",
    "Use the memory summary (if present) and recent history as ground truth.",
    "Do not re-ask facts unless they conflict.",
    `Reply in the user's language (locale hint: ${locale || "auto"}).`
  ].join(" ");

// ---------- ROUTES ----------
app.get("/health", async () => ({ ok: true }));

app.post("/v1/complete", async (req, reply) => {
  try {
    const { core_id, session_id, prompt, locale } = req.body || {};
    if (!core_id || !session_id || !prompt) {
      return reply.code(400).send({ error: "Missing core_id, session_id or prompt" });
    }

    const histKey = `hist:${core_id}:${session_id}`;
    const sumKey  = `sum:${core_id}:${session_id}`;

    // 1) Recent history (economy)
    const keepN = n(HISTORY_KEEP, 20);
    const recent = await redis.lrange(histKey, -keepN, -1); // returns array of strings
    const recentMsgs = recent.map((s) => JSON.parse(s));

    // 2) Long-term summary (if exists)
    const summary = await redis.get(sumKey);
    const messages = [];
    if (summary) {
      messages.push({
        role: "system",
        content: `Memory Summary (treat as true context): ${summary}`
      });
    }

    // 3) Append recent + new user message
    if (recentMsgs.length) messages.push(...recentMsgs);
    messages.push({ role: "user", content: prompt });

    // 4) Main answer from Claude
    const main = await callAnthropic({
      model: ANTHROPIC_MODEL,
      system: buildSystem(locale),
      messages,
      maxTokens: n(MAX_OUTPUT_TOKENS, 800),
      temperature: n(TEMPERATURE, 0.2),
      timeoutMs: n(TIMEOUT_MS, 20000)
    });

    const modelReply = main?.content?.[0]?.text;
    if (!modelReply) throw new Error("Invalid response from Claude");

    // 5) Persist (append)
    await redis.rpush(histKey, JSON.stringify({ role: "user", content: prompt }));
    await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: modelReply }));

    // 6) TTL rolling
    await redis.expire(histKey, n(TTL_SECONDS, 604800));
    await redis.expire(sumKey, n(TTL_SECONDS, 604800));

    // 7) Auto-summarize older part
    const totalLen = await redis.llen(histKey);
    const threshold = n(SUMMARIZE_THRESHOLD, 200);

    if (totalLen > threshold) {
      const olderSlice = await redis.lrange(histKey, 0, -keepN - 1); // all except last keepN
      if (olderSlice.length > 0) {
        const olderPlain = olderSlice.map((s) => JSON.parse(s));

        const sum = await callAnthropic({
          model: SUMMARIZER_MODEL,
          system:
            "Summarize the conversation into concise, durable facts and instructions. " +
            "Preserve identities, decisions, preferences, constraints, and dates. Keep it compact.",
          messages: [{ role: "user", content: JSON.stringify(olderPlain) }],
          maxTokens: n(SUMMARY_MAX_TOKENS, 400),
          temperature: 0,
          timeoutMs: n(TIMEOUT_MS, 20000)
        });

        const newSummary = sum?.content?.[0]?.text || "";
        if (newSummary) {
          await redis.set(sumKey, newSummary);
          // Trim history to last keepN
          const lastN = await redis.lrange(histKey, -keepN, -1);
          await redis.del(histKey);
          if (lastN.length) await redis.rpush(histKey, ...lastN);
        }
      }
    }

    return reply.send({ reply: modelReply });
  } catch (err) {
    req.log.error(err);
    return reply.code(500).send({ error: "Internal Server Error" });
  }
});

// ---------- BOOT ----------
app.listen({ port: Number(PORT), host: HOST }, (err, address) => {
  if (err) {
    app.log.error(err);
    process.exit(1);
  }
  app.log.info(`undrawn Core listening at ${address}`);
});
