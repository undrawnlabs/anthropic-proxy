// index.js
import Fastify from "fastify";
import cors from "@fastify/cors";
import Redis from "ioredis";
import fetch from "node-fetch";

// ---------- ENV ----------
const {
  REDIS_URL,
  ANTHROPIC_API_KEY,
  PORT = 3000,
  HOST = "0.0.0.0",

  // Tunables (safe defaults)
  ANTHROPIC_MODEL = "claude-3-opus-20240229",
  SUMMARIZER_MODEL = "claude-3-haiku-20240307",
  MAX_OUTPUT_TOKENS = "800",
  TEMPERATURE = "0.2",
  TIMEOUT_MS = "20000",
  HISTORY_KEEP = "20",            // how many recent messages to send to Claude
  SUMMARIZE_THRESHOLD = "200",    // when total messages exceed this, auto-summarize older part
  SUMMARY_MAX_TOKENS = "400"
} = process.env;

if (!REDIS_URL) throw new Error("Missing REDIS_URL");
if (!ANTHROPIC_API_KEY) throw new Error("Missing ANTHROPIC_API_KEY");

// ---------- SERVER ----------
const app = Fastify({ logger: true });
app.register(cors, { origin: true });

// ---------- REDIS ----------
const redis = new Redis(REDIS_URL, { tls: { rejectUnauthorized: false } });

// ---------- HELPERS ----------
function parseNumber(v, def) { const n = Number(v); return Number.isFinite(n) ? n : def; }

async function callAnthropic({ model, system, messages, maxTokens, temperature, timeoutMs }) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  const payload = {
    model,
    system,
    messages,
    max_tokens: maxTokens,
    temperature
  };
  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`Anthropic HTTP ${res.status}: ${text}`);
    }
    return await res.json();
  } catch (err) {
    // single retry (cold starts / transient network)
    try {
      const res2 = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "content-type": "application/json"
        },
        body: JSON.stringify({
          model, system, messages,
          max_tokens: maxTokens,
          temperature
        })
      });
      if (!res2.ok) {
        const text = await res2.text().catch(() => "");
        throw new Error(`Anthropic retry HTTP ${res2.status}: ${text}`);
      }
      return await res2.json();
    } finally {
      clearTimeout(t);
    }
  } finally {
    clearTimeout(t);
  }
}

function buildSystemPrompt(locale) {
  return [
    "You are undrawn Core.",
    "Use the provided memory summary and the recent message history as ground truth.",
    "Do not re-ask facts unless they conflict.",
    `Reply in the user's language (locale hint: ${locale || "auto"}).`
  ].join(" ");
}

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

    // 1) Load compact recent history (economy)
    const keepN = parseNumber(HISTORY_KEEP, 20);
    const recent = await redis.lrange(histKey, -keepN, -1);
    const recentMsgs = recent.map(x => JSON.parse(x)); // [{role, content}, ...]

    // 2) Load stored summary (long-term memory)
    const summary = await redis.get(sumKey);
    const messages = [];

    if (summary) {
      messages.push({
        role: "system",
        content: `Memory Summary (treat as true context): ${summary}`
      });
    }

    // 3) Append recent dialogue and the new user message
    if (recentMsgs.length) messages.push(...recentMsgs);
    messages.push({ role: "user", content: prompt });

    // 4) Call Claude for main reply
    const main = await callAnthropic({
      model: ANTHROPIC_MODEL,
      system: buildSystemPrompt(locale),
      messages,
      maxTokens: parseNumber(MAX_OUTPUT_TOKENS, 800),
      temperature: parseNumber(TEMPERATURE, 0.2),
      timeoutMs: parseNumber(TIMEOUT_MS, 20000)
    });

    const modelReply = main?.content?.[0]?.text;
    if (!modelReply) throw new Error("Invalid response from Claude");

    // 5) Persist messages (append)
    await redis.rpush(histKey, JSON.stringify({ role: "user", content: prompt }));
    await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: modelReply }));

    // 6) Expire keys (7 days rolling)
    const ttl = 60 * 60 * 24 * 7;
    await redis.expire(histKey, ttl);
    await redis.expire(sumKey, ttl);

    // 7) Auto-summarize older part when the thread grows
    const totalLen = await redis.llen(histKey);
    const threshold = parseNumber(SUMMARIZE_THRESHOLD, 200);

    if (totalLen > threshold) {
      // summarize ONLY the older slice; keep the most recent keepN untouched
      const olderSlice = await redis.lrange(histKey, 0, -keepN - 1); // everything except the last keepN
      if (olderSlice.length > 0) {
        const olderPlain = olderSlice.map(x => JSON.parse(x)); // array of {role, content}

        const sum = await callAnthropic({
          model: SUMMARIZER_MODEL,
          system: "Summarize the conversation into concise, durable facts and instructions. Preserve key decisions, preferences, identities, dates, and constraints.",
          messages: [
            { role: "user", content: JSON.stringify(olderPlain) }
          ],
          maxTokens: parseNumber(SUMMARY_MAX_TOKENS, 400),
          temperature: 0,
          timeoutMs: parseNumber(TIMEOUT_MS, 20000)
        });

        const newSummary = sum?.content?.[0]?.text || "";
        if (newSummary) {
          await redis.set(sumKey, newSummary);
          // hard-trim history to last keepN after summarizing
          const lastN = await redis.lrange(histKey, -keepN, -1);
          await redis.del(histKey);
          if (lastN.length) await redis.rpush(histKey, ...lastN);
        }
      }
    }

    // 8) Return API shape expected by Builder
    return reply.send({ reply: modelReply });

  } catch (err) {
    req.log.error(err);
    return reply.code(500).send({ error: "Internal Server Error" });
  }
});

app.addHook("onClose", async () => {
  try { await redis.quit(); } catch {}
});

// ---------- BOOT ----------
app.listen({ port: Number(PORT), host: HOST }, (err, address) => {
  if (err) {
    app.log.error(err);
    process.exit(1);
  }
  app.log.info(`undrawn Core listening at ${address}`);
});
