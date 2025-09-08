import type { FastifyInstance } from "fastify"
import { recordLatency, metrics } from "../config/metrics"
import { getRedis } from "../services/redis"
import { rateLimitMinute } from "../services/rateLimit"
import { buildSystem } from "../config/buildSystem"
import { callAnthropic } from "../services/anthropic"
import { loadEnv } from "../config/env"
import { sanitize } from "../utils/sanitize"

function jsonReply(text: string) {
  return {
    id: crypto.randomUUID(),
    object: "chat.completion",
    model: "hanna-core",
    created: Math.floor(Date.now() / 1000),
    choices: [{ index: 0, message: { role: "assistant", content: [{ type: "text", text }] }, finish_reason: "stop" }],
    usage: { prompt_tokens: 0, completion_tokens: text.length, total_tokens: text.length }
  }
}

export default async function completeRoutes(app: FastifyInstance) {
  const env = loadEnv()
  const redis = getRedis(env)

  app.post("/v1/complete", async (req, reply) => {
    const t0 = Date.now(); metrics.calls_total++
    const ct = String(req.headers["content-type"] || "").toLowerCase()
    if (!ct.includes("application/json")) {
      const ms = Date.now() - t0; recordLatency(ms)
      reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, reason: "non-json" }))
      return reply.code(200).send(jsonReply("CORE error: JSON only (use content-type: application/json)"))
    }
    try {
      if (await rateLimitMinute(redis, req.ip, env.RL_PER_MIN)) {
        const ms = Date.now() - t0; recordLatency(ms)
        reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, rate_limited: true }))
        return reply.code(200).send(jsonReply("CORE error: rate limit exceeded, try again later"))
      }

      const body: any = typeof req.body === "string" ? JSON.parse(req.body) : req.body
      if (!body || typeof body !== "object") {
        const ms = Date.now() - t0; recordLatency(ms)
        reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, reason: "empty body" }))
        return reply.code(200).send(jsonReply("CORE error: invalid or empty JSON body"))
      }

      const { core_id, session_id, prompt, locale } = body
      if (!core_id || !session_id || !prompt) {
        const ms = Date.now() - t0; recordLatency(ms)
        reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, reason: "missing fields" }))
        return reply.code(200).send(jsonReply("CORE error: missing core_id, session_id or prompt"))
      }

      const histKey = `hist:${core_id}:${session_id}`
      const sumKey  = `sum:${core_id}:${session_id}`

      const keepN = env.HISTORY_KEEP
      const safeParse = (v: string) => { try { return JSON.parse(v) } catch { return null } }

      const recent = await redis.lrange(histKey, -keepN, -1)
      const recentMsgs = (recent || []).map(safeParse).filter(Boolean)

      const summary = await redis.get(sumKey)
      const cap = env.MAX_INPUT_CHARS
      let cleanPrompt = String(prompt).replace(/\r/g,"").replace(/\t/g," ").replace(/ {2,}/g," ").trim()
      if (cap > 0 && cleanPrompt.length > cap) cleanPrompt = cleanPrompt.slice(0, cap)

      const messages = [...recentMsgs, { role: "user", content: cleanPrompt }]

      const res = await callAnthropic(env, {
        model: env.ANTHROPIC_MODEL,
        system: buildSystem(locale, summary, env.CORE_SYSTEM_PROMPT),
        messages,
        max_tokens: env.MAX_OUTPUT_TOKENS,
        temperature: env.TEMPERATURE
      }, env.TIMEOUT_MS)

      const modelReply = res?.content?.[0]?.text || ""
      if (!modelReply) throw new Error("Invalid Claude response")

      await redis.rpush(histKey, JSON.stringify({ role: "user", content: cleanPrompt }))
      await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: modelReply }))
      await redis.expire(histKey, env.TTL_SECONDS)
      await redis.expire(sumKey,  env.TTL_SECONDS)

      const total = await redis.llen(histKey)
      if (total > env.SUMMARIZE_THRESHOLD) {
        const older = await redis.lrange(histKey, 0, -keepN - 1)
        if (older.length) {
          const olderPlain = (older || []).map(safeParse).filter(Boolean)
          const sum = await callAnthropic(env, {
            model: env.SUMMARIZER_MODEL,
            system: "Summarize into compact, durable facts: identities, decisions, preferences, constraints, dates.",
            messages: [{ role: "user", content: JSON.stringify(olderPlain) }],
            max_tokens: env.SUMMARY_MAX_TOKENS,
            temperature: 0
          }, 8000)
          const text = sum?.content?.[0]?.text || ""
          if (text) {
            await redis.set(sumKey, sanitize(text))
            const lastN = await redis.lrange(histKey, -keepN, -1)
            await redis.del(histKey)
            if (lastN.length) await redis.rpush(histKey, ...lastN)
          }
        }
      }

      const ms = Date.now() - t0; recordLatency(ms)
      reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, has_summary: Boolean(summary), history_count: recentMsgs.length }).slice(0,256))

      return reply.code(200).send({
        id: "chatcmpl_" + crypto.randomUUID(),
        object: "chat.completion",
        model: env.ANTHROPIC_MODEL,
        created: Math.floor(Date.now() / 1000),
        choices: [{ index: 0, message: { role: "assistant", content: modelReply }, finish_reason: "stop" }],
        usage: { prompt_tokens: prompt.length, completion_tokens: modelReply.length, total_tokens: prompt.length + modelReply.length }
      })
    } catch (e: any) {
      metrics.errors_total++
      const ms = Date.now() - t0; recordLatency(ms)
      req.log.error({ err: String(e?.message || e) }, "core-fail")
      reply.header("X-Core-Diag", JSON.stringify({ ms, req_id: req.id, fail: true }).slice(0,256))
      return reply.code(200).send(jsonReply(`CORE error: ${String(e?.message || e)}`))
    }
  })
}
