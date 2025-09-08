import type { FastifyInstance } from "fastify"
import { loadEnv } from "../config/env"
import { getRedis } from "../services/redis"
import { mapOpenAIToAnthropic } from "../mappers/openaiToAnthropic"
import { buildSystem } from "../config/buildSystem"
import { callAnthropic } from "../services/anthropic"

export default async function chatCompletionsRoutes(app: FastifyInstance) {
  const env = loadEnv()
  const redis = getRedis(env)

  app.post("/v1/chat/completions", async (req, reply) => {
    const { model = env.ANTHROPIC_MODEL, messages = [], temperature, max_tokens, stream } = (req.body as any) || {}
    const session_id = (req.headers["x-session-id"] as string) || "sess_" + (req.headers.authorization || req.ip || "anon")
    const core_id = "webui"
    const histKey = `hist:${core_id}:${session_id}`
    const sumKey  = `sum:${core_id}:${session_id}`

    const { system: systemExtra, messages: mapped } = await mapOpenAIToAnthropic(messages as any[])
    const summary = await redis.get(sumKey)
    const system = [buildSystem((req.headers["x-locale"] as string) || "auto", summary, env.CORE_SYSTEM_PROMPT), systemExtra].filter(Boolean).join("\n\n")

    if (stream) {
      reply.raw.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" })
      const r = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "x-api-key": env.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
        body: JSON.stringify({
          model,
          system,
          messages: mapped.length ? mapped : [{ role: "user", content: [{ type: "text", text: "" }] }],
          max_tokens: Number(max_tokens ?? env.MAX_OUTPUT_TOKENS),
          temperature: Number(temperature ?? env.TEMPERATURE),
          stream: true
        })
      })
      if (!r.ok || !r.body) {
        const detail = r && !r.ok ? await r.text().catch(() => "") : "no body"
        reply.code(500).send({ error: { message: `Anthropic stream error: ${detail}` } })
        return
      }

      const decoder = new TextDecoder()
      const reader = r.body.getReader()
      let buffer = "", accText = ""

      const send = (obj: any) => reply.raw.write(`data: ${JSON.stringify(obj)}\n\n`)
      const base = () => ({
        id: "chatcmpl_" + crypto.randomUUID(),
        object: "chat.completion.chunk",
        model,
        created: Math.floor(Date.now()/1000),
        choices: [{ index: 0, delta: {}, finish_reason: null }]
      })

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const parts = buffer.split("\n\n"); buffer = parts.pop() || ""

        for (const p of parts) {
          const line = p.trim()
          if (!line.startsWith("data:")) continue
          const data = line.slice(5).trim()
          if (!data || data === "[DONE]") continue
          try {
            const evt = JSON.parse(data)
            if (evt.type === "content_block_delta" && evt.delta?.text) {
              accText += evt.delta.text
              const chunk = base()
              chunk.choices[0].delta = { content: evt.delta.text }
              send(chunk)
            }
            if (evt.type === "message_stop") {
              const lastUser = (messages as any[]).filter(m => m.role === "user").pop()
              const userText = typeof lastUser?.content === "string"
                ? lastUser.content
                : (Array.isArray(lastUser?.content) ? (lastUser.content.find((x:any)=>x.type === "text")?.text || "") : "")
              if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }))
              if (accText) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: accText }))
              await redis.expire(histKey, env.TTL_SECONDS)
              await redis.expire(sumKey,  env.TTL_SECONDS)

              reply.raw.write("data: [DONE]\n\n")
              reply.raw.end()
            }
          } catch {}
        }
      }
      return
    }

    const res = await callAnthropic(env, {
      model,
      system,
      messages: mapped.length ? mapped : [{ role: "user", content: [{ type: "text", text: "" }] }],
      max_tokens: Number(max_tokens ?? env.MAX_OUTPUT_TOKENS),
      temperature: Number(temperature ?? env.TEMPERATURE)
    }, env.TIMEOUT_MS)

    const text = res?.content?.[0]?.text || ""
    const lastUser = (messages as any[]).filter(m => m.role === "user").pop()
    const userText = typeof lastUser?.content === "string"
      ? lastUser.content
      : (Array.isArray(lastUser?.content) ? (lastUser.content.find((x:any)=>x.type === "text")?.text || "") : "")

    if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }))
    if (text) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: text }))
    await redis.expire(histKey, env.TTL_SECONDS)
    await redis.expire(sumKey,  env.TTL_SECONDS)

    return reply.send({
      id: "chatcmpl_" + crypto.randomUUID(),
      object: "chat.completion",
      model,
      created: Math.floor(Date.now() / 1000),
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
      usage: { prompt_tokens: userText?.length || 0, completion_tokens: text.length, total_tokens: (userText?.length || 0) + text.length }
    })
  })
}
