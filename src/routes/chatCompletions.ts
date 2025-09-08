// routes/chatCompletions.ts
import type { FastifyInstance } from "fastify";
import crypto from "crypto"; // needed for crypto.randomUUID()
import { loadEnv } from "../config/env";
import { getRedis } from "../services/redis";
import { mapOpenAIToAnthropic } from "../mappers/openaiToAnthropic";
import { buildSystem } from "../config/buildSystem";
import { callAnthropic } from "../services/anthropic";
import { resolveModelAlias } from "../utils/modelAlias";

export default async function chatCompletionsRoutes(app: FastifyInstance) {
  const env = loadEnv();
  const redis = getRedis(env);

  app.post("/v1/chat/completions", async (req, reply) => {
    const body = (req.body as any) || {};
    const {
      model = env.ANTHROPIC_MODEL,
      messages = [],
      temperature,
      max_tokens,
      stream
    } = body;

    const anthModel = resolveModelAlias(model, env.ANTHROPIC_MODEL);
    const session_id =
      (req.headers["x-session-id"] as string) ||
      "sess_" + (req.headers.authorization || req.ip || "anon");
    const core_id = "webui";
    const histKey = `hist:${core_id}:${session_id}`;
    const sumKey = `sum:${core_id}:${session_id}`;

    // map OpenAI messages → Anthropic format
    const { system: systemExtra, messages: mapped } = await mapOpenAIToAnthropic(
      messages as any[]
    );
    const summary = await redis.get(sumKey);
    const system = [
      buildSystem((req.headers["x-locale"] as string) || "auto", summary, env.CORE_SYSTEM_PROMPT),
      systemExtra
    ]
      .filter(Boolean)
      .join("\n\n");

    // --------------------- STREAM (SSE) ---------------------
    if (stream) {
      // 1) send SSE headers immediately to keep connection open
      reply.raw.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        // helpful behind some proxies (not strictly required)
        "X-Accel-Buffering": "no"
      });

      // 2) timeout + heartbeat (to avoid CF 524)
      const controller = new AbortController();
      const hardTimeout = Math.min(Number(env.TIMEOUT_MS) || 30000, 60000); // < 100s for CF
      const timer = setTimeout(() => controller.abort(), hardTimeout);
      const ping = setInterval(() => {
        try { reply.raw.write(": ping\n\n"); } catch {}
      }, 15000);

      // helpful: send a minimal first chunk so the UI renders the stream instantly
      const startChunk = {
        id: "chatcmpl_" + crypto.randomUUID(),
        object: "chat.completion.chunk",
        model: anthModel,
        created: Math.floor(Date.now() / 1000),
        choices: [{ index: 0, delta: { content: "" }, finish_reason: null }]
      };
      try { reply.raw.write(`data: ${JSON.stringify(startChunk)}\n\n`); } catch {}

      let r: Response | null = null;
      try {
        // 3) call Anthropic with stream: true
        r = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          signal: controller.signal,
          headers: {
            "x-api-key": env.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
          },
          body: JSON.stringify({
            model: anthModel,
            system,
            messages: mapped.length
              ? mapped
              : [{ role: "user", content: [{ type: "text", text: "" }] }],
            max_tokens: Number(max_tokens ?? env.MAX_OUTPUT_TOKENS),
            temperature: Number(temperature ?? env.TEMPERATURE),
            stream: true
          })
        });

        if (!r.ok || !r.body) {
          const detail = r && !r.ok ? await r.text().catch(() => "") : "no body";
          reply.raw.write(
            `data: ${JSON.stringify({ error: { message: `Anthropic stream error: ${detail}` } })}\n\n`
          );
          reply.raw.write("data: [DONE]\n\n");
          reply.raw.end();
          return;
        }

        // 4) stream Anthropic events → OpenAI-like chunks
        const decoder = new TextDecoder();
        const reader = r.body.getReader();
        let buffer = "";
        let accText = "";

        const sendChunk = (text: string) => {
          const chunk = {
            id: "chatcmpl_" + crypto.randomUUID(),
            object: "chat.completion.chunk",
            model: anthModel,
            created: Math.floor(Date.now() / 1000),
            choices: [{ index: 0, delta: { content: text }, finish_reason: null }]
          };
          reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";

          for (const p of parts) {
            const line = p.trim();
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim();
            if (!data || data === "[DONE]") continue;
            try {
              const evt = JSON.parse(data);
              if (evt.type === "content_block_delta" && evt.delta?.text) {
                accText += evt.delta.text;
                sendChunk(evt.delta.text);
              }
              if (evt.type === "message_stop") {
                // persist conversation (text-only is enough for memory)
                const lastUser = (messages as any[]).filter(m => m.role === "user").pop();
                const userText =
                  typeof lastUser?.content === "string"
                    ? lastUser.content
                    : Array.isArray(lastUser?.content)
                    ? (lastUser.content.find((x: any) => x.type === "text")?.text || "")
                    : "";

                if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }));
                if (accText) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: accText }));
                await redis.expire(histKey, env.TTL_SECONDS);
                await redis.expire(sumKey, env.TTL_SECONDS);

                reply.raw.write("data: [DONE]\n\n");
                reply.raw.end();
              }
            } catch { /* ignore parse noise */ }
          }
        }
      } catch (err: any) {
        const msg = err?.name === "AbortError" ? "upstream timeout" : String(err?.message || err);
        try {
          reply.raw.write(`data: ${JSON.stringify({ error: { message: msg } })}\n\n`);
          reply.raw.write("data: [DONE]\n\n");
          reply.raw.end();
        } catch {}
      } finally {
        clearTimeout(timer);
        clearInterval(ping);
      }
      return; // stream branch finished
    }

    // --------------------- NON-STREAM ---------------------
    const res = await callAnthropic(
      env,
      {
        model: anthModel,
        system,
        messages: mapped.length
          ? mapped
          : [{ role: "user", content: [{ type: "text", text: "" }] }],
        max_tokens: Number(max_tokens ?? env.MAX_OUTPUT_TOKENS),
        temperature: Number(temperature ?? env.TEMPERATURE)
      },
      env.TIMEOUT_MS
    );

    const text = res?.content?.[0]?.text || "";
    const lastUser = (messages as any[]).filter(m => m.role === "user").pop();
    const userText =
      typeof lastUser?.content === "string"
        ? lastUser.content
        : Array.isArray(lastUser?.content)
        ? (lastUser.content.find((x: any) => x.type === "text")?.text || "")
        : "";

    if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }));
    if (text) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: text }));
    await redis.expire(histKey, env.TTL_SECONDS);
    await redis.expire(sumKey, env.TTL_SECONDS);

    return reply.send({
      id: "chatcmpl_" + crypto.randomUUID(),
      object: "chat.completion",
      model: anthModel,
      created: Math.floor(Date.now() / 1000),
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
      usage: {
        prompt_tokens: userText?.length || 0,
        completion_tokens: text.length,
        total_tokens: (userText?.length || 0) + text.length
      }
    });
  });
}
