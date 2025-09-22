// routes/chatCompletions.ts
import type { FastifyInstance } from "fastify";
import { randomUUID } from "crypto";
import { loadEnv } from "../config/env";
import { getRedis } from "../services/redis";
import { mapOpenAIToAnthropic } from "../mappers/openaiToAnthropic";
import { buildSystem } from "../config/buildSystem";
import { callAnthropic } from "../services/anthropic";
import { resolveModelAlias } from "../utils/modelAlias";
import { AnthropicTools } from "../tools/schema";
import { executeTool } from "../tools/executeTool";

type AnthMsg = { role: "user" | "assistant"; content: any[] };

export default async function chatCompletionsRoutes(app: FastifyInstance) {
  const env = loadEnv();
  const redis = getRedis(env);

  app.post("/v1/chat/completions", async (req, reply) => {
    const body = (req.body as any) || {};
    let {
      model = env.ANTHROPIC_MODEL,
      messages = [],
      temperature,
      max_tokens,
      stream,
    } = body;

    const anthModel = resolveModelAlias(model, env.ANTHROPIC_MODEL);
    const toolsEnabled = process.env.WEB_SEARCH_ENABLED === "true";
    const tools = toolsEnabled ? AnthropicTools : undefined;

    const session_id =
      (req.headers["x-session-id"] as string) ||
      "sess_" + (req.headers.authorization || req.ip || "anon");
    const core_id = "webui";
    const histKey = `hist:${core_id}:${session_id}`;
    const sumKey = `sum:${core_id}:${session_id}`;
    const ttl = Number(env.TTL_SECONDS ?? 604800);

    const { system: systemExtra, messages: mapped } = await mapOpenAIToAnthropic(
      messages as any[]
    );
    const summary = await redis.get(sumKey);
    const system = [
      buildSystem((req.headers["x-locale"] as string) || "auto", summary, env.CORE_SYSTEM_PROMPT),
      systemExtra,
    ]
      .filter(Boolean)
      .join("\n\n");

    // Если включены инструменты — используем non-stream (интерактивные tool_use/tool_result циклы)
    if (stream && toolsEnabled) {
      stream = false;
    }

    if (stream) {
      reply.raw.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      });

      const controller = new AbortController();
      const hardTimeout = Math.min(Number(env.TIMEOUT_MS) || 30000, 60000);
      const timer = setTimeout(() => controller.abort(), hardTimeout);
      const ping = setInterval(() => {
        try {
          reply.raw.write(": ping\n\n");
        } catch {}
      }, 15000);

      try {
        reply.raw.write(
          `data: ${JSON.stringify({
            id: "chatcmpl_" + randomUUID(),
            object: "chat.completion.chunk",
            model: anthModel,
            created: Math.floor(Date.now() / 1000),
            choices: [{ index: 0, delta: { content: "" }, finish_reason: null }],
          })}\n\n`
        );
      } catch {}

      try {
        const upstream = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          signal: controller.signal,
          headers: {
            "x-api-key": env.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
          },
          body: JSON.stringify({
            model: anthModel,
            system,
            messages: mapped.length
              ? mapped
              : [{ role: "user", content: [{ type: "text", text: "" }] }],
            max_tokens: Number(max_tokens ?? env.MAX_OUTPUT_TOKENS),
            temperature: Number(temperature ?? env.TEMPERATURE),
            stream: true,
          }),
        });

        if (!upstream.ok || !upstream.body) {
          const detail = upstream && !upstream.ok ? await upstream.text().catch(() => "") : "no body";
          reply.raw.write(`data: ${JSON.stringify({ error: { message: `Anthropic stream error: ${detail}` } })}\n\n`);
          reply.raw.write("data: [DONE]\n\n");
          reply.raw.end();
          clearTimeout(timer);
          clearInterval(ping);
          return;
        }

        const decoder = new TextDecoder();
        const reader = upstream.body.getReader();
        let buffer = "";
        let accText = "";

        const sendChunk = (text: string) => {
          const chunk = {
            id: "chatcmpl_" + randomUUID(),
            object: "chat.completion.chunk",
            model: anthModel,
            created: Math.floor(Date.now() / 1000),
            choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
          };
          reply.raw.write(`data: ${JSON.stringify(chunk)}\n\n`);
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";

          for (const blk of blocks) {
            const m = blk.match(/^data:\s*(.+)$/m);
            if (!m) continue;

            const payload = m[1].trim();
            if (!payload || payload === "[DONE]") continue;

            let evt: any;
            try {
              evt = JSON.parse(payload);
            } catch {
              continue;
            }

            if (evt.type === "content_block_delta" && evt.delta?.text) {
              accText += evt.delta.text;
              sendChunk(evt.delta.text);
            }

            if (evt.type === "message_stop") {
              const lastUser = (messages as any[]).filter((m) => m.role === "user").pop();
              const userText =
                typeof lastUser?.content === "string"
                  ? lastUser.content
                  : Array.isArray(lastUser?.content)
                  ? (lastUser.content.find((x: any) => x.type === "text")?.text || "")
                  : "";

              if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }));
              if (accText) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: accText }));
              if (ttl > 0) {
                await redis.expire(histKey, ttl);
                await redis.expire(sumKey, ttl);
              }

              reply.raw.write("data: [DONE]\n\n");
              reply.raw.end();
            }
          }
        }
      } catch (e: any) {
        const msg = e?.name === "AbortError" ? "upstream timeout" : String(e?.message || e);
        try {
          reply.raw.write(`data: ${JSON.stringify({ error: { message: msg } })}\n\n`);
          reply.raw.write("data: [DONE]\n\n");
          reply.raw.end();
        } catch {}
      } finally {
        clearTimeout(timer);
        clearInterval(ping);
      }

      return;
    }

    // ---------- NON-STREAM с поддержкой tools (web search) ----------
    reply.type("application/json; charset=utf-8");

    let convo: AnthMsg[] = (mapped.length
      ? mapped
      : [{ role: "user", content: [{ type: "text", text: "" }] }]) as AnthMsg[];

    let finalText = "";
    const maxToolIterations = 6;

    for (let i = 0; i < maxToolIterations; i++) {
      const res = await callAnthropic(
        undefined,
        {
          model: anthModel,
          system,
          messages: convo,
          max_tokens: Number(max_tokens ?? env.MAX_OUTPUT_TOKENS ?? 1024),
          temperature: Number(temperature ?? env.TEMPERATURE ?? 0.2),
          ...(tools ? { tools, tool_choice: { type: "auto" } } : {}),
        },
        Number(env.TIMEOUT_MS ?? 30000)
      );

      const blocks = res?.content ?? [];
      const toolCalls: Array<{ id: string; name: string; input: any }> = [];
      let textBuf = "";

      for (const b of blocks) {
        if (b.type === "tool_use") {
          toolCalls.push({ id: b.id, name: b.name, input: b.input });
        } else if (b.type === "text" && b.text) {
          textBuf += b.text;
        }
      }

      if (toolCalls.length && toolsEnabled) {
        convo.push({ role: "assistant", content: blocks });

        for (const tc of toolCalls) {
          let result: any;
          let is_error = false;
          try {
            result = await executeTool(tc.name, tc.input);
          } catch (e: any) {
            result = { error: String(e?.message || e) };
            is_error = true;
          }

          convo.push({
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: tc.id,
                content: typeof result === "string" ? result : JSON.stringify(result),
                is_error,
              },
            ],
          });
        }

        continue;
      }

      finalText = textBuf;
      break;
    }

    const text = finalText || "";

    const lastUser = (messages as any[]).filter((m) => m.role === "user").pop();
    const userText =
      typeof lastUser?.content === "string"
        ? lastUser.content
        : Array.isArray(lastUser?.content)
        ? (lastUser.content.find((x: any) => x.type === "text")?.text || "")
        : "";

    if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }));
    if (text) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: text }));
    if (ttl > 0) {
      await redis.expire(histKey, ttl);
      await redis.expire(sumKey, ttl);
    }

    return reply.send({
      id: "chatcmpl_" + randomUUID(),
      object: "chat.completion",
      model: anthModel,
      created: Math.floor(Date.now() / 1000),
      choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
      usage: {
        prompt_tokens: userText?.length || 0,
        completion_tokens: text.length,
        total_tokens: (userText?.length || 0) + text.length,
      },
    });
  });
}
