// src/helpers/chatStream.ts
import { randomUUID } from "crypto";
import type { FastifyReply } from "fastify";
import type { Redis } from "@upstash/redis";
import { persistHistory, type PreparedContext } from "./chatCommon";

type Env = {
  ANTHROPIC_API_KEY: string;
  TIMEOUT_MS?: string | number;
};

export async function handleStreamSSE(
  env: Env,
  reply: FastifyReply,
  redis: Redis,
  ctx: PreparedContext,
  body: any
) {
   console.log("STREAM MODE")
  // SSE headers
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

  // Early empty chunk to kick UI stream
  try {
    reply.raw.write(
      `data: ${JSON.stringify({
        id: "chatcmpl_" + randomUUID(),
        object: "chat.completion.chunk",
        model: ctx.anthModel,
        created: Math.floor(Date.now() / 1000),
        choices: [{ index: 0, delta: { content: "" }, finish_reason: null }],
      })}\n\n`
    );
  } catch {}

  let upstream: Response | null = null;
  let accText = "";

  try {
    upstream = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: ctx.anthModel,
        system: ctx.system,
        messages: ctx.mappedMessages.length
          ? ctx.mappedMessages
          : [{ role: "user", content: [{ type: "text", text: "" }] }],
        max_tokens: Number(body?.max_tokens),
        temperature: Number(body?.temperature),
        stream: true,
      }),
    });

    if (!upstream.ok || !upstream.body) {
      const detail =
        upstream && !upstream.ok ? await upstream.text().catch(() => "") : "no body";
      reply.raw.write(
        `data: ${JSON.stringify({ error: { message: `Anthropic stream error: ${detail}` } })}\n\n`
      );
      reply.raw.write("data: [DONE]\n\n");
      reply.raw.end();
      return;
    }

    const decoder = new TextDecoder();
    const reader = upstream.body.getReader();
    let buffer = "";

    const sendChunk = (text: string) => {
      const chunk = {
        id: "chatcmpl_" + randomUUID(),
        object: "chat.completion.chunk",
        model: ctx.anthModel,
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
          await persistHistory(redis, ctx.histKey, ctx.sumKey, ctx.ttl, ctx.lastUserText, accText);
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
}
