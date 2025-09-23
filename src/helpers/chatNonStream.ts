// src/helpers/chatNonStream.ts
import { randomUUID } from "crypto";
import type { FastifyReply } from "fastify";
import type { Redis } from "@upstash/redis";
import { callAnthropic } from "../services/anthropic";
import { persistHistory, type PreparedContext } from "./chatCommon";
import { webSearch, formatWebContext } from "../services/webSearch";

type Env = {
  MAX_OUTPUT_TOKENS?: string | number;
  TEMPERATURE?: string | number;
  TIMEOUT_MS?: string | number;
  TAVILY_API_KEY?: string;
};

export async function handleNonStream(
  env: Env,
  reply: FastifyReply,
  redis: Redis,
  ctx: PreparedContext,
  body: any
) {
  console.log("NON STREAM MODE");

  // ---- Always-on Tavily web search ----
  let augmentedSystem = ctx.system;
  if (env.TAVILY_API_KEY) {
    try {
      // Query: explicit body.web_search.query OR last user text
      const query =
        body?.web_search?.query?.trim?.() || ctx.lastUserText || "";
      console.log("TAB", query)
      console.log({query})
      if (query) {
        const results = await webSearch(env.TAVILY_API_KEY, {
          query,
          max_results: body?.web_search?.max_results ?? 5,
          include_domains: body?.web_search?.include_domains,
          exclude_domains: body?.web_search?.exclude_domains,
          search_depth: body?.web_search?.search_depth ?? "basic",
        });

        const webCtx = formatWebContext(results, query);
        augmentedSystem = [augmentedSystem, webCtx]
          .filter(Boolean)
          .join("\n\n");
      }
    } catch (err) {
      console.warn("web_search failed:", err);
    }
  }

  // ---- Call Anthropic (non-stream) ----
  const res = await callAnthropic(
    env as any,
    {
      model: ctx.anthModel,
      system: augmentedSystem,
      messages: ctx.mappedMessages.length
        ? ctx.mappedMessages
        : [{ role: "user", content: [{ type: "text", text: "" }] }],
      max_tokens: Number(body?.max_tokens ?? env.MAX_OUTPUT_TOKENS),
      temperature: Number(body?.temperature ?? env.TEMPERATURE),
    },
    env.TIMEOUT_MS
  );

  const text = res?.content?.[0]?.text || "";

  await persistHistory(
    redis,
    ctx.histKey,
    ctx.sumKey,
    ctx.ttl,
    ctx.lastUserText,
    text
  );

  return reply.send({
    id: "chatcmpl_" + randomUUID(),
    object: "chat.completion",
    model: ctx.anthModel,
    created: Math.floor(Date.now() / 1000),
    choices: [
      { index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" },
    ],
    usage: {
      prompt_tokens: ctx.lastUserText?.length || 0,
      completion_tokens: text.length,
      total_tokens: (ctx.lastUserText?.length || 0) + text.length,
    },
  });
}
