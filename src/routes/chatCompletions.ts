// src/routes/chatCompletions.ts
import type { FastifyInstance } from "fastify";
import { loadEnv } from "../config/env";
import { getRedis } from "../services/redis";
import { prepareContext } from "../helpers/chatCommon";
import { handleNonStream } from "../helpers/chatNonStream";
// (kept for future) import { handleStreamSSE } from "../helpers/chatStream";

export default async function chatCompletionsRoutes(app: FastifyInstance) {
  const env = loadEnv();
  const redis = getRedis(env);
  console.log("chatCompletionsRoutes initialized");
  app.post("/v1/chat/completions", async (req, reply) => {
    const body = (req.body as any) || {};

    // 🔒 Force non-stream mode regardless of client input
    // const useStream = Boolean(body?.stream); // old behavior
    const useStream = false; // <- always non-stream

    const ctx = await prepareContext(env as any, req, body, redis);

    if (useStream) {
      // not used for now, but kept for future needs:
      // return handleStreamSSE(env as any, reply, redis, ctx, body);
    }

    return handleNonStream(env as any, reply, redis, ctx, body);
  });
}
