// src/helpers/chatCommon.ts
import type { FastifyRequest } from "fastify";
import { buildSystem } from "../config/buildSystem";
import { resolveModelAlias } from "../utils/modelAlias";
import { mapOpenAIToAnthropic } from "../mappers/openaiToAnthropic";
import type { Redis } from "@upstash/redis";


type Env = {
  ANTHROPIC_MODEL: string;
  CORE_SYSTEM_PROMPT?: string;
  TTL_SECONDS?: string | number;
  MAX_OUTPUT_TOKENS?: string | number;
  TEMPERATURE?: string | number;
};

export type PreparedContext = {
  anthModel: string;
  system: string;
  mappedMessages: any[];
  histKey: string;
  sumKey: string;
  ttl: number;
  lastUserText: string;
};

export async function prepareContext(
  env: Env,
  req: FastifyRequest,
  body: any,
  redis: Redis
): Promise<PreparedContext> {
  const {
    model = env.ANTHROPIC_MODEL,
    messages = [],
    temperature,
    max_tokens,
  } = body || {};

  const anthModel = resolveModelAlias(model, env.ANTHROPIC_MODEL);

  const session_id =
    (req.headers["x-session-id"] as string) ||
    "sess_" + (req.headers.authorization || req.ip || "anon");
  const core_id = "webui";
  const histKey = `hist:${core_id}:${session_id}`;
  const sumKey = `sum:${core_id}:${session_id}`;
  const ttl = Number(env.TTL_SECONDS ?? 604800);

  const { system: systemExtra, messages: mappedMessages } =
    await mapOpenAIToAnthropic(messages as any[]);

  const summary = await redis.get(sumKey);
  const system = [
    buildSystem((req.headers["x-locale"] as string) || "auto", summary, env.CORE_SYSTEM_PROMPT),
    systemExtra,
  ]
    .filter(Boolean)
    .join("\n\n");

  const lastUser = (messages as any[]).filter((m) => m.role === "user").pop();
  const lastUserText =
    typeof lastUser?.content === "string"
      ? lastUser.content
      : Array.isArray(lastUser?.content)
      ? (lastUser.content.find((x: any) => x.type === "text")?.text || "")
      : "";

  return {
    anthModel,
    system,
    mappedMessages,
    histKey,
    sumKey,
    ttl,
    lastUserText,
  };
}

export async function persistHistory(
  redis: Redis,
  histKey: string,
  sumKey: string,
  ttl: number,
  userText: string,
  assistantText: string
) {
  if (userText) await redis.rpush(histKey, JSON.stringify({ role: "user", content: userText }));
  if (assistantText) await redis.rpush(histKey, JSON.stringify({ role: "assistant", content: assistantText }));
  if (ttl > 0) {
    await redis.expire(histKey, ttl);
    await redis.expire(sumKey, ttl);
  }
}
