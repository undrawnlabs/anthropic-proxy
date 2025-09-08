import { Redis } from "@upstash/redis"
import type { Env } from "../config/env"
let client: Redis

export function getRedis(env: Env) {
  if (!client) client = new Redis({ url: env.UPSTASH_REDIS_REST_URL, token: env.UPSTASH_REDIS_REST_TOKEN })
  return client
}
