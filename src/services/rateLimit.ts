import type { Redis } from "@upstash/redis"
import { hash } from "../utils/hash"

export async function rateLimitMinute(redis: Redis, ip: string, limit: number) {
  const key = `rl:${hash(ip || "unknown")}:${Math.floor(Date.now()/60000)}`
  const count = await redis.incr(key)
  if (count === 1) await redis.expire(key, 65)
  return count > limit
}
