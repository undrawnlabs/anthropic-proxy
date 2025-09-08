import type { Redis } from "@upstash/redis"

export type Msg = { role: "user" | "assistant", content: string }

const parse = (v: string) => { try { return JSON.parse(v) as Msg } catch { return null } }

export async function getRecent(redis: Redis, key: string, keepN: number) {
  const raw = await redis.lrange(key, -keepN, -1)
  return (raw || []).map(parse).filter(Boolean) as Msg[]
}
export async function pushBoth(redis: Redis, key: string, user?: string, assistant?: string) {
  const arr: string[] = []
  if (user) arr.push(JSON.stringify({ role: "user", content: user }))
  if (assistant) arr.push(JSON.stringify({ role: "assistant", content: assistant }))
  if (arr.length) await redis.rpush(key, ...arr)
}
export async function expireBoth(redis: Redis, keys: string[], ttl: number) {
  await Promise.all(keys.map(k => redis.expire(k, ttl)))
}
