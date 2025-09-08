import type { Env } from "../config/env"

export async function callAnthropic(env: Env, payload: any, timeoutMs: number) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
      },
      body: JSON.stringify(payload)
    })
    if (!res.ok) throw new Error(`Anthropic ${res.status}: ${await res.text().catch(()=> "")}`)
    return await res.json()
  } finally {
    clearTimeout(timer)
  }
}
