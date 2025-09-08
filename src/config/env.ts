export type Env = {
  UPSTASH_REDIS_REST_URL: string
  UPSTASH_REDIS_REST_TOKEN: string
  ANTHROPIC_API_KEY: string
  ANTHROPIC_MODEL: string
  SUMMARIZER_MODEL: string
  MAX_OUTPUT_TOKENS: number
  TEMPERATURE: number
  UI_PROFILE: string
  TIMEOUT_MS: number
  HISTORY_KEEP: number
  SUMMARIZE_THRESHOLD: number
  SUMMARY_MAX_TOKENS: number
  TTL_SECONDS: number
  CORE_SYSTEM_PROMPT: string
  BODY_LIMIT_BYTES: number
  MAX_INPUT_CHARS: number
  RL_PER_MIN: number
  PORT: number
}

const n = (v: any, d: number) => (Number.isFinite(Number(v)) ? Number(v) : d)

export function loadEnv(): Env {
  const e = process.env
  if (!e.UPSTASH_REDIS_REST_URL || !e.UPSTASH_REDIS_REST_TOKEN) throw new Error("Missing Upstash env")
  if (!e.ANTHROPIC_API_KEY) throw new Error("Missing ANTHROPIC_API_KEY")

  const UI_PROFILE = e.UI_PROFILE ?? ""
  return {
    UPSTASH_REDIS_REST_URL: e.UPSTASH_REDIS_REST_URL,
    UPSTASH_REDIS_REST_TOKEN: e.UPSTASH_REDIS_REST_TOKEN,
    ANTHROPIC_API_KEY: e.ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL: e.ANTHROPIC_MODEL ?? "claude-3-haiku-20240307",
    SUMMARIZER_MODEL: e.SUMMARIZER_MODEL ?? "claude-3-haiku-20240307",
    MAX_OUTPUT_TOKENS: n(e.MAX_OUTPUT_TOKENS, 300),
    TEMPERATURE: n(e.TEMPERATURE, 0.2),
    UI_PROFILE,
    TIMEOUT_MS: n(UI_PROFILE === "builder" ? "9000" : (e.TIMEOUT_MS ?? "30000"), 30000),
    HISTORY_KEEP: n(e.HISTORY_KEEP, 10),
    SUMMARIZE_THRESHOLD: n(e.SUMMARIZE_THRESHOLD, 120),
    SUMMARY_MAX_TOKENS: n(e.SUMMARY_MAX_TOKENS, 250),
    TTL_SECONDS: n(e.TTL_SECONDS, 60 * 60 * 24 * 7),
    CORE_SYSTEM_PROMPT: e.CORE_SYSTEM_PROMPT ?? "",
    BODY_LIMIT_BYTES: n(e.BODY_LIMIT_BYTES, 25 * 1024 * 1024),
    MAX_INPUT_CHARS: n(e.MAX_INPUT_CHARS, 0),
    RL_PER_MIN: n(e.RL_PER_MIN, 60),
    PORT: n(e.PORT, 3000),
  }
}

export const UPSTASH_REDIS_REST_URL = process.env.UPSTASH_REDIS_REST_URL;
export const UPSTASH_REDIS_REST_TOKEN = process.env.UPSTASH_REDIS_REST_TOKEN;
