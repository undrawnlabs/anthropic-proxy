import { sanitize } from "../utils/sanitize"

const baseRules =
  "Use Memory Summary (if present) and recent history as ground truth. Do not re-ask facts unless conflicting. Be concise."

export const buildSystem = (locale: string | undefined, memorySummary: string | null, coreSystemPrompt: string) =>
  [
    "You are undrawn Core.",
    baseRules,
    memorySummary ? `Memory Summary: ${memorySummary}` : "",
    `Reply in the user's language (locale hint: ${locale || "auto"}).`,
    sanitize(coreSystemPrompt)
  ].filter(Boolean).join(" ")
