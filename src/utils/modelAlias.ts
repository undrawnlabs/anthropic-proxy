export function resolveModelAlias(input: string | undefined, fallback: string) {
  if (!input) return fallback;
  const id = input.trim().toLowerCase();
  // aliases you want the UI to use (add more if needed)
  if (id === "hanna-core" || id === "hanna" || id === "default" || id === "webui") {
    return fallback; // map to env.ANTHROPIC_MODEL
  }
  return input;
}
