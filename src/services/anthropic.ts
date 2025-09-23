// services/anthropic.ts
export async function callAnthropic(_: any, payload: any, timeoutMs?: number) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error("Missing ANTHROPIC_API_KEY");

  const controller = new AbortController();
  const ms = Number(timeoutMs ?? process.env.TIMEOUT_MS ?? 30000);
  const timer = setTimeout(() => controller.abort(), ms);

  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      signal: controller.signal,
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "tools-2024-04-04",
        "content-type": "application/json",
        "accept": "application/json",
      },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Anthropic ${res.status}: ${await res.text().catch(()=> "")}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}
