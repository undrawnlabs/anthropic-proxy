type OpenAIMessage = { role: "system"|"user"|"assistant", content?: any, text?: string }

const isDataUrl = (u: string) => typeof u === "string" && u.startsWith("data:")
const parseDataUrl = (u: string) => {
  const m = /^data:([^;]+);base64,(.+)$/i.exec(u || "")
  return m ? { mediaType: m[1], b64: m[2] } : null
}
async function fetchToBase64(url: string) {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`fetch ${r.status}`)
  const buf = Buffer.from(await r.arrayBuffer())
  return { mediaType: r.headers.get("content-type") || "application/octet-stream", b64: buf.toString("base64") }
}

async function toAnthropicParts(content: any) {
  const out: any[] = []
  const pushText = (t: any) => { const s = String(t || "").trim(); if (s) out.push({ type: "text", text: s }) }
  if (typeof content === "string") { pushText(content); return out }
  if (!Array.isArray(content)) return out

  for (const c of content) {
    if (c?.type === "text") pushText(c.text)
    if (c?.type === "image_url") {
      const u = c.image_url?.url || c.image_url
      if (!u) continue
      let mediaType: string | undefined, b64: string | undefined
      if (isDataUrl(u)) { const p = parseDataUrl(u); if (p) { mediaType = p.mediaType; b64 = p.b64 } }
      else { const r = await fetchToBase64(u); mediaType = r.mediaType; b64 = r.b64 }
      if (b64) out.push({ type: "image", source: { type: "base64", media_type: mediaType || "image/png", data: b64 } })
    }
  }
  return out
}

export async function mapOpenAIToAnthropic(messages: OpenAIMessage[]) {
  const systemAll: string[] = []
  const mapped: any[] = []

  for (const m of messages || []) {
    if (m.role === "system") {
      const s = typeof m.content === "string"
        ? m.content
        : (m.content || []).filter((x:any)=>x?.type==="text").map((x:any)=>x.text).join("\n")
      if (s) systemAll.push(s)
      continue
    }
    if (m.role === "user") {
      const parts = await toAnthropicParts(m.content ?? m.text ?? "")
      mapped.push({ role: "user", content: parts.length ? parts : [{ type: "text", text: String(m.content ?? m.text ?? "") }] })
      continue
    }
    if (m.role === "assistant") {
      const t = typeof m.content === "string"
        ? m.content
        : (m.content || []).filter((x:any)=>x?.type==="text").map((x:any)=>x.text).join("")
      mapped.push({ role: "assistant", content: [{ type: "text", text: String(t || "") }] })
    }
  }
  return { system: systemAll.join("\n\n").trim() || null, messages: mapped }
}
