export const sanitize = (s: string) =>
  String(s || "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\u0000/g, "")
    .replace(/[^\S\r\n]+/g, " ")
    .trim()
    .slice(0, 8000)
