import crypto from "crypto"
export const hash = (s: string) => crypto.createHash("sha256").update(String(s)).digest("hex").slice(0, 16)
