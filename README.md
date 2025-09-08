# anthropic-proxy

A proxy server that transforms Anthropic API requests to OpenAI format.
It allows you to use Anthropic's API format while connecting to OpenAI-compatible endpoints (e.g. OpenRouter).

## Features

- 🚀 Fastify server with CORS
- 🔄 Request/response mapping (Anthropic ↔ OpenAI format)
- 📦 Redis (Upstash) for session history & summarization
- ⚡ TypeScript + tsup build pipeline
- 🎛 Configurable via environment variables

---

## Getting Started

### 1. Clone & install

```bash
git clone https://github.com/maxnowack/anthropic-proxy.git
cd anthropic-proxy
npm install
```
