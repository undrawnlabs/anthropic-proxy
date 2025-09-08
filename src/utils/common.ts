import crypto from 'crypto';

export const jsonReply = (replyObj) => {
  const text = String(replyObj ?? '');
  return {
    id: crypto.randomUUID(),
    object: 'chat.completion',
    model: 'hanna-core',
    created: Math.floor(Date.now() / 1000),
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: [
            { type: 'text', text }
          ]
        },
        finish_reason: 'stop'
      }
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: text.length,
      total_tokens: text.length
    }
  };
};

export const recordLatency = (ms, metrics, lastLatencies) => {
  metrics.last_ms = ms;
  lastLatencies.push(ms);
  if (lastLatencies.length > 200) lastLatencies.shift();
  const sorted = [...lastLatencies].sort((a, b) => a - b);
  const idx = Math.floor(0.95 * (sorted.length - 1));
  metrics.p95_ms = sorted[idx] || ms;
};

export const callAnthropic = async ({ model, system, messages, maxTokens, temperature, timeoutMs }) => {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error('Missing ANTHROPIC_API_KEY environment variable');
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      signal: controller.signal,
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json'
      },
      body: JSON.stringify({ model, system, messages, max_tokens: maxTokens, temperature })
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => '');
      throw new Error(`Anthropic ${res.status}: ${detail}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
};

export const buildSystem = (locale, memorySummary) => {
  const baseRules = 'Use Memory Summary (if present) and recent history as ground truth. Do not re-ask facts unless conflicting. Be concise.';
  return [
    'You are undrawn Core.',
    baseRules,
    memorySummary ? `Memory Summary: ${memorySummary}` : '',
    `Reply in the user's language (locale hint: ${locale || 'auto'}).`,
    process.env.CORE_SYSTEM_PROMPT
  ].filter(Boolean).join(' ');
};

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | { type: string; text: string }[];
}

export const mapOpenAIToAnthropic = async (openaiMessages: OpenAIMessage[]) => {
  let systemAll: string[] = [];
  const msgs: { role: string; content: { type: string; text: string }[] }[] = [];

  for (const m of openaiMessages || []) {
    if (m.role === 'system') {
      const s = typeof m.content === 'string'
        ? m.content
        : (m.content || []).filter((x: any) => x?.type === 'text').map((x: any) => x.text).join('\n');
      if (s) systemAll.push(s);
      continue;
    }
    if (m.role === 'user') {
      msgs.push({
        role: 'user',
        content: [{ type: 'text', text: String(m.content || '') }]
      });
      continue;
    }
    if (m.role === 'assistant') {
      msgs.push({
        role: 'assistant',
        content: [{ type: 'text', text: String(m.content || '') }]
      });
      continue;
    }
  }

  return { system: systemAll.join('\n\n').trim() || null, messages: msgs };
};
