// index.js — with legacy route aliases
import Fastify from 'fastify';
import cors from '@fastify/cors';
import fetch from 'node-fetch';
import { Redis } from '@upstash/redis';

const {
  ANTHROPIC_API_KEY,
  UPSTASH_REDIS_REST_URL,
  UPSTASH_REDIS_REST_TOKEN,
  CORE_SYSTEM_PROMPT,
  SYSTEM_PROMPT_BASE: SP_FALLBACK,
  OPENAI_API_KEY,
  PORT = 3000,
  ANTHROPIC_MODEL = 'claude-3-5-sonnet-20240620',
  ROUTE_PREFIX // optional: e.g. "/v1"
} = process.env;

const SYSTEM_PROMPT_BASE = CORE_SYSTEM_PROMPT || SP_FALLBACK;
if (!ANTHROPIC_API_KEY || !UPSTASH_REDIS_REST_URL || !UPSTASH_REDIS_REST_TOKEN || !SYSTEM_PROMPT_BASE) {
  throw new Error('Missing ENV: ANTHROPIC_API_KEY, UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN, CORE_SYSTEM_PROMPT (or SYSTEM_PROMPT_BASE)');
}

const app = Fastify({ logger: true });
await app.register(cors, { origin: true });

const redis = new Redis({ url: UPSTASH_REDIS_REST_URL, token: UPSTASH_REDIS_REST_TOKEN });

const HISTORY_TTL_SECONDS = 60 * 60 * 24 * 30;
const STB_MAX_ITEMS = Number(process.env.STB_MAX_ITEMS || 20);
const RECALL_TOP_K = Number(process.env.RECALL_TOP_K || 6);
const LTM_SCAN_LIMIT = Number(process.env.LTM_SCAN_LIMIT || 1500);
const TOKEN_BUDGET = Number(process.env.TOKEN_BUDGET || 170_000);
const MAX_OUTPUT_TOKENS = Number(process.env.MAX_OUTPUT_TOKENS || 1024);

// Keys
const keySTB = (coreId, sessionId) => `stb:exec:${coreId}:${sessionId}`;
const keyLTM = (coreId, sessionId) => `ltm:exec:${coreId}:${sessionId}`;
const keySUM = (coreId, sessionId) => `sum:exec:${coreId}:${sessionId}`;

// Utils
const toAnthropicMessage = (item) => ({ role: item.role === 'assistant' ? 'assistant' : 'user', content: [{ type: 'text', text: String(item.content ?? '') }] });
async function ensureListKey(c, k){ const t = await c.type(k); if (t && t !== 'none' && t !== 'list') await c.del(k); }
async function appendList(c, k, ...e){ if(!e.length) return; await ensureListKey(c,k); await c.rpush(k, e.map(x=>JSON.stringify(x))); await c.expire(k, HISTORY_TTL_SECONDS); }
async function lrangeJSON(c,k,a,b){ await ensureListKey(c,k); const raw=await c.lrange(k,a,b); return raw.map(s=>{try{return JSON.parse(s)}catch{return null}}).filter(Boolean); }
async function ltrimKeepLast(c,k,n){ await ensureListKey(c,k); await c.ltrim(k,-n,-1); await c.expire(k,HISTORY_TTL_SECONDS); }
async function readSummary(c,k){ const v=await c.get(k); return typeof v==='string'?v:''; }
async function writeSummary(c,k,t){ await c.set(k,t||''); await c.expire(k,HISTORY_TTL_SECONDS); }

async function embedText(text){
  if(!OPENAI_API_KEY) return null;
  const r = await fetch('https://api.openai.com/v1/embeddings',{method:'POST',headers:{authorization:`Bearer ${OPENAI_API_KEY}`,'content-type':'application/json'},body:JSON.stringify({model:'text-embedding-3-small',input:text})});
  if(!r.ok){ const e=await r.text().catch(()=> ''); throw new Error(`OpenAI embeddings error: ${r.status} ${e}`);}
  const d=await r.json(); return d.data?.[0]?.embedding || null;
}
function cosineSim(a,b){ let dot=0,na=0,nb=0; const n=Math.min(a.length,b.length); for(let i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];} return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-12); }
function keywordScore(q,doc){ const qT=new Set(String(q).toLowerCase().split(/\W+/).filter(Boolean)); const dT=new Set(String(doc).toLowerCase().split(/\W+/).filter(Boolean)); let inter=0; for(const t of qT) if(dT.has(t)) inter++; return inter/Math.sqrt((qT.size||1)*(dT.size||1)); }
async function countTokens({system,messages,model}){ const r=await fetch('https://api.anthropic.com/v1/messages/count_tokens',{method:'POST',headers:{'x-api-key':ANTHROPIC_API_KEY,'anthropic-version':'2023-06-01','content-type':'application/json'},body:JSON.stringify({model,system,messages})}); if(!r.ok){ const e=await r.text().catch(()=> ''); throw new Error(`count_tokens failed: ${r.status} ${e}`);} const d=await r.json(); return d.input_tokens ?? 0; }
function updateSummaryHeuristic(prev, entries){
  const facts=[]; const add=(l)=>{ const t=String(l||'').trim(); if(t&&t.length<=220) facts.push(`- ${t}`); };
  for(const e of entries){ if(e.role==='user'){ const t=String(e.content||''); for(const c of t.split(/\n+/).slice(0,3)){ if(/(^i\s(am|work|live|prefer|use)\b)|(\bmy\s[a-z]+)/i.test(c)||/\b(is|=|:)\b/.test(c)) add(c); } } }
  const head='Memory state (concise facts from earlier context):';
  const merged=[ prev?prev.trim():`${head}\n- (none yet)`, ...(facts.length?[`\n# New observations:\n${facts.join('\n')}`]:[]) ].join('\n');
  return merged.slice(0,1600);
}
async function recallFromLTM(coreId, sessionId, userMsg){
  const all=await lrangeJSON(redis,keyLTM(coreId,sessionId),-LTM_SCAN_LIMIT,-1); if(!all.length) return [];
  let qEmb=null; if(OPENAI_API_KEY){ try{ qEmb=await embedText(userMsg);}catch{} }
  const scored=all.map(e=>{ const c=String(e.content||''); const s=(qEmb&&Array.isArray(e.emb))?cosineSim(qEmb,e.emb):keywordScore(userMsg,c); return {s,e}; }).sort((a,b)=>b.s-a.s);
  const top=scored.slice(0,RECALL_TOP_K).map(x=>x.e);
  const seen=new Set(), out=[]; for(const e of top){ const k=`${e.role}:${e.content.slice(0,120)}`; if(!seen.has(k)){ seen.add(k); out.push(e); } }
  return out;
}
async function buildPrompt({coreId,sessionId,locale,userEntry}){
  const stb=await lrangeJSON(redis,keySTB(coreId,sessionId),0,-1);
  const summary=await readSummary(redis,keySUM(coreId,sessionId));
  const recalled=await recallFromLTM(coreId,sessionId,userEntry.content);
  const summaryBlock=summary?`\n\n${summary}`:''; const recalledBlock=recalled.length?`\n\nRelevant context (retrieved):\n${recalled.map(e=>`- ${e.role}: ${e.content}`).join('\n')}`:'';
  let system=`${SYSTEM_PROMPT_BASE}\nLocale: ${locale}${summaryBlock}${recalledBlock}`;
  let messages=[...stb.map(toAnthropicMessage), toAnthropicMessage(userEntry)];
  let tokens=await countTokens({system,messages,model:ANTHROPIC_MODEL});
  if(tokens+MAX_OUTPUT_TOKENS<=TOKEN_BUDGET) return {system,messages};
  const work=[...stb]; while(work.length>0){ work.shift(); messages=[...work.map(toAnthropicMessage), toAnthropicMessage(userEntry)]; tokens=await countTokens({system,messages,model:ANTHROPIC_MODEL}); if(tokens+MAX_OUTPUT_TOKENS<=TOKEN_BUDGET) break; }
  if(tokens+MAX_OUTPUT_TOKENS>TOKEN_BUDGET && recalledBlock){ system=`${SYSTEM_PROMPT_BASE}\nLocale: ${locale}${summaryBlock}`; tokens=await countTokens({system,messages,model:ANTHROPIC_MODEL}); }
  if(tokens+MAX_OUTPUT_TOKENS>TOKEN_BUDGET && summaryBlock){ system=`${SYSTEM_PROMPT_BASE}\nLocale: ${locale}`; }
  return {system,messages};
}
async function rolloverSTBIfNeeded(coreId,sessionId){
  const k=keySTB(coreId,sessionId); await ensureListKey(redis,k); const len=await redis.llen(k); if(len<=STB_MAX_ITEMS) return;
  const excess=len-STB_MAX_ITEMS; const toMove=await lrangeJSON(redis,k,0,excess-1); await ltrimKeepLast(redis,k,STB_MAX_ITEMS);
  const enriched=[]; for(const e of toMove){ const item={...e}; if(OPENAI_API_KEY){ try{ const emb=await embedText(String(e.content||'')); if(emb) item.emb=emb; }catch{} } enriched.push(item); }
  await appendList(redis,keyLTM(coreId,sessionId),...enriched);
}
async function maybeRefreshSummary(coreId,sessionId,entries){ const prev=await readSummary(redis,keySUM(coreId,sessionId)); const next=updateSummaryHeuristic(prev,entries); if(next!==prev) await writeSummary(redis,keySUM(coreId,sessionId),next); }

// ---------------------- Core handler ----------------------
const chatHandler = async (req, reply) => {
  const body = req.body || {};
  const { core_id, session_id, user_message, locale = 'uk' } = body;
  if (!core_id || !session_id || !user_message) {
    return reply.code(400).send({ error: 'bad_request', detail: 'core_id, session_id, user_message are required' });
  }
  const userEntry = { role:'user', content:user_message, ts:Date.now(), locale };
  const { system, messages } = await buildPrompt({ coreId: core_id, sessionId: session_id, locale, userEntry });

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method:'POST',
    headers:{ 'x-api-key':ANTHROPIC_API_KEY, 'anthropic-version':'2023-06-01', 'content-type':'application/json' },
    body: JSON.stringify({ model: ANTHROPIC_MODEL, system, messages, max_tokens: MAX_OUTPUT_TOKENS, temperature: 0.2 })
  });
  if(!res.ok){ const t=await res.text().catch(()=> ''); req.log.error({status:res.status, body:t}, 'Anthropic API error'); return reply.code(502).send({ error:'upstream_error', detail:t }); }

  const data = await res.json();
  const assistantText = Array.isArray(data.content) && data.content[0]?.type === 'text' ? data.content[0].text : '';
  const assistantEntry = { role:'assistant', content:assistantText, ts:Date.now(), locale };

  await appendList(redis, keySTB(core_id, session_id), userEntry, assistantEntry);
  await ltrimKeepLast(redis, keySTB(core_id, session_id), STB_MAX_ITEMS);
  await rolloverSTBIfNeeded(core_id, session_id);
  await maybeRefreshSummary(core_id, session_id, [userEntry, assistantEntry]);

  return reply.send({ core_id, session_id, locale, reply: assistantText });
};

// ---------------------- Router (with optional prefix) ----------------------
const r = ROUTE_PREFIX || '';
app.get(`${r}/health`, async () => ({ ok: true }));
app.get(`${r}/debug/tokens`, async (req) => {
  const { core_id, session_id, locale = 'uk', prompt = 'ping' } = req.query;
  const userEntry = { role:'user', content:String(prompt||''), ts:Date.now(), locale };
  const { system, messages } = await buildPrompt({ coreId: core_id, sessionId: session_id, locale, userEntry });
  const tokens = await countTokens({ system, messages, model: ANTHROPIC_MODEL });
  return { input_tokens: tokens, system_chars: system.length, messages_count: messages.length };
});
app.post(`${r}/chat`, { schema: { body: { type:'object', required:['core_id','session_id','user_message'], properties:{
  core_id:{type:'string',minLength:1}, session_id:{type:'string',minLength:1}, user_message:{type:'string',minLength:1}, locale:{type:'string'} }, additionalProperties:false } } }, chatHandler);

// Legacy aliases for compatibility:
app.post(`${r}/v1/chat`, chatHandler);
app.post(`${r}/complete`, async (req, reply) => {  // accept {prompt} too
  const { core_id='exec', session_id='s1', prompt, user_message, locale='uk' } = req.body || {};
  req.body = { core_id, session_id, user_message: user_message ?? prompt, locale };
  return chatHandler(req, reply);
});
app.post(`${r}/v1/complete`, async (req, reply) => {  // same alias
  const { core_id='exec', session_id='s1', prompt, user_message, locale='uk' } = req.body || {};
  req.body = { core_id, session_id, user_message: user_message ?? prompt, locale };
  return chatHandler(req, reply);
});
app.post(`${r}/api/chat`, chatHandler);

// Log routes on startup
app.ready().then(() => app.log.info(app.printRoutes()));

// Start
app.listen({ port:Number(PORT), host:'0.0.0.0' })
  .then(addr => app.log.info(`listening on ${addr}`))
  .catch(err => { app.log.error(err); process.exit(1); });
