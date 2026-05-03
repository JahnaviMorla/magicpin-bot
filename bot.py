"""
Vera-challenger bot — magicpin AI Challenge
POST /v1/context  — receive context pushes
POST /v1/tick     — periodic wake-up, bot may initiate messages
POST /v1/reply    — receive merchant/customer reply, bot must respond
GET  /v1/healthz  — liveness probe
GET  /v1/metadata — bot identity
"""

import os
import time
import uuid
import json
import re
import httpx
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()
START = time.time()

contexts: dict = {}
conversations: dict = {}
sent_suppression: set = set()

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


async def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": ANTHROPIC_API_KEY,
    }
    async with httpx.AsyncClient(timeout=28.0) as client:
        r = await client.post(ANTHROPIC_API_URL, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block["text"]
        return text.strip()


def get_ctx(scope: str, context_id: str):
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None

def get_merchant(mid): return get_ctx("merchant", mid)
def get_category(slug): return get_ctx("category", slug)
def get_trigger(tid): return get_ctx("trigger", tid)
def get_customer(cid): return get_ctx("customer", cid)


AUTO_REPLY_PATTERNS = [
    "automated", "auto-reply", "i am currently", "out of office",
    "this is an automated", "thank you for contacting",
    "aapki jaankari ke liye", "main ek automated",
    "team tak pahuncha", "shukriya. main aapki"
]

def is_auto_reply(msg: str) -> bool:
    m = msg.lower()
    return any(p in m for p in AUTO_REPLY_PATTERNS)

def repeated_auto_reply(history: list) -> bool:
    msgs = [t["body"] for t in history if t.get("from") == "merchant"]
    return len(msgs) >= 2 and len(set(msgs[-3:])) <= 1

ACCEPT_SIGNALS = ["yes", "haan", "chalega", "ok", "okay", "sure", "go ahead",
                  "please do", "kar do", "send it", "bhejo"]
REJECT_SIGNALS = ["no", "nahi", "stop", "not interested", "band karo",
                  "mat bhejo", "unsubscribe", "bye"]

def detect_intent(msg: str) -> str:
    m = msg.lower().strip()
    for s in REJECT_SIGNALS:
        if s in m: return "reject"
    for s in ACCEPT_SIGNALS:
        if s in m: return "accept"
    return "neutral"


VERA_SYSTEM = """You are Vera, magicpin's merchant AI assistant. You compose WhatsApp messages to Indian merchants.

RULES:
1. Voice: peer/colleague tone, never promotional hype. Match category (clinical for dentists, trendy for salons, warm for restaurants, motivating for gyms, professional for pharmacies).
2. Language: Hindi-English code-mix when merchant languages include "hi".
3. Specificity: Anchor on a concrete number, date, or source. Use service+price like "Haircut @ Rs.99" not "X% off".
4. CTA: ONE call-to-action at the end. Action triggers: "Reply YES / STOP". Info triggers: open-ended question.
5. No preamble. Start with the hook.
6. No hallucination. Only cite facts from the context.
7. Length: 2-5 sentences, WhatsApp-appropriate.
8. No re-introduction after turn 1.

OUTPUT: respond ONLY with this JSON:
{
  "body": "<the WhatsApp message>",
  "cta": "open_ended" or "binary_yes_stop" or "none",
  "rationale": "<one sentence why>"
}"""

REPLY_SYSTEM = """You are Vera, magicpin's merchant AI assistant responding to a merchant's WhatsApp reply.

RULES:
1. Merchant accepted (yes/ok/go) -> ACTION: fulfill what was offered, add next natural step.
2. Merchant rejected (no/stop) -> end gracefully with one warm exit line.
3. Auto-reply detected -> try once to reach human, then end if repeated.
4. Merchant asked a question -> answer concisely with facts from context.
5. Never repeat a body you already sent.
6. Match category voice and merchant language.

OUTPUT: respond ONLY with ONE of these JSON formats:
{"action": "send", "body": "<message>", "cta": "open_ended", "rationale": "<why>"}
{"action": "wait", "wait_seconds": 1800, "rationale": "<why>"}
{"action": "end", "rationale": "<why>"}"""


async def compose_message(merchant, trigger, category, customer=None, history=None):
    m, t, cat = merchant, trigger, category

    merchant_summary = {
        "name": m["identity"]["name"],
        "owner": m["identity"].get("owner_first_name", ""),
        "city": m["identity"]["city"],
        "locality": m["identity"].get("locality", ""),
        "languages": m["identity"].get("languages", ["en"]),
        "subscription": m.get("subscription", {}),
        "performance_30d": m.get("performance", {}),
        "active_offers": [o["title"] for o in m.get("offers", []) if o.get("status") == "active"],
        "signals": m.get("signals", []),
        "customer_aggregate": m.get("customer_aggregate", {}),
    }

    category_summary = {
        "slug": cat.get("slug"),
        "voice_tone": cat.get("voice", {}).get("tone"),
        "code_mix": cat.get("voice", {}).get("code_mix"),
        "taboos": cat.get("voice", {}).get("vocab_taboo", []),
        "peer_stats": cat.get("peer_stats", {}),
        "offer_catalog": [o["title"] for o in cat.get("offer_catalog", [])[:5]],
        "digest_top": cat.get("digest", [])[:2],
        "seasonal_beats": cat.get("seasonal_beats", []),
        "trend_signals": cat.get("trend_signals", []),
    }

    trigger_summary = {
        "kind": t.get("kind"),
        "source": t.get("source"),
        "urgency": t.get("urgency"),
        "payload": t.get("payload", {}),
    }

    user_prompt = f"""CATEGORY:\n{json.dumps(category_summary, ensure_ascii=False, indent=2)}

MERCHANT:\n{json.dumps(merchant_summary, ensure_ascii=False, indent=2)}

TRIGGER:\n{json.dumps(trigger_summary, ensure_ascii=False, indent=2)}
"""
    if customer:
        cs = {
            "name": customer["identity"]["name"],
            "language_pref": customer["identity"].get("language_pref", "en"),
            "state": customer.get("state"),
            "last_visit": customer.get("relationship", {}).get("last_visit"),
            "services_received": customer.get("relationship", {}).get("services_received", []),
            "preferences": customer.get("preferences", {}),
        }
        user_prompt += f"\nCUSTOMER (send as merchant_on_behalf):\n{json.dumps(cs, ensure_ascii=False, indent=2)}\n"

    if history:
        user_prompt += f"\nPREVIOUS TURNS (do not repeat):\n{json.dumps(history, ensure_ascii=False)}\n"

    user_prompt += "\nCompose now. JSON only."

    raw = await call_claude(VERA_SYSTEM, user_prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        return {"body": "Namaste! Kuch important update hai — reply karein.", "cta": "open_ended", "rationale": "fallback"}


async def compose_reply(conversation_id, merchant_id, customer_id, from_role, message, turn_number):
    history = conversations.get(conversation_id, [])
    merchant = get_merchant(merchant_id)
    customer = get_customer(customer_id) if customer_id else None

    if from_role == "merchant" and is_auto_reply(message):
        if repeated_auto_reply(history):
            return {"action": "end", "rationale": "Auto-reply repeated, graceful exit"}
        return {
            "action": "send",
            "body": "Lagta hai yeh auto-reply hai. Agar aap khud dekh rahe hain toh bata dein — 1 min mein sab share kar deta/deti hoon.",
            "cta": "open_ended",
            "rationale": "First auto-reply — one attempt to reach human",
        }

    intent = detect_intent(message)
    if intent == "reject":
        return {"action": "end", "rationale": "Merchant not interested, exiting gracefully"}

    category_slug = merchant.get("category_slug", "") if merchant else ""
    category = get_category(category_slug)

    ctx = {
        "merchant_name": merchant["identity"]["name"] if merchant else merchant_id,
        "merchant_languages": merchant["identity"].get("languages", ["en"]) if merchant else ["en"],
        "category_slug": category_slug,
        "category_voice": category.get("voice", {}) if category else {},
        "digest_top": category.get("digest", [])[:2] if category else [],
        "merchant_signals": merchant.get("signals", []) if merchant else [],
        "active_offers": [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"] if merchant else [],
        "customer": {"name": customer["identity"]["name"], "state": customer.get("state")} if customer else None,
        "conversation_so_far": history[-6:],
        "intent_detected": intent,
    }

    user_prompt = f"""CONTEXT:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}

Merchant just said: "{message}"

What should Vera do? JSON only."""

    raw = await call_claude(REPLY_SYSTEM, user_prompt, max_tokens=400)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        return {"action": "send", "body": "Shukriya! Aur kuch help chahiye?", "cta": "open_ended", "rationale": "fallback"}


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Vera Challenger",
        "team_members": ["Participant"],
        "model": ANTHROPIC_MODEL,
        "approach": "4-context composer with trigger routing, auto-reply detection, intent transition, graceful exit",
        "contact_email": "participant@example.com",
        "version": "1.0.0",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in {"category", "merchant", "customer", "trigger"}:
        return JSONResponse(status_code=400, content={"accepted": False, "reason": "invalid_scope"})
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(status_code=409, content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]})
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}_{uuid.uuid4().hex[:6]}", "stored_at": datetime.now(timezone.utc).isoformat()}


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    seen_merchants = set()

    for trg_id in body.available_triggers:
        trg = get_trigger(trg_id)
        if not trg:
            continue
        sup_key = trg.get("suppression_key", trg_id)
        if sup_key in sent_suppression:
            continue
        merchant_id = trg.get("merchant_id") or trg.get("payload", {}).get("merchant_id")
        if not merchant_id or merchant_id in seen_merchants:
            continue
        merchant = get_merchant(merchant_id)
        if not merchant:
            continue
        category = get_category(merchant.get("category_slug", ""))
        if not category:
            continue
        customer_id = trg.get("customer_id")
        customer = get_customer(customer_id) if customer_id else None

        try:
            composed = await compose_message(merchant=merchant, trigger=trg, category=category, customer=customer)
        except Exception as e:
            print(f"[tick] error {merchant_id}/{trg_id}: {e}")
            continue

        body_text = composed.get("body", "")
        if not body_text:
            continue

        conv_id = f"conv_{merchant_id}_{trg_id}_{uuid.uuid4().hex[:6]}"
        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": "merchant_on_behalf" if customer else "vera",
            "trigger_id": trg_id,
            "template_name": f"vera_{trg.get('kind', 'generic')}_v1",
            "template_params": [merchant["identity"]["name"], trg.get("kind", ""), body_text[:50]],
            "body": body_text,
            "cta": composed.get("cta", "open_ended"),
            "suppression_key": sup_key,
            "rationale": composed.get("rationale", ""),
        }
        actions.append(action)
        sent_suppression.add(sup_key)
        seen_merchants.add(merchant_id)
        conversations[conv_id] = [{"from": "vera", "body": body_text, "ts": body.now}]
        if len(actions) >= 10:
            break

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    conversations.setdefault(body.conversation_id, []).append({
        "from": body.from_role, "body": body.message, "ts": body.received_at
    })
    result = await compose_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id or "",
        customer_id=body.customer_id,
        from_role=body.from_role,
        message=body.message,
        turn_number=body.turn_number,
    )
    if result.get("action") == "send":
        conversations[body.conversation_id].append({
            "from": "vera", "body": result.get("body", ""), "ts": body.received_at
        })
    return result


@app.post("/v1/teardown")
async def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression.clear()
    return {"status": "wiped"}