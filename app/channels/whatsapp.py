"""WhatsApp channel for Nexora AI (Meta WhatsApp Cloud API).

Flow: Meta → POST /whatsapp/webhook (signed with the app secret) → we answer 200 at once → in the background:
  1. who is this number?  AnalyticsIQ /api/ai/identity → the user + a delegation (unknown numbers are refused)
  2. the same assistant as the web chat runs with that user's role and scope (tools are scoped and audited by AnalyticsIQ)
  3. the answer is formatted for WhatsApp and sent with the Graph API; question and answer are audited in AnalyticsIQ

Environment:
  WHATSAPP_VERIFY_TOKEN       any secret string; type the same in the Meta app webhook settings
  WHATSAPP_ACCESS_TOKEN       permanent (system user) access token of the WhatsApp Business account
  WHATSAPP_PHONE_NUMBER_ID    the sender phone number id (Meta app → WhatsApp → API setup)
  WHATSAPP_APP_SECRET         Meta app secret; used to verify X-Hub-Signature-256 (required)
  WHATSAPP_GRAPH_VERSION      default v21.0
  WHATSAPP_RATE_PER_MINUTE    questions per number per minute (default 6)
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import logging
import os
import re
import time
from typing import Any

import httpx

from ..assistant.source import AnalyticsIQ, SourceError
from ..schemas import Message

log = logging.getLogger("pulseiq8_ai.whatsapp")
MAX_BODY = 3800          # WhatsApp allows 4096 characters per text message
MEMORY_TTL = 30 * 60     # forget a conversation after 30 minutes of silence
MEMORY_TURNS = 8

HELP = """*Nexora AI · AnalyticsIQ*
Ask about sales in plain words, for example:
• How are we doing this month?
• Which region is behind target?
• Will we hit the target?
• Which customers stopped buying?
• How much did ADAKIM buy this month?
Commands: *brief* (management briefing / my summary), *reset* (new conversation), *help*.
I only read data you are allowed to see; I cannot approve, change or delete anything."""

NOT_REGISTERED = ("This number is not registered for the AnalyticsIQ assistant. Ask your AnalyticsIQ administrator to add your "
                  "phone number to your user and enable WhatsApp.")


class WhatsApp:
    def __init__(self, client: httpx.AsyncClient, assistant, source: AnalyticsIQ):
        self.client = client
        self.assistant = assistant
        self.src = source
        self.verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
        self.access_token = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
        self.phone_id = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
        self.app_secret = os.environ.get("WHATSAPP_APP_SECRET", "")
        self.graph = f"https://graph.facebook.com/{os.environ.get('WHATSAPP_GRAPH_VERSION', 'v21.0')}"
        self.rate = int(os.environ.get("WHATSAPP_RATE_PER_MINUTE", "6"))
        self._seen: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._memory: dict[str, tuple[float, list[Message]]] = {}
        self._hits: dict[str, collections.deque] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.verify_token and self.access_token and self.phone_id and self.app_secret)

    def status(self) -> dict:
        return {"enabled": self.enabled, "verifyToken": bool(self.verify_token), "accessToken": bool(self.access_token),
                "phoneNumberId": bool(self.phone_id), "appSecret": bool(self.app_secret), "analyticsiq": self.src.configured}

    # ---------------------------------------------------------------- webhook

    def verify(self, mode: str | None, token: str | None, challenge: str | None) -> str | None:
        """Meta's subscription check (GET): echo the challenge when the verify token matches."""
        if mode == "subscribe" and self.verify_token and token and hmac.compare_digest(token, self.verify_token):
            return challenge or ""
        return None

    def signature_ok(self, body: bytes, header: str | None) -> bool:
        if not self.app_secret or not header or not header.startswith("sha256="):
            return False
        expected = hmac.new(self.app_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(header[7:], expected)

    def accept(self, payload: dict) -> int:
        """Queue every new incoming message; returns how many were queued. Never raises (Meta must get 200)."""
        queued = 0
        for msg in incoming(payload):
            if msg["id"] in self._seen:          # Meta retries deliveries; answer each message once
                continue
            self._seen[msg["id"]] = None
            while len(self._seen) > 2000:
                self._seen.popitem(last=False)
            task = asyncio.create_task(self._safe(msg))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            queued += 1
        return queued

    async def _safe(self, msg: dict) -> None:
        try:
            await self.process(msg)
        except Exception:  # never let one message break the channel
            log.exception("WhatsApp message failed")

    # ---------------------------------------------------------------- one message

    async def process(self, msg: dict) -> None:
        phone, text = msg["from"], (msg.get("text") or "").strip()
        await self.mark_read(msg["id"])
        if not self._allow(phone):
            await self.send(phone, "You are sending questions too quickly. Please wait a minute and try again.")
            return
        try:
            who = await self.src.identity(phone)
        except SourceError as e:
            log.warning("identity lookup failed: %s", e)
            await self.send(phone, "The assistant cannot reach AnalyticsIQ right now. Please try again later.")
            return
        if not who:
            await self.send(phone, NOT_REGISTERED)
            return
        user, delegation = who.get("user") or {}, who.get("delegation")
        if msg.get("type") != "text" or not text:
            await self.send(phone, "Please send your question as a text message.")
            return

        command = text.lower().strip(" .!?")
        if command in ("help", "hi", "hello", "menu", "start"):
            first = str(user.get("displayName") or "").split(" ")[0]
            await self.send(phone, (f"Hello {first}! " if first else "") + HELP)
            return
        if command in ("reset", "new", "clear"):
            self._memory.pop(phone, None)
            await self.send(phone, "Started a new conversation. What would you like to know?")
            return
        if command in ("brief", "briefing", "summary"):
            text = ("Give me the management briefing for this month: performance vs target, projection, regions and reps at risk, "
                    "unclaimed distributor sales and the key actions." if user.get("role") in ("ADMIN", "MANAGER")
                    else "Give me my summary for this month: my sales vs target, pace, projection and which customers to follow up.")

        history = self._history(phone)
        history.append(Message(role="user", content=text[:2000]))
        t0 = time.monotonic()
        await self.src.audit(delegation, "question", "whatsapp", text[:2000])
        res = await self.assistant.ask(history[-MEMORY_TURNS:], {"user": user, "channel": "whatsapp", "delegation": delegation})
        ms = int((time.monotonic() - t0) * 1000)
        if res.get("type") == "error" or not res.get("answer"):
            reply = "Sorry, I could not answer that right now. Please try again, or rephrase the question."
            await self.src.audit(delegation, "error", "whatsapp", str(res.get("error", "no answer"))[:500], False, ms)
        else:
            reply = format_answer(res)
            history.append(Message(role="assistant", content=res["answer"][:2000]))
            await self.src.audit(delegation, "answer", str(res.get("model") or "whatsapp"), res["answer"][:3000], True, ms,
                                 len(res.get("steps") or []))
        self._memory[phone] = (time.monotonic(), history[-MEMORY_TURNS:])
        for part in split(reply):
            await self.send(phone, part)

    def _history(self, phone: str) -> list[Message]:
        at, hist = self._memory.get(phone, (0.0, []))
        return list(hist) if time.monotonic() - at < MEMORY_TTL else []

    def _allow(self, phone: str) -> bool:
        now = time.monotonic()
        q = self._hits.setdefault(phone, collections.deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.rate:
            return False
        q.append(now)
        return True

    # ---------------------------------------------------------------- Graph API

    async def send(self, to: str, body: str) -> None:
        if not (self.access_token and self.phone_id):
            log.warning("WhatsApp is not configured; reply to %s dropped", mask(to))
            return
        r = await self.client.post(f"{self.graph}/{self.phone_id}/messages",
                                   headers={"Authorization": f"Bearer {self.access_token}"},
                                   json={"messaging_product": "whatsapp", "recipient_type": "individual", "to": to,
                                         "type": "text", "text": {"preview_url": False, "body": body}}, timeout=30)
        if r.status_code >= 400:
            log.warning("WhatsApp send to %s failed: %s %s", mask(to), r.status_code, r.text[:300])

    async def mark_read(self, message_id: str) -> None:
        if not (self.access_token and self.phone_id):
            return
        try:
            await self.client.post(f"{self.graph}/{self.phone_id}/messages", headers={"Authorization": f"Bearer {self.access_token}"},
                                   json={"messaging_product": "whatsapp", "status": "read", "message_id": message_id}, timeout=15)
        except httpx.HTTPError:
            pass


# -------------------------------------------------------------------- helpers

def incoming(payload: Any) -> list[dict]:
    """The user messages inside a webhook payload (status updates are ignored)."""
    out = []
    if not isinstance(payload, dict):
        return out
    for entry in payload.get("entry") or []:
        for change in (entry or {}).get("changes") or []:
            value = (change or {}).get("value") or {}
            for m in value.get("messages") or []:
                if not isinstance(m, dict) or not m.get("id") or not m.get("from"):
                    continue
                text = ""
                if m.get("type") == "text":
                    text = (m.get("text") or {}).get("body", "")
                elif m.get("type") == "button":
                    text = (m.get("button") or {}).get("text", "")
                elif m.get("type") == "interactive":
                    i = m.get("interactive") or {}
                    text = (i.get("button_reply") or i.get("list_reply") or {}).get("title", "")
                out.append({"id": m["id"], "from": str(m["from"]), "type": "text" if text else m.get("type"), "text": text})
    return out


def mask(phone: str) -> str:
    return "…" + str(phone)[-3:]


def to_whatsapp(md: str) -> str:
    """Markdown → WhatsApp formatting."""
    t = md.replace("\r", "")
    t = re.sub(r"^#{1,6}\s*(.+)$", r"*\1*", t, flags=re.M)          # headings → bold
    t = re.sub(r"\*\*(.+?)\*\*", r"*\1*", t)                         # **bold** → *bold*
    t = re.sub(r"__(.+?)__", r"*\1*", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)                   # links → text
    t = re.sub(r"^\s*[-*]\s+", "• ", t, flags=re.M)                  # bullets
    t = re.sub(r"`([^`]+)`", r"\1", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _cell(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 1000:
        return f"{v:,}"
    return str(v if v is not None else "")


def table_text(table: dict, max_rows: int = 10, max_cols: int = 4) -> str:
    cols = [str(c) for c in table.get("columns", [])][:max_cols]
    rows = [[_cell(v) for v in r[:max_cols]] for r in table.get("rows", [])[:max_rows]]
    if not cols or not rows:
        return ""
    lines = []
    for r in rows:
        lines.append("• " + " · ".join(f"{c}: {v}" if i else v for i, (c, v) in enumerate(zip(cols, r))))
    more = len(table.get("rows", [])) - max_rows
    if more > 0:
        lines.append(f"… and {more} more (see AnalyticsIQ)")
    return "\n".join(lines)


def format_answer(res: dict) -> str:
    parts = [to_whatsapp(res.get("answer", ""))]
    if isinstance(res.get("table"), dict):
        tbl = table_text(res["table"])
        if tbl and tbl not in parts[0]:
            parts.append(tbl)
    fu = res.get("followups") or []
    if fu:
        parts.append("_You can also ask:_\n" + "\n".join(f"• {q}" for q in fu[:3]))
    return "\n\n".join(p for p in parts if p)


def split(text: str, size: int = MAX_BODY) -> list[str]:
    if len(text) <= size:
        return [text]
    out, buf = [], ""
    for para in text.split("\n"):
        while len(para) > size:
            if buf:
                out.append(buf)
                buf = ""
            out.append(para[:size])
            para = para[size:]
        if len(buf) + len(para) + 1 > size:
            out.append(buf)
            buf = para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        out.append(buf)
    return out[:5]
