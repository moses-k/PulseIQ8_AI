"""PulseIQ8_AI HTTP API (FastAPI).

  GET  /                          chat UI (sidebar of chats, model picker, image upload)
  GET  /console                   models, routing status and usage
  GET  /health                     liveness
  GET  /v1/tiers                   tiers, their models and which are ready
  POST /v1/chat                    routed chat (tier "auto" or a named tier, optional task, json, images)
  POST /v1/tasks/{task}            a named task: PulseIQ8_AI writes the prompt around your data
  POST /v1/vision/analyze          upload an image (multipart) + question
  GET  /v1/usage                   calls per tier / model, latency, failures (last 24 h)
  POST /v1/chat/completions        OpenAI-compatible (model = pulseiq8-auto|fast|coding|general|reasoning|vision)
  GET  /v1/models                  OpenAI-compatible model list
  GET  /assistant                  data assistant test page (needs a delegation from AnalyticsIQ)
  GET  /v1/assistant/tools         the approved AnalyticsIQ tools for the user in X-AI-Delegation
  POST /v1/assistant/chat          ask a question; the assistant fetches data with tools and answers
  POST /v1/assistant/chat/stream   the same as server-sent events: step → result … → answer | error
  GET  /whatsapp/webhook           Meta WhatsApp Cloud API subscription check (hub.verify_token)
  POST /whatsapp/webhook           Meta WhatsApp messages (signed with the app secret) → Nexora AI answers on WhatsApp
  GET  /v1/whatsapp/status         which WhatsApp settings are present
Every /v1 call needs X-Api-Key (or Authorization: Bearer <key>).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import hmac
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path

from . import __version__, config
from .keys import KeyStore
from .router import NoModelError, Router
from .assistant.agent import Assistant
from .assistant.source import AnalyticsIQ
from .assistant.tools import Ctx as ToolCtx
from .assistant.source import SourceError
from .channels.whatsapp import WhatsApp
from .schemas import AssistantRequest, ChatRequest, ChatResponse, Image, Message, TaskRequest
from .tasks import TASK_TIERS, TASKS, build_user_message
from .usage import Usage

log = logging.getLogger("pulseiq8_ai")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
TIER_ALIASES = {"pulseiq8-auto": "auto", "pulseiq8-fast": "fast", "pulseiq8-coding": "coding",
                "pulseiq8-general": "general", "pulseiq8-reasoning": "reasoning", "pulseiq8-vision": "vision"}


def _no_proxy_for_local() -> None:
    """Office proxies (HTTP(S)_PROXY) must not be used for Ollama on this machine / Docker."""
    local = ["localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama"]
    for var in ("NO_PROXY", "no_proxy"):
        have = [x.strip() for x in os.environ.get(var, "").split(",") if x.strip()]
        os.environ[var] = ",".join(have + [h for h in local if h not in have])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _no_proxy_for_local()
    settings = config.load()
    async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
        app.state.settings = settings
        app.state.usage = Usage()
        app.state.router = Router(settings, client, app.state.usage, TASK_TIERS)
        app.state.assistant = Assistant(app.state.router, AnalyticsIQ(client))
        app.state.whatsapp = WhatsApp(client, app.state.assistant, app.state.assistant.src)
        app.state.keys = KeyStore(os.environ.get("PULSEIQ8_KEYS_FILE",
                                                 Path(__file__).resolve().parent.parent / "data" / "keys.json"))
        yield


app = FastAPI(title="PulseIQ8_AI", version=__version__, lifespan=lifespan,
              description="Multi-model AI service for PulseIQ8 mobile analytics and Nexora: Fast, Coding, General, "
                          "Reasoning and Vision tiers with routing, fallback and escalation.")


def require_key(request: Request, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)) -> None:
    keys = app.state.settings.api_keys
    given = x_api_key or (authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None)
    if given and app.state.keys.check(given):
        return
    if not given or not any(hmac.compare_digest(given.encode(), k.encode()) for k in keys):
        # say why, without printing the key itself
        why = "no X-Api-Key / Bearer header" if not given else f"wrong key ({len(given)} chars, starts {given[:4]}…)"
        log.warning("401 %s %s — %s; expected one of %d key(s) from PULSEIQ8_AI_KEYS (%s)",
                    request.method, request.url.path, why, len(keys), ", ".join(f"{len(k)} chars, starts {k[:4]}…" for k in keys))
        raise HTTPException(status_code=401, detail="Missing or wrong API key (X-Api-Key)")


def _is_local(request: Request) -> bool:
    """A call from this computer itself (not from Docker, the network or another machine)."""
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1") and os.environ.get("PULSEIQ8_LOCAL_ADMIN", "true").lower() != "false"


def require_admin(request: Request, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)) -> None:
    """Managing keys: allowed with a valid key, or from this computer itself (so the first key can be created
    in the browser without copying one from .env). Turn the second off with PULSEIQ8_LOCAL_ADMIN=false."""
    if _is_local(request):
        return
    require_key(request, x_api_key, authorization)


@app.exception_handler(NoModelError)
async def no_model(_, exc: NoModelError):
    return JSONResponse(status_code=503, content={"error": str(exc)})


@app.get("/", include_in_schema=False)
async def chat_ui():
    """PulseIQ8 AI chat (conversations are kept in the browser)."""
    return FileResponse(Path(__file__).parent / "static" / "chat.html")


@app.get("/console", include_in_schema=False)
async def console():
    """Models, routing status and usage (the API key is typed in the page)."""
    return FileResponse(Path(__file__).parent / "static" / "console.html")


@app.get("/assistant", include_in_schema=False)
async def assistant_ui():
    """Data assistant chat over AnalyticsIQ (the API key is typed in the page)."""
    return FileResponse(Path(__file__).parent / "static" / "assistant.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "PulseIQ8_AI", "version": __version__}


class NewKey(BaseModel):
    name: str = "API key"


@app.get("/v1/keys", dependencies=[Depends(require_admin)])
async def list_keys():
    """Keys created here (only their first characters). Keys from PULSEIQ8_AI_KEYS in .env are not listed."""
    return {"keys": app.state.keys.list(), "env_keys": len(app.state.settings.api_keys)}


@app.post("/v1/keys", dependencies=[Depends(require_admin)])
async def create_key(body: NewKey):
    """Create a key. The full key is in this answer only — it is stored as a hash."""
    return app.state.keys.create(body.name)


@app.delete("/v1/keys/{key_id}", dependencies=[Depends(require_admin)])
async def revoke_key(key_id: str):
    if not app.state.keys.revoke(key_id):
        raise HTTPException(status_code=404, detail="No such key")
    return {"revoked": key_id}


@app.get("/v1/tiers", dependencies=[Depends(require_key)])
async def tiers():
    return await app.state.router.status()


@app.post("/v1/chat", response_model=ChatResponse, response_model_by_alias=True, dependencies=[Depends(require_key)])
async def chat(req: ChatRequest):
    _check_images(req.messages)
    return await app.state.router.chat(req)


@app.post("/v1/chat/stream", dependencies=[Depends(require_key)])
async def chat_stream(req: ChatRequest):
    """Server-sent events: meta → delta … → done | error. A ': ping' comment every 10 s while the model loads."""
    _check_images(req.messages)

    async def events():
        gen = app.state.router.chat_stream(req)
        nxt = asyncio.ensure_future(gen.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({nxt}, timeout=10)
                if not done:
                    yield ": ping\n\n"
                    continue
                try:
                    ev = nxt.result()
                except StopAsyncIteration:
                    break
                yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                nxt = asyncio.ensure_future(gen.__anext__())
        finally:
            if not nxt.done():
                nxt.cancel()
            await gen.aclose()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/v1/tasks", dependencies=[Depends(require_key)])
async def list_tasks():
    return {name: {"tier": t.tier, "json": t.json_output} for name, t in TASKS.items()}


@app.post("/v1/tasks/{name}", dependencies=[Depends(require_key)])
async def run_task(name: str, body: TaskRequest):
    task = TASKS.get(name)
    if not task:
        raise HTTPException(status_code=404, detail=f"Unknown task {name}")
    msg = Message(role="user", content=build_user_message(task, body.data, body.question), images=body.images)
    _check_images([msg])
    req = ChatRequest(messages=[msg], system=task.system, tier=task.tier, task=name, json=task.json_output,
                      max_tokens=body.max_tokens or task.max_tokens,
                      sensitive=task.sensitive if body.sensitive is None else body.sensitive)
    res = await app.state.router.chat(req)
    return {"result": res.json_data if task.json_output else res.text, "tier": res.tier, "model": res.model,
            "provider": res.provider, "route_reason": res.route_reason, "latency_ms": res.latency_ms,
            "attempts": [a.model_dump() for a in res.attempts]}


@app.post("/v1/vision/analyze", dependencies=[Depends(require_key)])
async def vision(file: UploadFile = File(...), question: str = Form("What does this show? Is anything wrong?"),
                 sensitive: bool = Form(True)):
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 8 MB")
    media = file.content_type or "image/png"
    if media not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
        raise HTTPException(status_code=415, detail="PNG, JPEG, WebP or GIF only")
    body = TaskRequest(question=question[:2000], images=[Image(data=base64.b64encode(raw).decode(), media_type=media)],
                       sensitive=sensitive)
    return await run_task("vision.analyze", body)


@app.get("/v1/usage", dependencies=[Depends(require_key)])
async def usage(hours: float = 24):
    return app.state.usage.summary(max(0.1, min(hours, 24 * 7)))


# ---------------- data assistant (AnalyticsIQ) ----------------

@app.get("/v1/assistant/tools", dependencies=[Depends(require_key)])
async def assistant_tools(x_ai_delegation: Optional[str] = Header(None)):
    """The tools for the user in X-AI-Delegation (from AnalyticsIQ's approved catalogue) plus system_help."""
    a: Assistant = app.state.assistant
    if not x_ai_delegation:
        return {"source": a.src.title, "api": a.src.base, "connected": a.src.configured, "tools": [],
                "note": "send X-AI-Delegation (issued by AnalyticsIQ for a user) to list that user's tools"}
    try:
        tools = await ToolCtx(a.src, x_ai_delegation).tools()
    except SourceError as e:
        raise HTTPException(status_code=e.status or 502, detail=str(e))
    return {"source": a.src.title, "api": a.src.base, "connected": True,
            "tools": [{"name": t.name, "description": t.description, "args": t.params} for t in tools]}


@app.post("/v1/assistant/chat", dependencies=[Depends(require_key)])
async def assistant_chat(req: AssistantRequest):
    if not any(m.role == "user" and m.content.strip() for m in req.messages):
        raise HTTPException(status_code=400, detail="Ask a question (a user message)")
    res = await app.state.assistant.ask(req.messages, req.context, req.tier, req.local_only)
    if res.get("type") == "error":
        return JSONResponse(status_code=502, content=res)
    return res


@app.post("/v1/assistant/chat/stream", dependencies=[Depends(require_key)])
async def assistant_stream(req: AssistantRequest):
    """Server-sent events: step (tool chosen) → result (tool done) … → answer | error; ': ping' every 10 s."""
    if not any(m.role == "user" and m.content.strip() for m in req.messages):
        raise HTTPException(status_code=400, detail="Ask a question (a user message)")

    async def events():
        gen = app.state.assistant.run(req.messages, req.context, req.tier, req.local_only)
        nxt = asyncio.ensure_future(gen.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({nxt}, timeout=10)
                if not done:
                    yield ": ping\n\n"
                    continue
                try:
                    ev = nxt.result()
                except StopAsyncIteration:
                    break
                yield f"event: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"
                nxt = asyncio.ensure_future(gen.__anext__())
        finally:
            if not nxt.done():
                nxt.cancel()
            await gen.aclose()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------- WhatsApp (Meta Cloud API) ----------------

@app.get("/whatsapp/webhook", include_in_schema=False)
async def whatsapp_verify(request: Request):
    q = request.query_params
    challenge = app.state.whatsapp.verify(q.get("hub.mode"), q.get("hub.verify_token"), q.get("hub.challenge"))
    if challenge is None:
        raise HTTPException(status_code=403, detail="Verification failed")
    return PlainTextResponse(challenge)


@app.post("/whatsapp/webhook", include_in_schema=False)
async def whatsapp_webhook(request: Request):
    """Meta delivers messages here. Only signed payloads are accepted; work happens in the background."""
    wa: WhatsApp = app.state.whatsapp
    body = await request.body()
    if not wa.signature_ok(body, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=401, detail="Bad signature")
    try:
        payload = json.loads(body or b"{}")
    except ValueError:
        raise HTTPException(status_code=400, detail="Bad JSON")
    return {"queued": wa.accept(payload)}


@app.get("/v1/whatsapp/status", dependencies=[Depends(require_key)])
async def whatsapp_status():
    return app.state.whatsapp.status()


# ---------------- OpenAI-compatible (aider, IDE plugins, any OpenAI SDK) ----------------

@app.get("/v1/models", dependencies=[Depends(require_key)])
async def models():
    return {"object": "list", "data": [{"id": k, "object": "model", "owned_by": "pulseiq8"} for k in TIER_ALIASES]}


@app.post("/v1/chat/completions", dependencies=[Depends(require_key)])
async def chat_completions(body: dict):
    tier = TIER_ALIASES.get(str(body.get("model", "pulseiq8-auto")), "auto")
    msgs = []
    for m in body.get("messages") or []:
        content = m.get("content")
        images = []
        if isinstance(content, list):   # [{type: text}, {type: image_url}]
            texts = []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and ";base64," in url:
                        head, data = url.split(";base64,", 1)
                        images.append(Image(data=data, media_type=head[5:] or "image/png"))
            content = "\n".join(texts)
        role = m.get("role") if m.get("role") in ("system", "user", "assistant") else "user"
        msgs.append(Message(role=role, content=str(content or ""), images=images))
    fmt = (body.get("response_format") or {}).get("type")
    req = ChatRequest(messages=msgs, tier=tier, task="openai_compat", json=fmt == "json_object",
                      max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096),
                      temperature=float(body.get("temperature", 0.2)))
    _check_images(req.messages)
    res = await app.state.router.chat(req)
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
            "model": f"{res.provider}/{res.model}",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": res.text}}],
            "usage": {"prompt_tokens": res.usage.get("input_tokens", 0), "completion_tokens": res.usage.get("output_tokens", 0),
                      "total_tokens": res.usage.get("input_tokens", 0) + res.usage.get("output_tokens", 0)},
            "pulseiq8": {"tier": res.tier, "route_reason": res.route_reason, "latency_ms": res.latency_ms}}


def _check_images(messages: list[Message]) -> None:
    for m in messages:
        for img in m.images:
            if len(img.data) > MAX_IMAGE_BYTES * 4 // 3 + 4:
                raise HTTPException(status_code=413, detail="Image larger than 8 MB")
            if img.media_type not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                raise HTTPException(status_code=415, detail="PNG, JPEG, WebP or GIF only")
