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
Every /v1 call needs X-Api-Key (or Authorization: Bearer <key>).
"""
from __future__ import annotations

import base64
import logging
import os
import hmac
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from . import __version__, config
from .router import NoModelError, Router
from .schemas import ChatRequest, ChatResponse, Image, Message, TaskRequest
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
        yield


app = FastAPI(title="PulseIQ8_AI", version=__version__, lifespan=lifespan,
              description="Multi-model AI service for PulseIQ8 mobile analytics and Nexora: Fast, Coding, General, "
                          "Reasoning and Vision tiers with routing, fallback and escalation.")


def require_key(request: Request, x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)) -> None:
    keys = app.state.settings.api_keys
    given = x_api_key or (authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None)
    if not given or not any(hmac.compare_digest(given.encode(), k.encode()) for k in keys):
        # say why, without printing the key itself
        why = "no X-Api-Key / Bearer header" if not given else f"wrong key ({len(given)} chars, starts {given[:4]}…)"
        log.warning("401 %s %s — %s; expected one of %d key(s) from PULSEIQ8_AI_KEYS (%s)",
                    request.method, request.url.path, why, len(keys), ", ".join(f"{len(k)} chars, starts {k[:4]}…" for k in keys))
        raise HTTPException(status_code=401, detail="Missing or wrong API key (X-Api-Key)")


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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "PulseIQ8_AI", "version": __version__}


@app.get("/v1/tiers", dependencies=[Depends(require_key)])
async def tiers():
    return await app.state.router.status()


@app.post("/v1/chat", response_model=ChatResponse, response_model_by_alias=True, dependencies=[Depends(require_key)])
async def chat(req: ChatRequest):
    _check_images(req.messages)
    return await app.state.router.chat(req)


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
