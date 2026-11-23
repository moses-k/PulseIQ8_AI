"""Calls to the model providers: Ollama (local), Anthropic, OpenAI (and OpenAI-compatible), Google Gemini."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import httpx

from . import gate
from .config import ModelSpec, Provider

# how long Ollama keeps the model loaded after a request (its default is 5 min; loading again is slow)
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE_MODEL", "30m")


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = True, status: int = 0):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class Result:
    text: str
    usage: dict


_THINK = re.compile(r"<think>.*?</think>", re.S)


def _clean(text: str) -> str:
    # reasoning models (qwen3, deepseek-r1) may print their thinking first
    return _THINK.sub("", text or "").strip()


def _check(r: httpx.Response, who: str) -> dict:
    if r.status_code >= 400:
        body = r.text[:300]
        retry = r.status_code in (408, 409, 425, 429) or r.status_code >= 500 or r.status_code == 404
        raise ProviderError(f"{who} answered {r.status_code}: {body}", retryable=retry, status=r.status_code)
    try:
        return r.json()
    except ValueError as e:
        raise ProviderError(f"{who} returned non-JSON") from e


async def call(client: httpx.AsyncClient, provider: Provider, spec: ModelSpec, system: str, messages: list[dict],
               *, max_tokens: int, temperature: float, json_output: bool, timeout: float) -> Result:
    kind = provider.type
    if kind == "ollama":
        return await _ollama(client, provider, spec, system, messages, max_tokens, temperature, json_output, timeout)
    if kind == "anthropic":
        return await _anthropic(client, provider, spec, system, messages, max_tokens, temperature, timeout)
    if kind == "openai":
        return await _openai(client, provider, spec, system, messages, max_tokens, temperature, json_output, timeout)
    if kind == "gemini":
        return await _gemini(client, provider, spec, system, messages, max_tokens, temperature, json_output, timeout)
    raise ProviderError(f"unknown provider type {kind}", retryable=False)


def _has_json(text: str) -> bool:
    """True when the text holds a NON-EMPTY JSON object / array (small models often answer just {})."""
    t = (text or "").strip()
    for o, c in (("{", "}"), ("[", "]")):
        a, b = t.find(o), t.rfind(c)
        if a >= 0 and b > a:
            try:
                v = json.loads(t[a:b + 1])
            except ValueError:
                continue
            if v:
                return True
    return False


async def _ollama(client, p, spec, system, messages, max_tokens, temperature, json_output, timeout) -> Result:
    """JSON requests: first WITHOUT Ollama's format=json (coding models such as deepseek-coder-v2 often answer only {}
    with it), then once WITH it if the first answer holds no usable JSON."""
    first = await _ollama_once(client, p, spec, system, messages, max_tokens, temperature, False, timeout)
    if not json_output or _has_json(first.text):
        return first
    second = await _ollama_once(client, p, spec, system, messages, max_tokens, temperature, True, timeout)
    if _has_json(second.text):
        return second
    raise ProviderError(f"{spec.model} returned no usable JSON (answered: {first.text[:120]!r})")


async def _ollama_once(client, p, spec, system, messages, max_tokens, temperature, json_output, timeout) -> Result:
    msgs = [{"role": "system", "content": system}] if system else []
    for m in messages:
        item = {"role": m["role"], "content": m["content"]}
        if m.get("images"):
            item["images"] = [i["data"] for i in m["images"]]
        msgs.append(item)
    body = {"model": spec.model, "messages": msgs, "stream": False, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": spec.context, "num_predict": max_tokens, "temperature": temperature}}
    if json_output:
        body["format"] = "json"
    try:
        async with gate.ollama.slot():
            r = await client.post(f"{p.base_url}/api/chat", json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise _ollama_net_error(p, e, timeout) from e
    if r.status_code == 404 and "not found" in r.text.lower():
        raise ProviderError(f"model {spec.model} is not installed (run: ollama pull {spec.model})", status=404)
    low = r.text.lower()
    if r.status_code >= 500 and ("signal: killed" in low or "out of memory" in low or "cudamalloc" in low or "unable to allocate" in low):
        raise ProviderError(f"Ollama ran out of memory running {spec.model} (context {spec.context} tokens). Lower the context "
                            "(PULSEIQ8_FIXED_CONTEXT / *_CONTEXT in .env, e.g. 4096), close other apps, give Docker more "
                            "memory, or use a smaller model.", status=r.status_code)
    data = _check(r, "Ollama")
    text = _clean(data.get("message", {}).get("content", ""))
    return Result(text, {"input_tokens": data.get("prompt_eval_count", 0), "output_tokens": data.get("eval_count", 0)})


async def _anthropic(client, p, spec, system, messages, max_tokens, temperature, timeout) -> Result:
    msgs = []
    for m in messages:
        if m["role"] == "system":
            continue
        parts = [{"type": "image", "source": {"type": "base64", "media_type": i["media_type"], "data": i["data"]}}
                 for i in m.get("images") or []]
        parts.append({"type": "text", "text": m["content"]})
        msgs.append({"role": m["role"], "content": parts})
    body = {"model": spec.model, "max_tokens": max_tokens, "temperature": temperature, "messages": msgs}
    if system:
        body["system"] = system
    headers = {"x-api-key": p.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    try:
        r = await client.post(f"{p.base_url}/v1/messages", json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"Anthropic not reachable: {e.__class__.__name__}") from e
    data = _check(r, "Anthropic")
    text = "".join(c.get("text", "") for c in data.get("content", []) if c.get("type") == "text")
    u = data.get("usage") or {}
    return Result(_clean(text), {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0)})


async def _openai(client, p, spec, system, messages, max_tokens, temperature, json_output, timeout) -> Result:
    msgs = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m.get("images"):
            parts = [{"type": "text", "text": m["content"]}] + [
                {"type": "image_url", "image_url": {"url": f"data:{i['media_type']};base64,{i['data']}"}} for i in m["images"]]
            msgs.append({"role": m["role"], "content": parts})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    body = {"model": spec.model, "messages": msgs, "max_completion_tokens": max_tokens}
    if not spec.thinking:
        body["temperature"] = temperature
    if json_output:
        body["response_format"] = {"type": "json_object"}
    try:
        r = await client.post(f"{p.base_url}/v1/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {p.api_key}"}, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"OpenAI not reachable: {e.__class__.__name__}") from e
    data = _check(r, "OpenAI")
    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    u = data.get("usage") or {}
    return Result(_clean(text), {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)})


async def _gemini(client, p, spec, system, messages, max_tokens, temperature, json_output, timeout) -> Result:
    contents = []
    for m in messages:
        if m["role"] == "system":
            continue
        parts = [{"inline_data": {"mime_type": i["media_type"], "data": i["data"]}} for i in m.get("images") or []]
        parts.append({"text": m["content"]})
        contents.append({"role": "model" if m["role"] == "assistant" else "user", "parts": parts})
    gen = {"maxOutputTokens": max_tokens, "temperature": temperature}
    if json_output:
        gen["responseMimeType"] = "application/json"
    body = {"contents": contents, "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    try:
        r = await client.post(f"{p.base_url}/v1beta/models/{spec.model}:generateContent", json=body,
                              headers={"x-goog-api-key": p.api_key}, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"Gemini not reachable: {e.__class__.__name__}") from e
    data = _check(r, "Gemini")
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(part.get("text", "") for part in (cand.get("content") or {}).get("parts", []))
    if not text and cand.get("finishReason") in ("SAFETY", "RECITATION"):
        raise ProviderError(f"Gemini refused ({cand.get('finishReason')})")
    u = data.get("usageMetadata") or {}
    return Result(_clean(text), {"input_tokens": u.get("promptTokenCount", 0), "output_tokens": u.get("candidatesTokenCount", 0)})


def _ollama_net_error(p: Provider, e: Exception, timeout: float) -> ProviderError:
    if isinstance(e, httpx.TimeoutException) and not isinstance(e, httpx.ConnectTimeout):
        return ProviderError(f"Ollama at {p.base_url} is running but took longer than {int(timeout)} s to answer. The model is "
                             "too slow on this computer (Ollama in Docker on a Mac uses the CPU only) or busy with other "
                             "requests. Use the Ollama Mac app (uses the GPU, much faster) or raise PULSEIQ8_TIMEOUT in .env.")
    return ProviderError(f"Ollama not reachable at {p.base_url}: {e.__class__.__name__} — is Ollama running?")


def _ollama_error(status: int, body: str, spec: ModelSpec) -> ProviderError:
    low = body.lower()
    if status == 404 and "not found" in low:
        return ProviderError(f"model {spec.model} is not installed (run: ollama pull {spec.model})", status=404)
    if status >= 500 and ("signal: killed" in low or "out of memory" in low or "unable to allocate" in low):
        return ProviderError(f"Ollama ran out of memory running {spec.model} (context {spec.context} tokens). Lower the "
                             "context (PULSEIQ8_FIXED_CONTEXT / *_CONTEXT in .env, e.g. 4096), close other apps, give Docker "
                             "more memory, or use a smaller model.", status=status)
    return ProviderError(f"Ollama answered {status}: {body[:300]}", retryable=status >= 500 or status == 429, status=status)


async def stream(client: httpx.AsyncClient, provider: Provider, spec: ModelSpec, system: str, messages: list[dict],
                 *, max_tokens: int, temperature: float, timeout: float, usage: dict, prio: int | None = None):
    """Yields the answer piece by piece. Ollama streams token by token; other providers answer in one piece."""
    if provider.type != "ollama":
        res = await call(client, provider, spec, system, messages, max_tokens=max_tokens, temperature=temperature,
                         json_output=False, timeout=timeout)
        usage.update(res.usage)
        yield res.text
        return
    msgs = [{"role": "system", "content": system}] if system else []
    for m in messages:
        item = {"role": m["role"], "content": m["content"]}
        if m.get("images"):
            item["images"] = [i["data"] for i in m["images"]]
        msgs.append(item)
    body = {"model": spec.model, "messages": msgs, "stream": True, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": spec.context, "num_predict": max_tokens, "temperature": temperature}}
    try:
        async with gate.ollama.slot(prio), client.stream("POST", f"{provider.base_url}/api/chat", json=body,
                                 timeout=httpx.Timeout(timeout, connect=10)) as r:
            if r.status_code >= 400:
                raise _ollama_error(r.status_code, (await r.aread()).decode(errors="replace"), spec)
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    part = json.loads(line)
                except ValueError:
                    continue
                if part.get("error"):
                    raise _ollama_error(500, str(part["error"]), spec)
                text = (part.get("message") or {}).get("content") or ""
                if text:
                    yield text
                if part.get("done"):
                    usage.update({"input_tokens": part.get("prompt_eval_count", 0), "output_tokens": part.get("eval_count", 0)})
                    return
    except httpx.HTTPError as e:
        raise _ollama_net_error(provider, e, timeout) from e


async def ollama_tags(client: httpx.AsyncClient, base_url: str):
    """(names of the models pulled into Ollama, None) or (None, why it could not be reached)."""
    try:
        r = await client.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        names = set()
        for m in r.json().get("models", []):
            n = m.get("name") or m.get("model") or ""
            names.add(n)
            if n.endswith(":latest"):
                names.add(n[: -len(":latest")])
        return names, None
    except httpx.HTTPStatusError as e:
        return None, f"answered HTTP {e.response.status_code} (is this really Ollama?)"
    except httpx.HTTPError as e:
        return None, f"{e.__class__.__name__}: {str(e)[:160] or 'no answer'}"
    except ValueError:
        return None, "answered something that is not Ollama JSON"


async def ollama_installed(client: httpx.AsyncClient, p: Provider):
    names, _ = await ollama_tags(client, p.base_url)
    return names
