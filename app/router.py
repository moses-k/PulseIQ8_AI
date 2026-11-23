"""The orchestrator: picks a tier for each request, then a model inside it, with fallback and escalation.

  1. tier   – images → vision; a known task → its tier; an explicit tier; else rules on the text
              (code → coding, complex analysis → reasoning, short simple jobs → fast, the rest → general);
              optionally the Fast model classifies requests the rules are unsure about.
  2. model  – the tier's models in order, minus: cloud models when the policy / request forbids them, models whose
              context window is too small for the input, Ollama models not pulled, models whose circuit is open.
  3. call   – on an error / timeout / rate limit / invalid JSON: the next model (fallback), then the next tier up
              (cascade: fast → general → reasoning, coding → reasoning). Reasoning is reached only when needed.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import time
from dataclasses import dataclass

import httpx

from . import gate, privacy, providers
from .config import ModelSpec, Settings
from .schemas import Attempt, ChatRequest, ChatResponse
from .usage import Usage

ESCALATION = {"fast": ["general", "reasoning"], "general": ["reasoning"], "coding": ["reasoning"],
              "reasoning": [], "vision": []}

_CODE = re.compile(
    r"```|\bswift(ui)?\b|\bkotlin\b|jetpack|\bcompose\b|spring ?boot|\bjava\b|stack ?trace|exception|nullpointer|"
    r"\bcrash|compile|refactor|unit ?tests?|\bdebug|\bbug\b|\bfunction\b|\bmethod\b|\bclass\b|endpoint|\bsql\b|regex|"
    r"\.swift\b|\.kt\b|\.java\b|xcode|gradle|android studio|pull request|code review|\bapi\b|json schema|dockerfile",
    re.I)
_REASON = re.compile(
    r"root[- ]cause|architect|system design|design (a|the) system|trade-?offs?|\bcompare\b|pros and cons|"
    r"step[- ]by[- ]step plan|multi[- ]step|plan (the|a) migration|financial|cash ?flow|valuation|forecast|"
    r"intermittent|why does .* sometimes|strategy|evaluate (the )?options|best approach|which (option|approach) is better|"
    r"investigate|post-?mortem", re.I)
_FAST = re.compile(
    r"summari[sz]e|summary|tl;?dr|\btitle\b|classif|categori[sz]e|intent|sentiment|translate|rephrase|rewrite|"
    r"spell|grammar|yes or no|one word|keywords|extract (the )?(name|date|amount)", re.I)


@dataclass
class Route:
    tier: str
    reason: str
    confident: bool = True


class NoModelError(Exception):
    pass


def decide(req: ChatRequest, task_tiers: dict[str, str]) -> Route:
    if any(m.images for m in req.messages):
        return Route("vision", "the request has images")
    if req.task and req.task in task_tiers:
        return Route(task_tiers[req.task], f"task {req.task}")
    if req.tier != "auto":
        return Route(req.tier, "tier requested by the caller")
    text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
    n = len(text)
    code = len(_CODE.findall(text))
    reason = len(_REASON.findall(text))
    if reason >= 2 or (reason >= 1 and (n > 1500 or code >= 2)):
        return Route("reasoning", "complex problem (analysis / architecture / root cause)")
    if code >= 1:
        return Route("coding", "software engineering request")
    if _FAST.search(text) and n < 4000:
        return Route("fast", "simple job (summary / title / classification / rewrite)")
    if reason == 1:
        return Route("general", "analysis question", confident=False)
    if n < 60 or (n < 160 and text.rstrip().endswith("?") and text.count("?") == 1):
        return Route("fast", "short question", confident=False)
    return Route("general", "general request", confident=False)


def estimate_tokens(req: ChatRequest, system: str) -> int:
    chars = len(system) + sum(len(m.content) for m in req.messages)
    images = sum(len(m.images) for m in req.messages)
    return chars // 4 + images * 1500


def extract_json(text: str):
    """The first NON-EMPTY JSON object / array in the text ({} or [] count as no answer)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t)
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = t.find(opener), t.rfind(closer)
        if a >= 0 and b > a:
            try:
                v = json.loads(t[a:b + 1])
            except ValueError:
                continue
            if v:
                return v
    return None


class Router:
    def __init__(self, settings: Settings, client: httpx.AsyncClient, usage: Usage, task_tiers: dict[str, str]):
        self.s = settings
        self.client = client
        self.usage = usage
        self.task_tiers = task_tiers
        self._fails: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._installed: set[str] | None = None
        self._installed_at = 0.0
        self.ollama_error: str | None = None

    # ---------------- model choice ----------------

    async def installed(self) -> set[str] | None:
        """Models pulled into Ollama (cached 30 s). If the configured address does not answer, the usual local
        addresses are tried (localhost / 127.0.0.1 / host.docker.internal, port 11434) and the first that works is used."""
        if time.monotonic() - self._installed_at > 30:
            name, ollama = next(((n, p) for n, p in self.s.providers.items() if p.type == "ollama"), (None, None))
            self.ollama_error = None
            if ollama is None:
                self._installed = None
            else:
                names, err = await providers.ollama_tags(self.client, ollama.base_url)
                if names is None:
                    tried = [f"{ollama.base_url}: {err}"]
                    for alt in ("http://localhost:11434", "http://127.0.0.1:11434", "http://host.docker.internal:11434"):
                        if alt == ollama.base_url:
                            continue
                        alt_names, alt_err = await providers.ollama_tags(self.client, alt)
                        if alt_names is not None:
                            self.s.providers[name] = dataclasses.replace(ollama, base_url=alt)
                            names = alt_names
                            break
                        tried.append(f"{alt}: {alt_err}")
                    if names is None:
                        self.ollama_error = " · ".join(tried)
                self._installed = names
            self._installed_at = time.monotonic()
        return self._installed

    def available(self, spec: ModelSpec, installed: set[str] | None) -> tuple[bool, str]:
        p = self.s.providers[spec.provider]
        if not p.configured:
            return False, "no API key"
        if p.type == "ollama" and installed is not None and spec.model not in installed:
            return False, f"not pulled (ollama pull {spec.model})"
        if self._open_until.get(spec.key, 0) > time.monotonic():
            return False, "paused after repeated failures"
        return True, ""

    async def candidates(self, tier: str, req: ChatRequest, tokens: int) -> list[ModelSpec]:
        local_only = self.s.data_policy == "local_only" or req.sensitive
        models = [m for m in self.s.tiers[tier].models if not (local_only and m.cloud)]
        if tier == "vision":
            models = [m for m in models if m.vision]
        if self.s.data_policy == "local_first":
            models.sort(key=lambda m: m.cloud)          # stable: local first, configured order kept
        installed = await self.installed()
        ok = [m for m in models if self.available(m, installed)[0]]
        need = tokens + req.max_tokens
        fits = [m for m in ok if m.context >= need]
        if not fits and ok:
            fits = sorted(ok, key=lambda m: -m.context)[:1]   # nothing big enough: the biggest window
        return fits

    async def classify(self, req: ChatRequest) -> str | None:
        """Ask the Fast model which tier a request needs (used only when the rules are unsure)."""
        text = next((m.content for m in reversed(req.messages) if m.role == "user"), "")[:1500]
        sys = ("Route a request to one model tier. fast = quick answers, summaries, classification, short writing. "
               "coding = programming, debugging, code review. general = conversation, research, business analysis, "
               "documents, planning, explanation. reasoning = only hard multi-step problems: architecture, root-cause "
               "investigations, financial analysis, comparing several solutions. Return JSON {\"tier\": \"...\"}.")
        for spec in await self.candidates("fast", ChatRequest(messages=[], max_tokens=20), len(text) // 4):
            try:
                r = await providers.call(self.client, self.s.providers[spec.provider], spec, sys,
                                         [{"role": "user", "content": f"<data>{text}</data>"}],
                                         max_tokens=20, temperature=0, json_output=True, timeout=20)
                tier = (extract_json(r.text) or {}).get("tier")
                return tier if tier in ("fast", "coding", "general", "reasoning") else None
            except providers.ProviderError:
                continue
        return None

    # ---------------- the call ----------------

    async def chat(self, req: ChatRequest) -> ChatResponse:
        t0 = time.monotonic()
        gate.priority.set(gate.priority_for(req.task))
        if not self.s.routing:
            return await self._fixed(req, t0)
        route = decide(req, self.task_tiers)
        if not route.confident and self.s.llm_classifier:
            tier = await self.classify(req)
            if tier and tier != route.tier:
                route = Route(tier, f"classified by the Fast model ({route.reason} by rules)")
        system = "\n\n".join(x for x in (req.system, privacy.INJECTION_GUARD) if x)
        tokens = estimate_tokens(req, system)
        escalate = self.s.escalate if req.escalate is None else req.escalate
        chain = [route.tier] + (ESCALATION[route.tier] if escalate else [])
        attempts: list[Attempt] = []
        base_msgs = [m.model_dump() for m in req.messages if m.role != "system"]
        sys_extra = "\n\n".join(m.content for m in req.messages if m.role == "system")
        if sys_extra:
            system = sys_extra + "\n\n" + system
        if req.json_output:
            system += "\n\nAnswer with ONE valid JSON object only — no markdown, no text before or after it."

        plan = [(tier, spec) for tier in chain for spec in await self.candidates(tier, req, tokens)]
        if not plan:
            # last resort: nothing in this tier chain is installed / allowed — use any other available model
            # (e.g. only the coding model is pulled yet), so the caller still gets an answer
            seen = set()
            for tier_name, t in self.s.tiers.items():
                if tier_name == "vision" and route.tier != "vision":
                    continue
                for spec in await self.candidates(tier_name, req, tokens):
                    if spec.key not in seen:
                        seen.add(spec.key)
                        plan.append((tier_name, spec))
            if plan:
                route = Route(route.tier, f"{route.reason}; no {route.tier} model installed, used what is available")
        for tier, spec in plan:
            msgs = base_msgs
            if spec.cloud:   # personal data never leaves for a cloud provider unmasked
                msgs = [{**m, "content": privacy.mask(m["content"])} for m in base_msgs]
            a0 = time.monotonic()
            try:
                res = await providers.call(self.client, self.s.providers[spec.provider], spec, system, msgs,
                                           max_tokens=req.max_tokens, temperature=req.temperature,
                                           json_output=req.json_output, timeout=self.s.request_timeout)
                parsed = extract_json(res.text) if req.json_output else None
                if req.json_output and parsed is None:
                    raise providers.ProviderError("answer was not valid JSON")
                if not res.text.strip():
                    raise providers.ProviderError("empty answer")
            except (providers.ProviderError, asyncio.TimeoutError, httpx.HTTPError) as e:
                ms = int((time.monotonic() - a0) * 1000)
                attempts.append(Attempt(tier=tier, model=spec.key, ok=False, ms=ms, error=str(e)[:300]))
                self._failed(spec)
                self.usage.record(req.task, tier, spec.key, False, ms, {}, str(e))
                continue   # fallback: the next model, then the next tier
            ms = int((time.monotonic() - a0) * 1000)
            self._fails[spec.key] = 0
            attempts.append(Attempt(tier=tier, model=spec.key, ok=True, ms=ms))
            self.usage.record(req.task, tier, spec.key, True, ms, res.usage, None)
            why = route.reason if tier == route.tier else f"{route.reason}; answered by the {tier} tier"
            return ChatResponse(text=res.text, json=parsed, tier=tier, model=spec.model, provider=spec.provider,
                                route_reason=why, latency_ms=int((time.monotonic() - t0) * 1000),
                                attempts=attempts, usage=res.usage)
        detail = "; ".join(f"{a.model}: {a.error}" for a in attempts) or "no model is available for this tier"
        raise NoModelError(f"No model could answer ({route.tier}): {detail}")

    # ---------------- streaming (chat UI) ----------------

    async def chat_stream(self, req: ChatRequest):
        """Like chat(), but yields events: meta (which model) → delta (pieces of text) … → done | error.
        Falls back to the next model only while nothing has been sent yet."""
        t0 = time.monotonic()
        prio = gate.priority_for(req.task)
        if not self.s.routing:
            if self.s.fixed is None:
                yield {"type": "error", "message": "Routing is off but router.fixed has no model (set PULSEIQ8_FIXED_MODEL)"}
                return
            await self.installed()
            plan, reason = [("fixed", self.s.fixed)], "routing off — fixed model"
            if self.s.fixed.cloud and (self.s.data_policy == "local_only" or req.sensitive):
                yield {"type": "error", "message": "The fixed model is a cloud model, which the data policy / Local only does not allow"}
                return
        else:
            route = decide(req, self.task_tiers)
            tokens = estimate_tokens(req, req.system or "")
            chain = [route.tier] + (ESCALATION[route.tier] if (self.s.escalate if req.escalate is None else req.escalate) else [])
            plan = [(tier, spec) for tier in chain for spec in await self.candidates(tier, req, tokens)]
            if not plan:
                for tier_name in self.s.tiers:
                    if tier_name == "vision" and route.tier != "vision":
                        continue
                    for spec in await self.candidates(tier_name, req, tokens):
                        if all(spec.key != p[1].key for p in plan):
                            plan.append((tier_name, spec))
            reason = route.reason
        system = "\n\n".join(x for x in (req.system, privacy.INJECTION_GUARD) if x)
        sys_extra = "\n\n".join(m.content for m in req.messages if m.role == "system")
        if sys_extra:
            system = sys_extra + "\n\n" + system
        base_msgs = [m.model_dump() for m in req.messages if m.role != "system"]
        errors = []
        for tier, spec in plan:
            msgs = [{**m, "content": privacy.mask(m["content"])} for m in base_msgs] if spec.cloud else base_msgs
            usage: dict = {}
            sent = False
            a0 = time.monotonic()
            try:
                async for piece in providers.stream(self.client, self.s.providers[spec.provider], spec, system, msgs,
                                                    max_tokens=req.max_tokens, temperature=req.temperature,
                                                    timeout=self.s.request_timeout, usage=usage, prio=prio):
                    if not sent:
                        sent = True
                        yield {"type": "meta", "tier": tier, "model": spec.model, "provider": spec.provider,
                               "route_reason": reason if tier in ("fixed", plan[0][0]) else f"{reason}; answered by the {tier} tier",
                               "first_token_ms": int((time.monotonic() - a0) * 1000)}
                    yield {"type": "delta", "text": piece}
                if not sent:
                    raise providers.ProviderError("empty answer")
            except (providers.ProviderError, asyncio.TimeoutError, httpx.HTTPError) as e:
                ms = int((time.monotonic() - a0) * 1000)
                self._failed(spec)
                self.usage.record(req.task, tier, spec.key, False, ms, {}, str(e))
                if sent:
                    yield {"type": "error", "message": f"{spec.model} stopped: {str(e)[:300]}"}
                    return
                errors.append(f"{spec.key}: {str(e)[:300]}")
                continue
            ms = int((time.monotonic() - a0) * 1000)
            self._fails[spec.key] = 0
            self.usage.record(req.task, tier, spec.key, True, ms, usage, None)
            yield {"type": "done", "latency_ms": int((time.monotonic() - t0) * 1000), "usage": usage}
            return
        yield {"type": "error", "message": "No model could answer: " + ("; ".join(errors) or "no model is available")}

    async def _fixed(self, req: ChatRequest, t0: float) -> ChatResponse:
        """Routing switched off: every request goes to the one configured model (router.fixed)."""
        spec = self.s.fixed
        if spec is None:
            raise NoModelError("Routing is off but router.fixed has no model (set PULSEIQ8_FIXED_MODEL)")
        if spec.cloud and (self.s.data_policy == "local_only" or req.sensitive):
            raise NoModelError(f"Routing is off and the fixed model {spec.key} is a cloud model, "
                               "which the data policy / a sensitive request does not allow")
        if not self.s.providers[spec.provider].configured:
            raise NoModelError(f"Routing is off and {spec.provider} has no API key")
        await self.installed()   # finds Ollama on another local address if needed
        system = "\n\n".join(x for x in (req.system, privacy.INJECTION_GUARD) if x)
        sys_extra = "\n\n".join(m.content for m in req.messages if m.role == "system")
        if sys_extra:
            system = sys_extra + "\n\n" + system
        if req.json_output:
            system += "\n\nAnswer with ONE valid JSON object only — no markdown, no text before or after it."
        msgs = [m.model_dump() for m in req.messages if m.role != "system"]
        if spec.cloud:
            msgs = [{**m, "content": privacy.mask(m["content"])} for m in msgs]
        a0 = time.monotonic()
        try:
            res = await providers.call(self.client, self.s.providers[spec.provider], spec, system, msgs,
                                       max_tokens=req.max_tokens, temperature=req.temperature,
                                       json_output=req.json_output, timeout=self.s.request_timeout)
            parsed = extract_json(res.text) if req.json_output else None
            if req.json_output and parsed is None:
                raise providers.ProviderError("answer was not valid JSON")
        except (providers.ProviderError, asyncio.TimeoutError, httpx.HTTPError) as e:
            ms = int((time.monotonic() - a0) * 1000)
            self.usage.record(req.task, "fixed", spec.key, False, ms, {}, str(e))
            raise NoModelError(f"The fixed model {spec.key} could not answer: {str(e)[:300]}") from e
        ms = int((time.monotonic() - a0) * 1000)
        self.usage.record(req.task, "fixed", spec.key, True, ms, res.usage, None)
        return ChatResponse(text=res.text, json=parsed, tier="fixed", model=spec.model, provider=spec.provider,
                            route_reason="routing off — fixed model", latency_ms=int((time.monotonic() - t0) * 1000),
                            attempts=[Attempt(tier="fixed", model=spec.key, ok=True, ms=ms)], usage=res.usage)

    def _failed(self, spec: ModelSpec) -> None:
        n = self._fails.get(spec.key, 0) + 1
        self._fails[spec.key] = n
        if n >= 3:
            self._open_until[spec.key] = time.monotonic() + self.s.breaker_seconds
            self._fails[spec.key] = 0

    async def status(self) -> dict:
        installed = await self.installed()
        tiers = {}
        for name, t in self.s.tiers.items():
            rows = []
            for m in t.models:
                ok, why = self.available(m, installed)
                if ok and installed is None and self.s.providers[m.provider].type == "ollama":
                    ok, why = False, "Ollama not reachable"
                if ok and (self.s.data_policy == "local_only" and m.cloud):
                    ok, why = False, "blocked by data_policy local_only"
                rows.append({"provider": m.provider, "model": m.model, "cloud": m.cloud, "context": m.context,
                             "available": ok, "reason": why})
            tiers[name] = {"title": t.title, "summary": t.summary, "models": rows,
                           "ready": any(r["available"] for r in rows)}
        ollama = next((p for p in self.s.providers.values() if p.type == "ollama"), None)
        fixed = self.s.fixed
        return {"routing": self.s.routing,
                "fixed_model": None if fixed is None else {"provider": fixed.provider, "model": fixed.model,
                                                           "context": fixed.context,
                                                           "available": self.available(fixed, installed)[0],
                                                           "reason": self.available(fixed, installed)[1]},
                "ollama_reachable": installed is not None, "ollama_url": ollama.base_url if ollama else None,
                "ollama_error": self.ollama_error, "ollama_models": sorted(installed or []),
                "data_policy": self.s.data_policy, "tiers": tiers}
