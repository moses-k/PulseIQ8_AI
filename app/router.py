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

from . import privacy, providers
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
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t)
    for opener, closer in (("{", "}"), ("[", "]")):
        a, b = t.find(opener), t.rfind(closer)
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except ValueError:
                continue
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

    def _fixed_only(self, req: ChatRequest) -> bool:
        """Routing off: the fixed model answers, unless the caller picked a mode (tier) explicitly."""
        return not self.s.routing and (req.tier or "auto") == "auto" and not req.model

    async def chosen_model(self, req: ChatRequest) -> ModelSpec:
        """The model the caller picked by name ("qwen3:8b" or "provider:model"). Settings (context, thinking) come
        from the first tier that lists it; other installed Ollama models are allowed with the fixed model's context."""
        name = (req.model or "").strip()
        provider, model = ("ollama", name)
        if ":" in name and name.split(":", 1)[0] in self.s.providers:
            provider, model = name.split(":", 1)
        known = [m for t in self.s.tiers.values() for m in t.models if m.provider == provider and m.model == model]
        if req.tier not in (None, "auto"):
            known = sorted(known, key=lambda m: m.tier != req.tier)        # the chosen mode's settings first
        fixed = self.s.fixed
        if known:
            spec = known[0]
        elif fixed is not None and fixed.provider == provider and fixed.model == model:
            spec = fixed
        elif self.s.providers.get(provider) is not None and self.s.providers[provider].type == "ollama":
            spec = ModelSpec(tier="model", provider=provider, model=model, context=fixed.context if fixed else 8192)
        else:
            raise NoModelError(f"Model {name} is not configured")
        if spec.cloud and (self.s.data_policy == "local_only" or req.sensitive):
            raise NoModelError(f"{spec.key} is a cloud model, which the data policy / a sensitive request does not allow")
        ok, why = self.available(spec, await self.installed())
        if not ok:
            raise NoModelError(f"Model {spec.model} cannot answer: {why}")
        return spec

    async def _plan(self, req: ChatRequest):
        """Route the request: (route, system prompt, messages, [(tier, model) in the order to try])."""
        if req.model:
            spec = await self.chosen_model(req)
            system = "\n\n".join(x for x in (req.system, privacy.INJECTION_GUARD) if x)
            sys_extra = "\n\n".join(m.content for m in req.messages if m.role == "system")
            if sys_extra:
                system = sys_extra + "\n\n" + system
            if req.json_output:
                system += "\n\nAnswer with ONE valid JSON object only — no markdown, no text before or after it."
            tier = spec.tier if spec.tier in self.s.tiers else (req.tier if req.tier != "auto" else "general")
            return (Route(tier, f"model {spec.model} chosen by the user"), system,
                    [m.model_dump() for m in req.messages if m.role != "system"], [(tier, spec)])
        route = decide(req, self.task_tiers)
        if not route.confident and self.s.llm_classifier:
            tier = await self.classify(req)
            if tier and tier != route.tier:
                route = Route(tier, f"classified by the Fast model ({route.reason} by rules)")
        system = "\n\n".join(x for x in (req.system, privacy.INJECTION_GUARD) if x)
        tokens = estimate_tokens(req, system)
        escalate = self.s.escalate if req.escalate is None else req.escalate
        chain = [route.tier] + (ESCALATION[route.tier] if escalate else [])
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
        return route, system, base_msgs, plan

    async def chat(self, req: ChatRequest) -> ChatResponse:
        t0 = time.monotonic()
        if self._fixed_only(req):
            return await self._fixed(req, t0)
        route, system, base_msgs, plan = await self._plan(req)
        attempts: list[Attempt] = []
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

    async def chat_stream(self, req: ChatRequest):
        """Streams one answer as events: meta (model chosen) → delta (text pieces) … → done | error.
        Same routing, privacy masking and fallback as chat(); a model that fails before writing anything is
        skipped for the next one, a failure after text was sent ends the stream with an error."""
        t0 = time.monotonic()
        if self._fixed_only(req):
            spec = self.s.fixed
            if spec is None:
                yield {"type": "error", "message": "Routing is off but router.fixed has no model (set PULSEIQ8_FIXED_MODEL)"}
                return
            if spec.cloud and (self.s.data_policy == "local_only" or req.sensitive):
                yield {"type": "error", "message": f"The fixed model {spec.key} is a cloud model, which the data policy does not allow"}
                return
            await self.installed()
            system = "\n\n".join(x for x in (req.system, privacy.INJECTION_GUARD) if x)
            sys_extra = "\n\n".join(m.content for m in req.messages if m.role == "system")
            if sys_extra:
                system = sys_extra + "\n\n" + system
            route, base_msgs, plan = Route("fixed", "routing off — fixed model"), [m.model_dump() for m in req.messages if m.role != "system"], [("fixed", spec)]
        else:
            try:
                route, system, base_msgs, plan = await self._plan(req)
            except Exception as e:  # noqa: BLE001 — report, never break the stream
                yield {"type": "error", "message": str(e)[:300]}
                return
        errors: list[str] = []
        for tier, spec in plan:
            msgs = base_msgs
            if spec.cloud:
                msgs = [{**m, "content": privacy.mask(m["content"])} for m in base_msgs]
            why = route.reason if tier == route.tier else f"{route.reason}; answered by the {tier} tier"
            usage: dict = {}
            a0 = time.monotonic()
            sent = False
            try:
                async for piece in providers.stream(self.client, self.s.providers[spec.provider], spec, system, msgs,
                                                    max_tokens=req.max_tokens, temperature=req.temperature,
                                                    timeout=self.s.request_timeout, usage=usage):
                    if not sent:
                        yield {"type": "meta", "tier": tier, "model": spec.model, "provider": spec.provider, "route_reason": why}
                        sent = True
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
                errors.append(f"{spec.key}: {str(e)[:200]}")
                continue
            ms = int((time.monotonic() - a0) * 1000)
            self._fails[spec.key] = 0
            self.usage.record(req.task, tier, spec.key, True, ms, usage, None)
            yield {"type": "done", "latency_ms": int((time.monotonic() - t0) * 1000), "usage": usage}
            return
        yield {"type": "error", "message": f"No model could answer ({route.tier}): " + ("; ".join(errors) or "no model is available for this tier")}

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
