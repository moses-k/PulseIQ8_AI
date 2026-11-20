"""Loads config/models.yaml (with ${VAR:-default} from the environment) into typed settings."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TIERS = ("fast", "coding", "general", "reasoning", "vision")
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(text: str) -> str:
    return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)


@dataclass(frozen=True)
class Provider:
    name: str
    type: str          # ollama | openai | anthropic | gemini
    base_url: str
    api_key: str = ""

    @property
    def configured(self) -> bool:
        return self.type == "ollama" or bool(self.api_key)


@dataclass(frozen=True)
class ModelSpec:
    tier: str
    provider: str
    model: str
    context: int = 8192
    cloud: bool = False
    vision: bool = False
    thinking: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass
class Tier:
    name: str
    title: str
    summary: str
    models: list[ModelSpec] = field(default_factory=list)


@dataclass
class Settings:
    api_keys: list[str]
    data_policy: str
    request_timeout: float
    llm_classifier: bool
    escalate: bool
    routing: bool
    fixed: "ModelSpec | None"
    breaker_seconds: int
    providers: dict[str, Provider]
    tiers: dict[str, Tier]


def _bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def load(path: str | os.PathLike | None = None) -> Settings:
    path = Path(path or os.environ.get("PULSEIQ8_AI_CONFIG", Path(__file__).resolve().parent.parent / "config" / "models.yaml"))
    raw = yaml.safe_load(_expand(path.read_text(encoding="utf-8"))) or {}
    svc, router = raw.get("service", {}) or {}, raw.get("router", {}) or {}

    providers = {
        name: Provider(name=name, type=str(p.get("type", name)), base_url=str(p.get("base_url", "")).rstrip("/"),
                       api_key=str(p.get("api_key") or ""))
        for name, p in (raw.get("providers") or {}).items()
    }
    tiers: dict[str, Tier] = {}
    for name in TIERS:
        t = (raw.get("tiers") or {}).get(name) or {}
        models = []
        for m in t.get("models") or []:
            model = str(m.get("model") or "").strip()
            if not model or m.get("provider") not in providers:
                continue                       # empty name = switched off
            models.append(ModelSpec(tier=name, provider=m["provider"], model=model, context=int(m.get("context") or 8192),
                                    cloud=bool(m.get("cloud", False)), vision=bool(m.get("vision", False)),
                                    thinking=bool(m.get("thinking", False))))
        tiers[name] = Tier(name=name, title=str(t.get("title", name.title())), summary=str(t.get("summary", "")), models=models)

    policy = str(svc.get("data_policy") or "local_first").strip()
    if policy not in ("local_only", "local_first", "cloud_ok"):
        policy = "local_first"
    keys = [k for k in (svc.get("api_keys") or []) if k]
    # PULSEIQ8_AI_KEYS may hold several comma-separated keys
    keys = [part.strip() for k in keys for part in str(k).split(",") if part.strip()]
    fx = router.get("fixed") or {}
    fixed = None
    if str(fx.get("model") or "").strip() and fx.get("provider") in providers:
        ptype = providers[fx["provider"]].type
        fixed = ModelSpec(tier="fixed", provider=fx["provider"], model=str(fx["model"]).strip(),
                          context=int(fx.get("context") or 16384), cloud=ptype != "ollama", vision=True)
    return Settings(api_keys=keys, data_policy=policy, request_timeout=float(svc.get("request_timeout_seconds") or 300),
                    llm_classifier=_bool(router.get("llm_classifier", False)), escalate=_bool(router.get("escalate", True)),
                    routing=_bool(router.get("enabled", True)), fixed=fixed,
                    breaker_seconds=int(router.get("breaker_seconds") or 60), providers=providers, tiers=tiers)
