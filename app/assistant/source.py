"""How the assistant reaches AnalyticsIQ.

The assistant has NO database access and never writes SQL. It may only call the approved tools AnalyticsIQ publishes at
/api/ai/tools; AnalyticsIQ runs them with its own business logic, limits them to the user's role and scope, and audits
every call. Each call carries:

  X-AI-Key          the service key shared by PulseIQ8_AI and AnalyticsIQ (ANALYTICSIQ_AI_KEY here, AI_SERVICE_KEY there)
  X-AI-Delegation   a short-lived token AnalyticsIQ issued for the signed-in user (web) or the WhatsApp number's owner

Configure with:
  ANALYTICSIQ_URL      http://localhost:8090/api   (in Docker: http://host.docker.internal:8090/api)
  ANALYTICSIQ_AI_KEY   the shared service key
"""
from __future__ import annotations

import os
from typing import Any

import httpx


class SourceError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class AnalyticsIQ:
    name = "analyticsiq"
    title = "AnalyticsIQ sales analytics"

    def __init__(self, client: httpx.AsyncClient, base_url: str | None = None, key: str | None = None):
        self.client = client
        self.base = (base_url or os.environ.get("ANALYTICSIQ_URL", "http://localhost:8090/api")).rstrip("/")
        self.key = key if key is not None else os.environ.get("ANALYTICSIQ_AI_KEY", "")

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def _headers(self, delegation: str | None) -> dict[str, str]:
        h = {"X-AI-Key": self.key}
        if delegation:
            h["X-AI-Delegation"] = delegation
        return h

    async def _send(self, method: str, path: str, delegation: str | None = None, body: Any = None) -> Any:
        if not self.key:
            raise SourceError("The assistant is not connected to AnalyticsIQ (set ANALYTICSIQ_AI_KEY)")
        try:
            r = await self.client.request(method, f"{self.base}{path}", json=body, headers=self._headers(delegation), timeout=60)
        except httpx.HTTPError as e:
            raise SourceError(f"AnalyticsIQ is not reachable at {self.base} ({type(e).__name__})") from e
        if r.status_code >= 400:
            detail = ""
            try:
                detail = (r.json() or {}).get("detail", "")
            except ValueError:
                detail = r.text[:200]
            raise SourceError(detail or f"AnalyticsIQ answered {r.status_code}", r.status_code)
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    async def catalogue(self, delegation: str | None) -> list[dict]:
        """The approved tools for this user's role."""
        data = await self._send("GET", "/ai/tools", delegation)
        return list((data or {}).get("tools", []))

    async def call(self, name: str, args: dict, delegation: str) -> Any:
        """Run one approved tool as the delegated user (read-only, scoped and audited by AnalyticsIQ)."""
        return await self._send("POST", f"/ai/tools/{name}", delegation, args or {})

    async def identity(self, phone: str, channel: str = "whatsapp") -> dict | None:
        """Who owns this phone number? None when it is not registered (or WhatsApp is not enabled for the user)."""
        try:
            return await self._send("POST", "/ai/identity", None, {"phone": phone, "channel": channel})
        except SourceError as e:
            if e.status == 404:
                return None
            raise

    async def audit(self, delegation: str | None, kind: str, name: str | None, detail: str, ok: bool = True,
                    ms: int | None = None, rows: int | None = None) -> None:
        """Best effort: record a question / answer from a channel AnalyticsIQ does not see (WhatsApp)."""
        if not delegation:
            return
        try:
            await self._send("POST", "/ai/audit", delegation,
                             {"kind": kind, "name": name, "detail": detail[:4000], "ok": ok, "ms": ms, "rows": rows})
        except SourceError:
            pass
