"""The tools the assistant may use.

All data tools come from AnalyticsIQ's approved catalogue (/api/ai/tools) for the user's role, and run there as that
user. There is no SQL tool and no direct database access. The only local tool is `system_help` (product guidance).
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from . import knowledge
from .source import AnalyticsIQ, SourceError

MAX_RESULT_CHARS = 12000
NAME_ARGS = {"region": "regions", "segment": "segments", "rep": "reps"}

HELP_TOOL = {
    "name": "system_help",
    "description": "How AnalyticsIQ works and how to use it (uploads, customers to review, how sales are credited, claims, "
                   "targets and pace, regions, reports, sign-in and roles, what the assistant can do). No sales data.",
    "args": {"question": "what the user wants to know"},
}


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]
    remote: bool = True


@dataclasses.dataclass
class Ctx:
    src: AnalyticsIQ
    delegation: str | None
    _tools: list[Tool] | None = None
    _names: dict | None = None

    async def tools(self) -> list[Tool]:
        if self._tools is None:
            found = [Tool(t["name"], t.get("description", ""), dict(t.get("args") or {})) for t in await self.src.catalogue(self.delegation)]
            found.append(Tool(HELP_TOOL["name"], HELP_TOOL["description"], HELP_TOOL["args"], remote=False))
            self._tools = found
        return self._tools

    async def names(self) -> dict:
        if self._names is None:
            try:
                self._names = await self.src.call("get_filter_options", {}, self.delegation) or {}
            except SourceError:
                self._names = {}
        return self._names

    async def resolve(self, kind: str, value: Any) -> Any:
        """Map what the model wrote ("Zone d", "synthia", "hospitals") onto the exact master name."""
        if value in (None, ""):
            return value
        v = str(value).strip().upper()
        options = [str(x) for x in (await self.names()).get(kind, [])]
        if not options or v in options:
            return v
        if v.endswith("S") and v[:-1] in options:
            return v[:-1]
        hits = [o for o in options if v in o] or [o for o in options if o in v]
        return hits[0] if len(hits) == 1 else v


def describe(tools: list[Tool]) -> str:
    lines = []
    for t in tools:
        params = ", ".join(f"{k}: {v}" for k, v in t.params.items()) or "no arguments"
        lines.append(f"- {t.name}({params}) — {t.description}")
    return "\n".join(lines)


async def call(ctx: Ctx, name: str, args: dict) -> tuple[bool, Any]:
    """Returns (ok, result). Errors are returned for the model to correct, never raised."""
    if not isinstance(args, dict):
        args = {}
    try:
        tools = {t.name: t for t in await ctx.tools()}
    except SourceError as e:
        return False, {"error": str(e)}
    tool = tools.get(name)
    if tool is None:
        return False, {"error": f"unknown or not allowed tool '{name}'; use one of: {', '.join(tools)}"}
    if not tool.remote:
        return True, {"guides": knowledge.search(str(args.get("question") or args.get("topic") or ""))}
    clean = {k: v for k, v in args.items() if v not in (None, "", [])}
    for arg, kind in NAME_ARGS.items():
        if arg in clean:
            clean[arg] = await ctx.resolve(kind, clean[arg])
    try:
        return True, await ctx.src.call(name, clean, ctx.delegation)
    except SourceError as e:
        if e.status == 403:
            return False, {"error": "not allowed for this user: " + str(e), "permission": True}
        return False, {"error": str(e)}


def compact(result: Any) -> str:
    text = json.dumps(result, default=str, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f'… [truncated {len(text) - MAX_RESULT_CHARS} chars; ask with filters or a smaller limit]'
