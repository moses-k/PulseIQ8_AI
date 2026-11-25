"""The data assistant: a plan → fetch → answer loop over the router.

Each turn the model replies with ONE JSON object: either a tool call {"action": name, "args": {...}, "why": "..."} or the
final answer {"answer": markdown, "table"?, "chart"?, "followups"?}. Tool results go back to the model inside <data>
tags (untrusted). The router's tiers, fallback, escalation and data policy apply to every call, so this works with
local Ollama models and cloud models alike.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from typing import Any, AsyncIterator

from .. import privacy
from ..schemas import ChatRequest, Message
from . import tools as T
from .source import AnalyticsIQ, SourceError

MAX_STEPS = int(os.environ.get("ASSISTANT_MAX_STEPS", "6"))
_HARD = re.compile(r"\bwhy\b|compare|versus|\bvs\b|trend|forecast|predict|root cause|explain|strategy|recommend|plan", re.I)

GLOSSARY = """Business rules of AnalyticsIQ (Moscom pharma sales, amounts in KES, VAT inclusive):
- Each sales line has a date, document, customer, rep, region and segment (CHEMIST, DISTRIBUTORS, EQA, HOSPITAL, INSTITUTIONS, MEDS).
- Direct sales count for the customer's region and rep. Distributor purchases (customers under distributor regions such as JACARANDA, MEDS, KENTONS LIMITED) are company revenue but count for no region or rep until claimed.
- Claims: reps claim sales distributors made to non-onboarded customers. Claim lines in sales files ("<DISTRIBUTOR> CLAIMS") and approved claims count for the claiming rep and region.
- Company totals = all invoiced lines. Region / rep totals = credited sales = direct + claims.
- Achievement = sales / target. Projection = sales so far / days elapsed x days in month. Targets are monthly (segment, region, rep)."""


CHANNEL_STYLE = {
    "web": "The reply is shown in the AnalyticsIQ web app: markdown is fine; add a table for lists and a chart for trends.",
    "whatsapp": "The reply is sent on WhatsApp: keep it short (under 120 words), no markdown headings or links, use *bold* sparingly; "
                "a small table (max 10 rows, max 4 columns) is fine; no chart.",
}

READ_ONLY = """You are read-only. You cannot approve or reject claims, change targets, edit customers, reps, regions or segments, upload
or delete files, or manage users. If asked, say that clearly, explain which page and role can do it (use system_help), and never claim
an action was done."""


def _who(user: dict) -> str:
    if not user:
        return "The user is not identified."
    role = user.get("role", "")
    scope = user.get("scope") or ("all data" if role in ("ADMIN", "MANAGER") else "own data")
    name = user.get("displayName") or user.get("username") or "the user"
    return f"You are talking to {name} (role {role}; data scope: {scope}). Address them by first name now and then."


def system_prompt(tools: list[T.Tool], context: dict, today: str) -> str:
    user = context.get("user") if isinstance(context.get("user"), dict) else {}
    channel = str(context.get("channel") or "web")
    filters = {k: v for k, v in context.items() if k not in ("user", "channel", "delegation") and v not in (None, "", [], {})}
    ctx = "; ".join(f"{k}={v}" for k, v in filters.items()) or "no dashboard filters"
    return f"""You are Nexora AI, the assistant inside AnalyticsIQ (Moscom pharma sales analytics). You answer questions about sales,
targets, reps, regions, segments, customers, distributors and claims by calling approved tools, and questions about how to use
the system with system_help. Explain results in clear business English.

Today is {today}. {_who(user)}
Channel: {channel}. {CHANNEL_STYLE.get(channel, CHANNEL_STYLE["web"])}
Page filters: {ctx}.
{GLOSSARY}

TOOLS (the only ones you may use; they already apply the user's permissions)
{T.describe(tools)}

HOW TO REPLY — always exactly ONE JSON object, nothing else:
1) to call a tool: {{"action": "<tool name>", "args": {{...}}, "why": "<a few words>"}}
2) to answer:      {{"answer": "<markdown>", "table": {{"columns": [...], "rows": [[...]]}}, "chart": {{"type": "bar"|"line", "title": "...", "x": [...], "series": [{{"name": "...", "values": [...]}}]}}, "followups": ["...", "..."]}}
   table, chart and followups are optional.

RULES
- Never invent figures: every number (sales, targets, claims, projections) must come from a tool result in this conversation.
  If no tool gives it, say you do not have it.
- Pick the most specific tool: get_company_sales for totals, get_sales_by_region / _rep / _segment for rankings, get_target_achievement
  for who is behind, get_month_end_projection for "will we hit target", get_inactive_customers / get_declining_customers for customers
  to follow up, get_customer_sales for one customer, get_unclaimed_sales / get_distributor_claims for distributors,
  generate_management_summary for a briefing, get_filter_options for exact names or the date range.
- "This month" means the month of the latest data unless the user says otherwise.
- If a tool says the user is not allowed, tell them politely that their role does not give access; do not try to get around it.
- {READ_ONLY}
- Show money as exact amounts with two decimals and KES (e.g. KES 1,275,153.09); percentages with one decimal.
- Lead with the direct answer in one sentence, then key figures, then a short "what to do next" when useful (who to call,
  which customers to visit, what to fix). Be concise.
- If a tool returns an error, fix the arguments or try another tool; if the data cannot answer, say so plainly.
- At most {MAX_STEPS} tool calls per question.
{privacy.INJECTION_GUARD}"""


def _clean_answer(obj: dict, fallback_text: str) -> dict:
    answer = obj.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = fallback_text.strip() or "I could not produce an answer."
    out: dict[str, Any] = {"answer": answer.strip()}
    tbl = obj.get("table")
    if isinstance(tbl, dict) and isinstance(tbl.get("columns"), list) and isinstance(tbl.get("rows"), list):
        cols = [str(c) for c in tbl["columns"]][:12]
        rows = [r[:len(cols)] for r in tbl["rows"] if isinstance(r, list)][:200]
        if cols and rows:
            out["table"] = {"columns": cols, "rows": rows}
    ch = obj.get("chart")
    if isinstance(ch, dict) and isinstance(ch.get("x"), list) and isinstance(ch.get("series"), list):
        series = [{"name": str(s.get("name", "")), "values": [v if isinstance(v, (int, float)) else None for v in s.get("values", [])][:200]}
                  for s in ch["series"] if isinstance(s, dict)][:6]
        if series:
            out["chart"] = {"type": "line" if ch.get("type") == "line" else "bar", "title": str(ch.get("title", "")),
                            "x": [str(x) for x in ch["x"]][:200], "series": series}
    fu = obj.get("followups")
    if isinstance(fu, list):
        out["followups"] = [str(x) for x in fu if str(x).strip()][:4]
    return out


def _summary(name: str, ok: bool, res: Any) -> str:
    if not ok:
        return str((res or {}).get("error", "failed"))[:200] if isinstance(res, dict) else "failed"
    if isinstance(res, dict):
        for key in ("rows", "lines", "reps", "targets", "claims", "customers", "days", "guides", "trend", "byMonth"):
            if isinstance(res.get(key), list):
                return f"{len(res[key])} {key}"
        if "kpi" in res:
            k = res["kpi"]
            return f"sales {k.get('totalSales')}" + (f", {k.get('achievementPct')}% of target" if k.get("achievementPct") is not None else "")
    return "done"


class Assistant:
    def __init__(self, router, source: AnalyticsIQ):
        self.router = router
        self.src = source

    async def run(self, messages: list[Message], context: dict | None = None, tier: str = "auto",
                  local_only: bool | None = None, model: str | None = None) -> AsyncIterator[dict]:
        """Yields events: step (tool chosen) → result (tool done) … → answer | error."""
        t0 = time.monotonic()
        context = dict(context or {})
        delegation = context.pop("delegation", None)
        if not self.src.configured:
            yield {"type": "error", "error": "The assistant is not connected to AnalyticsIQ (set ANALYTICSIQ_AI_KEY)", "steps": []}
            return
        if not delegation:
            yield {"type": "error", "error": "Ask from AnalyticsIQ (Ask AI) or registered WhatsApp: the assistant only answers for a signed-in user",
                   "steps": []}
            return
        ctx = T.Ctx(self.src, delegation)
        try:
            tools = await ctx.tools()
        except SourceError as e:
            yield {"type": "error", "error": str(e), "steps": []}
            return
        question = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if tier == "auto":
            tier = "reasoning" if _HARD.search(question) and len(question) > 40 else "general"
        sensitive = (os.environ.get("ASSISTANT_LOCAL_ONLY", "true").lower() != "false") if local_only is None else local_only
        system = system_prompt(tools, context, date.today().isoformat())
        convo = [m for m in messages if m.role in ("user", "assistant")][-12:]
        scratch: list[Message] = []
        seen: set[str] = set()
        steps: list[dict] = []
        last = None

        for step in range(MAX_STEPS + 1):
            final_turn = step == MAX_STEPS
            extra = [Message(role="user", content="You have used all tool calls. Reply now with the final answer JSON.")] if final_turn else []
            req = ChatRequest(messages=convo + scratch + extra, system=system, tier=tier, model=model or None, task="assistant.nexora",
                              json_output=True, max_tokens=2000, temperature=0.1, sensitive=sensitive)
            try:
                res = await self.router.chat(req)
            except Exception as e:  # NoModelError, provider failures after all fallbacks
                yield {"type": "error", "error": str(e), "steps": steps}
                return
            last = res
            obj = res.json_data if isinstance(res.json_data, dict) else {}
            action = obj.get("action") or obj.get("tool")
            if action and not final_turn and "answer" not in obj:
                args = obj.get("args") or obj.get("arguments") or {}
                key = f"{action}:{json.dumps(args, sort_keys=True, default=str)}"
                yield {"type": "step", "n": len(steps) + 1, "tool": action, "args": args, "why": str(obj.get("why", ""))[:160]}
                s0 = time.monotonic()
                if key in seen:
                    ok, result = False, {"error": "you already called this tool with these arguments; use the earlier result"}
                else:
                    seen.add(key)
                    ok, result = await T.call(ctx, str(action), args)
                ms = int((time.monotonic() - s0) * 1000)
                info = {"tool": action, "args": args, "ok": ok, "ms": ms, "summary": _summary(action, ok, result)}
                steps.append(info)
                yield {"type": "result", **info}
                scratch.append(Message(role="assistant", content=json.dumps({"action": action, "args": args})))
                scratch.append(Message(role="user", content=f'<data tool="{action}">{T.compact(result)}</data>\n'
                                                             "Continue: call another tool or give the final answer JSON."))
                continue
            answer = _clean_answer(obj, res.text if not obj else "")
            yield {"type": "answer", **answer, "steps": steps, "tier": res.tier, "model": res.model,
                   "provider": res.provider, "latency_ms": int((time.monotonic() - t0) * 1000)}
            return
        yield {"type": "error", "error": "no answer after the maximum number of steps", "steps": steps,
               "model": getattr(last, "model", None)}

    async def ask(self, messages: list[Message], context: dict | None = None, tier: str = "auto",
                  local_only: bool | None = None, model: str | None = None) -> dict:
        final: dict = {}
        async for ev in self.run(messages, context, tier, local_only, model):
            if ev["type"] in ("answer", "error"):
                final = ev
        return final
