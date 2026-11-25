"""Nexora AI assistant: approved tools via a fake AnalyticsIQ AI API, delegation and permissions, the agent loop,
the help tool, the WhatsApp channel and the API (no real models, no real Meta)."""
import asyncio
import hashlib
import hmac
import json
import os

import httpx
import pytest

os.environ.setdefault("PULSEIQ8_AI_KEYS", "test-key")

from app.assistant import knowledge, tools as T  # noqa: E402
from app.assistant.agent import Assistant, system_prompt  # noqa: E402
from app.assistant.source import AnalyticsIQ  # noqa: E402
from app.channels import whatsapp as W  # noqa: E402
from app.schemas import ChatResponse, Message  # noqa: E402

SERVICE_KEY = "svc-key"
GOOD = "deleg-admin"
REP = "deleg-rep"
USERS = {GOOD: {"id": 1, "username": "admin", "displayName": "Mary Admin", "role": "ADMIN", "scope": "all data"},
         REP: {"id": 2, "username": "synthia", "displayName": "Synthia W", "role": "REP", "rep": "SYNTHIA", "scope": "rep SYNTHIA"}}
CATALOGUE = [
    {"name": "get_filter_options", "description": "names", "args": {}},
    {"name": "get_company_sales", "description": "totals", "args": {"from": "date", "region": "region", "rep": "rep", "segment": "segment"}},
    {"name": "get_sales_by_region", "description": "per region", "args": {}},
    {"name": "get_inactive_customers", "description": "stopped buying", "args": {"days": "14"}},
    {"name": "get_upload_issues", "description": "staff only", "args": {}},
]
KPI = {"kpi": {"totalSales": 35463516.98, "target": 58500000, "achievementPct": 60.6}}


class FakeAiq:
    """AnalyticsIQ /api/ai/*: checks the service key and the delegation like the real controller."""

    def __init__(self, phones=None):
        self.calls = []
        self.audits = []
        self.phones = phones or {"254712345678": GOOD}

    def __call__(self, request: httpx.Request):
        path = request.url.path.removeprefix("/api")
        body = json.loads(request.content or b"{}") if request.method == "POST" else None
        self.calls.append((request.method, path, body, request.headers.get("X-AI-Delegation")))
        if request.headers.get("X-AI-Key") != SERVICE_KEY:
            return httpx.Response(401, json={"detail": "Wrong X-AI-Key"})
        if path == "/ai/identity":
            d = self.phones.get(body["phone"])
            return httpx.Response(200, json={"delegation": d, "user": USERS[d]}) if d else httpx.Response(404, json={"detail": "not registered"})
        deleg = request.headers.get("X-AI-Delegation")
        if deleg not in USERS:
            return httpx.Response(401, json={"detail": "The delegation has expired or is invalid"})
        role = USERS[deleg]["role"]
        if path == "/ai/tools":
            return httpx.Response(200, json={"tools": [t for t in CATALOGUE if role != "REP" or t["name"] != "get_upload_issues"],
                                             "user": USERS[deleg]})
        if path == "/ai/audit":
            self.audits.append(body)
            return httpx.Response(204)
        if path.startswith("/ai/tools/"):
            name = path.rsplit("/", 1)[1]
            if name == "get_upload_issues" and role == "REP":
                return httpx.Response(403, json={"detail": "Your role cannot use get_upload_issues"})
            if name == "get_filter_options":
                return httpx.Response(200, json={"regions": ["ZONE D", "LOWER EASTERN"], "segments": ["HOSPITAL", "CHEMIST"],
                                                 "reps": ["SYNTHIA", "JOSHUA MWALIMU"]})
            if name == "get_company_sales":
                return httpx.Response(200, json={"filters": body, **KPI})
            return httpx.Response(200, json={"customers": [{"name": "A"}]})
        return httpx.Response(404, json={"detail": "no such endpoint"})


def source(fake=None, key=SERVICE_KEY):
    fake = fake or FakeAiq()
    return AnalyticsIQ(httpx.AsyncClient(transport=httpx.MockTransport(fake)), base_url="http://aiq/api", key=key), fake


class FakeRouter:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    async def chat(self, req):
        self.requests.append(req)
        if not self.replies:
            raise RuntimeError("no more replies")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return ChatResponse(text=json.dumps(r), json=r, tier=req.tier, model="fake", provider="ollama",
                            route_reason="test", latency_ms=1, attempts=[])


def ask(router, src, question="How is Zone D doing?", delegation=GOOD, **kw):
    ctx = {"user": USERS.get(delegation, {}), "channel": "web", "delegation": delegation, "region": "ZONE D"}
    return asyncio.run(Assistant(router, src).ask([Message(role="user", content=question)], ctx, **kw))


# ---------------- tools ----------------

def test_tools_come_from_the_catalogue_for_the_role():
    src, _ = source()
    admin = {t.name for t in asyncio.run(T.Ctx(src, GOOD).tools())}
    rep = {t.name for t in asyncio.run(T.Ctx(src, REP).tools())}
    assert "get_upload_issues" in admin and "get_upload_issues" not in rep
    assert "system_help" in admin and "system_help" in rep
    assert not any("sql" in n for n in admin | rep)           # no SQL tool, ever


def test_names_are_resolved_and_delegation_is_sent():
    src, fake = source()
    ok, res = asyncio.run(T.call(T.Ctx(src, GOOD), "get_company_sales", {"region": "zone d", "rep": "synthia", "segment": "hospitals", "from": ""}))
    assert ok and res["kpi"]["totalSales"] == 35463516.98
    call = [c for c in fake.calls if c[1] == "/ai/tools/get_company_sales"][0]
    assert call[2] == {"region": "ZONE D", "rep": "SYNTHIA", "segment": "HOSPITAL"} and call[3] == GOOD


def test_unknown_tool_and_permission_errors_are_returned_not_raised():
    src, _ = source()
    ok, res = asyncio.run(T.call(T.Ctx(src, GOOD), "run_sql", {"sql": "select 1"}))
    assert not ok and "not allowed" in res["error"]
    ok, res = asyncio.run(T.call(T.Ctx(src, REP), "get_upload_issues", {}))
    assert not ok and "not allowed" in res["error"]       # hidden from the rep's catalogue


def test_server_side_denial_is_reported_as_permission():
    src, _ = source()
    ctx = T.Ctx(src, REP)
    ctx._tools = [T.Tool("get_upload_issues", "", {})]       # even if the model somehow names it, AnalyticsIQ refuses
    ok, res = asyncio.run(T.call(ctx, "get_upload_issues", {}))
    assert not ok and res.get("permission")


def test_help_tool_is_local():
    src, fake = source()
    ok, res = asyncio.run(T.call(T.Ctx(src, GOOD), "system_help", {"question": "how do I upload the excel file?"}))
    assert ok and res["guides"][0]["topic"] == "upload"
    assert not any(c[1].startswith("/ai/tools/system_help") for c in fake.calls)
    assert knowledge.search("approve a claim")[0]["topic"] == "claims"


def test_unreachable_source_is_a_clear_error():
    def down(_):
        raise httpx.ConnectError("refused")
    src = AnalyticsIQ(httpx.AsyncClient(transport=httpx.MockTransport(down)), base_url="http://aiq/api", key=SERVICE_KEY)
    ok, res = asyncio.run(T.call(T.Ctx(src, GOOD), "get_company_sales", {}))
    assert not ok and "not reachable" in res["error"]


def test_large_results_are_truncated():
    text = T.compact({"rows": ["x" * 100] * 500})
    assert len(text) < T.MAX_RESULT_CHARS + 200 and "truncated" in text


# ---------------- agent ----------------

def test_fetches_then_answers_with_steps():
    src, fake = source()
    router = FakeRouter([
        {"action": "get_company_sales", "args": {"region": "Zone D"}, "why": "totals"},
        {"answer": "Zone D is at **60.6%** of target.", "table": {"columns": ["Region", "Sales"], "rows": [["ZONE D", 1.0]]},
         "followups": ["Which reps are behind?"]},
    ])
    res = ask(router, src)
    assert res["type"] == "answer" and "60.6%" in res["answer"] and res["steps"][0]["ok"]
    second = router.requests[1].messages
    assert second[-1].content.startswith('<data tool="get_company_sales">')
    assert router.requests[0].json_output and router.requests[0].sensitive   # local models only by default
    system = router.requests[0].system
    assert "Mary Admin" in system and "Nexora AI" in system and "read-only" in system
    assert GOOD not in system                                                  # the delegation never reaches the model
    assert "region=ZONE D" in system


def test_requires_a_delegated_user():
    src, fake = source()
    res = ask(FakeRouter([{"answer": "x"}]), src, delegation=None)
    assert res["type"] == "error" and "signed-in user" in res["error"] and not fake.calls


def test_not_configured_is_an_error():
    src, _ = source(key="")
    res = ask(FakeRouter([{"answer": "x"}]), src)
    assert res["type"] == "error" and "ANALYTICSIQ_AI_KEY" in res["error"]


def test_expired_delegation_is_an_error():
    src, _ = source()
    res = ask(FakeRouter([{"answer": "x"}]), src, delegation="expired")
    assert res["type"] == "error" and "expired" in res["error"]


def test_repeated_call_is_not_run_twice():
    src, fake = source()
    res = ask(FakeRouter([{"action": "get_sales_by_region", "args": {}}, {"action": "get_sales_by_region", "args": {}}, {"answer": "Done."}]), src)
    assert [s["ok"] for s in res["steps"]] == [True, False]
    assert sum(1 for c in fake.calls if c[1] == "/ai/tools/get_sales_by_region") == 1


def test_step_limit_forces_final_answer():
    src, _ = source()
    router = FakeRouter([{"action": "get_inactive_customers", "args": {"days": i}} for i in range(1, 7)] + [{"answer": "Final."}])
    res = ask(router, src)
    assert res["answer"] == "Final." and len(res["steps"]) == 6


def test_router_failure_is_an_error_event():
    src, _ = source()
    res = ask(FakeRouter([RuntimeError("No model available")]), src)
    assert res["type"] == "error" and "No model" in res["error"]


def test_hard_questions_go_to_reasoning_tier():
    src, _ = source()
    router = FakeRouter([{"answer": "ok"}])
    ask(router, src, question="Why did Hospital sales drop compared with last month in the coast region?")
    assert router.requests[0].tier == "reasoning"


def test_bad_chart_and_table_are_dropped():
    src, _ = source()
    res = ask(FakeRouter([{"answer": "x", "table": {"columns": "no"}, "chart": {"x": [1], "series": "bad"}}]), src)
    assert "table" not in res and "chart" not in res


def test_whatsapp_prompt_style():
    p = system_prompt([], {"user": USERS[REP], "channel": "whatsapp"}, "2026-11-25")
    assert "WhatsApp" in p and "Synthia W" in p and "rep SYNTHIA" in p


# ---------------- WhatsApp ----------------

def payload(text, mid="wamid.1", phone="254712345678", kind="text"):
    msg = {"id": mid, "from": phone, "type": kind}
    if kind == "text":
        msg["text"] = {"body": text}
    return {"object": "whatsapp_business_account",
            "entry": [{"changes": [{"value": {"messages": [msg], "statuses": [{"id": "x", "status": "read"}]}}]}]}


class Sent:
    def __init__(self):
        self.out = []

    def __call__(self, request):
        b = json.loads(request.content)
        if b.get("type") == "text":
            self.out.append((b["to"], b["text"]["body"]))
        return httpx.Response(200, json={"messages": [{"id": "ok"}]})


def wa(router, fake=None, env=None):
    src, fake = source(fake)
    sent = Sent()
    os.environ.update({"WHATSAPP_VERIFY_TOKEN": "vt", "WHATSAPP_ACCESS_TOKEN": "at", "WHATSAPP_PHONE_NUMBER_ID": "123",
                       "WHATSAPP_APP_SECRET": "sec", **(env or {})})
    ch = W.WhatsApp(httpx.AsyncClient(transport=httpx.MockTransport(sent)), Assistant(router, src), src)
    return ch, sent, fake


def test_whatsapp_parsing_and_formatting():
    msgs = W.incoming(payload("hi"))
    assert msgs == [{"id": "wamid.1", "from": "254712345678", "type": "text", "text": "hi"}]
    assert W.incoming({"entry": [{"changes": [{"value": {"statuses": [{}]}}]}]}) == []
    assert W.to_whatsapp("## Title\n**KES 1.00** and [link](http://x)\n- a") == "*Title*\n*KES 1.00* and link\n• a"
    parts = W.split("a" * 9000)
    assert len(parts) == 3 and all(len(p) <= W.MAX_BODY for p in parts)
    assert "ZONE D" in W.table_text({"columns": ["Region", "Sales"], "rows": [["ZONE D", 1234.5]]})


def test_whatsapp_verify_and_signature():
    ch, _, _ = wa(FakeRouter([]))
    assert ch.verify("subscribe", "vt", "42") == "42" and ch.verify("subscribe", "bad", "42") is None
    body = b'{"a":1}'
    good = "sha256=" + hmac.new(b"sec", body, hashlib.sha256).hexdigest()
    assert ch.signature_ok(body, good) and not ch.signature_ok(body, "sha256=00") and not ch.signature_ok(body, None)


def test_whatsapp_answers_registered_user_with_audit():
    ch, sent, fake = wa(FakeRouter([{"action": "get_company_sales", "args": {}}, {"answer": "**Sales** are KES 35,463,516.98."}]))
    asyncio.run(ch.process(W.incoming(payload("How are we doing?"))[0]))
    assert sent.out[-1] == ("254712345678", "*Sales* are KES 35,463,516.98.")
    assert any(c[1] == "/ai/tools/get_company_sales" and c[3] == GOOD for c in fake.calls)
    assert [a["kind"] for a in fake.audits] == ["question", "answer"]


def test_whatsapp_refuses_unknown_numbers():
    router = FakeRouter([])
    ch, sent, fake = wa(router)
    asyncio.run(ch.process(W.incoming(payload("sales?", phone="254700000000"))[0]))
    assert "not registered" in sent.out[-1][1] and not router.requests
    assert not any(c[1].startswith("/ai/tools") for c in fake.calls)


def test_whatsapp_commands_rate_limit_and_dedupe():
    ch, sent, _ = wa(FakeRouter([]), env={"WHATSAPP_RATE_PER_MINUTE": "2"})
    asyncio.run(ch.process(W.incoming(payload("help"))[0]))
    assert "Nexora AI" in sent.out[-1][1] and "Hello Mary" in sent.out[-1][1]
    asyncio.run(ch.process(W.incoming(payload("reset"))[0]))
    asyncio.run(ch.process(W.incoming(payload("help"))[0]))
    assert "too quickly" in sent.out[-1][1]
    os.environ["WHATSAPP_RATE_PER_MINUTE"] = "6"

    async def twice():
        n = ch.accept(payload("x", mid="dup")) + ch.accept(payload("x", mid="dup"))
        await asyncio.sleep(0)
        return n
    assert asyncio.run(twice()) == 1


def test_whatsapp_brief_command_uses_role():
    router = FakeRouter([{"answer": "Brief."}])
    ch, sent, _ = wa(router)
    asyncio.run(ch.process(W.incoming(payload("brief"))[0]))
    assert "management briefing" in router.requests[0].messages[-1].content and sent.out[-1][1] == "Brief."


# ---------------- API ----------------

def test_api_chat_stream_tools_and_webhook():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        src, _ = source()
        app.state.assistant = Assistant(FakeRouter([{"action": "get_filter_options", "args": {}}, {"answer": "There are 2 regions."}]), src)
        h = {"X-Api-Key": "test-key"}
        msg = {"messages": [{"role": "user", "content": "How many regions?"}], "context": {"delegation": GOOD, "user": USERS[GOOD]}}
        assert c.post("/v1/assistant/chat", json={"messages": []}, headers=h).status_code == 400
        r = c.post("/v1/assistant/chat", json=msg, headers=h)
        assert r.status_code == 200 and r.json()["answer"] == "There are 2 regions."
        assert c.post("/v1/assistant/chat", json=msg).status_code == 401
        app.state.assistant = Assistant(FakeRouter([{"action": "get_filter_options", "args": {}}, {"answer": "Two."}]), src)
        with c.stream("POST", "/v1/assistant/chat/stream", json=msg, headers=h) as s:
            body = "".join(s.iter_text())
        assert body.index("event: step") < body.index("event: result") < body.index("event: answer")
        tools = c.get("/v1/assistant/tools", headers={**h, "X-AI-Delegation": GOOD}).json()
        assert {"get_company_sales", "system_help"} <= {t["name"] for t in tools["tools"]}
        assert c.get("/v1/assistant/tools", headers=h).json()["tools"] == []

        ch, _, _ = wa(FakeRouter([]))
        app.state.whatsapp = ch
        assert c.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "vt", "hub.challenge": "99"}).text == "99"
        assert c.get("/whatsapp/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "no", "hub.challenge": "99"}).status_code == 403
        raw = json.dumps({"entry": []}).encode()
        assert c.post("/whatsapp/webhook", content=raw).status_code == 401
        sig = "sha256=" + hmac.new(b"sec", raw, hashlib.sha256).hexdigest()
        r = c.post("/whatsapp/webhook", content=raw, headers={"X-Hub-Signature-256": sig})
        assert r.status_code == 200 and r.json() == {"queued": 0}
        assert c.get("/v1/whatsapp/status", headers=h).json()["enabled"] is True


def test_preferred_model_is_used_and_listed():
    src, _ = source()
    router = FakeRouter([{"answer": "ok"}])
    asyncio.run(Assistant(router, src).ask([Message(role="user", content="hi")],
                                           {"user": USERS[GOOD], "delegation": GOOD}, model="deepseek-coder-v2:16b"))
    assert router.requests[0].model == "deepseek-coder-v2:16b" and router.requests[0].sensitive
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        async def installed():
            return {"deepseek-coder-v2:16b", "qwen3:8b", "llama3.1:8b"}
        app.state.router.installed = installed
        ids = [m["id"] for m in c.get("/v1/assistant/models", headers={"X-Api-Key": "test-key"}).json()["models"]]
        assert ids == ["qwen3:8b", "deepseek-coder-v2:16b", "llama3.1:8b"]
