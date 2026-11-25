"""Routing, fallback, escalation, privacy and the API — against fake providers (no real models needed)."""
import asyncio
import json
import os

import httpx
import pytest

os.environ.setdefault("PULSEIQ8_AI_KEYS", "test-key")

from app import config  # noqa: E402
from app.router import Router, decide  # noqa: E402
from app.schemas import ChatRequest, Message  # noqa: E402
from app.tasks import TASK_TIERS  # noqa: E402
from app.usage import Usage  # noqa: E402


def settings(**env):
    for k, v in env.items():
        os.environ[k] = v
    try:
        return config.load()
    finally:
        for k in env:
            os.environ.pop(k, None)


class Fake:
    """Pretends to be Ollama + Anthropic. `broken` models fail; `bad_json` models answer non-JSON."""

    def __init__(self, installed, broken=(), bad_json=()):
        self.installed, self.broken, self.bad_json = set(installed), set(broken), set(bad_json)
        self.calls = []

    def __call__(self, request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.installed]})
        body = json.loads(request.content)
        model = body["model"]
        self.calls.append((request.url.host, model, body))
        if model in self.broken:
            return httpx.Response(503, text="overloaded")
        text = "not json at all" if model in self.bad_json else json.dumps({"answer": f"from {model}"})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": text}, "prompt_eval_count": 10, "eval_count": 5})
        if request.url.path == "/v1/messages":
            return httpx.Response(200, json={"content": [{"type": "text", "text": text}], "usage": {}})
        return httpx.Response(404)


def run(router, req):
    return asyncio.run(router.chat(req))


def make(fake, **env):
    s = settings(**env)
    client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    return Router(s, client, Usage(), TASK_TIERS)


def user(text, **kw):
    return ChatRequest(messages=[Message(role="user", content=text)], **kw)


ALL_LOCAL = ["qwen3:4b", "llama3.2:3b", "gemma3:4b", "deepseek-coder-v2:16b", "qwen2.5-coder:14b", "qwen3:8b",
             "llama3.1:8b", "qwen3:14b", "deepseek-r1:14b", "qwen2.5vl:7b"]


# ---------------- tier decisions ----------------

@pytest.mark.parametrize("text,tier", [
    ("Summarize this paragraph about loan repayments please", "fast"),
    ("Give this chat a title", "fast"),
    ("Why does my SwiftUI view crash with a NullPointer exception in the stack trace?", "coding"),
    ("Refactor this Kotlin Jetpack Compose screen", "coding"),
    ("Investigate the root cause of intermittent payment failures and compare the architecture options", "reasoning"),
    ("Write a friendly announcement for our customers about the new savings product and how it helps families "
     "plan their monthly budget better across the year", "general"),
])
def test_rules(text, tier):
    assert decide(user(text), TASK_TIERS).tier == tier


def test_images_go_to_vision_and_tasks_to_their_tier():
    req = ChatRequest(messages=[Message(role="user", content="why?", images=[{"data": "AAAA"}])])
    assert decide(req, TASK_TIERS).tier == "vision"
    assert decide(user("hi", task="analytics.anomaly"), TASK_TIERS).tier == "reasoning"
    assert decide(user("hi", tier="coding"), TASK_TIERS).tier == "coding"


# ---------------- fallback / escalation ----------------

def test_fallback_to_next_model_in_tier():
    fake = Fake(ALL_LOCAL, broken={"deepseek-coder-v2:16b"})
    res = run(make(fake), user("fix this swift function", json=True))
    assert res.tier == "coding" and res.model == "qwen2.5-coder:14b"
    assert [a.ok for a in res.attempts] == [False, True]


def test_last_resort_uses_any_installed_model():
    fake = Fake(["deepseek-coder-v2:16b"])     # only the coding model pulled so far
    res = run(make(fake), user("Write a friendly note to the team about the new release schedule for next month please"))
    assert res.model == "deepseek-coder-v2:16b" and "used what is available" in res.route_reason


def test_invalid_json_falls_back_then_escalates():
    fake = Fake(ALL_LOCAL, bad_json={"qwen3:8b", "qwen3:4b", "llama3.2:3b", "gemma3:4b"})   # every Quick model fails
    res = run(make(fake), user("summarize this", json=True))
    assert res.tier == "general" and "answered by the general tier" in res.route_reason
    assert res.json_data == {"answer": "from llama3.1:8b"}


def test_models_not_pulled_are_skipped():
    fake = Fake(["qwen3:8b"])
    res = run(make(fake), user("summarize this"))
    assert res.model == "qwen3:8b"           # no fast model pulled → escalated to general
    assert all(c[1] != "qwen3:4b" for c in fake.calls)


def test_no_model_gives_clear_error():
    fake = Fake([])
    with pytest.raises(Exception) as e:
        run(make(fake), user("summarize this"))
    assert "No model could answer" in str(e.value)


# ---------------- privacy ----------------

def test_cloud_fallback_is_masked_and_sensitive_stays_local():
    fake = Fake([])   # nothing local → the cloud general model
    r = make(fake, ANTHROPIC_API_KEY="k", ANTHROPIC_GENERAL_MODEL="claude-x")
    res = run(r, user("Customer 0712345678 jane@example.com writes a long message about her loan " * 3))
    assert res.provider == "anthropic"
    sent = fake.calls[-1][2]["messages"][0]["content"][-1]["text"]
    assert "0712345678" not in sent and "jane@example.com" not in sent and "#" in sent
    with pytest.raises(Exception):
        run(r, user("Customer 0712345678 about her loan", sensitive=True))   # sensitive: never cloud


def test_local_only_policy_blocks_cloud():
    fake = Fake([])
    r = make(fake, ANTHROPIC_API_KEY="k", ANTHROPIC_GENERAL_MODEL="claude-x", PULSEIQ8_DATA_POLICY="local_only")
    with pytest.raises(Exception):
        run(r, user("hello there, how are you doing today my friend, tell me something interesting " * 3))
    assert not any(c[0] == "api.anthropic.com" for c in fake.calls)


def test_circuit_breaker_pauses_failing_model():
    fake = Fake(ALL_LOCAL, broken={"deepseek-coder-v2:16b"})
    r = make(fake)
    for _ in range(3):
        run(r, user("fix this swift function"))
    before = len([c for c in fake.calls if c[1] == "deepseek-coder-v2:16b"])
    run(r, user("fix this swift function"))
    assert len([c for c in fake.calls if c[1] == "deepseek-coder-v2:16b"]) == before   # skipped while paused


# ---------------- API ----------------

def test_api_auth_chat_task_and_openai_compat():
    from fastapi.testclient import TestClient
    from app import main
    fake = Fake(ALL_LOCAL)
    with TestClient(main.app) as c:
        main.app.state.router = Router(main.app.state.settings, httpx.AsyncClient(transport=httpx.MockTransport(fake)),
                                       main.app.state.usage, TASK_TIERS)
        assert c.post("/v1/chat", json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 401
        h = {"X-Api-Key": "test-key"}
        r = c.post("/v1/chat", headers=h, json={"messages": [{"role": "user", "content": "fix my kotlin code"}], "json": True})
        assert r.status_code == 200 and r.json()["tier"] == "coding" and r.json()["json"]["answer"]
        r = c.post("/v1/tasks/analytics.anomaly", headers=h, json={"data": {"metric": "api failure rate"}})
        assert r.status_code == 200 and r.json()["tier"] == "reasoning"
        r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer test-key"},
                   json={"model": "pulseiq8-coding", "messages": [{"role": "user", "content": "write a swift func"}]})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"]
        r = c.post("/v1/vision/analyze", headers=h, files={"file": ("s.png", b"\x89PNG....", "image/png")},
                   data={"question": "Why is this Android app crashing?"})
        assert r.status_code == 200 and r.json()["tier"] == "vision"
        u = c.get("/v1/usage", headers=h).json()
        assert u["calls"] >= 4 and "coding" in u["by_tier"]
        t = c.get("/v1/tiers", headers=h).json()
        assert t["tiers"]["coding"]["ready"] is True


def test_finds_ollama_on_another_local_address():
    def handler(request):
        if request.url.host == "wrong-host":
            raise httpx.ConnectError("refused")
        if request.url.host == "127.0.0.1" and request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "deepseek-coder-v2:16b"}]})
        if request.url.host == "127.0.0.1":
            return httpx.Response(200, json={"message": {"content": "hello"}})
        raise httpx.ConnectError("refused")
    s = settings(OLLAMA_BASE_URL="http://wrong-host:11434", GENERAL_MODEL="deepseek-coder-v2:16b")
    r = Router(s, httpx.AsyncClient(transport=httpx.MockTransport(handler)), Usage(), TASK_TIERS)
    st = asyncio.run(r.status())
    assert st["ollama_reachable"] and st["ollama_url"] == "http://127.0.0.1:11434"
    res = run(r, user("Write a friendly note to the team about the new release schedule for next month please"))
    assert res.text == "hello"


def test_reports_why_ollama_is_unreachable():
    def handler(request):
        raise httpx.ConnectError("Connection refused")
    r = Router(settings(), httpx.AsyncClient(transport=httpx.MockTransport(handler)), Usage(), TASK_TIERS)
    st = asyncio.run(r.status())
    assert not st["ollama_reachable"] and "Connection refused" in st["ollama_error"]


def test_routing_off_sends_everything_to_the_fixed_model():
    fake = Fake(ALL_LOCAL)
    r = make(fake, PULSEIQ8_ROUTING="false", PULSEIQ8_FIXED_MODEL="deepseek-coder-v2:16b")
    for text in ("summarize this", "fix my swift code", "investigate the root cause and compare the architecture options"):
        res = run(r, user(text, json=True))
        assert res.tier == "fixed" and res.model == "deepseek-coder-v2:16b" and "routing off" in res.route_reason
    assert {c[1] for c in fake.calls} == {"deepseek-coder-v2:16b"}
    st = asyncio.run(r.status())
    assert st["routing"] is False and st["fixed_model"]["model"] == "deepseek-coder-v2:16b"


def test_routing_off_fixed_cloud_model_respects_policy():
    fake = Fake([])
    r = make(fake, PULSEIQ8_ROUTING="false", PULSEIQ8_FIXED_PROVIDER="anthropic", PULSEIQ8_FIXED_MODEL="claude-x",
             ANTHROPIC_API_KEY="k", PULSEIQ8_DATA_POLICY="local_only")
    with pytest.raises(Exception) as e:
        run(r, user("hello"))
    assert "cloud model" in str(e.value)


def test_empty_json_is_retried_with_format_json_then_rejected():
    calls = []
    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "deepseek-coder-v2:16b"}]})
        body = json.loads(request.content); calls.append(body.get("format"))
        text = "{}" if body.get("format") is None else '{"headline": "ok", "findings": [1]}'
        return httpx.Response(200, json={"message": {"content": text}})
    s = settings(PULSEIQ8_ROUTING="false", PULSEIQ8_FIXED_MODEL="deepseek-coder-v2:16b")
    r = Router(s, httpx.AsyncClient(transport=httpx.MockTransport(handler)), Usage(), TASK_TIERS)
    res = run(r, user("analyse", json=True))
    assert res.json_data == {"headline": "ok", "findings": [1]} and calls == [None, "json"]

    def always_empty(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "deepseek-coder-v2:16b"}]})
        return httpx.Response(200, json={"message": {"content": "{}"}})
    r = Router(s, httpx.AsyncClient(transport=httpx.MockTransport(always_empty)), Usage(), TASK_TIERS)
    with pytest.raises(Exception) as e:
        run(r, user("analyse", json=True))
    assert "no usable JSON" in str(e.value)


def test_keys_created_in_the_service(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setenv("PULSEIQ8_KEYS_FILE", str(tmp_path / "keys.json"))
    fake = Fake(ALL_LOCAL)
    with TestClient(main.app) as c:     # TestClient's client host is "testclient": not local
        main.app.state.router = Router(main.app.state.settings, httpx.AsyncClient(transport=httpx.MockTransport(fake)),
                                       main.app.state.usage, TASK_TIERS)
        assert c.post("/v1/keys", json={"name": "x"}).status_code == 401          # remote: needs a key
        h = {"X-Api-Key": "test-key"}
        made = c.post("/v1/keys", headers=h, json={"name": "Chat in Chrome"}).json()
        assert made["key"].startswith("piq8_") and len(made["key"]) == 53
        new = {"X-Api-Key": made["key"]}
        assert c.get("/v1/tiers", headers=new).status_code == 200                # the new key works
        listed = c.get("/v1/keys", headers=h).json()["keys"]
        assert listed[0]["name"] == "Chat in Chrome" and "key" not in listed[0] and listed[0]["last_used"]
        assert made["key"] not in (tmp_path / "keys.json").read_text()           # only the hash is stored
        assert c.delete(f"/v1/keys/{made['id']}", headers=h).status_code == 200
        assert c.get("/v1/tiers", headers=new).status_code == 401                # revoked


# ---------------- streaming (/v1/chat/stream) ----------------

def stream(router, req):
    async def go():
        return [ev async for ev in router.chat_stream(req)]
    return asyncio.run(go())


def test_stream_meta_delta_done_with_fallback():
    fake = Fake(ALL_LOCAL, broken={"deepseek-coder-v2:16b"})
    evs = stream(make(fake), user("fix this swift function"))
    types = [e["type"] for e in evs]
    assert types[0] == "meta" and types[-1] == "done" and "delta" in types
    assert evs[0]["model"] == "qwen2.5-coder:14b"          # the broken model was skipped before any text
    assert "".join(e["text"] for e in evs if e["type"] == "delta")


def test_stream_reports_error_when_nothing_answers():
    evs = stream(make(Fake([])), user("hello there"))
    assert evs[-1]["type"] == "error" and "No model could answer" in evs[-1]["message"]


def test_stream_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        app.state.router = make(Fake(ALL_LOCAL))
        with c.stream("POST", "/v1/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}, headers={"X-Api-Key": "test-key"}) as s:
            body = "".join(s.iter_text())
        assert body.index("event: meta") < body.index("event: delta") < body.index("event: done")


# ---------------- modes (tiers) with Qwen3 ----------------

QWEN = {"FAST_MODEL": "qwen3:8b", "GENERAL_MODEL": "qwen3:8b", "REASONING_MODEL": "qwen3:8b", "CODING_MODEL": "qwen3:8b"}


def test_qwen3_answers_directly_except_in_reasoning_mode():
    fake = Fake(["qwen3:8b"])
    r = make(fake, **QWEN)
    run(r, ChatRequest(messages=[Message(role="user", content="hello")], tier="general"))
    general = fake.calls[-1][2]
    assert general["messages"][0]["content"].endswith("/no_think")
    run(r, ChatRequest(messages=[Message(role="user", content="hello")], tier="reasoning", max_tokens=1000))
    deep = fake.calls[-1][2]
    assert "/no_think" not in deep["messages"][0]["content"] and deep["options"]["num_predict"] == 1000 + 4096


def test_explicit_mode_works_when_routing_is_off():
    fake = Fake(["qwen3:8b", "llama3.2:3b"])
    r = make(fake, PULSEIQ8_ROUTING="false", PULSEIQ8_FIXED_MODEL="llama3.2:3b", **QWEN)
    assert run(r, user("hello")).tier == "fixed"                                                # Auto → fixed model
    res = run(r, ChatRequest(messages=[Message(role="user", content="hello")], tier="reasoning"))
    assert res.tier == "reasoning" and res.model == "qwen3:8b"                                  # a chosen mode wins


def test_quick_mode_defaults_to_qwen3_8b():
    fake = Fake(["qwen3:8b"])
    res = run(make(fake), ChatRequest(messages=[Message(role="user", content="hi")], tier="fast"))
    assert res.tier == "fast" and res.model == "qwen3:8b"


def test_user_can_pick_an_installed_model():
    fake = Fake(["qwen3:8b", "mistral:7b"])
    r = make(fake, PULSEIQ8_ROUTING="false", PULSEIQ8_FIXED_MODEL="qwen3:8b")
    res = run(r, ChatRequest(messages=[Message(role="user", content="hi")], model="mistral:7b"))
    assert res.model == "mistral:7b" and "chosen by the user" in res.route_reason
    with pytest.raises(Exception):
        run(r, ChatRequest(messages=[Message(role="user", content="hi")], model="llama9:70b"))     # not installed
    evs = stream(r, ChatRequest(messages=[Message(role="user", content="hi")], model="ollama:mistral:7b"))
    assert evs[0]["type"] == "meta" and evs[0]["model"] == "mistral:7b"
