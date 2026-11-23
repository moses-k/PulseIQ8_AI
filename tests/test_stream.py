import asyncio, json
import httpx
from app import providers
from app.config import ModelSpec, Provider


def _spec():
    return ModelSpec(tier="fixed", provider="ollama", model="deepseek-coder-v2:16b", context=8192)


def test_ollama_stream_yields_pieces():
    lines = [{"message": {"content": "Hel"}}, {"message": {"content": "lo"}},
             {"message": {"content": ""}, "done": True, "prompt_eval_count": 5, "eval_count": 2}]
    body = "\n".join(json.dumps(x) for x in lines).encode()

    def handler(req):
        assert json.loads(req.content)["stream"] is True
        return httpx.Response(200, content=body)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            p = Provider(name="ollama", type="ollama", base_url="http://x")
            usage = {}
            out = [t async for t in providers.stream(c, p, _spec(), "sys", [{"role": "user", "content": "hi"}],
                                                     max_tokens=50, temperature=0.2, timeout=5, usage=usage)]
            return out, usage
    out, usage = asyncio.run(run())
    assert out == ["Hel", "lo"] and usage["output_tokens"] == 2


def test_ollama_stream_oom_message():
    def handler(req):
        return httpx.Response(500, content=b'{"error":"llama runner process has terminated: signal: killed"}')

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            p = Provider(name="ollama", type="ollama", base_url="http://x")
            return [t async for t in providers.stream(c, p, _spec(), "", [{"role": "user", "content": "hi"}],
                                                      max_tokens=5, temperature=0, timeout=5, usage={})]
    try:
        asyncio.run(run()); assert False
    except providers.ProviderError as e:
        assert "out of memory" in str(e)
