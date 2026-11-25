import sys
p = sys.argv[1] + '/app/providers.py'; s = open(p).read()
def r(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a)); s = s.replace(a, b)
r('''def _ollama_error(status: int, body: str, spec: ModelSpec) -> ProviderError:''', '''def _ollama_net_error(p: Provider, e: Exception, timeout: float) -> ProviderError:
    if isinstance(e, httpx.TimeoutException) and not isinstance(e, httpx.ConnectTimeout):
        return ProviderError(f"Ollama at {p.base_url} is running but took longer than {int(timeout)} s to answer. The model is "
                             "too slow on this computer (Ollama in Docker on a Mac uses the CPU only) or busy with other "
                             "requests. Use the Ollama Mac app (uses the GPU, much faster) or raise PULSEIQ8_TIMEOUT in .env.")
    return ProviderError(f"Ollama not reachable at {p.base_url}: {e.__class__.__name__} — is Ollama running?")


def _ollama_error(status: int, body: str, spec: ModelSpec) -> ProviderError:''')
r('''        r = await client.post(f"{p.base_url}/api/chat", json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise ProviderError(f"Ollama not reachable at {p.base_url}: {e.__class__.__name__}") from e''', '''        r = await client.post(f"{p.base_url}/api/chat", json=body, timeout=timeout)
    except httpx.HTTPError as e:
        raise _ollama_net_error(p, e, timeout) from e''')
r('''    except httpx.HTTPError as e:
        raise ProviderError(f"Ollama not reachable at {provider.base_url}: {e.__class__.__name__}") from e''', '''    except httpx.HTTPError as e:
        raise _ollama_net_error(provider, e, timeout) from e''')
# keep the model in memory between requests (loading it again takes long on a CPU)
r('''"stream": False,
            "options"''', '''"stream": False, "keep_alive": KEEP_ALIVE,
            "options"''')
r('''"stream": True,
            "options"''', '''"stream": True, "keep_alive": KEEP_ALIVE,
            "options"''')
r('''from .config import ModelSpec, Provider
''', '''from .config import ModelSpec, Provider

# how long Ollama keeps the model loaded after a request (its default is 5 min; loading again is slow)
KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE_MODEL", "30m")
''')
r('import json\nimport re\n', 'import json\nimport os\nimport re\n')
open(p, 'w').write(s)
print("ok")
