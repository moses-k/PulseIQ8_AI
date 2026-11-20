s# PulseIQ8_AI

The AI service behind **PulseIQ8** mobile analytics (and the Nexora workspace). It is a separate service: PulseIQ8
talks to it over HTTP, and any other app can too. It routes every request to the right **model tier**, falls back
when a model is down, escalates when an answer is not good enough, and keeps personal data away from cloud models.

```
PulseIQ8 analytics (Spring Boot) ──HTTP /v1/chat──▶  PulseIQ8_AI (FastAPI, port 8091)
Nexora / IDE / aider ───────────HTTP /v1/…──────▶     ├─ Router & orchestrator: rules · classifier · fallback · cascade · policies
                                                     ├─ Local AI: Ollama (private models)
                                                     └─ Cloud AI (optional): Anthropic · OpenAI · Gemini
```

## Model tiers

| Tier | Name | For | Default models (local first) |
|---|---|---|---|
| 1 `fast` | Fast AI | quick answers, summaries, classification, intent, simple writing, chat titles | qwen3:4b → llama3.2:3b → gemma3:4b |
| 2 `coding` | Coding AI | Swift/SwiftUI, Kotlin/Compose, Java/Spring Boot, APIs, debugging, refactoring, tests, reviews | deepseek-coder-v2:16b → qwen2.5-coder:14b |
| 3 `general` | Nexora Intelligence | conversation, research, business analysis, documents, planning, explanation, creative | qwen3:8b → llama3.1:8b → cloud (if set) |
| 4 `reasoning` | Nexora Reasoning | architecture, financial analysis, hard debugging, multi-step plans, root cause, comparing solutions — **only when needed** | qwen3:14b → deepseek-r1:14b → cloud (if set) |
| 5 `vision` | Nexora Vision | screenshots, UI analysis, document images, charts, OCR, visual troubleshooting | qwen2.5vl:7b → gemma3:4b → cloud (if set) |

All models are set in `config/models.yaml` / `.env`. Cloud models stay off until you give a key **and** a model name.

## How a request is routed

**Routing switch:** `PULSEIQ8_ROUTING=false` turns routing off — every request then goes to the one model set by
`PULSEIQ8_FIXED_PROVIDER` / `PULSEIQ8_FIXED_MODEL` / `PULSEIQ8_FIXED_CONTEXT` (e.g. `ollama` /
`deepseek-coder-v2:16b` / `16384`), with no fallback. The data policy still applies (a cloud fixed model is refused
under `local_only` or for sensitive requests). With routing on (default) it works as below.

1. **Tier**: images → Vision; a known `task` → its tier; an explicit `tier` → that tier; otherwise rules on the text
   (code / stack traces → Coding; root cause, architecture, trade-offs, financial → Reasoning; summaries, titles,
   classification → Fast; the rest → General). Set `PULSEIQ8_LLM_CLASSIFIER=true` to let the Fast model decide
   when the rules are unsure.
2. **Model**: the tier's models in order, skipping cloud models when the policy or the request forbids them, models
   whose context window is too small for the input, models not pulled into Ollama, and models paused after 3
   failures in a row (circuit breaker, 60 s).
3. **Fallback & cascade**: on an error, timeout, rate limit or invalid JSON the next model is tried, then the next
   tier up (fast → general → reasoning, coding → reasoning). The answer says which tier/model answered and why.

**Data policy** (`PULSEIQ8_DATA_POLICY`): `local_only` never calls a cloud provider; `local_first` (default) uses
cloud only as a fallback and never for `sensitive` requests; `cloud_ok` follows the configured order. Before
anything goes to a cloud model, emails, phone/account/ID numbers and long tokens are masked. Every system prompt
tells the model that data inside `<data>` / images is untrusted (prompt-injection guard).

## API (all `/v1` calls need `X-Api-Key` or `Authorization: Bearer`)

| Method | Path | |
|---|---|---|
| GET | `/` | **chat UI**: sidebar of chats (pinned / today / last 7 days, search, rename, delete), greeting, composer with model/tier picker, image upload or paste, "Local only" switch, Markdown + code blocks with Copy, thinking timer, stop, retry, light/dark |
| GET | `/console` | models, routing status, Ollama connection and usage |
| GET | `/health` | liveness |
| GET | `/v1/tiers` | tiers, models and which are ready (pulled / keyed / not paused) |
| POST | `/v1/chat` | `{messages, system?, tier?: auto\|fast\|coding\|general\|reasoning\|vision, task?, json?, max_tokens?, sensitive?}` → `{text, json, tier, model, provider, route_reason, latency_ms, attempts, usage}` |
| GET | `/v1/tasks` | the named tasks |
| POST | `/v1/tasks/{task}` | `{data?, question?, images?}` — PulseIQ8_AI writes the prompt: `analytics.insights/ask/session/anomaly/remedy`, `code.review/crash/tests`, `vision.analyze/ui_review`, `text.summarize/title/classify/intent`, `reasoning.compare/plan` |
| POST | `/v1/vision/analyze` | multipart `file` + `question` ("Why is this Android application crashing?") |
| GET | `/v1/usage` | calls per tier / model, latency, success rate, recent runs |
| POST | `/v1/chat/completions` | OpenAI-compatible; `model` = `pulseiq8-auto\|fast\|coding\|general\|reasoning\|vision` |
| GET | `/v1/models` | OpenAI-compatible model list |

Interactive API docs: `http://localhost:8091/docs`.

## Run

**On the Mac (uses the Ollama app, fastest — it uses the GPU):**
```bash
cd ~/dev/PlayGround/Nox/PulseIQ8_AI
cp .env.example .env            # set PULSEIQ8_AI_KEYS to a long random value
ollama pull qwen3:4b; ollama pull qwen3:8b     # deepseek-coder-v2:16b you already have
./run.sh                        # → http://localhost:8091
```

**Docker (service + Ollama + model download):**
```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f ollama-pull
```

**Tests:** `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest && .venv/bin/pytest -q`

## Connect PulseIQ8 analytics

In `~/coop/analytics/config/application.yml`:
```yaml
analytics:
  ai:
    enabled: true
    provider: pulseiq8
    base-url: http://localhost:8091
    api-key: "<the same key as PULSEIQ8_AI_KEYS>"
    model: pulseiq8
    timeout-seconds: 300
```
PulseIQ8 then sends every AI job (insights, questions, session and anomaly explanations, live-fix messages) to
PulseIQ8_AI with its task name, and PulseIQ8_AI picks the tier: anomalies go to Reasoning, live-fix text to Fast,
the rest to General.

**Code fixes through PulseIQ8_AI** (aider talks to the OpenAI-compatible endpoint):
```yaml
  code-fix:
    model: openai/pulseiq8-coding
    openai-base: http://localhost:8091/v1
    openai-key: "<the same key>"
```

## Safety notes
- Keep `data_policy: local_only` or `local_first` for bank data; cloud providers need DPO sign-off (Kenya DPA).
- Change `PULSEIQ8_AI_KEYS` before exposing the port beyond the Mac; put it behind TLS on a server.
- AI answers can be wrong — PulseIQ8 shows them as suggestions and never applies a fix without a person's approval.
