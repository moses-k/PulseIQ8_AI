#!/usr/bin/env bash
# Start PulseIQ8_AI in the background: Ollama (Docker or the Ollama app) + the API on PORT (.env, default 8091).
#   ./start.sh            start (runs ./run.sh in the background, log in logs/pulseiq.log)
#   ./start.sh --app      use the Ollama Mac app (GPU, faster) instead of Ollama in Docker (docker-compose.ollama.yml)
# Stop with ./stop.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || cp .env.example .env
PORT="$(grep -E '^PORT=' .env | tail -1 | cut -d= -f2 | tr -d '"' || true)"; PORT="${PORT:-8091}"
OLLAMA_URL="http://localhost:11434"
mkdir -p logs

ollama_up() { curl -fsS -m 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

# 1) Ollama
if ollama_up; then
  echo "✓ Ollama already running on 11434"
elif [ "${1:-}" != "--app" ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "… starting Ollama in Docker (nexora-ollama)"
  docker compose -f docker-compose.ollama.yml up -d --remove-orphans
elif [ -d /Applications/Ollama.app ]; then
  echo "… starting the Ollama app"; open -a Ollama
else
  echo "✗ No Ollama: install the Ollama app or start Docker Desktop"; exit 1
fi
for i in $(seq 1 60); do ollama_up && break; sleep 1; done
ollama_up || { echo "✗ Ollama did not answer on 11434"; exit 1; }
MODELS="$(curl -fsS "$OLLAMA_URL/api/tags" | python3 -c 'import sys,json; print(" ".join(m["name"] for m in json.load(sys.stdin)["models"]))' 2>/dev/null || true)"
echo "  models: ${MODELS:-none yet (still downloading? docker compose -f docker-compose.ollama.yml logs -f ollama)}"

# 2) PulseIQ8_AI API
if [ -f logs/pulseiq.pid ] && kill -0 "$(cat logs/pulseiq.pid)" 2>/dev/null; then
  echo "✓ PulseIQ8_AI already running (pid $(cat logs/pulseiq.pid)) on http://localhost:$PORT"; exit 0
fi
OLD="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
if [ -n "$OLD" ]; then echo "… port $PORT busy (pid $OLD) – stopping it"; kill $OLD 2>/dev/null || true; sleep 2; fi
echo "… starting PulseIQ8_AI on port $PORT (log: logs/pulseiq.log)"
nohup ./run.sh >> logs/pulseiq.log 2>&1 &
echo $! > logs/pulseiq.pid
for i in $(seq 1 90); do
  curl -fsS -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "✓ PulseIQ8_AI up: http://localhost:$PORT  (pid $(cat logs/pulseiq.pid))"; exit 0; }
  kill -0 "$(cat logs/pulseiq.pid)" 2>/dev/null || { echo "✗ PulseIQ8_AI exited – last log lines:"; tail -20 logs/pulseiq.log; exit 1; }
  sleep 1
done
echo "✗ PulseIQ8_AI did not answer on /health in 90s – see logs/pulseiq.log"; exit 1
