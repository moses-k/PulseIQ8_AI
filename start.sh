#!/usr/bin/env bash
# Start PulseIQ8_AI in the background: Ollama (Docker or the Ollama app) + the API on PORT (.env, default 8091).
#   ./start.sh            start (runs ./run.sh in the background, log in logs/pulseiq.log)
#   ./start.sh --app      use the Ollama Mac app (GPU, faster) instead of Ollama in Docker (the ollama service in docker-compose.yml)
# Stop with ./stop.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || cp .env.example .env
PORT="$(grep -E '^PORT=' .env | tail -1 | cut -d= -f2 | tr -d '"' || true)"; PORT="${PORT:-8091}"
OLLAMA_URL="http://localhost:11434"
mkdir -p logs
# PIDs listening on a port, except Docker Desktop's own processes (it holds the ports of running containers:
# killing it stops Docker for everything)
port_pids() {
  for p in $(lsof -ti tcp:"$1" -sTCP:LISTEN 2>/dev/null); do
    case "$(ps -p "$p" -o comm= 2>/dev/null)" in
      *com.docker*|*docker*|*vpnkit*|*Docker*) ;;
      *) echo "$p" ;;
    esac
  done
}

ollama_up() { curl -fsS -m 2 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; }

# 1) Ollama
if ollama_up; then
  echo "✓ Ollama already running on 11434"
elif [ "${1:-}" != "--app" ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "… starting Ollama in Docker (nexora-ollama) and pulling the models"
  # an old nexora-ollama container from the former docker-compose.ollama.yml: replace it (models stay in the volume)
  proj="$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project" }}' nexora-ollama 2>/dev/null || true)"
  if [ -n "$proj" ] && [ "$proj" != "pulseiq8" ]; then docker rm -f nexora-ollama >/dev/null; fi
  docker compose up -d ollama
elif [ -d /Applications/Ollama.app ]; then
  echo "… starting the Ollama app"; open -a Ollama
else
  echo "✗ No Ollama: install the Ollama app or start Docker Desktop"; exit 1
fi
for i in $(seq 1 60); do ollama_up && break; sleep 1; done
ollama_up || { echo "✗ Ollama did not answer on 11434"; exit 1; }
MODELS="$(curl -fsS "$OLLAMA_URL/api/tags" | python3 -c 'import sys,json; print(" ".join(m["name"] for m in json.load(sys.stdin)["models"]))' 2>/dev/null || true)"
echo "  models: ${MODELS:-none yet (still downloading? docker compose logs -f ollama)}"

# the chat's default model must be in this Ollama (the Ollama app has its own model store, separate from Docker)
DEFAULT_MODEL="$(grep -E '^PULSEIQ8_DEFAULT_MODEL=' .env | tail -1 | cut -d= -f2 | tr -d '"' || true)"; DEFAULT_MODEL="${DEFAULT_MODEL:-qwen3:8b}"
case " $MODELS " in
  *" $DEFAULT_MODEL "*) echo "✓ default model $DEFAULT_MODEL is installed" ;;
  *)
    if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^nexora-ollama$' && ! docker ps --format '{{.Names}}' | grep -q '^nexora-ollama$' \
       && command -v ollama >/dev/null && [ -d "$HOME/.ollama" ]; then
      echo "… copying $DEFAULT_MODEL from the Docker model volume to the Ollama app (no download)"
      mkdir -p "$HOME/.ollama/models"
      docker cp nexora-ollama:/root/.ollama/models/. "$HOME/.ollama/models/" >/dev/null 2>&1 || true
      MODELS="$(curl -fsS "$OLLAMA_URL/api/tags" | python3 -c 'import sys,json; print(" ".join(m["name"] for m in json.load(sys.stdin)["models"]))' 2>/dev/null || true)"
    fi
    case " $MODELS " in
      *" $DEFAULT_MODEL "*) echo "✓ default model $DEFAULT_MODEL is installed" ;;
      *)
        if command -v ollama >/dev/null; then
          echo "… pulling $DEFAULT_MODEL (first time only)"; ollama pull "$DEFAULT_MODEL" || echo "! pull failed — run: ollama pull $DEFAULT_MODEL"
        else
          echo "… pulling $DEFAULT_MODEL in Docker (first time only)"; docker exec nexora-ollama ollama pull "$DEFAULT_MODEL" || true
        fi ;;
    esac ;;
esac

# 2) PulseIQ8_AI API
if [ -f logs/pulseiq.pid ] && kill -0 "$(cat logs/pulseiq.pid)" 2>/dev/null; then
  echo "✓ PulseIQ8_AI already running (pid $(cat logs/pulseiq.pid)) on http://localhost:$PORT"; exit 0
fi
# the PulseIQ8_AI container publishes the same port: stop it (not Docker) so ./run.sh can use the port
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^pulseiq8-ai$'; then
  echo "… stopping the PulseIQ8_AI container (it uses port $PORT)"; docker compose stop pulseiq8-ai >/dev/null || true
fi
OLD="$(port_pids "$PORT" | tr '\n' ' ')"
if [ -n "${OLD// /}" ]; then echo "… port $PORT busy (pid $OLD) – stopping it"; kill $OLD 2>/dev/null || true; sleep 2; fi
echo "… starting PulseIQ8_AI on port $PORT (log: logs/pulseiq.log)"
nohup ./run.sh >> logs/pulseiq.log 2>&1 &
echo $! > logs/pulseiq.pid
for i in $(seq 1 90); do
  curl -fsS -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "✓ PulseIQ8_AI up: http://localhost:$PORT  (pid $(cat logs/pulseiq.pid))"; exit 0; }
  kill -0 "$(cat logs/pulseiq.pid)" 2>/dev/null || { echo "✗ PulseIQ8_AI exited – last log lines:"; tail -20 logs/pulseiq.log; exit 1; }
  sleep 1
done
echo "✗ PulseIQ8_AI did not answer on /health in 90s – see logs/pulseiq.log"; exit 1
