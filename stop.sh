#!/usr/bin/env bash
# Stop PulseIQ8_AI.
#   ./stop.sh          stop the API (Ollama keeps running, models stay loaded)
#   ./stop.sh --all    also stop Ollama (Docker container nexora-ollama, or the Ollama app)
set -uo pipefail
cd "$(dirname "$0")"
PORT="$(grep -E '^PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"')"; PORT="${PORT:-8091}"
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

PIDS=""
[ -f logs/pulseiq.pid ] && PIDS="$(cat logs/pulseiq.pid)"
PIDS="$PIDS $(port_pids "$PORT")"
PIDS="$(echo $PIDS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
if [ -n "${PIDS// /}" ]; then
  kill $PIDS 2>/dev/null; sleep 2
  for p in $PIDS; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done
  echo "✓ PulseIQ8_AI stopped (port $PORT)"
else
  echo "· PulseIQ8_AI (./run.sh) was not running"
fi
rm -f logs/pulseiq.pid
# PulseIQ8_AI running as a Docker container: stop the container, never Docker itself
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^pulseiq8-ai$'; then
  docker compose stop pulseiq8-ai >/dev/null && echo "✓ PulseIQ8_AI container stopped"
fi

if [ "${1:-}" = "--all" ]; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^nexora-ollama$'; then
    docker compose stop ollama && echo "✓ Ollama (Docker) stopped – models kept in the volume"
  elif pgrep -xq Ollama; then
    osascript -e 'quit app "Ollama"' && echo "✓ Ollama app quit"
  else
    echo "· Ollama was not running"
  fi
fi
