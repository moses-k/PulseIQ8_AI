#!/usr/bin/env bash
# Stop PulseIQ8_AI.
#   ./stop.sh          stop the API (Ollama keeps running, models stay loaded)
#   ./stop.sh --all    also stop Ollama (Docker container nexora-ollama, or the Ollama app)
set -uo pipefail
cd "$(dirname "$0")"
PORT="$(grep -E '^PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"')"; PORT="${PORT:-8091}"

PIDS=""
[ -f logs/pulseiq.pid ] && PIDS="$(cat logs/pulseiq.pid)"
PIDS="$PIDS $(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
PIDS="$(echo $PIDS | tr ' ' '\n' | sort -u | tr '\n' ' ')"
if [ -n "${PIDS// /}" ]; then
  kill $PIDS 2>/dev/null; sleep 2
  for p in $PIDS; do kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null; done
  echo "✓ PulseIQ8_AI stopped (port $PORT)"
else
  echo "· PulseIQ8_AI was not running"
fi
rm -f logs/pulseiq.pid

if [ "${1:-}" = "--all" ]; then
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^nexora-ollama$'; then
    docker compose -f docker-compose.ollama.yml down && echo "✓ Ollama (Docker) stopped – models kept in the volume"
  elif pgrep -xq Ollama; then
    osascript -e 'quit app "Ollama"' && echo "✓ Ollama app quit"
  else
    echo "· Ollama was not running"
  fi
fi
