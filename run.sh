#!/usr/bin/env bash
# Run PulseIQ8_AI on this Mac without Docker (uses the Ollama app on localhost:11434).
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || cp .env.example .env
# needs Python 3.9+; prefer a newer one (Homebrew / python.org) over the Xcode python3 (3.9)
PY=""
for c in python3.13 python3.12 python3.11 python3.10 /opt/homebrew/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.9+ is needed (brew install python@3.12)"; exit 1; }
# re-create the venv if it was made with a different Python than the one found
ver() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])'; }
if [ -x .venv/bin/python ] && [ "$(ver .venv/bin/python)" != "$(ver "$PY")" ]; then
  echo "Re-creating .venv with $("$PY" --version)"; rm -rf .venv
fi
[ -d .venv ] || "$PY" -m venv .venv
echo "PulseIQ8_AI on $(.venv/bin/python --version)"
.venv/bin/pip install -q -r requirements.txt
set -a; . ./.env; set +a
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8091}"
