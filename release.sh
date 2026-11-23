#!/usr/bin/env bash
# Builds PulseIQ8_AI and pushes it to Docker Hub with a version tag (+ latest), for Intel/AMD and ARM servers.
#
#   ./release.sh              version from app/__init__.py
#   ./release.sh 0.2.0
#   IMAGE=kipyegonk/pulseiq8-ai ./release.sh
#
# Make the Docker Hub repository PRIVATE before the first push (hub.docker.com → Create repository → Private).
# Nothing secret is in the image: .env, data/ (keys) and .venv are excluded by .dockerignore.
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin:$PATH"

IMAGE="${IMAGE:-kipyegonk/pulseiq8-ai}"
VERSION="${1:-$(sed -n 's/^__version__ *= *["'\'']\([^"'\'']*\)["'\''].*/\1/p' app/__init__.py 2>/dev/null | head -1)}"
VERSION="${VERSION:-0.1.0}"
[[ "$VERSION" =~ ^[0-9A-Za-z._-]+$ ]] || { echo "Bad version: $VERSION"; exit 1; }
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "Image: $IMAGE:$VERSION (+ latest)   Platforms: $PLATFORMS   Commit: $COMMIT"
docker info >/dev/null 2>&1 || { echo "Docker is not running"; exit 1; }
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  read -rp "Uncommitted changes will be in the image. Continue? [y/N] " a; [[ "$a" == [yY] ]] || exit 1
fi
docker buildx inspect pulseiq >/dev/null 2>&1 || docker buildx create --name pulseiq --driver docker-container >/dev/null
docker buildx build --builder pulseiq --platform "$PLATFORMS" \
  --label org.opencontainers.image.version="$VERSION" --label org.opencontainers.image.revision="$COMMIT" \
  -t "$IMAGE:$VERSION" -t "$IMAGE:latest" --push .
echo "Pushed $IMAGE:$VERSION — on the server: ./start.sh (it pulls $IMAGE too)"
