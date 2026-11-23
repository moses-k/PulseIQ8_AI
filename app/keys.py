"""API keys created inside PulseIQ8_AI (besides the ones in PULSEIQ8_AI_KEYS).

Stored in data/keys.json as SHA-256 hashes only — the full key is shown once, when it is created.
Keys can be listed and revoked; each remembers when it was last used.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
import uuid
from pathlib import Path


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore:
    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._keys: list[dict] = []
        self._dirty_used = 0.0
        if self.path.exists():
            try:
                self._keys = json.loads(self.path.read_text(encoding="utf-8")).get("keys", [])
            except (ValueError, OSError):
                self._keys = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"keys": self._keys}, indent=2), encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def create(self, name: str) -> dict:
        key = "piq8_" + secrets.token_hex(24)
        row = {"id": uuid.uuid4().hex[:12], "name": (name or "API key").strip()[:60] or "API key",
               "prefix": key[:12], "hash": _hash(key), "created": int(time.time()), "last_used": None}
        with self._lock:
            self._keys.append(row)
            self._save()
        return {**self.public(row), "key": key}   # the full key: only now

    def revoke(self, key_id: str) -> bool:
        with self._lock:
            before = len(self._keys)
            self._keys = [k for k in self._keys if k["id"] != key_id]
            if len(self._keys) != before:
                self._save()
                return True
        return False

    def check(self, key: str) -> bool:
        h = _hash(key)
        with self._lock:
            for k in self._keys:
                if secrets.compare_digest(k["hash"], h):
                    k["last_used"] = int(time.time())
                    if time.time() - self._dirty_used > 60:     # don't write the file on every call
                        self._dirty_used = time.time()
                        self._save()
                    return True
        return False

    def list(self) -> list[dict]:
        with self._lock:
            return [self.public(k) for k in self._keys]

    @staticmethod
    def public(k: dict) -> dict:
        return {"id": k["id"], "name": k["name"], "prefix": k["prefix"] + "…", "created": k["created"],
                "last_used": k["last_used"]}
