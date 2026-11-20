"""In-memory usage log: what was routed where, how long it took, what failed (for dashboards and tuning)."""
from __future__ import annotations

import threading
import time
from collections import deque


class Usage:
    def __init__(self, keep: int = 1000):
        self._lock = threading.Lock()
        self._runs: deque = deque(maxlen=keep)

    def record(self, task, tier, model, ok, ms, usage, error):
        with self._lock:
            self._runs.appendleft({"at": time.time(), "task": task, "tier": tier, "model": model, "ok": ok, "ms": ms,
                                   "input_tokens": (usage or {}).get("input_tokens", 0),
                                   "output_tokens": (usage or {}).get("output_tokens", 0),
                                   "error": (error or "")[:200] or None})

    def summary(self, hours: float = 24) -> dict:
        since = time.time() - hours * 3600
        with self._lock:
            runs = [r for r in self._runs if r["at"] >= since]
            recent = list(self._runs)[:30]
        by_tier: dict = {}
        by_model: dict = {}
        for r in runs:
            for bucket, key in ((by_tier, r["tier"]), (by_model, r["model"])):
                b = bucket.setdefault(key, {"calls": 0, "failed": 0, "ms": 0, "input_tokens": 0, "output_tokens": 0})
                b["calls"] += 1
                b["failed"] += 0 if r["ok"] else 1
                b["ms"] += r["ms"]
                b["input_tokens"] += r["input_tokens"]
                b["output_tokens"] += r["output_tokens"]
        for bucket in (by_tier, by_model):
            for b in bucket.values():
                ok = b["calls"] - b["failed"]
                b["avg_ms"] = round(b.pop("ms") / b["calls"]) if b["calls"] else 0
                b["success_rate"] = round(100 * ok / b["calls"], 1) if b["calls"] else 0
        return {"hours": hours, "calls": len(runs), "by_tier": by_tier, "by_model": by_model,
                "recent": [{**r, "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(r["at"]))} for r in recent]}
