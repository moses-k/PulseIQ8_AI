"""Lets questions a person is waiting for go ahead of background work.

A local model answers one request at a time; everything else waits in Ollama's queue in arrival order, so a
dashboard question could sit behind a long background analysis. This gate holds requests here instead and always
lets the most urgent one go next: chat / ask (a person waiting) before insights, anomaly scans and remedies.
"""
from __future__ import annotations

import asyncio
import contextvars
import heapq
import itertools
import os
from contextlib import asynccontextmanager

INTERACTIVE = 0
BACKGROUND = 1

# tasks a person is actively waiting for
_INTERACTIVE_TASKS = ("nexora.chat", "nexora.code", "analytics.ask", "analytics.session", "analytics.explain",
                      "chat", "ask", "openai")

priority: contextvars.ContextVar[int] = contextvars.ContextVar("pulseiq8_priority", default=INTERACTIVE)


def priority_for(task: str | None) -> int:
    t = (task or "").lower()
    if not t:
        return INTERACTIVE
    return INTERACTIVE if any(t == x or t.startswith(x + ".") or t.endswith("." + x) for x in _INTERACTIVE_TASKS) \
        else BACKGROUND


class Gate:
    def __init__(self, slots: int):
        self.slots = max(1, slots)
        self.busy = 0
        self._waiting: list = []
        self._seq = itertools.count()

    @property
    def queued(self) -> int:
        return sum(1 for *_, f in self._waiting if not f.done())

    async def acquire(self, prio: int) -> None:
        if self.busy < self.slots and not self.queued:
            self.busy += 1
            return
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiting, (prio, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()          # the slot was handed over just as the caller gave up
            else:
                fut.cancel()
            raise

    def release(self) -> None:
        while self._waiting:
            *_, fut = heapq.heappop(self._waiting)
            if not fut.done():
                fut.set_result(None)    # the slot passes straight to the next waiter
                return
        self.busy = max(0, self.busy - 1)

    @asynccontextmanager
    async def slot(self, prio: int | None = None):
        await self.acquire(priority.get() if prio is None else prio)
        try:
            yield
        finally:
            self.release()


ollama = Gate(int(os.getenv("PULSEIQ8_OLLAMA_PARALLEL", "1") or 1))
