import asyncio
from app import gate


def test_interactive_goes_before_background():
    order = []

    async def job(g, name, prio, hold):
        async with g.slot(prio):
            order.append(name)
            await asyncio.sleep(hold)

    async def run():
        g = gate.Gate(1)
        first = asyncio.ensure_future(job(g, "insights", gate.BACKGROUND, 0.05))
        await asyncio.sleep(0.01)
        bg = asyncio.ensure_future(job(g, "anomalies", gate.BACKGROUND, 0))
        await asyncio.sleep(0.005)
        ask = asyncio.ensure_future(job(g, "ask", gate.INTERACTIVE, 0))
        await asyncio.gather(first, bg, ask)
        assert g.busy == 0
    asyncio.run(run())
    assert order == ["insights", "ask", "anomalies"]


def test_cancelled_waiter_frees_nothing_twice():
    async def run():
        g = gate.Gate(1)
        await g.acquire(1)
        w = asyncio.ensure_future(g.acquire(0)); await asyncio.sleep(0)
        w.cancel()
        try:
            await w
        except asyncio.CancelledError:
            pass
        g.release()
        assert g.busy == 0 and g.queued == 0
    asyncio.run(run())


def test_priorities():
    assert gate.priority_for("analytics.ask") == gate.INTERACTIVE
    assert gate.priority_for("nexora.chat") == gate.INTERACTIVE
    assert gate.priority_for("analytics.insights") == gate.BACKGROUND
    assert gate.priority_for("analytics.remedy") == gate.BACKGROUND
