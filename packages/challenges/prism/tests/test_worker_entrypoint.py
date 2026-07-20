from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from prism_challenge import worker
from prism_challenge.config import PrismSettings


def _fake_app(process_results):
    database = SimpleNamespace(init=AsyncMock(), close=AsyncMock())
    worker_obj = SimpleNamespace(process_next=AsyncMock(side_effect=process_results))
    state = SimpleNamespace(database=database, worker=worker_obj)
    return SimpleNamespace(state=state)


async def test_run_worker_processes_then_closes_on_cancel(monkeypatch):
    app = _fake_app(["sub-1", asyncio.CancelledError()])
    monkeypatch.setattr(worker, "create_app", lambda settings: app)

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(worker.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_worker(PrismSettings(shared_token="x"), interval_seconds=0.01)

    app.state.database.init.assert_awaited_once()
    # Database must be closed in the finally block even though the loop was cancelled.
    app.state.database.close.assert_awaited_once()
    assert app.state.worker.process_next.await_count == 2
    assert sleeps == [0.01]


async def test_run_worker_skips_log_when_no_submission(monkeypatch):
    app = _fake_app([None, asyncio.CancelledError()])
    monkeypatch.setattr(worker, "create_app", lambda settings: app)

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr(worker.asyncio, "sleep", fake_sleep)
    info = MagicMock()
    monkeypatch.setattr(worker.logger, "info", info)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_worker(PrismSettings(shared_token="x"), interval_seconds=0.0)

    info.assert_not_called()
    app.state.database.close.assert_awaited_once()


def test_main_wires_run_worker_with_parsed_interval(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prism-worker", "--interval-seconds", "12.5"])

    captured: dict[str, object] = {}

    def fake_run(coro):
        # Close the coroutine so it is never actually executed (no real DB/event loop).
        coro.close()
        captured["ran"] = True

    monkeypatch.setattr(worker.asyncio, "run", fake_run)

    recorded: dict[str, float] = {}

    def fake_run_worker(settings, *, interval_seconds):
        recorded["interval"] = interval_seconds

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(worker, "run_worker", fake_run_worker)
    monkeypatch.setattr(worker, "PrismSettings", lambda: PrismSettings(shared_token="x"))

    worker.main()

    assert captured["ran"] is True
    assert recorded["interval"] == 12.5


def test_main_uses_default_interval(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prism-worker"])
    monkeypatch.setattr(worker.asyncio, "run", lambda coro: coro.close())
    recorded: dict[str, float] = {}

    def fake_run_worker(settings, *, interval_seconds):
        recorded["interval"] = interval_seconds

        async def _noop():
            return None

        return _noop()

    monkeypatch.setattr(worker, "run_worker", fake_run_worker)
    monkeypatch.setattr(worker, "PrismSettings", lambda: PrismSettings(shared_token="x"))

    worker.main()

    assert recorded["interval"] == 5.0
