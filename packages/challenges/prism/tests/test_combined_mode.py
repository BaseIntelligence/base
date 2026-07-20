from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from base.challenge_sdk import app_factory
from fastapi import APIRouter
from fastapi.testclient import TestClient

from prism_challenge import worker as worker_module
from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import PrismContainerEvaluator
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.worker import run_worker_loop


def _settings(tmp_path: Path, **overrides: object) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        **overrides,
    )


async def _empty_weights() -> dict[str, float]:
    return {}


def test_combined_mode_default_is_off() -> None:
    settings = PrismSettings(shared_token="x")
    assert settings.combined_mode is False
    assert settings.combined_worker_interval_seconds == 5.0


def test_combined_mode_reads_exact_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRISM_COMBINED_MODE", "true")
    monkeypatch.setenv("PRISM_COMBINED_WORKER_INTERVAL_SECONDS", "2.5")
    settings = PrismSettings(shared_token="x")
    assert settings.combined_mode is True
    assert settings.combined_worker_interval_seconds == 2.5


def test_combined_off_is_api_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, combined_mode=False)
    spy = AsyncMock()
    monkeypatch.setattr(worker_module, "run_worker_loop", spy)

    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200

    spy.assert_not_awaited()


def test_combined_on_launches_worker_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, combined_mode=True, combined_worker_interval_seconds=1.5)
    recorded: dict[str, object] = {}

    async def fake_loop(worker: object, *, interval_seconds: float, resilient: bool) -> None:
        recorded["worker"] = worker
        recorded["interval_seconds"] = interval_seconds
        recorded["resilient"] = resilient
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            recorded["cancelled"] = True
            raise

    monkeypatch.setattr(worker_module, "run_worker_loop", fake_loop)

    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert recorded["worker"] is app.state.worker
        assert recorded["interval_seconds"] == 1.5
        assert recorded["resilient"] is True

    assert recorded.get("cancelled") is True


def test_combined_shutdown_cancels_worker_before_db_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, combined_mode=True)
    order: list[str] = []

    async def fake_loop(worker: object, *, interval_seconds: float, resilient: bool) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("worker_cancelled")
            raise

    async def fake_close(self: object) -> None:
        order.append("db_closed")

    monkeypatch.setattr(worker_module, "run_worker_loop", fake_loop)
    monkeypatch.setattr("prism_challenge.db.Database.close", fake_close)

    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200

    assert order == ["worker_cancelled", "db_closed"]


def test_combined_dispatch_uses_broker_no_local_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The combined worker orchestrates GPU work via the broker, so no local GPU is required."""
    # Isolate from ambient CHALLENGE_/PRISM_DOCKER_* env used by other local suites.
    for name in (
        "PRISM_DOCKER_BACKEND",
        "CHALLENGE_DOCKER_BACKEND",
        "PRISM_DOCKER_BROKER_TOKEN",
        "PRISM_DOCKER_BROKER_TOKEN_FILE",
        "CHALLENGE_DOCKER_BROKER_TOKEN",
        "CHALLENGE_DOCKER_BROKER_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = _settings(tmp_path, combined_mode=True)
    # Code default: the eval dispatch backend is the broker (not the local `cli`/`--gpus` path).
    assert settings.docker_backend == "broker"

    ctx = PrismContext(
        sequence_length=settings.sequence_length,
        max_layers=settings.max_layers,
        max_parameters=settings.max_parameters,
    )
    evaluator = PrismContainerEvaluator(settings=settings, ctx=ctx)
    executor = evaluator._executor()
    assert executor.backend == "broker"
    # The broker URL/token the separate `-worker` service used to carry must reach the in-process
    # worker via the same broker settings (platform sets PRISM_DOCKER_BROKER_URL + token).
    assert hasattr(executor, "broker_url")
    assert hasattr(executor, "broker_token")


async def test_run_worker_loop_resilient_continues_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_next = AsyncMock(side_effect=[RuntimeError("boom"), "sub-1", asyncio.CancelledError()])
    worker = SimpleNamespace(process_next=process_next)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(worker_module.asyncio, "sleep", fake_sleep)
    logged: dict[str, int] = {"exception": 0, "info": 0}
    monkeypatch.setattr(
        worker_module.logger,
        "exception",
        lambda *a, **k: logged.__setitem__("exception", logged["exception"] + 1),
    )
    monkeypatch.setattr(
        worker_module.logger,
        "info",
        lambda *a, **k: logged.__setitem__("info", logged["info"] + 1),
    )

    with pytest.raises(asyncio.CancelledError):
        await run_worker_loop(worker, interval_seconds=0.0, resilient=True)

    assert process_next.await_count == 3
    assert logged["exception"] == 1
    assert logged["info"] == 1


async def test_run_worker_loop_not_resilient_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    process_next = AsyncMock(side_effect=[RuntimeError("boom")])
    worker = SimpleNamespace(process_next=process_next)

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(worker_module.asyncio, "sleep", fake_sleep)

    with pytest.raises(RuntimeError):
        await run_worker_loop(worker, interval_seconds=0.0, resilient=False)

    assert process_next.await_count == 1


async def test_log_unexpected_background_exit_logs_and_signals_on_error(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    raised: list[signal.Signals] = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))

    async def boom() -> None:
        raise RuntimeError("worker died")

    task: asyncio.Task[None] = asyncio.create_task(boom())
    with pytest.raises(RuntimeError):
        await task

    with caplog.at_level(logging.CRITICAL, logger="prism.sdk.app_factory"):
        app_factory._log_unexpected_background_exit(task)

    assert any(r.levelno == logging.CRITICAL for r in caplog.records)
    assert any("exited unexpectedly" in r.message for r in caplog.records)
    assert raised == [signal.SIGTERM]


async def test_log_unexpected_background_exit_logs_and_signals_on_clean_return(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    raised: list[signal.Signals] = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))

    async def clean() -> None:
        return None

    task: asyncio.Task[None] = asyncio.create_task(clean())
    await task

    with caplog.at_level(logging.CRITICAL, logger="prism.sdk.app_factory"):
        app_factory._log_unexpected_background_exit(task)

    assert any("without error" in r.message for r in caplog.records)
    assert raised == [signal.SIGTERM]


async def test_log_unexpected_background_exit_ignores_cancelled(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    raised: list[signal.Signals] = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))

    async def sleeper() -> None:
        await asyncio.sleep(3600)

    task: asyncio.Task[None] = asyncio.create_task(sleeper())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with caplog.at_level(logging.CRITICAL, logger="prism.sdk.app_factory"):
        app_factory._log_unexpected_background_exit(task)

    assert not [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert raised == []


def test_combined_shutdown_no_double_log_when_task_fails(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A background task that dies just before shutdown is surfaced exactly once.

    The done-callback logs CRITICAL + raises SIGTERM; the lifespan shutdown then consumes the
    already-failed task without re-logging it (no cosmetic double-log).
    """
    raised: list[signal.Signals] = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))

    class _DB:
        async def init(self) -> None:
            return None

        async def close(self) -> None:
            return None

    async def _boom(app: object) -> None:
        raise RuntimeError("drainer died")

    app = app_factory.create_challenge_app(
        settings=PrismSettings(shared_token="x"),
        database=_DB(),
        public_router=APIRouter(),
        get_weights_fn=_empty_weights,
        background_tasks=(_boom,),
    )

    with caplog.at_level(logging.CRITICAL, logger="prism.sdk.app_factory"):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    critical = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert len(critical) == 1
    assert "exited unexpectedly" in critical[0].message
    assert "crashed during shutdown" not in caplog.text
    assert raised == [signal.SIGTERM]
