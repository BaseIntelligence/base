from __future__ import annotations

import asyncio
import functools
import logging
import signal

import pytest
from fastapi import APIRouter
from sqlalchemy import select
from test_evaluation_worker import (
    RecordingExecutor,
    create_submission_with_job,
    patch_worker_environment,
)

from agent_challenge.api.app import build_worker_main
from agent_challenge.evaluation.worker import WorkerIteration, run_worker_loop
from agent_challenge.models import EvaluationJob
from agent_challenge.sdk.app_factory import _handle_worker_task_done, create_challenge_app
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.db import Database

WORKER_TASK_NAME = "combined-worker-loop"


async def _empty_weights() -> dict[str, float]:
    return {}


def _fresh_app(tmp_path, worker_main):
    """Build an app on a throwaway DB so lifespan teardown never disposes the
    suite-wide singleton engine that the rest of the tests share."""

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'combined.sqlite3'}")
    settings = ChallengeSettings()
    app = create_challenge_app(
        settings=settings,
        database=db,
        public_router=APIRouter(),
        get_weights_fn=_empty_weights,
        worker_main=worker_main,
    )
    return app, db


def _worker_task() -> asyncio.Task[None] | None:
    for task in asyncio.all_tasks():
        if task.get_name() == WORKER_TASK_NAME:
            return task
    return None


def test_build_worker_main_off_returns_none():
    assert build_worker_main(ChallengeSettings()) is None


def test_build_worker_main_on_reuses_run_worker_loop():
    worker_main = build_worker_main(ChallengeSettings(combined_worker=True))

    assert isinstance(worker_main, functools.partial)
    assert worker_main.func is run_worker_loop
    assert worker_main.keywords == {"manage_database": False}


def test_exact_env_var_enables_combined_worker(monkeypatch):
    monkeypatch.delenv("CHALLENGE_COMBINED_WORKER", raising=False)
    assert ChallengeSettings().combined_worker is False

    monkeypatch.setenv("CHALLENGE_COMBINED_WORKER", "true")
    settings = ChallengeSettings()
    assert settings.combined_worker is True
    assert build_worker_main(settings) is not None


def test_similarly_named_env_var_does_not_enable_combined_worker(monkeypatch):
    monkeypatch.delenv("CHALLENGE_COMBINED_WORKER", raising=False)
    # prism uses PRISM_COMBINED_MODE; a near-miss name must NOT flip this flag.
    monkeypatch.setenv("CHALLENGE_COMBINED_MODE", "true")
    monkeypatch.setenv("COMBINED_WORKER", "true")

    assert ChallengeSettings().combined_worker is False


async def test_combined_off_is_api_only(tmp_path):
    app, _db = _fresh_app(tmp_path, worker_main=None)

    async with app.router.lifespan_context(app):
        assert _worker_task() is None


async def test_combined_on_launches_and_drives_loop(tmp_path):
    started = asyncio.Event()
    polls = 0

    async def stub_worker() -> None:
        nonlocal polls
        started.set()
        while True:
            polls += 1
            await asyncio.sleep(0.01)

    app, _db = _fresh_app(tmp_path, worker_main=stub_worker)

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task = _worker_task()
        assert task is not None
        assert not task.done()
        await asyncio.sleep(0.05)
        assert polls >= 1


async def test_combined_shutdown_cancels_worker_before_db_close(tmp_path):
    order: list[str] = []
    started = asyncio.Event()

    async def stub_worker() -> None:
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("worker_cancelled")
            raise

    app, db = _fresh_app(tmp_path, worker_main=stub_worker)

    original_close = db.close

    async def tracked_close() -> None:
        order.append("db_closed")
        await original_close()

    db.close = tracked_close  # type: ignore[method-assign]

    captured: dict[str, asyncio.Task[None]] = {}
    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task = _worker_task()
        assert task is not None
        captured["task"] = task

    assert captured["task"].cancelled()
    assert order == ["worker_cancelled", "db_closed"]


async def test_worker_task_done_handler_fail_loud(monkeypatch, caplog):
    raised: list[signal.Signals] = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))

    async def boom() -> None:
        raise RuntimeError("worker died")

    task = asyncio.ensure_future(boom())
    with pytest.raises(RuntimeError):
        await task

    with caplog.at_level(logging.CRITICAL):
        _handle_worker_task_done(task)

    assert raised == [signal.SIGTERM]
    assert "combined-mode worker loop exited unexpectedly" in caplog.text


async def test_worker_task_done_handler_clean_return_fails_loud(monkeypatch, caplog):
    raised: list[signal.Signals] = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))

    async def clean_return() -> None:
        return None

    task = asyncio.ensure_future(clean_return())
    await task

    with caplog.at_level(logging.CRITICAL):
        _handle_worker_task_done(task)

    assert raised == [signal.SIGTERM]
    assert "combined-mode worker loop returned unexpectedly" in caplog.text


async def test_worker_task_done_handler_cancelled_is_noop(monkeypatch):
    raised: list[signal.Signals] = []
    monkeypatch.setattr(signal, "raise_signal", lambda sig: raised.append(sig))

    async def sleeper() -> None:
        await asyncio.sleep(3600)

    task = asyncio.ensure_future(sleeper())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _handle_worker_task_done(task)

    assert raised == []


async def test_run_worker_loop_respects_manage_database_flag(monkeypatch):
    calls: list[str] = []

    async def fake_init() -> None:
        calls.append("init")

    async def fake_close() -> None:
        calls.append("close")

    async def fake_run_worker_once(*, worker_id, lease_seconds) -> WorkerIteration:
        return WorkerIteration(stale_jobs=0, summary=None)

    monkeypatch.setattr("agent_challenge.evaluation.worker.database.init", fake_init)
    monkeypatch.setattr("agent_challenge.evaluation.worker.database.close", fake_close)
    monkeypatch.setattr("agent_challenge.evaluation.worker.run_worker_once", fake_run_worker_once)

    await run_worker_loop(once=True, manage_database=False)
    assert calls == []

    await run_worker_loop(once=True, manage_database=True)
    assert calls == ["init", "close"]


async def test_combined_worker_loop_drains_queued_job(database_session, monkeypatch, tmp_path):
    patch_worker_environment(monkeypatch)
    job_id = await create_submission_with_job(
        database_session, tmp_path, job_id="combined-drain-job"
    )
    executor = RecordingExecutor()
    monkeypatch.setattr(
        "agent_challenge.evaluation.runner.build_docker_executor",
        lambda: executor,
    )

    # manage_database=False mirrors combined mode: the API lifespan owns the DB.
    await run_worker_loop(once=True, manage_database=False, worker_id="combined-worker")

    assert executor.tasks == ["analyzer", "task-a"]
    async with database_session() as session:
        job = await session.scalar(select(EvaluationJob).where(EvaluationJob.job_id == job_id))
    assert job is not None
    assert job.status == "completed"
