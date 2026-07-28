"""Lifespan wiring for agent-challenge raw-weight push (production path).

Covers: enabled client + background task, disabled no-op, clean shutdown, and
ledger table registration via shared Database.create_all.
"""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import text

from agent_challenge.api import app as app_module
from agent_challenge.evaluation import raw_weight_push as push_module
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.sdk.db import Database

PUSH_TASK_NAME = "raw-weight-push-loop"


def _settings(tmp_path: Path, **overrides: object) -> ChallengeSettings:
    defaults: dict[str, object] = {
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'ac-push-life.sqlite3'}",
        "shared_token": "ac-lifespan-token",
        "shared_token_file": None,
        "combined_worker": False,
        "raw_weight_push_enabled": True,
        "master_base_url": "http://master.test",
        "raw_weight_push_interval_seconds": 30.0,
    }
    defaults.update(overrides)
    return ChallengeSettings(**defaults)  # type: ignore[arg-type]


def _push_tasks() -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task.get_name() == PUSH_TASK_NAME]


@pytest.mark.asyncio
async def test_enabled_starts_push_task_and_exposes_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given push enabled with master+token, When lifespan starts, Then client
    is on app.state and the named background task is running."""

    started = asyncio.Event()
    recorded: dict[str, object] = {}

    async def fake_loop(client: object, *, interval_seconds: float, resilient: bool) -> None:
        recorded["client"] = client
        recorded["interval_seconds"] = interval_seconds
        recorded["resilient"] = resilient
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            recorded["cancelled"] = True
            raise

    monkeypatch.setattr(push_module, "run_raw_weight_push_loop", fake_loop)

    settings = _settings(tmp_path, raw_weight_push_interval_seconds=12.5)
    db = Database(settings.database_url)
    app = app_module.create_app(challenge_settings=settings, db=db)

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(started.wait(), timeout=2.0)
        client = getattr(app.state, "raw_weight_push_client", None)
        assert client is not None
        assert recorded["client"] is client
        assert recorded["interval_seconds"] == 12.5
        assert recorded["resilient"] is True
        tasks = _push_tasks()
        assert len(tasks) == 1
        assert not tasks[0].done()

    assert recorded.get("cancelled") is True


@pytest.mark.asyncio
async def test_disabled_does_not_build_client_or_start_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given raw_weight_push_enabled=False, When lifespan starts, Then no client
    and no push background task."""

    loop_spy = AsyncMock()
    monkeypatch.setattr(push_module, "run_raw_weight_push_loop", loop_spy)

    settings = _settings(tmp_path, raw_weight_push_enabled=False)
    db = Database(settings.database_url)
    app = app_module.create_app(challenge_settings=settings, db=db)

    async with app.router.lifespan_context(app):
        assert getattr(app.state, "raw_weight_push_client", None) is None
        assert _push_tasks() == []

    loop_spy.assert_not_awaited()


@pytest.mark.asyncio
async def test_shutdown_cancels_push_task_without_pending_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given a running push loop, When lifespan exits, Then the task is cancelled
    and no 'Task was destroyed but it is pending' warning is emitted."""

    started = asyncio.Event()

    async def fake_loop(client: object, *, interval_seconds: float, resilient: bool) -> None:
        del client, interval_seconds, resilient
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(push_module, "run_raw_weight_push_loop", fake_loop)

    settings = _settings(tmp_path)
    db = Database(settings.database_url)
    app = app_module.create_app(challenge_settings=settings, db=db)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(started.wait(), timeout=2.0)
            task = _push_tasks()[0]
        # Allow the event loop a tick to surface destroy warnings if any leaked.
        await asyncio.sleep(0)
        pending_msgs = [
            str(w.message)
            for w in caught
            if "destroyed but it is pending" in str(w.message).lower()
            or "task was destroyed" in str(w.message).lower()
        ]
        assert pending_msgs == []
        assert task.cancelled() or task.done()


@pytest.mark.asyncio
async def test_ledger_table_exists_after_startup(tmp_path: Path) -> None:
    """Given model registration, When lifespan runs database.init, Then
    raw_weight_push_ledger exists on the shared Database."""

    settings = _settings(
        tmp_path,
        # Disabled still creates tables via metadata; no push loop needed.
        raw_weight_push_enabled=False,
    )
    db = Database(settings.database_url)
    app = app_module.create_app(challenge_settings=settings, db=db)

    async with app.router.lifespan_context(app):
        async with db.engine.connect() as connection:
            name = (
                await connection.execute(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name = 'raw_weight_push_ledger'"
                    )
                )
            ).scalar_one_or_none()
        assert name == "raw_weight_push_ledger"


def test_raw_weight_push_settings_defaults_match_prism() -> None:
    """Settings names/defaults/validators mirror Prism's raw_weight_push_* knobs."""

    settings = ChallengeSettings(
        shared_token="x",
        shared_token_file=None,
    )
    assert settings.raw_weight_push_enabled is True
    assert settings.raw_weight_push_interval_seconds == 30.0
    assert settings.raw_weight_push_freshness_seconds == 300
    assert settings.raw_weight_push_timeout_seconds == 10.0


def test_raw_weight_push_interval_rejects_below_minimum() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChallengeSettings(
            shared_token="x",
            shared_token_file=None,
            raw_weight_push_interval_seconds=0.05,
        )


def test_maybe_build_push_client_loads_token_from_shared_token_file(
    tmp_path: Path,
) -> None:
    """File-backed challenge token alone is enough to construct the push client."""

    token_path = tmp_path / "challenge_token"
    token_path.write_text("file-backed-ac-token-value", encoding="utf-8")
    settings = ChallengeSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'ac-push-file-token.sqlite3'}",
        shared_token=None,
        shared_token_file=str(token_path),
        raw_weight_push_enabled=True,
        master_base_url="http://master.test",
        epoch_seconds=360,
    )
    db = Database(settings.database_url)
    client = push_module.maybe_build_push_client_from_settings(
        settings=settings,
        database=db,
    )
    assert client is not None
    assert client.shared_token == "file-backed-ac-token-value"
    assert client.master_base_url == "http://master.test"
    assert client.challenge_slug == "agent-challenge"
