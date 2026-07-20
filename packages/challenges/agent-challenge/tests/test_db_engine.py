from __future__ import annotations

from unittest.mock import patch

import agent_challenge.sdk.db as db_module
from agent_challenge.sdk.db import Database


def _capture_engine_kwargs(database_url: str) -> dict[str, object]:
    captured: dict[str, object] = {}
    real = db_module.create_async_engine

    def capture(url, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        captured["url"] = url
        # Build a harmless in-memory sqlite engine so Database.__init__ succeeds
        # regardless of the URL under test.
        return real("sqlite+aiosqlite:///:memory:")

    with patch.object(db_module, "create_async_engine", side_effect=capture):
        Database(database_url)
    return captured


def test_postgresql_engine_has_connection_hygiene() -> None:
    captured = _capture_engine_kwargs("postgresql+asyncpg://user:pass@host:5432/db")

    assert captured.get("pool_pre_ping") is True
    assert captured.get("pool_recycle") == 300
    connect_args = captured.get("connect_args") or {}
    assert isinstance(connect_args, dict)
    # command_timeout bounds any single statement so a black-holed cross-node
    # TCP connection fails fast instead of hanging indefinitely.
    assert connect_args.get("command_timeout") == 60


def test_sqlite_engine_keeps_existing_connect_args() -> None:
    captured = _capture_engine_kwargs("sqlite+aiosqlite:///:memory:")

    connect_args = captured.get("connect_args") or {}
    assert isinstance(connect_args, dict)
    assert connect_args.get("check_same_thread") is False
    assert connect_args.get("timeout") == 30.0
    # sqlite is in-process; PostgreSQL-only hygiene must not leak in.
    assert "command_timeout" not in connect_args
    assert captured.get("pool_pre_ping") is None


def test_postgresql_engine_uses_bounded_connection_pool() -> None:
    captured = _capture_engine_kwargs("postgresql+*****************************/db")

    # Defense in depth for the combined worker: concurrent SSE streams plus the
    # worker loop must not exhaust the pool.
    assert captured.get("pool_size") == 10
    assert captured.get("max_overflow") == 20


def test_file_sqlite_engine_uses_bounded_connection_pool() -> None:
    captured = _capture_engine_kwargs("sqlite+aiosqlite:////tmp/agent-challenge.sqlite3")

    assert captured.get("pool_size") == 10
    assert captured.get("max_overflow") == 20


def test_memory_sqlite_engine_keeps_staticpool_without_sizing() -> None:
    captured = _capture_engine_kwargs("sqlite+aiosqlite:///:memory:")

    # In-memory SQLite uses a StaticPool (single shared connection); pool sizing
    # would break the shared-memory database the test suite relies on.
    assert "pool_size" not in captured
    assert "max_overflow" not in captured
