from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db import (
    Validator,
    ValidatorHealthEvent,
    ValidatorHealthEventType,
    ValidatorStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.master.validator_coordination import (
    ValidatorCoordinationService,
    ValidatorNotRegisteredError,
)

pytestmark = pytest.mark.postgres

BASE_TS = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


async def test_register_and_heartbeat_flow_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = ValidatorCoordinationService(
        session_factory, heartbeat_interval_seconds=30, now_fn=clock.now
    )
    hotkey = "5PgValidatorFlow"

    try:
        # First registration: row online with capabilities + registered/online.
        await service.register(
            hotkey=hotkey,
            uid=7,
            capabilities=["cpu", "gpu"],
            version="1.0.0",
            last_seen_meta={"broker": "ok"},
        )
        async with session_factory() as session:
            validator = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one()
            assert validator.status == ValidatorStatus.ONLINE
            assert validator.capabilities == ["cpu", "gpu"]
            assert validator.version == "1.0.0"
            assert validator.registered_at is not None
            assert validator.last_heartbeat_at is not None
            events = await _events(session_factory, hotkey)
            assert events == [
                ValidatorHealthEventType.REGISTERED,
                ValidatorHealthEventType.ONLINE,
            ]
        registered_at = validator.registered_at

        # Idempotent re-register updates the same row, preserves registered_at.
        clock.value = BASE_TS + timedelta(seconds=30)
        await service.register(
            hotkey=hotkey,
            uid=7,
            capabilities=["cpu"],
            version="2.0.0",
            last_seen_meta=None,
        )
        async with session_factory() as session:
            count = await session.scalar(
                select(func.count(Validator.id)).where(Validator.hotkey == hotkey)
            )
            assert count == 1
            validator = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one()
            assert validator.capabilities == ["cpu"]
            assert validator.version == "2.0.0"
            assert validator.registered_at == registered_at
            assert validator.last_seen_meta == {"broker": "ok"}

        # Heartbeat persists metadata + advances last_heartbeat_at.
        clock.value = BASE_TS + timedelta(seconds=60)
        _, now = await service.heartbeat(
            hotkey=hotkey, last_seen_meta={"concurrency": 2}
        )
        assert now == clock.value
        async with session_factory() as session:
            validator = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one()
            assert validator.last_heartbeat_at == clock.value
            assert validator.last_seen_meta == {"concurrency": 2}

        # Offline -> heartbeat flips back to online + emits online event.
        async with session_scope(session_factory) as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one()
            row.status = ValidatorStatus.OFFLINE
        clock.value = BASE_TS + timedelta(seconds=90)
        validator, _ = await service.heartbeat(hotkey=hotkey)
        assert validator.status == ValidatorStatus.ONLINE
        events = await _events(session_factory, hotkey)
        assert events[-1] == ValidatorHealthEventType.ONLINE
    finally:
        await engine.dispose()


async def test_heartbeat_unknown_hotkey_raises_and_creates_no_row_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    service = ValidatorCoordinationService(session_factory, now_fn=lambda: BASE_TS)
    try:
        with pytest.raises(ValidatorNotRegisteredError):
            await service.heartbeat(hotkey="5UnknownHotkey")
        async with session_factory() as session:
            count = await session.scalar(select(func.count(Validator.id)))
        assert count == 0
    finally:
        await engine.dispose()


async def _events(
    session_factory: async_sessionmaker[AsyncSession], hotkey: str
) -> list[ValidatorHealthEventType]:
    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(ValidatorHealthEvent)
                    .where(ValidatorHealthEvent.validator_hotkey == hotkey)
                    .order_by(ValidatorHealthEvent.created_at)
                )
            )
            .scalars()
            .all()
        )
    return [row.event for row in rows]
