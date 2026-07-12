from __future__ import annotations

import asyncio
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
        _, now, _idem = await service.heartbeat(
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
        validator, _now, _idem = await service.heartbeat(hotkey=hotkey)
        assert validator.status == ValidatorStatus.ONLINE
        events = await _events(session_factory, hotkey)
        assert events[-1] == ValidatorHealthEventType.ONLINE
    finally:
        await engine.dispose()


async def test_crash_detection_and_recovery_flow_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    """VAL-VREG-023: register -> detect-crash -> recover parity on Postgres."""

    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = ValidatorCoordinationService(
        session_factory,
        heartbeat_interval_seconds=30,
        heartbeat_timeout_seconds=100,
        now_fn=clock.now,
    )
    hotkey = "****************"

    try:
        await service.register(
            hotkey=hotkey,
            uid=3,
            capabilities=["cpu", "gpu"],
            version="1.0.0",
            last_seen_meta={"broker": "ok"},
        )
        assert await _status(session_factory, hotkey) == ValidatorStatus.ONLINE

        # Recent validator within the window is NOT marked offline.
        clock.value = BASE_TS + timedelta(seconds=100)
        assert await service.detect_offline_validators() == []
        assert await _status(session_factory, hotkey) == ValidatorStatus.ONLINE

        # Stale heartbeat past the timeout flips to offline with a crash event.
        clock.value = BASE_TS + timedelta(seconds=200)
        assert await service.detect_offline_validators() == [hotkey]
        assert await _status(session_factory, hotkey) == ValidatorStatus.OFFLINE

        # Edge-triggered: a second pass records no duplicate crash event.
        clock.value = BASE_TS + timedelta(seconds=400)
        assert await service.detect_offline_validators() == []

        # Recovery via heartbeat flips back to online.
        clock.value = BASE_TS + timedelta(seconds=450)
        validator, _now, _idem = await service.heartbeat(hotkey=hotkey)
        assert validator.status == ValidatorStatus.ONLINE

        events = await _events(session_factory, hotkey)
        assert events == [
            ValidatorHealthEventType.REGISTERED,
            ValidatorHealthEventType.ONLINE,
            ValidatorHealthEventType.CRASH_DETECTED,
            ValidatorHealthEventType.ONLINE,
        ]
        listed = await service.list_validators()
        assert [v.hotkey for v in listed] == [hotkey]
        assert listed[0].status == ValidatorStatus.ONLINE
        assert listed[0].capabilities == ["cpu", "gpu"]
    finally:
        await engine.dispose()


async def _status(
    session_factory: async_sessionmaker[AsyncSession], hotkey: str
) -> ValidatorStatus:
    async with session_factory() as session:
        row = (
            await session.execute(select(Validator).where(Validator.hotkey == hotkey))
        ).scalar_one()
        return ValidatorStatus(row.status)


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


async def test_concurrent_first_register_single_row_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    """Concurrent first-register of one new hotkey yields a single row, no error."""

    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = ValidatorCoordinationService(
        session_factory, heartbeat_interval_seconds=30, now_fn=clock.now
    )
    hotkey = "****************"
    try:
        results = await asyncio.gather(
            *(
                service.register(
                    hotkey=hotkey,
                    uid=1,
                    capabilities=["cpu"],
                    version="1.0.0",
                )
                for _ in range(6)
            ),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"concurrent register raised: {errors!r}"

        async with session_factory() as session:
            count = await session.scalar(
                select(func.count(Validator.id)).where(Validator.hotkey == hotkey)
            )
        assert count == 1

        events = await _events(session_factory, hotkey)
        assert events.count(ValidatorHealthEventType.REGISTERED) == 1
        assert events.count(ValidatorHealthEventType.ONLINE) == 1
    finally:
        await engine.dispose()


async def test_same_instant_events_have_monotonic_seq_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    """Same-created_at events read back in deterministic, append-only order."""

    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    clock = _Clock(BASE_TS)
    service = ValidatorCoordinationService(
        session_factory, heartbeat_interval_seconds=30, now_fn=clock.now
    )
    hotkey = "****************"
    try:
        # register emits registered + online at the SAME stubbed instant.
        await service.register(
            hotkey=hotkey, uid=1, capabilities=["cpu"], version="1.0.0"
        )
        async with session_scope(session_factory) as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one()
            row.status = ValidatorStatus.OFFLINE
        # heartbeat recovery at the SAME instant -> a third online event.
        await service.heartbeat(hotkey=hotkey)

        events = await service.list_health_events(hotkey)
        created = [event.created_at for event in events]
        seqs = [event.seq for event in events]

        assert len(set(created)) == 1, created
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)
        assert [event.event for event in events] == [
            ValidatorHealthEventType.REGISTERED,
            ValidatorHealthEventType.ONLINE,
            ValidatorHealthEventType.ONLINE,
        ]
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
