"""Hardening tests for the validator coordination control plane.

Covers the misc-hardening feature:
- first-registration is atomic (concurrent first-register of the same new hotkey
  yields a single ``validators`` row and never a 500 / unhandled error);
- ``validator_health_events`` carry a monotonic ``seq`` so same-instant events
  read back in deterministic, append-only order;
- schema polish: ``ix_validators_registered_at`` index + ``Validator.version``
  is non-null with a server default.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from base.db import (
    Base,
    Validator,
    ValidatorHealthEventType,
    ValidatorStatus,
)
from base.db.session import create_engine, create_session_factory, session_scope
from base.master.validator_coordination import ValidatorCoordinationService

NOW_EPOCH = 1_750_000_000.0
HEARTBEAT_INTERVAL = 45
HEARTBEAT_TIMEOUT = 100


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class FakeClock:
    def __init__(self, epoch: float) -> None:
        self.epoch = float(epoch)

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.epoch, UTC)


async def _build_file_service(
    tmp_path: Any,
    *,
    name: str = "registry.sqlite3",
    timeout: int = HEARTBEAT_TIMEOUT,
    epoch: float = NOW_EPOCH,
) -> tuple[ValidatorCoordinationService, Any, FakeClock, Any]:
    """Build a service backed by a file SQLite DB (multi-connection safe)."""

    db_path = tmp_path / name
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)
    clock = FakeClock(epoch)
    service = ValidatorCoordinationService(
        session_factory,
        heartbeat_interval_seconds=HEARTBEAT_INTERVAL,
        heartbeat_timeout_seconds=timeout,
        now_fn=clock.now,
    )
    return service, session_factory, clock, engine


async def _count(session_factory: Any, hotkey: str) -> int:
    async with session_factory() as session:
        return await session.scalar(
            select(func.count(Validator.id)).where(Validator.hotkey == hotkey)
        )


# ---------------------------------------------------------------------------
# Atomic first-registration
# ---------------------------------------------------------------------------


async def test_concurrent_first_register_yields_single_row_no_error(
    tmp_path: Any,
) -> None:
    service, session_factory, _clock, engine = await _build_file_service(tmp_path)
    try:
        results = await asyncio.gather(
            *(
                service.register(
                    hotkey="permitted",
                    uid=1,
                    capabilities=["cpu"],
                    version="1.0.0",
                )
                for _ in range(5)
            ),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"concurrent register raised: {errors!r}"

        assert await _count(session_factory, "permitted") == 1

        events = await service.list_health_events("permitted")
        event_types = [event.event for event in events]
        assert event_types.count(ValidatorHealthEventType.REGISTERED) == 1
        assert event_types.count(ValidatorHealthEventType.ONLINE) == 1

        async with session_factory() as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == "permitted")
                )
            ).scalar_one()
        assert row.status == ValidatorStatus.ONLINE
    finally:
        await engine.dispose()


async def test_register_retries_as_update_on_integrity_error(tmp_path: Any) -> None:
    """A lost first-register race (INSERT conflict) retries as an idempotent update."""

    service, session_factory, clock, engine = await _build_file_service(tmp_path)
    try:
        now = clock.now()
        original = service._register_in_session
        state = {"calls": 0}

        async def racing(session: Any, **kwargs: Any) -> Any:
            state["calls"] += 1
            if state["calls"] == 1:
                # A concurrent first-register commits the row first, then our
                # INSERT would conflict on the unique hotkey -> IntegrityError.
                async with session_scope(session_factory) as other:
                    other.add(
                        Validator(
                            hotkey=kwargs["hotkey"],
                            uid=None,
                            status=ValidatorStatus.ONLINE,
                            capabilities=["cpu"],
                            version="1.0.0",
                            registered_at=now,
                            last_heartbeat_at=now,
                            last_seen_meta={},
                        )
                    )
                    await ValidatorCoordinationService._add_event(
                        other,
                        kwargs["hotkey"],
                        ValidatorHealthEventType.REGISTERED,
                        now,
                    )
                    await ValidatorCoordinationService._add_event(
                        other,
                        kwargs["hotkey"],
                        ValidatorHealthEventType.ONLINE,
                        now,
                    )
                raise IntegrityError("INSERT", {}, Exception("duplicate hotkey"))
            return await original(session, **kwargs)

        service._register_in_session = racing  # type: ignore[method-assign]

        result, _ = await service.register(
            hotkey="permitted",
            uid=7,
            capabilities=["cpu", "gpu"],
            version="2.0.0",
        )

        assert state["calls"] == 2
        assert result.status == ValidatorStatus.ONLINE
        assert await _count(session_factory, "permitted") == 1

        async with session_factory() as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == "permitted")
                )
            ).scalar_one()
        # The retry applied the second payload as an update.
        assert row.capabilities == ["cpu", "gpu"]
        assert row.version == "2.0.0"

        events = await service.list_health_events("permitted")
        event_types = [event.event for event in events]
        # Events from the winning register are preserved; the retry adds none.
        assert event_types.count(ValidatorHealthEventType.REGISTERED) == 1
    finally:
        await engine.dispose()


async def test_register_with_none_version_coalesces_to_default(tmp_path: Any) -> None:
    service, session_factory, _clock, engine = await _build_file_service(tmp_path)
    try:
        validator, _ = await service.register(
            hotkey="permitted",
            uid=1,
            capabilities=["cpu"],
            version=None,
        )
        assert validator.version is not None
        assert validator.version != ""

        async with session_factory() as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == "permitted")
                )
            ).scalar_one()
        assert row.version is not None
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Deterministic same-instant ordering
# ---------------------------------------------------------------------------


async def test_same_instant_events_have_monotonic_seq(tmp_path: Any) -> None:
    service, session_factory, clock, engine = await _build_file_service(tmp_path)
    try:
        # register emits registered + online both at the SAME clock instant.
        await service.register(
            hotkey="permitted", uid=1, capabilities=["cpu"], version="1.0.0"
        )

        # flip offline then heartbeat at the SAME instant -> another online event.
        async with session_scope(session_factory) as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == "permitted")
                )
            ).scalar_one()
            row.status = ValidatorStatus.OFFLINE
        await service.heartbeat(hotkey="permitted")

        events = await service.list_health_events("permitted")
        created = [_as_utc(event.created_at) for event in events]
        seqs = [event.seq for event in events]

        # All events share the same instant but seq is strictly increasing,
        # so the audit read order is deterministic and append-only.
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


async def test_list_health_events_orders_by_created_at_then_seq(tmp_path: Any) -> None:
    service, session_factory, clock, engine = await _build_file_service(tmp_path)
    try:
        await service.register(
            hotkey="permitted", uid=1, capabilities=["cpu"], version="1.0.0"
        )
        # advance the clock and force a crash event at a later instant.
        clock.epoch = NOW_EPOCH + HEARTBEAT_TIMEOUT + 1
        await service.detect_offline_validators()

        events = await service.list_health_events("permitted")
        keys = [(_as_utc(event.created_at), event.seq) for event in events]
        assert keys == sorted(keys)
        assert [event.event for event in events] == [
            ValidatorHealthEventType.REGISTERED,
            ValidatorHealthEventType.ONLINE,
            ValidatorHealthEventType.CRASH_DETECTED,
        ]
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Schema polish (model-level)
# ---------------------------------------------------------------------------


def test_validators_registered_at_is_indexed() -> None:
    table = Base.metadata.tables["validators"]
    index_names = {index.name for index in table.indexes}
    assert "ix_validators_registered_at" in index_names


def test_validator_version_is_non_null_with_server_default() -> None:
    column = Base.metadata.tables["validators"].c.version
    assert column.nullable is False
    assert column.server_default is not None


def test_validator_health_event_seq_is_non_null_monotonic_column() -> None:
    column = Base.metadata.tables["validator_health_events"].c.seq
    assert column.nullable is False


@pytest.mark.parametrize("missing", ["seq"])
def test_validator_health_event_index_includes_seq(missing: str) -> None:
    table = Base.metadata.tables["validator_health_events"]
    composite = next(
        index
        for index in table.indexes
        if index.name == "ix_validator_health_events_hotkey_created"
    )
    column_names = [column.name for column in composite.columns]
    assert missing in column_names
    assert column_names == ["validator_hotkey", "created_at", "seq"]
