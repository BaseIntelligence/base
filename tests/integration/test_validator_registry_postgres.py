from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from base.db import (
    Validator,
    ValidatorHealthEvent,
    ValidatorHealthEventType,
    ValidatorRequestNonce,
    ValidatorStatus,
    create_engine,
    create_session_factory,
    session_scope,
)

pytestmark = pytest.mark.postgres

VALIDATOR_TABLES = (
    "validators",
    "validator_health_events",
    "validator_request_nonces",
)


async def test_migration_creates_validator_tables(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    try:
        async with engine.connect() as connection:
            present = {
                row[0]
                for row in (
                    await connection.execute(
                        text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = current_schema()"
                        )
                    )
                ).all()
            }
            status_column = (
                (
                    await connection.execute(
                        text(
                            """
                            SELECT data_type
                            FROM information_schema.columns
                            WHERE table_schema = current_schema()
                              AND table_name = 'validators'
                              AND column_name = 'status'
                            """
                        )
                    )
                )
                .mappings()
                .one()
            )
            native_enum_count = (
                await connection.execute(
                    text(
                        "SELECT count(*) FROM pg_type "
                        "WHERE typname = 'validator_status'"
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    assert set(VALIDATOR_TABLES).issubset(present)
    assert status_column["data_type"] == "character varying"
    assert native_enum_count == 0


async def test_validator_lifecycle_round_trips_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    hotkey = "5FpgValidator"
    try:
        async with session_scope(session_factory) as session:
            session.add(
                Validator(
                    hotkey=hotkey,
                    uid=2,
                    status=ValidatorStatus.ONLINE,
                    capabilities=["cpu", "gpu"],
                    version="9.9.9",
                    registered_at=datetime.now(UTC),
                    last_heartbeat_at=datetime.now(UTC),
                    last_seen_meta={"broker": "ok"},
                )
            )
            session.add(
                ValidatorHealthEvent(
                    validator_hotkey=hotkey,
                    event=ValidatorHealthEventType.REGISTERED,
                )
            )
            session.add(
                ValidatorHealthEvent(
                    validator_hotkey=hotkey,
                    event=ValidatorHealthEventType.ONLINE,
                )
            )

        async with session_factory() as session:
            stored = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one()
            assert stored.status == ValidatorStatus.ONLINE
            assert stored.capabilities == ["cpu", "gpu"]
            assert stored.last_seen_meta == {"broker": "ok"}

            events = (
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
            assert [event.event for event in events] == [
                ValidatorHealthEventType.REGISTERED,
                ValidatorHealthEventType.ONLINE,
            ]
    finally:
        await engine.dispose()


async def test_duplicate_hotkey_violates_unique_constraint_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    hotkey = "5FpgDuplicate"
    try:
        async with session_scope(session_factory) as session:
            session.add(Validator(hotkey=hotkey, status=ValidatorStatus.ONLINE))

        with pytest.raises(IntegrityError):
            async with session_scope(session_factory) as session:
                session.add(Validator(hotkey=hotkey, status=ValidatorStatus.OFFLINE))
    finally:
        await engine.dispose()


async def test_duplicate_nonce_violates_unique_constraint_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    try:
        async with session_scope(session_factory) as session:
            session.add(
                ValidatorRequestNonce(
                    id=uuid.uuid4(),
                    hotkey="5FpgNonce",
                    nonce="dup",
                    body_hash="h",
                    created_at=datetime.now(UTC),
                )
            )

        with pytest.raises(IntegrityError):
            async with session_scope(session_factory) as session:
                session.add(
                    ValidatorRequestNonce(
                        id=uuid.uuid4(),
                        hotkey="5FpgNonce",
                        nonce="dup",
                        body_hash="h2",
                        created_at=datetime.now(UTC),
                    )
                )

        async with session_factory() as session:
            count = await session.scalar(select(func.count(ValidatorRequestNonce.id)))
        assert count == 1
    finally:
        await engine.dispose()
