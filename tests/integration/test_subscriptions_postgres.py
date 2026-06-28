"""Postgres parity for the validator subscriptions column (0008).

``alembic upgrade head`` from an empty Postgres DB creates the
``validators.subscriptions`` JSON column, the default is ``[]`` for inserted
rows, and a non-empty subscription set round-trips on Postgres.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text

from base.db import (
    Validator,
    ValidatorStatus,
    create_engine,
    create_session_factory,
    session_scope,
)

pytestmark = pytest.mark.postgres


async def test_subscriptions_column_exists_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    try:
        async with engine.connect() as connection:
            column = (
                (
                    await connection.execute(
                        text(
                            """
                            SELECT is_nullable
                            FROM information_schema.columns
                            WHERE table_schema = current_schema()
                              AND table_name = 'validators'
                              AND column_name = 'subscriptions'
                            """
                        )
                    )
                )
                .mappings()
                .one()
            )
    finally:
        await engine.dispose()

    assert column["is_nullable"] == "NO"


async def test_subscriptions_default_and_round_trip_on_postgres(
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    session_factory = create_session_factory(engine)
    try:
        # a row inserted without subscriptions defaults to the empty list.
        async with session_scope(session_factory) as session:
            session.add(
                Validator(
                    hotkey="default-validator",
                    status=ValidatorStatus.ONLINE,
                    capabilities=["cpu"],
                    version="1.0.0",
                    registered_at=datetime.now(UTC),
                    last_heartbeat_at=datetime.now(UTC),
                )
            )
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == "default-validator")
                )
            ).scalar_one()
            assert row.subscriptions == []

        # a non-empty subscription set persists and reads back.
        async with session_scope(session_factory) as session:
            session.add(
                Validator(
                    hotkey="subscribed-validator",
                    status=ValidatorStatus.ONLINE,
                    capabilities=["cpu", "gpu"],
                    subscriptions=["prism", "agent-challenge"],
                    version="1.0.0",
                    registered_at=datetime.now(UTC),
                    last_heartbeat_at=datetime.now(UTC),
                )
            )
        async with session_factory() as session:
            row = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == "subscribed-validator")
                )
            ).scalar_one()
            assert row.subscriptions == ["prism", "agent-challenge"]
    finally:
        await engine.dispose()
        await cleanup_postgres_database()
