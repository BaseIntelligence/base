from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from pathlib import Path

import pytest

from base.db.session import create_engine, create_session_factory
from base.master.registry import DatabaseChallengeRegistry
from base.schemas.challenge import ChallengeCreate, ChallengeStatus


def _payload(
    *, slug: str, name: str, status: ChallengeStatus, emission_percent: Decimal
) -> ChallengeCreate:
    return ChallengeCreate(
        slug=slug,
        name=name,
        image="ghcr.io/baseintelligence/demo:1.0.0",
        version="1.0.0",
        emission_percent=emission_percent,
        status=status,
    )


@pytest.mark.postgres
async def test_database_registry_active_only_list_uses_postgres_asyncpg(
    tmp_path: Path,
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    registry = DatabaseChallengeRegistry(
        create_session_factory(engine),
        secret_dir=tmp_path / "secrets",
    )

    try:
        await registry.create(
            _payload(
                slug="test-active-challenge-001",
                name="Active Challenge Regression",
                status=ChallengeStatus.ACTIVE,
                emission_percent=Decimal("100"),
            )
        )
        await registry.create(
            _payload(
                slug="test-inactive-challenge-001",
                name="Inactive Challenge Regression",
                status=ChallengeStatus.INACTIVE,
                emission_percent=Decimal("0"),
            )
        )

        active_records = await registry.list(active_only=True)

        assert [record.slug for record in active_records] == [
            "test-active-challenge-001"
        ]
        assert active_records[0].name == "Active Challenge Regression"
        assert active_records[0].status == ChallengeStatus.ACTIVE
        assert "test-inactive-challenge-001" not in {
            record.slug for record in active_records
        }
    finally:
        await engine.dispose()
        await cleanup_postgres_database()


@pytest.mark.postgres
async def test_database_registry_set_status_serializes_server_updated_timestamp(
    tmp_path: Path,
    migrated_postgres_database: str,
    cleanup_postgres_database: Callable[[], Awaitable[None]],
) -> None:
    engine = create_engine(migrated_postgres_database)
    registry = DatabaseChallengeRegistry(
        create_session_factory(engine),
        secret_dir=tmp_path / "secrets",
    )

    try:
        await registry.create(
            _payload(
                slug="test-status-challenge-001",
                name="Status Challenge Regression",
                status=ChallengeStatus.DRAFT,
                emission_percent=Decimal("0"),
            )
        )

        record = await registry.set_status(
            "test-status-challenge-001", ChallengeStatus.ACTIVE
        )

        assert record.status == ChallengeStatus.ACTIVE
        assert record.updated_at is not None
    finally:
        await engine.dispose()
        await cleanup_postgres_database()
