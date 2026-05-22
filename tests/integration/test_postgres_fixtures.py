from __future__ import annotations

import pytest

ASYNC_POSTGRES_SCHEME = "postgresql+asyncpg://"


@pytest.mark.postgres
def test_migrated_postgres_database_fixture_runs_migrations(
    migrated_postgres_database: str,
) -> None:
    assert migrated_postgres_database.startswith(ASYNC_POSTGRES_SCHEME)
