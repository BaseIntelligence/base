from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import text

from base.db import create_engine
from base.db.migrations import downgrade, upgrade

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[2]

WORKER_REGISTRATION_COLUMNS = {
    "worker_id",
    "worker_pubkey",
    "miner_hotkey",
    "binding_signature",
    "provider",
    "provider_instance_ref",
    "capabilities",
    "status",
    "last_heartbeat_at",
    "created_at",
}
# Legacy tables the worker migration MUST NOT alter.
LEGACY_TABLES = ("validators", "work_assignments", "challenges")


async def _columns(database_url: str, table: str) -> set[str]:
    engine = create_engine(database_url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = :t"
                    ),
                    {"t": table},
                )
            ).scalars()
            return set(rows.all())
    finally:
        await engine.dispose()


async def _table_exists(database_url: str, table: str) -> bool:
    engine = create_engine(database_url)
    try:
        async with engine.connect() as connection:
            found = (
                await connection.execute(
                    text(
                        "SELECT 1 FROM information_schema.tables "
                        "WHERE table_schema = current_schema() AND table_name = :t"
                    ),
                    {"t": table},
                )
            ).first()
            return found is not None
    finally:
        await engine.dispose()


# VAL-MASTER-015
async def test_worker_tables_present_after_upgrade(
    migrated_postgres_database: str,
) -> None:
    registration_columns = await _columns(
        migrated_postgres_database, "worker_registrations"
    )
    assert WORKER_REGISTRATION_COLUMNS <= registration_columns
    assert await _table_exists(migrated_postgres_database, "worker_faults")

    legacy_before = {
        table: await _columns(migrated_postgres_database, table)
        for table in LEGACY_TABLES
    }
    for table, columns in legacy_before.items():
        assert columns, f"legacy table {table} unexpectedly empty/missing"


# VAL-MASTER-015: downgrade removes the worker tables, upgrade restores them.
async def test_worker_migration_downgrade_upgrade_roundtrip(
    migrated_postgres_database: str,
) -> None:
    config_path = ROOT / "alembic.ini"

    legacy_before = {
        table: await _columns(migrated_postgres_database, table)
        for table in LEGACY_TABLES
    }

    # alembic env runs asyncio.run internally, so drive it from a worker thread
    # to avoid nesting event loops inside this async test.
    await asyncio.to_thread(
        downgrade, config_path, database_url=migrated_postgres_database, revision="-1"
    )
    try:
        assert not await _table_exists(
            migrated_postgres_database, "worker_registrations"
        )
        assert not await _table_exists(migrated_postgres_database, "worker_faults")
        assert not await _table_exists(
            migrated_postgres_database, "worker_request_nonces"
        )
        # legacy tables untouched by the downgrade.
        for table, columns in legacy_before.items():
            assert await _columns(migrated_postgres_database, table) == columns
    finally:
        # restore head so the shared session-scoped test database is intact.
        await asyncio.to_thread(
            upgrade,
            config_path,
            database_url=migrated_postgres_database,
            revision="head",
        )

    assert await _table_exists(migrated_postgres_database, "worker_registrations")
    assert await _table_exists(migrated_postgres_database, "worker_faults")
    for table, columns in legacy_before.items():
        assert await _columns(migrated_postgres_database, table) == columns
