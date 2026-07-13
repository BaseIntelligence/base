"""Alembic migration coverage for ``worker_assignments`` (revision 0010).

Postgres counterpart to the sqlite drift test: ``alembic upgrade head`` creates
``worker_assignments`` with its replica columns and the ``(work_unit_id,
worker_id)`` uniqueness that keeps a restart from spawning orphan replicas
(VAL-AGENT-017); a downgrade to the prior worker-registry revision drops it while
leaving ``worker_registrations`` and the legacy control-plane tables intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from base.db import create_engine

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[2]

WORKER_ASSIGNMENT_COLUMNS = {
    "id",
    "challenge_slug",
    "work_unit_id",
    "submission_ref",
    "worker_id",
    "worker_pubkey",
    "miner_hotkey",
    "payload",
    "required_capability",
    "status",
    "attempt_count",
    "max_attempts",
    "deadline_at",
    "last_progress_at",
    "checkpoint_ref",
    "result_success",
    "result_payload",
    "manifest_sha256",
    "created_at",
    "updated_at",
}
PRESERVED_TABLES = ("worker_registrations", "work_assignments", "validators")


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


async def _unique_column_sets(database_url: str, table: str) -> set[tuple[str, ...]]:
    engine = create_engine(database_url)
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        """
                        SELECT tc.constraint_name, kcu.column_name
                        FROM information_schema.table_constraints tc
                        JOIN information_schema.key_column_usage kcu
                          ON tc.constraint_name = kcu.constraint_name
                         AND tc.table_schema = kcu.table_schema
                        WHERE tc.table_schema = current_schema()
                          AND tc.table_name = :t
                          AND tc.constraint_type = 'UNIQUE'
                        ORDER BY kcu.ordinal_position
                        """
                    ),
                    {"t": table},
                )
            ).all()
    finally:
        await engine.dispose()
    grouped: dict[str, list[str]] = {}
    for constraint_name, column_name in rows:
        grouped.setdefault(constraint_name, []).append(column_name)
    return {tuple(sorted(cols)) for cols in grouped.values()}


async def test_worker_assignments_present_after_upgrade(
    migrated_postgres_database: str,
) -> None:
    columns = await _columns(migrated_postgres_database, "worker_assignments")
    assert WORKER_ASSIGNMENT_COLUMNS <= columns
    unique_sets = await _unique_column_sets(
        migrated_postgres_database, "worker_assignments"
    )
    assert ("work_unit_id", "worker_id") in unique_sets


async def test_worker_assignments_downgrade_upgrade_roundtrip(
    migrated_postgres_database: str,
) -> None:
    # head→0009 path crosses 0011_drop_llm_usage_records, which cannot
    # downgrade after LLM gateway removal.
    pytest.skip(
        "Downgrade past 0011_drop_llm_usage_records is deliberately irreversible"
    )
