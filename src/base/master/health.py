from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alembic.script import ScriptDirectory
from sqlalchemy import text

from base.challenge_sdk.health import ReadinessProbe
from base.db.migrations import alembic_config


def migration_head(config_path: str, database_url: str) -> str:
    """Return the single expected migration head without connecting to the database."""

    head = ScriptDirectory.from_config(
        alembic_config(config_path, database_url)
    ).get_current_head()
    if head is None:
        raise RuntimeError("master migration head is unavailable")
    return head


def postgres_readiness_probe(
    session_factory: Callable[[], Any],
    *,
    expected_migration_revision: str,
) -> ReadinessProbe:
    """Check both PostgreSQL connectivity and the applied schema revision."""

    async def check() -> bool:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
            result = await session.execute(
                text("SELECT version_num FROM alembic_version")
            )
            return result.scalar_one_or_none() == expected_migration_revision

    return ReadinessProbe(name="postgresql", check=check)


__all__ = ["migration_head", "postgres_readiness_probe"]
