from __future__ import annotations

import re
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from platform_network.db.session import create_engine

_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


@pytest.mark.postgres
async def test_multi_challenge_postgres_schema_contexts_are_isolated(
    migrated_postgres_database: str,
) -> None:
    engine = create_engine(migrated_postgres_database)
    schema_a = f"challenge_isolation_a_{uuid.uuid4().hex}"
    schema_b = f"challenge_isolation_b_{uuid.uuid4().hex}"

    try:
        await _create_schema_context(engine, schema_a, "value-from-a")
        await _create_schema_context(engine, schema_b, "value-from-b")

        assert await _read_schema_value(engine, schema_a) == "value-from-a"
        assert await _read_schema_value(engine, schema_b) == "value-from-b"
        assert await _schema_value_count(engine, schema_a, "value-from-b") == 0
        assert await _schema_value_count(engine, schema_b, "value-from-a") == 0
    finally:
        await _drop_schemas(engine, schema_a, schema_b)
        await engine.dispose()


async def _create_schema_context(
    engine: AsyncEngine, schema_name: str, value: str
) -> None:
    schema = _schema_identifier(schema_name)
    async with engine.begin() as connection:
        await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        await connection.execute(
            text("CREATE TABLE challenge_values (value text PRIMARY KEY)")
        )
        await connection.execute(
            text("INSERT INTO challenge_values (value) VALUES (:value)"),
            {"value": value},
        )


async def _read_schema_value(engine: AsyncEngine, schema_name: str) -> str:
    schema = _schema_identifier(schema_name)
    async with engine.begin() as connection:
        await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        value = await connection.scalar(text("SELECT value FROM challenge_values"))
    assert isinstance(value, str)
    return value


async def _schema_value_count(engine: AsyncEngine, schema_name: str, value: str) -> int:
    schema = _schema_identifier(schema_name)
    async with engine.begin() as connection:
        await connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        count = await connection.scalar(
            text("SELECT count(*) FROM challenge_values WHERE value = :value"),
            {"value": value},
        )
    assert isinstance(count, int)
    return count


async def _drop_schemas(engine: AsyncEngine, *schema_names: str) -> None:
    async with engine.begin() as connection:
        for schema_name in schema_names:
            schema = _schema_identifier(schema_name)
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


def _schema_identifier(schema_name: str) -> str:
    assert _SCHEMA_RE.fullmatch(schema_name)
    return schema_name
