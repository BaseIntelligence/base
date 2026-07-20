"""Prism SQLite policy, migration ledger, and legacy LLM restore safety.

Covers VAL-WEIGHT-092, VAL-GATE-043/044, VAL-COMPOSE-067 critically:
every connection enables foreign keys, WAL compatibility, and busy timeout;
schema revision is ordered and checksummed; legacy held/quarantine rows never
become scored solely because a restored volume contained them.
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from prism_challenge.db import (
    PRISM_BUSY_TIMEOUT_MS,
    PRISM_SCHEMA_REVISION,
    Database,
    _record_schema_revision,
)


@pytest.mark.asyncio
async def test_every_connection_applies_sqlite_policy(tmp_path: Path) -> None:
    db = Database(tmp_path / "prism.sqlite3")
    await db.init()
    async with db.connect() as conn:
        fk = await conn.execute_fetchall("PRAGMA foreign_keys")
        busy = await conn.execute_fetchall("PRAGMA busy_timeout")
        journal = await conn.execute_fetchall("PRAGMA journal_mode")
    assert int(fk[0][0]) == 1
    assert int(busy[0][0]) == PRISM_BUSY_TIMEOUT_MS
    assert str(journal[0][0]).lower() in {"wal", "memory", "delete"}  # WAL preferred


@pytest.mark.asyncio
async def test_schema_revision_ledger_is_idempotent(tmp_path: Path) -> None:
    db = Database(tmp_path / "prism.sqlite3")
    await db.init()
    async with db.connect() as conn:
        rows = await conn.execute_fetchall("SELECT revision, checksum FROM prism_schema_migrations")
        assert any(row[0] == PRISM_SCHEMA_REVISION for row in rows)
        # Rerun is a no-op.
        await _record_schema_revision(conn, PRISM_SCHEMA_REVISION)
        again = await conn.execute_fetchall(
            "SELECT count(*) FROM prism_schema_migrations WHERE revision=?",
            (PRISM_SCHEMA_REVISION,),
        )
        assert int(again[0][0]) == 1


@pytest.mark.asyncio
async def test_unknown_future_schema_revision_refuses(tmp_path: Path) -> None:
    db = Database(tmp_path / "prism.sqlite3")
    await db.init()
    async with db.connect() as conn:
        await conn.execute(
            "INSERT INTO prism_schema_migrations(revision, checksum, applied_at) "
            "VALUES ('prism-schema.v99', 'x', '2099-01-01T00:00:00+00:00')"
        )
        with pytest.raises(RuntimeError, match="unknown future"):
            await _record_schema_revision(conn, PRISM_SCHEMA_REVISION)


@pytest.mark.asyncio
async def test_legacy_held_rows_rejected_not_scored(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    # Simulate a pre-removal database with held/quarantined rows.
    async with aiosqlite.connect(path) as conn:
        await conn.execute(
            "CREATE TABLE submissions ("
            "id TEXT PRIMARY KEY, hotkey TEXT, epoch_id INTEGER, filename TEXT,"
            "code TEXT, code_hash TEXT, arch_hash TEXT, name TEXT, metadata TEXT,"
            "status TEXT, error TEXT, created_at TEXT, updated_at TEXT, claimed_at TEXT)"
        )
        await conn.execute(
            "INSERT INTO submissions VALUES ("
            "'s1','hk',1,'f.py','print(1)','h','a',NULL,'{}',"
            "'held',NULL,'t','t',NULL)"
        )
        await conn.execute(
            "INSERT INTO submissions VALUES ("
            "'s2','hk',1,'g.py','print(2)','h2','a',NULL,'{}',"
            "'quarantined',NULL,'t','t',NULL)"
        )
        await conn.execute(
            "INSERT INTO submissions VALUES ("
            "'s3','hk',1,'ok.py','print(3)','h3','a',NULL,'{}',"
            "'completed',NULL,'t','t',NULL)"
        )
        await conn.commit()

    db = Database(path)
    await db.init()
    async with db.connect() as conn:
        rows = await conn.execute_fetchall("SELECT id, status, error FROM submissions ORDER BY id")
    by_id = {row[0]: (row[1], row[2]) for row in rows}
    assert by_id["s1"][0] == "rejected"
    assert "legacy" in (by_id["s1"][1] or "").lower()
    assert by_id["s2"][0] == "rejected"
    assert by_id["s3"][0] == "completed"
    # No gated/scored promotion from legacy hold.
    assert by_id["s1"][0] not in {"completed", "scored", "approved"}
