#!/usr/bin/env python3
"""Wipe all Agent Challenge data rows while preserving the schema.

This is intended for clearing test/seed data from a deployment. It enumerates
every challenge-owned table from the SQLAlchemy metadata (so the list always
matches the models), prints current row counts, and on ``--execute`` truncates
them all. The schema itself is left intact and is in any case recreated by
``Database.init()`` (``Base.metadata.create_all``) on the next app start.

Safety model
------------
- Dry-run by default: prints the table list, per-table row counts, and the exact
  statement(s) that *would* run. Nothing is modified.
- ``--execute`` requires ``--yes`` to actually perform the wipe.
- After a wipe it re-counts every table and fails loudly if any row remains.

Backups
-------
This script does NOT take the backup for you, because on a production host the
canonical, dependency-free way is ``pg_dump`` against the running container:

    docker exec <postgres-container> \\
        pg_dump -U <user> -d <db> --no-owner --format=custom \\
        > agent-challenge-backup-$(date +%Y%m%dT%H%M%SZ).dump

Take that backup FIRST, then run this script with ``--execute --yes``.

Usage
-----
    # Dry run (no changes) — reads CHALLENGE_DATABASE_URL or --database-url:
    python scripts/wipe_db.py

    # Actually wipe (after taking a backup):
    python scripts/wipe_db.py --execute --yes

    # Explicit URL:
    python scripts/wipe_db.py --database-url postgresql+asyncpg://user:pw@host:5432/db \\
        --execute --yes
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from importlib import import_module

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Import the models package so every table registers on Base.metadata.
import_module("agent_challenge.core.models")
from agent_challenge.sdk.db import Base  # noqa: E402


def _resolve_database_url(explicit: str | None) -> str:
    url = explicit or os.environ.get("CHALLENGE_DATABASE_URL")
    if not url:
        raise SystemExit("No database URL. Pass --database-url or set CHALLENGE_DATABASE_URL.")
    return url


def _all_tables() -> list[str]:
    # Unordered: the models have FK cycles so no dependency sort exists. postgres
    # TRUNCATE CASCADE is order-independent; the sqlite path disables FK checks.
    return sorted(Base.metadata.tables.keys())


async def _count_rows(connection, tables: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in tables:
        result = await connection.execute(text(f'SELECT count(*) FROM "{name}"'))
        counts[name] = int(result.scalar_one())
    return counts


def _print_counts(title: str, counts: dict[str, int]) -> None:
    total = sum(counts.values())
    print(f"\n{title} (total rows: {total})")
    width = max((len(name) for name in counts), default=0)
    for name, count in counts.items():
        marker = "" if count == 0 else "  <-- non-empty"
        print(f"  {name.ljust(width)}  {count}{marker}")


async def _run(args: argparse.Namespace) -> int:
    database_url = _resolve_database_url(args.database_url)
    backend = database_url.split("://", 1)[0]
    tables = _all_tables()
    if not tables:
        raise SystemExit("No tables found on Base.metadata — wrong import path?")

    is_sqlite = backend.startswith("sqlite")
    is_postgresql = backend.startswith("postgresql")
    if not (is_sqlite or is_postgresql):
        raise SystemExit(f"Unsupported backend for wipe: {backend!r}")

    connect_args: dict[str, object] = {}
    if is_sqlite:
        connect_args = {"check_same_thread": False, "timeout": 30.0}
    engine = create_async_engine(database_url, connect_args=connect_args)

    # Redact credentials when echoing the URL.
    safe_url = database_url
    if "@" in safe_url and "://" in safe_url:
        scheme, rest = safe_url.split("://", 1)
        if "@" in rest:
            safe_url = f"{scheme}://<redacted>@{rest.split('@', 1)[1]}"
    print(f"[wipe] backend={backend} url={safe_url}")
    print(f"[wipe] {len(tables)} challenge-owned tables")

    try:
        async with engine.begin() as connection:
            before = await _count_rows(connection, tables)
        _print_counts("Current row counts", before)

        if is_postgresql:
            quoted = ", ".join(f'"{name}"' for name in tables)
            statements = [f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE"]
        else:
            statements = [f'DELETE FROM "{name}"' for name in tables]
            statements.append("DELETE FROM sqlite_sequence")

        if not args.execute:
            print("\n[dry-run] No changes made. Statements that WOULD run:")
            for statement in statements:
                print(f"  {statement};")
            print("\nRe-run with --execute --yes to perform the wipe.")
            return 0

        if not args.yes:
            raise SystemExit("Refusing to wipe without --yes.")

        async with engine.begin() as connection:
            if is_sqlite:
                await connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            for statement in statements:
                if statement == "DELETE FROM sqlite_sequence":
                    exists = await connection.execute(
                        text(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE type='table' AND name='sqlite_sequence'"
                        )
                    )
                    if exists.scalar_one_or_none() is None:
                        continue
                await connection.execute(text(statement))

        async with engine.begin() as connection:
            after = await _count_rows(connection, tables)
        _print_counts("Row counts after wipe", after)

        remaining = {name: count for name, count in after.items() if count}
        if remaining:
            print(f"\n[wipe] FAILED — rows remain in: {remaining}", file=sys.stderr)
            return 1
        print(f"\n[wipe] OK — all {len(tables)} tables emptied, schema preserved.")
        return 0
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        help="SQLAlchemy async URL (default: CHALLENGE_DATABASE_URL env).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the wipe (default is a dry run).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation alongside --execute.",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
