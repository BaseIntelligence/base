"""Migration tests for validator subscriptions (0008) on SQLite.

``alembic upgrade head`` from an empty DB adds the ``validators.subscriptions``
JSON-list column (NOT NULL, server default ``[]``); the history has a single
head; the migration matches the ORM models (empty ``compare_metadata`` diff);
and a downgrade to the prior revision / re-upgrade round-trips cleanly.
"""

from __future__ import annotations

from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import inspect

from base.db import Base, migrations

ROOT_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI = ROOT_DIR / "alembic.ini"
PRIOR_REVISION = "0007_harden_validator_registry"


def _async_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


def _sync_url(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def test_single_head() -> None:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ROOT_DIR / "alembic"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1


def test_upgrade_adds_subscriptions_column(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.sqlite3"
    migrations.upgrade(ALEMBIC_INI, database_url=_async_url(db_path), revision="head")

    engine = create_sync_engine(_sync_url(db_path))
    try:
        columns = {c["name"]: c for c in inspect(engine).get_columns("validators")}
        assert "subscriptions" in columns
        assert columns["subscriptions"]["nullable"] is False
        assert columns["subscriptions"]["default"] is not None
    finally:
        engine.dispose()


def test_migration_matches_models_no_drift(tmp_path: Path) -> None:
    db_path = tmp_path / "compare.sqlite3"
    migrations.upgrade(ALEMBIC_INI, database_url=_async_url(db_path), revision="head")

    engine = create_sync_engine(_sync_url(db_path))
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection,
                opts={
                    "compare_type": True,
                    "compare_server_default": True,
                    "render_as_batch": True,
                    "target_metadata": Base.metadata,
                },
            )
            diff = compare_metadata(context, Base.metadata)
    finally:
        engine.dispose()

    assert diff == []


def test_downgrade_to_prior_reverts_and_reupgrade_restores(tmp_path: Path) -> None:
    db_path = tmp_path / "roundtrip.sqlite3"
    async_url = _async_url(db_path)

    migrations.upgrade(ALEMBIC_INI, database_url=async_url, revision="head")

    migrations.downgrade(ALEMBIC_INI, database_url=async_url, revision=PRIOR_REVISION)
    engine = create_sync_engine(_sync_url(db_path))
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("validators")}
        assert "subscriptions" not in columns
        # earlier control-plane columns remain intact.
        assert "capabilities" in columns
        assert "last_seen_meta" in columns
    finally:
        engine.dispose()

    migrations.upgrade(ALEMBIC_INI, database_url=async_url, revision="head")
    engine = create_sync_engine(_sync_url(db_path))
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("validators")}
        assert "subscriptions" in columns
    finally:
        engine.dispose()
