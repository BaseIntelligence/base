"""Migration tests for the validator-registry hardening (0007) on SQLite.

``alembic upgrade head`` from an empty DB adds the monotonic ``seq`` column to
``validator_health_events``, the ``ix_validators_registered_at`` index, and makes
``validators.version`` non-null; the history has a single head; the migration
matches the ORM models (empty ``compare_metadata`` diff); and a downgrade to the
prior revision / re-upgrade round-trips cleanly.
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
PRIOR_REVISION = "0006_create_work_results"


def _async_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


def _sync_url(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def test_single_head() -> None:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ROOT_DIR / "alembic"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1


def test_upgrade_adds_seq_index_and_non_null_version(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.sqlite3"
    migrations.upgrade(ALEMBIC_INI, database_url=_async_url(db_path), revision="head")

    engine = create_sync_engine(_sync_url(db_path))
    try:
        inspector = inspect(engine)

        event_columns = {
            c["name"]: c for c in inspector.get_columns("validator_health_events")
        }
        assert "seq" in event_columns
        assert event_columns["seq"]["nullable"] is False

        event_indexes = {
            idx["name"]: idx for idx in inspector.get_indexes("validator_health_events")
        }
        composite = event_indexes["ix_validator_health_events_hotkey_created"]
        assert composite["column_names"] == [
            "validator_hotkey",
            "created_at",
            "seq",
        ]

        validator_indexes = {idx["name"] for idx in inspector.get_indexes("validators")}
        assert "ix_validators_registered_at" in validator_indexes

        version_column = next(
            c for c in inspector.get_columns("validators") if c["name"] == "version"
        )
        assert version_column["nullable"] is False
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
        inspector = inspect(engine)
        event_columns = {
            c["name"] for c in inspector.get_columns("validator_health_events")
        }
        assert "seq" not in event_columns
        validator_indexes = {idx["name"] for idx in inspector.get_indexes("validators")}
        assert "ix_validators_registered_at" not in validator_indexes
        # earlier control-plane tables remain intact.
        assert "validators" in inspector.get_table_names()
        assert "work_results" in inspector.get_table_names()
    finally:
        engine.dispose()

    migrations.upgrade(ALEMBIC_INI, database_url=async_url, revision="head")
    engine = create_sync_engine(_sync_url(db_path))
    try:
        event_columns = {
            c["name"] for c in inspect(engine).get_columns("validator_health_events")
        }
        assert "seq" in event_columns
    finally:
        engine.dispose()
