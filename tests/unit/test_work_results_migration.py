"""Migration + model tests for the ``work_results`` table on SQLite.

``alembic upgrade head`` from an empty DB creates ``work_results`` with its key
columns/indices; the history has a single head; the migration matches the ORM
models (empty ``compare_metadata`` diff); and a downgrade/re-upgrade round-trips
cleanly while leaving the earlier control-plane tables intact.
"""

from __future__ import annotations

from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine as create_sync_engine
from sqlalchemy import inspect

from base.db import Base, WorkResult, migrations

ROOT_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI = ROOT_DIR / "alembic.ini"


def _async_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


def _sync_url(db_path: Path) -> str:
    return f"sqlite:///{db_path}"


def _table_names(db_path: Path) -> set[str]:
    engine = create_sync_engine(_sync_url(db_path))
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_single_head() -> None:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(ROOT_DIR / "alembic"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1


def test_work_results_model_has_documented_columns() -> None:
    columns = set(Base.metadata.tables["work_results"].c.keys())
    assert {
        "id",
        "assignment_id",
        "challenge_slug",
        "work_unit_id",
        "submission_ref",
        "validator_hotkey",
        "success",
        "payload",
        "created_at",
    } <= columns
    assert WorkResult.__tablename__ == "work_results"


def test_upgrade_from_empty_creates_work_results(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.sqlite3"
    migrations.upgrade(ALEMBIC_INI, database_url=_async_url(db_path), revision="head")

    assert "work_results" in _table_names(db_path)

    engine = create_sync_engine(_sync_url(db_path))
    try:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("work_results")}
        assert {
            "id",
            "assignment_id",
            "challenge_slug",
            "work_unit_id",
            "submission_ref",
            "validator_hotkey",
            "success",
            "payload",
            "created_at",
        } <= columns

        index_names = {idx["name"] for idx in inspector.get_indexes("work_results")}
        assert {
            "ix_work_results_assignment_id",
            "ix_work_results_challenge_slug",
            "ix_work_results_validator_hotkey",
        } <= index_names
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


def test_downgrade_removes_work_results_and_reupgrade_recreates(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "roundtrip.sqlite3"
    async_url = _async_url(db_path)

    migrations.upgrade(ALEMBIC_INI, database_url=async_url, revision="head")
    assert "work_results" in _table_names(db_path)

    migrations.downgrade(
        ALEMBIC_INI,
        database_url=async_url,
        revision="0005_create_work_assignments",
    )
    tables_after_downgrade = _table_names(db_path)
    assert "work_results" not in tables_after_downgrade
    assert "work_assignments" in tables_after_downgrade
    assert "validators" in tables_after_downgrade

    migrations.upgrade(ALEMBIC_INI, database_url=async_url, revision="head")
    assert "work_results" in _table_names(db_path)
