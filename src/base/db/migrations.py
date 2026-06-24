"""Programmatic Alembic migration helpers."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config

from alembic import command


def alembic_config(config_path: str | Path, database_url: str | None = None) -> Config:
    """Build an Alembic config with an optional runtime database URL override."""

    config = Config(str(config_path))
    if database_url is not None:
        config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade(
    config_path: str | Path, database_url: str | None = None, revision: str = "head"
) -> None:
    """Upgrade the master database schema to the requested Alembic revision."""

    command.upgrade(alembic_config(config_path, database_url), revision)


def downgrade(
    config_path: str | Path, database_url: str | None = None, revision: str = "-1"
) -> None:
    """Downgrade the master database schema to the requested Alembic revision."""

    command.downgrade(alembic_config(config_path, database_url), revision)
