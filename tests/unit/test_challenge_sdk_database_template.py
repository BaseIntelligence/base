from __future__ import annotations

from pathlib import Path

from platform_network.template_engine import (
    ChallengeTemplateContext,
    render_challenge_template,
)


def _render_demo_challenge(tmp_path: Path) -> Path:
    out = tmp_path / "challenge"
    render_challenge_template(out, ChallengeTemplateContext.from_slug("demo-challenge"))
    return out


def _read(rendered_root: Path, relative_path: str) -> str:
    return (rendered_root / relative_path).read_text(encoding="utf-8")


def test_generated_sdk_database_helper_uses_async_sqlalchemy_orm_contract(
    tmp_path: Path,
) -> None:
    out = _render_demo_challenge(tmp_path)

    sdk_db = _read(out, "src/demo_challenge/sdk/db.py")
    challenge_db = _read(out, "src/demo_challenge/db.py")
    app_factory = _read(out, "src/demo_challenge/sdk/app_factory.py")
    app_entrypoint = _read(out, "src/demo_challenge/app.py")

    assert (
        "from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, "
        "create_async_engine"
    ) in sdk_db
    assert "from sqlalchemy.orm import DeclarativeBase" in sdk_db
    assert "class Base(DeclarativeBase):" in sdk_db
    assert "self.engine = create_async_engine(" in sdk_db
    assert (
        'connect_args={"check_same_thread": False} '
        'if database_url.startswith("sqlite") else {}'
    ) in sdk_db
    assert "self._session_factory = async_sessionmaker(" in sdk_db
    assert "expire_on_commit=False" in sdk_db
    assert "autoflush=False" in sdk_db
    assert "await connection.run_sync(Base.metadata.create_all)" in sdk_db
    assert "async def session(self) -> AsyncIterator[AsyncSession]:" in sdk_db
    assert (
        "async def session_dependency(self) -> AsyncIterator[AsyncSession]:"
    ) in sdk_db
    assert "async with self.session() as session:" in sdk_db
    assert "await self.engine.dispose()" in sdk_db

    assert "database = Database(settings.database_url)" in challenge_db
    assert "await database.init()" in app_factory
    assert "await database.close()" in app_factory
    assert "lifespan=lifespan" in app_factory
    assert "from . import models as _models" in app_entrypoint


def test_generated_routes_use_fastapi_async_session_dependency(tmp_path: Path) -> None:
    out = _render_demo_challenge(tmp_path)

    routes = _read(out, "src/demo_challenge/routes.py")

    assert "from fastapi import APIRouter, Depends" in routes
    assert "from sqlalchemy.ext.asyncio import AsyncSession" in routes
    assert "from .db import database" in routes
    assert (
        "DatabaseSession = Annotated[AsyncSession, "
        "Depends(database.session_dependency)]"
    ) in routes
    assert "session: DatabaseSession" in routes
    assert "await session.commit()" in routes
    assert "await session.execute(" in routes


def test_generated_dependencies_and_settings_support_postgres_and_sqlite(
    tmp_path: Path,
) -> None:
    out = _render_demo_challenge(tmp_path)

    pyproject = _read(out, "pyproject.toml")
    sdk_config = _read(out, "src/demo_challenge/sdk/config.py")
    challenge_config = _read(out, "config.example.yaml")
    dockerfile = _read(out, "Dockerfile")
    test_conftest = _read(out, "tests/conftest.py")

    assert '"sqlalchemy[asyncio]>=2.0.32"' in pyproject
    assert '"aiosqlite>=0.20.0"' in pyproject
    assert "asyncpg" in pyproject

    assert "database_url: str" in sdk_config
    assert (
        'database_url: str = "sqlite+aiosqlite:////data/challenge.sqlite3"'
    ) in sdk_config
    assert (
        'database_url: "sqlite+aiosqlite:////data/challenge.sqlite3"'
    ) in challenge_config
    assert (
        "CHALLENGE_DATABASE_URL=sqlite+aiosqlite:////data/challenge.sqlite3"
    ) in dockerfile
    assert (
        'os.environ.setdefault("CHALLENGE_DATABASE_URL", '
        'f"sqlite+aiosqlite:///{_TEST_DB}")'
    ) in test_conftest


def test_generated_database_template_avoids_alternate_orms(tmp_path: Path) -> None:
    out = _render_demo_challenge(tmp_path)

    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.suffix in {"", ".py", ".toml", ".yaml"}
    )

    forbidden_orm_terms = (
        "SQLModel",
        "sqlmodel",
        "Tortoise",
        "tortoise",
        "Django",
        "django",
        "Peewee",
        "peewee",
    )
    for forbidden in forbidden_orm_terms:
        assert forbidden not in generated_text
