"""Async SQLAlchemy engine and session helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(
    database_url: str, *, echo: bool = False, pool_pre_ping: bool = True
) -> AsyncEngine:
    """Create an async SQLAlchemy engine for the platform master database.

    Args:
        database_url: Async SQLAlchemy database URL.
        echo: Whether SQLAlchemy should echo SQL statements.
        pool_pre_ping: Whether pooled connections should be pre-pinged.

    Returns:
        Configured async SQLAlchemy engine.
    """

    return create_async_engine(database_url, echo=echo, pool_pre_ping=pool_pre_ping)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to an async engine."""

    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Provide a transactional async session scope.

    The session is committed on success and rolled back on error.
    """

    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
