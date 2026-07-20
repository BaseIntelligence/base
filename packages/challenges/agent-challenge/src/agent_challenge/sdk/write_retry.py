"""Retry helper for transient SQLite writer-lock contention.

On the combined-worker SQLite deployment (multiple replicas sharing one WAL
file), two writers that each hold a SHARED lock and then try to upgrade to
RESERVED collide with an *immediate* ``database is locked`` that ``busy_timeout``
cannot resolve by waiting. This helper reruns a write transaction a few times
with a short jittered backoff so those momentary collisions succeed instead of
surfacing as HTTP 500s.

Inert on PostgreSQL (asyncpg): its ``OperationalError`` messages never contain
the SQLite lock text, so the retry branch is never taken and the write runs
exactly once. No SQLite-only SQL is issued here, so nothing leaks to PostgreSQL.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

#: Total attempts (1 initial + up to 4 retries) before a lock error propagates.
WRITE_LOCK_MAX_ATTEMPTS = 5
#: First backoff window; doubles each retry up to ``WRITE_LOCK_MAX_DELAY_SECONDS``.
WRITE_LOCK_BASE_DELAY_SECONDS = 0.025
WRITE_LOCK_MAX_DELAY_SECONDS = 0.4

#: Lowercased substrings that mark a transient SQLite writer-lock collision.
_LOCK_ERROR_MARKERS = ("database is locked", "database is busy")


def is_sqlite_write_lock_error(exc: OperationalError) -> bool:
    """Return True when ``exc`` is a transient SQLite writer-lock contention error.

    Matches on the DBAPI ``orig`` message so only genuine lock/busy collisions are
    retried; every other ``OperationalError`` (bad SQL, missing table, PostgreSQL
    errors, ...) returns False and is re-raised untouched by the caller.
    """

    orig = getattr(exc, "orig", None)
    message = str(orig if orig is not None else exc).lower()
    return any(marker in message for marker in _LOCK_ERROR_MARKERS)


def _backoff_delay(attempt: int) -> float:
    # Exponential target (base, 2*base, 4*base, ...) capped, with jitter across the
    # upper half of the window so colliding writers do not resynchronize on the
    # same retry instant (thundering herd).
    ceiling = min(WRITE_LOCK_MAX_DELAY_SECONDS, WRITE_LOCK_BASE_DELAY_SECONDS * 2**attempt)
    return random.uniform(ceiling / 2, ceiling)


async def run_write_with_lock_retry[T](
    session: AsyncSession,
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = WRITE_LOCK_MAX_ATTEMPTS,
) -> T:
    """Run ``operation`` then ``session.commit()``, retrying transient lock errors.

    ``operation`` performs the writes for ONE transaction and must be safe to run
    again from a clean session, because a ``session.rollback()`` precedes every
    retry (any uncommitted rows/counters from the failed attempt are discarded, so
    replaying is exact and never double-counts). On an ``OperationalError`` that is
    a SQLite writer-lock collision the session is rolled back, a short jittered
    backoff elapses, and the transaction is retried up to ``max_attempts`` times
    before the error propagates. Non-lock ``OperationalError``s (and every other
    exception) propagate immediately and are never swallowed.
    """

    for attempt in range(max_attempts):
        try:
            result = await operation()
            await session.commit()
            return result
        except OperationalError as exc:
            if not is_sqlite_write_lock_error(exc):
                raise
            await session.rollback()
            if attempt == max_attempts - 1:
                raise
            await asyncio.sleep(_backoff_delay(attempt))
    raise RuntimeError(  # pragma: no cover - max_attempts >= 1 always returns/raises above
        "run_write_with_lock_retry requires max_attempts >= 1"
    )
