# Challenge Integration Guide

![Platform Banner](../assets/banner.jpg)

## Implement weights

In the generated challenge repository, implement:

```python
async def get_weights() -> dict[str, float]:
    return {"5F...hotkey": 1.0}
```

The master normalizes returned values, so raw scores are acceptable as long as they are finite and non-negative.

## Challenge database contract

Generated challenges use the async SQLAlchemy SDK and read their runtime database URL from `CHALLENGE_DATABASE_URL`.

The challenge runtime is SQLite-backed. Platform injects `CHALLENGE_DATABASE_URL` pointing at the SQLite file on the challenge `/data` Swarm volume:

```text
sqlite+aiosqlite:////data/challenge.sqlite3
```

The same URL is used for local generated challenge runs and for the deployed Swarm service. There is no Postgres server per challenge; each challenge mounts its own `/data` volume for the SQLite file and artifacts.

Challenges must never receive `PLATFORM_DATABASE_URL`, master database URLs, or any central control-plane PostgreSQL credentials. The shared control-plane PostgreSQL is only for master and validator state.

## Async SQLAlchemy usage

Generated challenge templates export a `Base` and `database` helper. Use normal SQLAlchemy 2.x async ORM patterns with `AsyncSession`, `select()`, model registration, and the FastAPI session dependency.

```python
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    hotkey: Mapped[str] = mapped_column(String(128), index=True)
    score: Mapped[int] = mapped_column(Integer, default=0)


# In generated challenges, import Base and database instead:
# from .core.db import Base, database

router = APIRouter()
DatabaseSession = Annotated[AsyncSession, Depends(database.session_dependency)]


@router.get("/submissions/{hotkey}")
async def list_submissions(hotkey: str, session: DatabaseSession) -> list[int]:
    result = await session.execute(
        select(Submission).where(Submission.hotkey == hotkey)
    )
    return [submission.score for submission in result.scalars()]
```

Generated applications call `Base.metadata.create_all` through the async engine during startup after models are imported. That creates missing tables for the current model set. Challenge Alembic migration automation is not part of this implementation.

## Persistent storage

Challenge services get a `/data` Swarm volume. Use `/data` for the SQLite database, artifacts, analyzer output, uploaded files, and any local state that should survive restarts.

The `/data` Swarm volume is the only persistent store for a challenge. By default Platform retains the `/data` volume when a challenge service is removed. That retention protects challenge state and the SQLite database from accidental deletion.

## Operator cleanup and purge

Normal challenge stop removes the Swarm service but keeps the `/data` volume available for reuse. If an operator intentionally wants to purge a challenge database, inspect the volume first, then delete only the matching slug volume.

```bash
docker volume ls --filter label=platform.challenge.slug=<slug>

docker volume rm <challenge-data-volume>
```

These commands are manual and destructive. Confirm the slug and volume before running them. Platform does not provide automated destructive purge in this implementation.

## Out of scope

This implementation does not provide a Postgres server per challenge, Docker Compose or stack-file Postgres support, automatic backups, restore workflows, high availability, connection pooling, storage resize workflows, challenge Alembic migration automation, or automated destructive purge.

## Build and publish

The generated CI workflow tests the challenge and pushes its Docker image to GHCR on main/tags.
