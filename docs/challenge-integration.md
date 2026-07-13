# Challenge Integration Guide

![BASE Banner](../assets/banner.jpg)

## Implement weights

In the generated challenge repository, implement scoring that produces raw
hotkey weights. The supported target path has the challenge **push** an
authenticated raw-weight payload to the master (challenge-scoped credential,
versioned epoch/revision, idempotent, replay-protected). The master normalizes
returned values, so raw scores are acceptable as long as they are finite and
non-negative.

```python
async def get_weights() -> dict[str, float]:
    return {"5F...hotkey": 1.0}
```

Challenges never submit final UID vectors and never open the master control-plane
PostgreSQL.

## Challenge database contract

Generated challenges use the async SQLAlchemy SDK and read their runtime database
URL from `CHALLENGE_DATABASE_URL`. The runtime is SQLite-backed; BASE points that
URL at the SQLite file on the challenge `/data` **Compose volume**:

```text
sqlite+aiosqlite:////data/challenge.sqlite3
```

The same URL is used for local generated runs and the deployed long-lived
Compose challenge service. There is no Postgres server per challenge; each
challenge mounts its own `/data` volume for the SQLite file and artifacts.

Challenges must never receive `BASE_DATABASE_URL`, master database URLs, or any
central control-plane PostgreSQL credentials. The shared control-plane PostgreSQL
is only for master state.

## Async SQLAlchemy usage

Generated templates export a `Base` and `database` helper. Use normal SQLAlchemy
2.x async ORM patterns with `AsyncSession`, `select()`, model registration, and
the FastAPI session dependency.

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

Generated apps call `Base.metadata.create_all` through the async engine during
startup after models are imported, creating missing tables for the current model
set. Challenge Alembic migration automation is not part of this implementation.

## Persistent storage

Challenge services get a named **Compose volume** mounted at `/data`. Use `/data`
for the SQLite database, artifacts, analyzer output, uploaded files, and any local
state that should survive restarts. It is the only persistent store for a
challenge, and BASE retains it by default when the service is removed so state
survives accidental deletion.

## Operator cleanup and purge

A normal stop or deactivation removes the long-lived Compose service but keeps
the `/data` volume for reuse. To intentionally purge a challenge database,
inspect the volume first, then delete only the matching slug volume:

```bash
docker volume ls --filter label=base.challenge.slug=<slug>
# or list volumes owned by the master Compose project and identify the slug volume

docker volume rm <challenge-data-volume>
```

These commands are manual and destructive. Confirm the slug and volume first.
BASE provides no automated destructive purge in this implementation.

## Runtime topology notes

- Challenges join the master project's private `app` network only.
- They never join the master `db` network.
- Evaluation is not performed by short-lived evaluator containers created from
  Base or Prism; Prism verifies/ingests external results when used.
- Production images for challenges installed via registry adoption must use
  digest pins; the master challenge watcher rolls forward only with controlled
  pull + recreate + health/version verify and rolls back on failure.

## Out of scope

No Postgres server per challenge, no Swarm service graph for challenge
lifecycle on the target path, and no automatic backups, high availability,
connection pooling, storage resize workflows, challenge Alembic migration
automation, or automated destructive purge beyond the explicit operator scripts
documented for the master Compose project ([compose.md](compose.md)).

## Build and publish

The generated CI workflow tests the challenge and pushes its Docker image to GHCR
on main/tags. Pin published digests in the master registry for production adopts.
