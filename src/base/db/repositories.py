"""Async repositories for the base master database."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from base.db.models import Challenge, ChallengeHealthEvent, ChallengeStatus


class ChallengeRepository:
    """Repository for challenge registry persistence operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository with an async SQLAlchemy session."""

        self._session = session

    async def add(self, challenge: Challenge) -> Challenge:
        """Add a challenge to the current unit of work."""

        self._session.add(challenge)
        await self._session.flush()
        return challenge

    async def get(self, challenge_id: uuid.UUID) -> Challenge | None:
        """Return a challenge by ID with all registry relationships loaded."""

        result = await self._session.execute(
            select(Challenge)
            .where(Challenge.id == challenge_id)
            .options(
                selectinload(Challenge.image),
                selectinload(Challenge.auth),
                selectinload(Challenge.resources),
                selectinload(Challenge.volumes),
                selectinload(Challenge.secrets),
                selectinload(Challenge.env),
                selectinload(Challenge.capabilities),
                selectinload(Challenge.routes),
            )
        )
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> Challenge | None:
        """Return a challenge by slug with all registry relationships loaded."""

        result = await self._session.execute(
            select(Challenge)
            .where(Challenge.slug == slug)
            .options(
                selectinload(Challenge.image),
                selectinload(Challenge.auth),
                selectinload(Challenge.resources),
                selectinload(Challenge.volumes),
                selectinload(Challenge.secrets),
                selectinload(Challenge.env),
                selectinload(Challenge.capabilities),
                selectinload(Challenge.routes),
            )
        )
        return result.scalar_one_or_none()

    async def list(
        self, *, status: ChallengeStatus | None = None
    ) -> Sequence[Challenge]:
        """List challenges, optionally filtered by lifecycle status."""

        query = select(Challenge).order_by(Challenge.slug)
        if status is not None:
            query = query.where(Challenge.status == status)

        result = await self._session.execute(
            query.options(
                selectinload(Challenge.image),
                selectinload(Challenge.routes),
                selectinload(Challenge.capabilities),
            )
        )
        return result.scalars().all()

    async def list_active(self) -> Sequence[Challenge]:
        """List active challenges for registry and aggregation flows."""

        return await self.list(status=ChallengeStatus.ACTIVE)

    async def delete(self, challenge: Challenge) -> None:
        """Delete a challenge from the current unit of work."""

        await self._session.delete(challenge)
        await self._session.flush()

    async def record_health_event(
        self, event: ChallengeHealthEvent
    ) -> ChallengeHealthEvent:
        """Persist a health observation for a challenge."""

        self._session.add(event)
        await self._session.flush()
        return event
