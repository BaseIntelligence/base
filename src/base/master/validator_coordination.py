"""Validator coordination registration endpoints on the master app.

Implements the hotkey-signed, metagraph-permit-gated ``register`` and
``heartbeat`` routes of the coordination plane (architecture.md sec 4). The
master only records validator liveness here; it never executes work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.db.models import (
    Validator,
    ValidatorHealthEvent,
    ValidatorHealthEventType,
    ValidatorStatus,
)
from base.db.session import session_scope
from base.schemas.validator import (
    ValidatorHeartbeatRequest,
    ValidatorHeartbeatResponse,
    ValidatorRegisterRequest,
    ValidatorRegisterResponse,
    ValidatorView,
)
from base.security.validator_auth import ValidatorIdentity

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60


class ValidatorNotRegisteredError(LookupError):
    """Heartbeat received for a hotkey without a ``validators`` row (HTTP 404)."""


class ValidatorCoordinationService:
    """Persist validator registration and heartbeat liveness transitions."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._session_factory = session_factory
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._now_fn = now_fn

    async def register(
        self,
        *,
        hotkey: str,
        uid: int | None,
        capabilities: list[str],
        version: str | None,
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> Validator:
        """Create or update the validator row and emit lifecycle events.

        First registration appends ``registered`` + ``online`` events; an
        idempotent re-register updates the same row (capabilities, version,
        ``last_heartbeat_at``) and preserves ``registered_at``. Re-registering a
        previously-offline validator records the ``online`` recovery.
        """

        now = self._now_fn()
        async with session_scope(self._session_factory) as session:
            existing = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one_or_none()

            if existing is None:
                validator = Validator(
                    hotkey=hotkey,
                    uid=uid,
                    status=ValidatorStatus.ONLINE,
                    capabilities=list(capabilities),
                    version=version,
                    registered_at=now,
                    last_heartbeat_at=now,
                    last_seen_meta=dict(last_seen_meta or {}),
                )
                session.add(validator)
                self._add_event(
                    session,
                    hotkey,
                    ValidatorHealthEventType.REGISTERED,
                    now,
                )
                self._add_event(session, hotkey, ValidatorHealthEventType.ONLINE, now)
                return validator

            was_offline = existing.status == ValidatorStatus.OFFLINE
            existing.uid = uid
            existing.status = ValidatorStatus.ONLINE
            existing.capabilities = list(capabilities)
            existing.version = version
            existing.last_heartbeat_at = now
            if last_seen_meta is not None:
                existing.last_seen_meta = dict(last_seen_meta)
            if was_offline:
                self._add_event(
                    session,
                    hotkey,
                    ValidatorHealthEventType.ONLINE,
                    now,
                    message="re-registered after offline",
                )
            return existing

    async def heartbeat(
        self,
        *,
        hotkey: str,
        last_seen_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Validator, datetime]:
        """Refresh liveness; flip an offline validator back to online.

        Raises :class:`ValidatorNotRegisteredError` when the hotkey has no
        registered row (the validator must ``register`` first).
        """

        now = self._now_fn()
        async with session_scope(self._session_factory) as session:
            validator = (
                await session.execute(
                    select(Validator).where(Validator.hotkey == hotkey)
                )
            ).scalar_one_or_none()
            if validator is None:
                raise ValidatorNotRegisteredError(hotkey)

            was_offline = validator.status == ValidatorStatus.OFFLINE
            validator.status = ValidatorStatus.ONLINE
            validator.last_heartbeat_at = now
            if last_seen_meta is not None:
                validator.last_seen_meta = dict(last_seen_meta)
            if was_offline:
                self._add_event(
                    session,
                    hotkey,
                    ValidatorHealthEventType.ONLINE,
                    now,
                    message="recovered via heartbeat",
                )
            return validator, now

    @staticmethod
    def _add_event(
        session: AsyncSession,
        hotkey: str,
        event: ValidatorHealthEventType,
        created_at: datetime,
        *,
        message: str | None = None,
    ) -> None:
        session.add(
            ValidatorHealthEvent(
                validator_hotkey=hotkey,
                event=event,
                message=message,
                created_at=created_at,
            )
        )


def validator_to_view(validator: Validator) -> ValidatorView:
    """Convert a persisted validator row to its public view."""

    return ValidatorView(
        hotkey=validator.hotkey,
        uid=validator.uid,
        status=ValidatorStatus(validator.status).value,
        capabilities=list(validator.capabilities),
        version=validator.version,
        registered_at=validator.registered_at,
        last_heartbeat_at=validator.last_heartbeat_at,
        last_seen_meta=dict(validator.last_seen_meta),
    )


def build_validator_coordination_router(
    *,
    service: ValidatorCoordinationService,
    auth_dependency: Callable[..., Any],
) -> APIRouter:
    """Build the validator coordination router (register + heartbeat).

    ``auth_dependency`` is the FastAPI dependency from
    :func:`base.security.validator_auth.build_validator_auth_dependency`; it
    yields a :class:`ValidatorIdentity` for an authenticated, eligible validator.
    """

    router = APIRouter()

    @router.post("/v1/validators/register", response_model=ValidatorRegisterResponse)
    async def register_validator(
        payload: ValidatorRegisterRequest,
        identity: ValidatorIdentity = Depends(auth_dependency),
    ) -> ValidatorRegisterResponse:
        validator = await service.register(
            hotkey=identity.hotkey,
            uid=identity.uid,
            capabilities=payload.capabilities,
            version=payload.version,
            last_seen_meta=payload.last_seen_meta,
        )
        return ValidatorRegisterResponse(
            validator=validator_to_view(validator),
            heartbeat_interval_seconds=service.heartbeat_interval_seconds,
        )

    @router.post("/v1/validators/heartbeat", response_model=ValidatorHeartbeatResponse)
    async def heartbeat_validator(
        payload: ValidatorHeartbeatRequest,
        identity: ValidatorIdentity = Depends(auth_dependency),
    ) -> ValidatorHeartbeatResponse:
        try:
            validator, now = await service.heartbeat(
                hotkey=identity.hotkey,
                last_seen_meta=payload.last_seen_meta,
            )
        except ValidatorNotRegisteredError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="validator not registered",
            ) from exc
        return ValidatorHeartbeatResponse(
            status=ValidatorStatus(validator.status).value,
            now=now,
        )

    return router
