"""Non-authoritative master-side storage of validator chain-submission observations.

Validators report accepted/rejected/unknown outcomes keyed by their public
identity and the immutable vector identity. The master never treats these as
chain finality and never invokes ``set_weights``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.challenge_sdk.roles import Capability, Role, role_contract
from base.db.models import ValidatorSubmissionObservation
from base.db.session import session_scope
from base.schemas.weights import (
    ValidatorSubmissionObservationRequest,
    ValidatorSubmissionObservationResponse,
)

logger = logging.getLogger(__name__)


class SubmissionObservationError(RuntimeError):
    """Observation validation or conflict failure."""


class SubmissionObservationConflictError(SubmissionObservationError):
    """Same operation identity with a conflicting outcome payload."""


class ValidatorSubmissionObservationService:
    """Persist validator-owned chain outcome observations (non-authoritative)."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    @role_contract(
        role=Role.MASTER, capability=Capability.MASTER_COORDINATION
    )
    async def record(
        self,
        *,
        validator_hotkey: str,
        request: ValidatorSubmissionObservationRequest,
    ) -> ValidatorSubmissionObservationResponse:
        """Idempotently store or reject a conflicting observation."""

        async with session_scope(self._session_factory) as session:
            existing = await session.scalar(
                select(ValidatorSubmissionObservation).where(
                    ValidatorSubmissionObservation.validator_hotkey
                    == validator_hotkey,
                    ValidatorSubmissionObservation.vector_id == request.vector_id,
                    ValidatorSubmissionObservation.vector_digest
                    == request.vector_digest,
                    ValidatorSubmissionObservation.outcome == request.outcome,
                    ValidatorSubmissionObservation.attempt == request.attempt,
                )
            )
            if existing is not None:
                if (
                    int(existing.netuid) != int(request.netuid)
                    or str(existing.chain_endpoint or "")
                    != str(request.chain_endpoint or "")
                    or str(existing.error_code or "") != str(request.error_code or "")
                ):
                    raise SubmissionObservationConflictError(
                        "conflicting observation under the same operation identity"
                    )
                return ValidatorSubmissionObservationResponse(
                    observation_id=str(existing.id),
                    validator_hotkey=existing.validator_hotkey,
                    vector_id=existing.vector_id,
                    vector_digest=existing.vector_digest,
                    outcome=existing.outcome,
                    attempt=int(existing.attempt),
                    created_at=existing.created_at,
                    idempotent=True,
                )

            row = ValidatorSubmissionObservation(
                id=uuid.uuid4(),
                validator_hotkey=validator_hotkey,
                vector_id=request.vector_id,
                vector_digest=request.vector_digest,
                netuid=int(request.netuid),
                chain_endpoint=str(request.chain_endpoint or ""),
                outcome=request.outcome,
                attempt=int(request.attempt),
                error_code=request.error_code,
                observed_at=request.observed_at or datetime.now(UTC),
                created_at=datetime.now(UTC),
            )
            session.add(row)
            await session.flush()
            logger.info(
                "validator submission observation recorded: hotkey=%s vector=%s "
                "outcome=%s attempt=%s (non-authoritative)",
                validator_hotkey,
                request.vector_id,
                request.outcome,
                request.attempt,
            )
            return ValidatorSubmissionObservationResponse(
                observation_id=str(row.id),
                validator_hotkey=row.validator_hotkey,
                vector_id=row.vector_id,
                vector_digest=row.vector_digest,
                outcome=row.outcome,
                attempt=int(row.attempt),
                created_at=row.created_at,
                idempotent=False,
            )

    async def list_for_vector(self, vector_id: str) -> list[dict[str, Any]]:
        async with session_scope(self._session_factory) as session:
            rows = (
                await session.scalars(
                    select(ValidatorSubmissionObservation).where(
                        ValidatorSubmissionObservation.vector_id == vector_id
                    )
                )
            ).all()
            return [
                {
                    "observation_id": str(row.id),
                    "validator_hotkey": row.validator_hotkey,
                    "vector_id": row.vector_id,
                    "vector_digest": row.vector_digest,
                    "outcome": row.outcome,
                    "attempt": int(row.attempt),
                    "netuid": int(row.netuid),
                }
                for row in rows
            ]


__all__ = [
    "SubmissionObservationConflictError",
    "SubmissionObservationError",
    "ValidatorSubmissionObservationService",
]
