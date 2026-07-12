"""Master weight service: durable aggregation authority and vector serving.

Sealed epochs read only from durable ``raw_weight_snapshots``. Challenge
GET ``/internal/v1/get_weights`` pull is not used for sealed aggregation.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.bittensor.metagraph_cache import MetagraphCache
from base.challenge_sdk.roles import Capability, Role, role_contract
from base.master.aggregation import (
    AggregationService,
    EpochWithheldError,
    VectorNotFoundError,
    fractions_from_percent,
)
from base.master.aggregator import aggregate_challenge_weights
from base.master.challenge_client import ChallengeClient
from base.master.registry import record_to_registry_view
from base.schemas.challenge import RegistryChallenge
from base.schemas.weights import (
    MASTER_WEIGHTS_FRESHNESS_SECONDS,
    ChallengeWeightsResult,
    FinalWeights,
    MasterWeightsResponse,
)

logger = logging.getLogger(__name__)


async def _resolve(value):  # type: ignore[no-untyped-def]
    if inspect.isawaitable(value):
        return await value
    return value


async def active_challenge_inputs(
    registry,  # type: ignore[no-untyped-def]
) -> tuple[list[RegistryChallenge], dict[str, str]]:
    records = await _resolve(registry.list(active_only=True))
    challenges = [record_to_registry_view(record) for record in records]
    tokens = {
        record.slug: await _resolve(registry.get_token(record.slug))
        for record in records
    }
    return challenges, tokens


def _metagraph_updated_at(
    metagraph_cache: MetagraphCache, fallback: datetime
) -> datetime:
    updated_at = float(getattr(metagraph_cache, "_updated_at", 0.0) or 0.0)
    if updated_at <= 0:
        return fallback
    return datetime.fromtimestamp(updated_at, UTC)


class MasterWeightService:
    """Master aggregation + durable vector serving.

    When ``session_factory`` is provided, sealed-epoch aggregation and
    ``/v1/weights/latest`` use durable storage only. The optional challenge
    client remains available for diagnostics/tests of the legacy pull path, but
    is never used to seal or serve a canonical sealed vector.
    """

    def __init__(
        self,
        *,
        metagraph_cache: MetagraphCache,
        challenge_client: ChallengeClient | None = None,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        aggregation_service: AggregationService | None = None,
        freshness_seconds: int = MASTER_WEIGHTS_FRESHNESS_SECONDS,
    ) -> None:
        self.metagraph_cache = metagraph_cache
        self.challenge_client = challenge_client or ChallengeClient()
        self._session_factory = session_factory
        if aggregation_service is not None:
            self.aggregation: AggregationService | None = aggregation_service
        elif session_factory is not None:
            self.aggregation = AggregationService(
                session_factory,
                freshness_seconds=freshness_seconds,
            )
        else:
            self.aggregation = None
        self.freshness_seconds = freshness_seconds

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_RAW_WEIGHT_INGRESS)
    async def collect_weights(
        self, challenges: list[RegistryChallenge], tokens: dict[str, str]
    ) -> list[ChallengeWeightsResult]:
        """Legacy pull collector — not used for sealed-epoch aggregation.

        Retained for unit tests and transitional diagnostics. Can be invoked
        only outside the durable seal path.
        """

        results: list[ChallengeWeightsResult] = []
        for challenge in challenges:
            token = tokens.get(challenge.slug)
            if not token:
                raise RuntimeError(f"challenge {challenge.slug!r} is missing a token")
            result = await self.challenge_client.get_weights(
                slug=challenge.slug,
                base_url=challenge.internal_base_url,
                token=token,
                emission_percent=float(challenge.emission_percent),
            )
            if not result.ok:
                raise RuntimeError(
                    f"challenge {challenge.slug!r} failed to provide weights: "
                    f"{result.error or 'unknown error'}"
                )
            results.append(result)
        return results

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_AGGREGATION)
    async def collect_durable_weights(
        self,
        *,
        epoch: int,
        challenges: Sequence[RegistryChallenge],
    ) -> list[ChallengeWeightsResult]:
        """Select durable raw_weight_snapshots only (VAL-CROSS-067)."""

        if self.aggregation is None:
            raise RuntimeError(
                "durable weight collection requires session_factory / "
                "AggregationService"
            )
        slugs = [challenge.slug for challenge in challenges]
        shares = {
            challenge.slug: float(challenge.emission_percent)
            for challenge in challenges
        }
        selected = await self.aggregation.list_selected_snapshots(
            epoch, challenge_slugs=slugs
        )
        by_slug = {str(snap.challenge_slug): snap for snap in selected}
        results: list[ChallengeWeightsResult] = []
        missing: list[str] = []
        for challenge in challenges:
            snap = by_slug.get(challenge.slug)
            if snap is None:
                missing.append(challenge.slug)
                continue
            weights = {
                str(hotkey): float(value)
                for hotkey, value in dict(snap.weights or {}).items()
            }
            results.append(
                ChallengeWeightsResult(
                    slug=challenge.slug,
                    emission_percent=float(shares.get(challenge.slug, 0.0)),
                    weights=weights,
                    ok=True,
                )
            )
        if missing:
            raise EpochWithheldError(
                int(epoch),
                reason=f"withheld:{','.join(sorted(missing))}",
                outcomes=[
                    {
                        "challenge_slug": slug,
                        "outcome": "missing",
                        "reason_code": "missing",
                    }
                    for slug in missing
                ],
            )
        return results

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_AGGREGATION)
    async def compute_weights(
        self, challenges: list[RegistryChallenge], tokens: dict[str, str]
    ) -> tuple[FinalWeights, list[ChallengeWeightsResult]]:
        """Diagnostic aggregate path.

        Prefer :meth:`seal_epoch` for production publication. This path still
        supports in-memory tests that inject a stub challenge client.
        """

        hotkey_to_uid = self.metagraph_cache.get()
        results = await self.collect_weights(challenges, tokens)
        return aggregate_challenge_weights(results, hotkey_to_uid), results

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_AGGREGATION)
    async def seal_epoch(
        self,
        epoch: int,
        challenges: Sequence[RegistryChallenge],
        *,
        netuid: int,
        chain_endpoint: str = "",
        deadline_at: datetime | None = None,
    ) -> MasterWeightsResponse:
        """Open + seal from durable raw snapshots only."""

        if self.aggregation is None:
            raise RuntimeError(
                "seal_epoch requires session_factory / AggregationService"
            )
        expected = [challenge.slug for challenge in challenges]
        emission_percent = {
            challenge.slug: float(challenge.emission_percent)
            for challenge in challenges
        }
        shares = fractions_from_percent(emission_percent)
        await self.aggregation.open_epoch(
            int(epoch),
            expected_challenges=expected,
            emission_shares=shares,
            deadline_at=deadline_at,
        )
        hotkey_to_uid = self.metagraph_cache.get()
        vector = await self.aggregation.seal_epoch(
            int(epoch),
            hotkey_to_uid=hotkey_to_uid,
            netuid=int(netuid),
            chain_endpoint=chain_endpoint,
            metagraph_updated_at=_metagraph_updated_at(
                self.metagraph_cache, self.aggregation._now_fn()
            ),
            emission_percent_by_slug=emission_percent,
        )
        return self.aggregation.vector_to_response(vector)

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_VECTOR_READ)
    async def compute_latest_response(
        self,
        challenges: list[RegistryChallenge],
        tokens: dict[str, str],
        *,
        netuid: int,
        chain_endpoint: str,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> MasterWeightsResponse:
        """Serve the latest sealed vector from durable storage when available.

        If durable storage is configured, this never recomputes or pulls
        challenge get_weights. Without durable storage (unit tests), falls back
        to an in-memory aggregate over injected challenge clients.
        """

        if self.aggregation is not None:
            vector = await self.aggregation.get_latest_vector()
            if vector is None:
                # No sealed vector yet — do not fabricate via pull.
                raise VectorNotFoundError("latest")
            return self.aggregation.vector_to_response(vector)

        # Legacy diagnostic path for unit tests without a session factory.
        computed_at = now_fn()
        final, results = await self.compute_weights(challenges, tokens)
        return MasterWeightsResponse(
            netuid=netuid,
            chain_endpoint=chain_endpoint,
            uids=final.uids,
            weights=final.weights,
            hotkey_weights=final.hotkey_weights,
            computed_at=computed_at,
            expires_at=computed_at + timedelta(seconds=self.freshness_seconds),
            source_challenges=results,
            metagraph_updated_at=_metagraph_updated_at(
                self.metagraph_cache, computed_at
            ),
        )

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_VECTOR_READ)
    async def get_vector_response(self, vector_id: str) -> MasterWeightsResponse:
        if self.aggregation is None:
            raise RuntimeError("vector-by-id requires durable aggregation store")
        vector = await self.aggregation.get_vector_by_id(vector_id)
        return self.aggregation.vector_to_response(vector)

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_AGGREGATION)
    async def run_epoch(
        self,
        challenges: list[RegistryChallenge],
        tokens: dict[str, str],
        *,
        epoch: int | None = None,
        netuid: int | None = None,
        chain_endpoint: str = "",
    ) -> FinalWeights:
        """Run one aggregation epoch.

        When durable storage (``session_factory`` / ``AggregationService``) is
        configured, this **must** seal from durable ``raw_weight_snapshots`` only
        and never fall back to challenge ``get_weights`` (VAL-CROSS-067). Seal
        requires concrete ``epoch`` and ``netuid``.

        Without durable storage, the diagnostic pull aggregator remains available
        for unit/CLI dry-run tests that inject a stub challenge client.
        """

        if self.aggregation is not None:
            if epoch is None or netuid is None:
                raise RuntimeError(
                    "durable seal path requires concrete epoch and netuid; "
                    "refusing get_weights fallback for sealed aggregation "
                    "(VAL-CROSS-067)"
                )
            response = await self.seal_epoch(
                int(epoch),
                challenges,
                netuid=int(netuid),
                chain_endpoint=chain_endpoint,
            )
            logger.info(
                "sealed durable weights",
                extra={
                    "uids": len(response.uids),
                    "epoch": epoch,
                    "vector_id": response.vector_id,
                },
            )
            return FinalWeights(
                uids=list(response.uids),
                weights=list(response.weights),
                hotkey_weights=dict(response.hotkey_weights),
            )

        final, _results = await self.compute_weights(challenges, tokens)
        logger.info(
            "computed weights",
            extra={"uids": len(final.uids), "challenges": len(challenges)},
        )
        return final


__all__ = [
    "MasterWeightService",
    "active_challenge_inputs",
]
