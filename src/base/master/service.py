"""Master weight service: durable aggregation authority and vector serving.

Sealed epochs read only from durable ``raw_weight_snapshots``. Challenge
GET ``/internal/v1/get_weights`` pull is not used for sealed aggregation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.bittensor.metagraph_cache import MetagraphCache
from base.challenge_sdk.roles import Capability, Role, role_contract
from base.db.models import FinalWeightVector
from base.master.aggregation import (
    AggregationService,
    EpochWithheldError,
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
        epoch_interval_seconds: int = 360,
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
        self.freshness_seconds = int(freshness_seconds)
        self.epoch_interval_seconds = int(epoch_interval_seconds or 360)
        # Serialize background sealer ticks + lazy seal-on-GET heal path.
        self._seal_lock = asyncio.Lock()

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

    def _aware(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value

    def _vector_is_expired(self, vector: FinalWeightVector, *, now: datetime) -> bool:
        expires_at = getattr(vector, "expires_at", None)
        if expires_at is None:
            return True
        return self._aware(expires_at) <= self._aware(now)

    def resolve_epoch(
        self,
        *,
        now: datetime | None = None,
        epoch: int | None = None,
        epoch_interval_seconds: int | float | None = None,
    ) -> int:
        """Wall-clock epoch bucket (same identity as CLI master weights)."""

        if epoch is not None:
            return int(epoch)
        raw = (
            epoch_interval_seconds
            if epoch_interval_seconds is not None
            else self.epoch_interval_seconds
        )
        interval = max(1, int(raw or 360))
        clock = now if now is not None else datetime.now(UTC)
        return int(self._aware(clock).timestamp()) // interval

    async def _next_seal_epoch(
        self,
        *,
        now: datetime,
        epoch_interval_seconds: int | float | None = None,
        force_new: bool = False,
    ) -> int:
        """Pick an openable epoch identity for a fresh seal.

        Prefer the wall-clock bucket. When that bucket is already sealed (or
        the latest sealed vector is expired in-bucket), advance past the last
        sealed epoch so ``AggregationService.seal_epoch`` publishes a new TTL.
        """

        candidate = self.resolve_epoch(
            now=now, epoch_interval_seconds=epoch_interval_seconds
        )
        if self.aggregation is None:
            return candidate
        latest = await self.aggregation.get_latest_vector()
        if latest is None:
            return candidate
        last_epoch = int(getattr(latest, "epoch", 0) or 0)
        expired = self._vector_is_expired(latest, now=now)
        if force_new or expired or candidate <= last_epoch:
            return max(candidate, last_epoch + 1)
        return candidate

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_AGGREGATION)
    async def seal_fresh_if_needed(
        self,
        challenges: Sequence[RegistryChallenge],
        tokens: dict[str, str],
        *,
        netuid: int,
        chain_endpoint: str = "",
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        epoch_interval_seconds: int | float | None = None,
        force: bool = False,
    ) -> MasterWeightsResponse:
        """Seal when missing/expired (or always when ``force``), under lock.

        Background sealer uses ``force=True`` each tick so the durable TTL stays
        ahead of serve validation. Lazy GET heal uses ``force=False`` and only
        seals when there is no fresh vector. Zero-miner all-sources-missing seal
        behavior is retained via :meth:`seal_epoch`.
        """

        del tokens  # durable seal path does not pull challenge tokens
        if self.aggregation is None:
            raise RuntimeError(
                "seal_fresh_if_needed requires session_factory / AggregationService"
            )
        async with self._seal_lock:
            now = now_fn()
            latest = await self.aggregation.get_latest_vector()
            if (
                not force
                and latest is not None
                and not self._vector_is_expired(latest, now=now)
            ):
                return self.aggregation.vector_to_response(latest)
            epoch = await self._next_seal_epoch(
                now=now,
                epoch_interval_seconds=epoch_interval_seconds,
                force_new=force and latest is not None,
            )
            logger.info(
                "auto-sealing master weights",
                extra={
                    "epoch": epoch,
                    "force": force,
                    "had_latest": latest is not None,
                    "expired": (
                        self._vector_is_expired(latest, now=now)
                        if latest is not None
                        else True
                    ),
                },
            )
            return await self.seal_epoch(
                int(epoch),
                challenges,
                netuid=int(netuid),
                chain_endpoint=chain_endpoint or "",
            )

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

        When durable storage is configured, expiry or a missing vector triggers
        a lazy seal under lock (safety net for startup race / sealer lag) so
        production never maps pure TTL expiry to HTTP 502. Without durable
        storage (unit tests), falls back to an in-memory aggregate over injected
        challenge clients.
        """

        if self.aggregation is not None:
            # Heal path: reseal under lock when missing or past expires_at.
            # Never surface expiry alone as an unhandled serve failure.
            return await self.seal_fresh_if_needed(
                challenges,
                tokens,
                netuid=netuid,
                chain_endpoint=chain_endpoint,
                now_fn=now_fn,
                force=False,
            )

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
