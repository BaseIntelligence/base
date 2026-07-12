"""Durable master aggregation and final-vector provenance.

The master is the sole canonical aggregation authority. It:

* opens an epoch with the expected active challenge set and emission shares;
* seals only from durable selected ``raw_weight_snapshots``;
* withholds an epoch when any expected active source is missing or invalid;
* persists one immutable final vector (never recomputed on read).

Pull-on-read of challenge ``get_weights`` is not part of the sealed path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from base.challenge_sdk.roles import Capability, Role, role_contract
from base.db.models import (
    AggregationEpoch,
    AggregationEpochStatus,
    FinalWeightVector,
    RawWeightSnapshot,
)
from base.db.session import session_scope
from base.master.aggregator import (
    CHAIN_U16_MAX,
    ZERO_MINER_BURN_UID,
    aggregate_challenge_weights,
)
from base.schemas.weights import (
    MASTER_WEIGHTS_FRESHNESS_SECONDS,
    ChallengeWeightsResult,
    MasterWeightsResponse,
    SourceOutcome,
    SourceRef,
)

logger = logging.getLogger(__name__)

VECTOR_PROTOCOL_VERSION = "1.0"
EMISSION_POLICY_VERSION = "emission-shares.absolute.v1"
BURN_POLICY_VERSION = "burn-uid0.v1"
MAPPING_POLICY_VERSION = "hotkey-to-uid.v1"
SOURCE_OUTCOME_POLICY_VERSION = "source-outcome.v1"
DEFAULT_EPOCH_DEADLINE_SECONDS = 300

REASON_MISSING = "missing"
REASON_UNAVAILABLE = "unavailable"
REASON_EMPTY = "empty"
REASON_ACCEPTED = "accepted"
REASON_ZERO_CONTRIBUTION = "zero_contribution"
REASON_SEALED = "sealed"
REASON_WITHHELD = "withheld"


class AggregationError(RuntimeError):
    """Base aggregation failure."""


class EpochWithheldError(AggregationError):
    """Active sources incomplete; epoch is withheld (no new vector)."""

    def __init__(self, epoch: int, *, reason: str, outcomes: list[dict[str, Any]]):
        super().__init__(f"epoch {epoch} withheld: {reason}")
        self.epoch = epoch
        self.reason = reason
        self.outcomes = outcomes


class EmissionPolicyError(AggregationError):
    """Emission shares violate the absolute (sum <= 1) policy."""


class VectorNotFoundError(LookupError):
    """No persisted vector for the requested identity."""


@dataclass(frozen=True)
class EmisionShareAssignment:
    """Absolute emission share (fraction of total emission, 0..1) per challenge."""

    shares: dict[str, float]
    policy_version: str = EMISSION_POLICY_VERSION


def validate_emission_shares(shares: Mapping[str, float]) -> dict[str, float]:
    """Validate absolute finite non-negative shares summing to at most 1.0."""

    cleaned: dict[str, float] = {}
    for slug, raw in shares.items():
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise EmissionPolicyError(
                f"emission share for {slug!r} must be finite and non-negative"
            )
        cleaned[str(slug)] = value
    total = sum(cleaned.values())
    if total > 1.0 + 1e-12:
        raise EmissionPolicyError(
            f"emission shares sum to {total:.6f} which exceeds 1.0; "
            "over-allocation is rejected (no relative renormalization)"
        )
    return cleaned


def fractions_from_percent(
    emission_percent_by_slug: Mapping[str, float],
) -> dict[str, float]:
    """Convert registry percent shares (0..100) into absolute fractions (0..1)."""

    fractions = {
        str(slug): max(float(value), 0.0) / 100.0
        for slug, value in emission_percent_by_slug.items()
    }
    return validate_emission_shares(fractions)


def compute_vector_digest(
    *,
    protocol_version: str,
    epoch: int,
    revision: int,
    netuid: int,
    chain_endpoint: str,
    uids: Sequence[int],
    weights: Sequence[float],
    emission_policy_version: str,
    emission_shares: Mapping[str, float],
    burn_policy_version: str,
    mapping_policy_version: str,
    source_snapshot_ids: Sequence[str],
    source_snapshot_digests: Sequence[str],
    metagraph_hash: str | None,
) -> tuple[str, str, str]:
    """Return (digest_hex, canonical_json, chain_domain_json)."""

    chain_domain = {
        "netuid": int(netuid),
        "uids": [int(uid) for uid in uids],
        "weights": [float(weight) for weight in weights],
    }
    chain_domain_bytes = json.dumps(
        chain_domain, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    body = {
        "protocol_version": protocol_version,
        "epoch": int(epoch),
        "revision": int(revision),
        "netuid": int(netuid),
        "chain_endpoint": chain_endpoint,
        "uids": chain_domain["uids"],
        "weights": chain_domain["weights"],
        "emission_policy_version": emission_policy_version,
        "emission_shares": {
            str(slug): float(share) for slug, share in sorted(emission_shares.items())
        },
        "burn_policy_version": burn_policy_version,
        "mapping_policy_version": mapping_policy_version,
        "source_snapshot_ids": list(source_snapshot_ids),
        "source_snapshot_digests": list(source_snapshot_digests),
        "metagraph_hash": metagraph_hash,
        "chain_domain": chain_domain,
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest, canonical, chain_domain_bytes


def snapshot_to_challenge_result(
    snapshot: RawWeightSnapshot,
    *,
    emission_percent: float,
) -> ChallengeWeightsResult:
    """Project a durable raw snapshot into the aggregator input shape."""

    raw_weights = dict(snapshot.weights or {})
    weights = {str(hotkey): float(value) for hotkey, value in raw_weights.items()}
    return ChallengeWeightsResult(
        slug=str(snapshot.challenge_slug),
        emission_percent=float(emission_percent),
        weights=weights,
        ok=True,
        error=None,
    )


def _metagraph_hash(mapping: Mapping[str, int]) -> str:
    payload = json.dumps(
        {str(hotkey): int(uid) for hotkey, uid in sorted(mapping.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AggregationService:
    """Open/seal/withhold epochs and publish immutable final vectors."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
        freshness_seconds: int = MASTER_WEIGHTS_FRESHNESS_SECONDS,
        min_allowed_weights: int = 1,
        max_weight_limit: int = CHAIN_U16_MAX,
    ) -> None:
        self._session_factory = session_factory
        self._now_fn = now_fn
        self.freshness_seconds = freshness_seconds
        self.min_allowed_weights = min_allowed_weights
        self.max_weight_limit = max_weight_limit

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_AGGREGATION)
    async def open_epoch(
        self,
        epoch: int,
        *,
        expected_challenges: Sequence[str],
        emission_shares: Mapping[str, float],
        deadline_at: datetime | None = None,
    ) -> AggregationEpoch:
        """Open (or return existing open) epoch with barrier expectation."""

        shares = validate_emission_shares(emission_shares)
        expected = sorted({str(slug) for slug in expected_challenges})
        now = self._now_fn()
        deadline = deadline_at or (
            now + timedelta(seconds=DEFAULT_EPOCH_DEADLINE_SECONDS)
        )
        async with session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(AggregationEpoch).where(AggregationEpoch.epoch == int(epoch))
                )
            ).scalar_one_or_none()
            if row is not None:
                if row.status != AggregationEpochStatus.OPEN:
                    return row
                # Refresh open-epoch barrier metadata only while still open.
                row.expected_challenges = list(expected)
                row.emission_shares = dict(shares)
                row.emission_policy_version = EMISSION_POLICY_VERSION
                row.source_outcome_policy_version = SOURCE_OUTCOME_POLICY_VERSION
                row.burn_policy_version = BURN_POLICY_VERSION
                row.mapping_policy_version = MAPPING_POLICY_VERSION
                if row.deadline_at is None:
                    row.deadline_at = deadline
                await session.flush()
                await session.refresh(row)
                return row
            row = AggregationEpoch(
                id=uuid.uuid4(),
                epoch=int(epoch),
                status=AggregationEpochStatus.OPEN,
                deadline_at=deadline,
                expected_challenges=list(expected),
                emission_policy_version=EMISSION_POLICY_VERSION,
                emission_shares=dict(shares),
                source_outcome_policy_version=SOURCE_OUTCOME_POLICY_VERSION,
                burn_policy_version=BURN_POLICY_VERSION,
                mapping_policy_version=MAPPING_POLICY_VERSION,
                source_outcomes=[],
            )
            session.add(row)
            await session.flush()
            await session.refresh(row)
            return row

    async def list_selected_snapshots(
        self, epoch: int, *, challenge_slugs: Sequence[str] | None = None
    ) -> list[RawWeightSnapshot]:
        async with session_scope(self._session_factory) as session:
            stmt = select(RawWeightSnapshot).where(
                RawWeightSnapshot.epoch == int(epoch),
                RawWeightSnapshot.is_selected_source.is_(True),
            )
            if challenge_slugs is not None:
                slugs = list(challenge_slugs)
                if not slugs:
                    return []
                stmt = stmt.where(RawWeightSnapshot.challenge_slug.in_(slugs))
            rows = (
                (
                    await session.execute(
                        stmt.order_by(
                            RawWeightSnapshot.challenge_slug.asc(),
                            RawWeightSnapshot.revision.desc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            return list(rows)

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_AGGREGATION)
    async def seal_epoch(
        self,
        epoch: int,
        *,
        hotkey_to_uid: Mapping[str, int],
        netuid: int,
        chain_endpoint: str = "",
        metagraph_block: int | None = None,
        metagraph_updated_at: datetime | None = None,
        emission_percent_by_slug: Mapping[str, float] | None = None,
    ) -> FinalWeightVector:
        """Seal epoch from durable selected raw snapshots and publish one vector.

        Missing expected active sources withhold the epoch without publishing.
        """

        now = self._now_fn()
        async with session_scope(self._session_factory) as session:
            epoch_row = (
                await session.execute(
                    select(AggregationEpoch).where(AggregationEpoch.epoch == int(epoch))
                )
            ).scalar_one_or_none()
            if epoch_row is None:
                raise AggregationError(f"epoch {epoch} is not open")
            if epoch_row.status == AggregationEpochStatus.SEALED:
                if epoch_row.vector_id is None:
                    raise AggregationError(
                        f"epoch {epoch} is sealed without a vector identity"
                    )
                existing = (
                    await session.execute(
                        select(FinalWeightVector).where(
                            FinalWeightVector.id == epoch_row.vector_id
                        )
                    )
                ).scalar_one_or_none()
                if existing is None:
                    raise AggregationError(
                        f"epoch {epoch} sealed vector {epoch_row.vector_id} missing"
                    )
                return existing
            if epoch_row.status == AggregationEpochStatus.WITHHELD:
                raise EpochWithheldError(
                    int(epoch),
                    reason=epoch_row.outcome_reason or REASON_WITHHELD,
                    outcomes=list(epoch_row.source_outcomes or []),
                )
            if epoch_row.status != AggregationEpochStatus.OPEN:
                raise AggregationError(
                    f"epoch {epoch} has unexpected status {epoch_row.status!r}"
                )

            expected = [str(s) for s in (epoch_row.expected_challenges or [])]
            shares = dict(epoch_row.emission_shares or {})
            if emission_percent_by_slug is not None:
                # Convert remaining percent map for absolute shares if open row
                # did not capture fractions yet.
                shares = fractions_from_percent(emission_percent_by_slug)

            selected = (
                (
                    await session.execute(
                        select(RawWeightSnapshot)
                        .where(
                            RawWeightSnapshot.epoch == int(epoch),
                            RawWeightSnapshot.is_selected_source.is_(True),
                        )
                        .order_by(
                            RawWeightSnapshot.challenge_slug.asc(),
                            RawWeightSnapshot.revision.desc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            selected_by_slug: dict[str, RawWeightSnapshot] = {}
            for snap in selected:
                slug = str(snap.challenge_slug)
                # Highest revision first due to order; keep first only.
                if slug not in selected_by_slug:
                    selected_by_slug[slug] = snap

            outcomes: list[dict[str, Any]] = []
            missing: list[str] = []
            challenge_results: list[ChallengeWeightsResult] = []
            source_ids: list[str] = []
            source_digests: list[str] = []

            active_set = expected or sorted(selected_by_slug.keys())
            for slug in active_set:
                snap = selected_by_slug.get(slug)
                if snap is None:
                    missing.append(slug)
                    outcomes.append(
                        {
                            "challenge_slug": slug,
                            "outcome": REASON_MISSING,
                            "reason_code": REASON_MISSING,
                            "snapshot_id": None,
                            "payload_digest": None,
                            "revision": None,
                        }
                    )
                    continue
                weights = dict(snap.weights or {})
                positive = {
                    str(hotkey): float(value)
                    for hotkey, value in weights.items()
                    if float(value) > 0.0
                }
                share_fraction = float(shares.get(slug, 0.0))
                emission_percent = share_fraction * 100.0
                if not weights:
                    # Between ingress (empty rejected) and seal, treat missing
                    # map as invalid source.
                    missing.append(slug)
                    outcomes.append(
                        {
                            "challenge_slug": slug,
                            "outcome": REASON_EMPTY,
                            "reason_code": REASON_EMPTY,
                            "snapshot_id": str(snap.id),
                            "payload_digest": snap.payload_digest,
                            "revision": int(snap.revision),
                        }
                    )
                    continue
                outcome_code = (
                    REASON_ZERO_CONTRIBUTION if not positive else REASON_ACCEPTED
                )
                outcomes.append(
                    {
                        "challenge_slug": slug,
                        "outcome": outcome_code,
                        "reason_code": outcome_code,
                        "snapshot_id": str(snap.id),
                        "payload_digest": snap.payload_digest,
                        "revision": int(snap.revision),
                    }
                )
                challenge_results.append(
                    snapshot_to_challenge_result(
                        snap, emission_percent=emission_percent
                    )
                )
                source_ids.append(str(snap.id))
                source_digests.append(str(snap.payload_digest))

            if missing:
                epoch_row.status = AggregationEpochStatus.WITHHELD
                epoch_row.sealed_at = now
                epoch_row.source_outcomes = outcomes
                epoch_row.outcome_reason = (
                    f"{REASON_WITHHELD}:{','.join(sorted(missing))}"
                )
                epoch_row.emission_shares = dict(shares)
                epoch_row.emission_policy_version = EMISSION_POLICY_VERSION
                epoch_row.source_outcome_policy_version = SOURCE_OUTCOME_POLICY_VERSION
                await session.flush()
                raise EpochWithheldError(
                    int(epoch),
                    reason=epoch_row.outcome_reason or REASON_WITHHELD,
                    outcomes=outcomes,
                )

            mapping = {str(hotkey): int(uid) for hotkey, uid in hotkey_to_uid.items()}
            final = aggregate_challenge_weights(
                challenge_results,
                mapping,
                min_allowed_weights=self.min_allowed_weights,
                max_weight_limit=self.max_weight_limit,
            )
            # Absolute shares path already handled in aggregate_challenge_weights
            # via emission_percent on ChallengeWeightsResult; contract forbids
            # relative renormalization of master shares on seal.

            mg_hash = _metagraph_hash(mapping)
            digest, canonical, chain_domain = compute_vector_digest(
                protocol_version=VECTOR_PROTOCOL_VERSION,
                epoch=int(epoch),
                revision=1,
                netuid=int(netuid),
                chain_endpoint=str(chain_endpoint or ""),
                uids=final.uids,
                weights=final.weights,
                emission_policy_version=EMISSION_POLICY_VERSION,
                emission_shares=shares,
                burn_policy_version=BURN_POLICY_VERSION,
                mapping_policy_version=MAPPING_POLICY_VERSION,
                source_snapshot_ids=source_ids,
                source_snapshot_digests=source_digests,
                metagraph_hash=mg_hash,
            )
            expires_at = now + timedelta(seconds=self.freshness_seconds)
            vector = FinalWeightVector(
                id=uuid.uuid4(),
                epoch=int(epoch),
                revision=1,
                protocol_version=VECTOR_PROTOCOL_VERSION,
                netuid=int(netuid),
                chain_endpoint=str(chain_endpoint or ""),
                vector_digest=digest,
                uids=list(final.uids),
                weights=list(final.weights),
                hotkey_weights=dict(final.hotkey_weights),
                chain_domain_bytes=chain_domain,
                canonical_payload=canonical,
                source_snapshot_ids=source_ids,
                source_snapshot_digests=source_digests,
                source_outcomes=outcomes,
                emission_policy_version=EMISSION_POLICY_VERSION,
                emission_shares=dict(shares),
                burn_policy_version=BURN_POLICY_VERSION,
                mapping_policy_version=MAPPING_POLICY_VERSION,
                metagraph_block=metagraph_block,
                metagraph_hash=mg_hash,
                metagraph_identity={
                    "hash": mg_hash,
                    "block": metagraph_block,
                    "uid_count": len(mapping),
                    "burn_uid": ZERO_MINER_BURN_UID,
                },
                hotkey_to_uid=mapping,
                computed_at=now,
                expires_at=expires_at,
                metagraph_updated_at=metagraph_updated_at or now,
            )
            session.add(vector)
            await session.flush()

            epoch_row.status = AggregationEpochStatus.SEALED
            epoch_row.sealed_at = now
            epoch_row.vector_id = vector.id
            epoch_row.source_outcomes = outcomes
            epoch_row.outcome_reason = REASON_SEALED
            epoch_row.emission_shares = dict(shares)
            epoch_row.emission_policy_version = EMISSION_POLICY_VERSION
            epoch_row.source_outcome_policy_version = SOURCE_OUTCOME_POLICY_VERSION
            epoch_row.burn_policy_version = BURN_POLICY_VERSION
            epoch_row.mapping_policy_version = MAPPING_POLICY_VERSION
            await session.flush()
            await session.refresh(vector)
            logger.info(
                "sealed aggregation epoch",
                extra={
                    "epoch": int(epoch),
                    "vector_id": str(vector.id),
                    "vector_digest": digest[:16],
                    "sources": len(source_ids),
                },
            )
            return vector

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_VECTOR_READ)
    async def get_latest_vector(self) -> FinalWeightVector | None:
        """Load the newest sealed vector from durable storage (no recompute)."""

        async with session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(FinalWeightVector)
                    .order_by(
                        FinalWeightVector.epoch.desc(),
                        FinalWeightVector.created_at.desc(),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
            return row

    @role_contract(role=Role.MASTER, capability=Capability.MASTER_VECTOR_READ)
    async def get_vector_by_id(self, vector_id: str | uuid.UUID) -> FinalWeightVector:
        try:
            vid = uuid.UUID(str(vector_id))
        except ValueError as exc:
            raise VectorNotFoundError(vector_id) from exc
        async with session_scope(self._session_factory) as session:
            row = (
                await session.execute(
                    select(FinalWeightVector).where(FinalWeightVector.id == vid)
                )
            ).scalar_one_or_none()
            if row is None:
                raise VectorNotFoundError(str(vector_id))
            return row

    def vector_to_response(self, vector: FinalWeightVector) -> MasterWeightsResponse:
        """Project durable vector row into the validator API response shape."""

        def _aware(value: datetime | None) -> datetime:
            if value is None:
                return self._now_fn()
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value

        source_challenges = [
            ChallengeWeightsResult(
                slug=str(outcome.get("challenge_slug", "")),
                emission_percent=float(
                    (vector.emission_shares or {}).get(
                        str(outcome.get("challenge_slug", "")), 0.0
                    )
                )
                * 100.0,
                weights={},
                ok=str(outcome.get("outcome"))
                in {REASON_ACCEPTED, REASON_ZERO_CONTRIBUTION},
                error=(
                    None
                    if str(outcome.get("outcome"))
                    in {REASON_ACCEPTED, REASON_ZERO_CONTRIBUTION}
                    else str(outcome.get("reason_code") or outcome.get("outcome"))
                ),
            )
            for outcome in (vector.source_outcomes or [])
        ]
        source_snapshots = [
            SourceRef(
                challenge_slug=_slug_for_source(vector, idx),
                snapshot_id=str(sid),
                payload_digest=str(
                    (vector.source_snapshot_digests or [""])[idx]
                    if idx < len(vector.source_snapshot_digests or [])
                    else ""
                ),
                outcome=str(
                    ((vector.source_outcomes or [{}])[idx] or {}).get("outcome")
                    or REASON_ACCEPTED
                ),
            )
            for idx, sid in enumerate(vector.source_snapshot_ids or [])
        ]
        outcomes = [
            SourceOutcome(
                challenge_slug=str(item.get("challenge_slug", "")),
                outcome=str(item.get("outcome", "")),
                reason_code=str(item.get("reason_code") or item.get("outcome") or ""),
                snapshot_id=(
                    str(item["snapshot_id"])
                    if item.get("snapshot_id") is not None
                    else None
                ),
                payload_digest=(
                    str(item["payload_digest"])
                    if item.get("payload_digest") is not None
                    else None
                ),
                revision=(
                    int(item["revision"]) if item.get("revision") is not None else None
                ),
            )
            for item in (vector.source_outcomes or [])
        ]
        return MasterWeightsResponse(
            protocol_version=str(vector.protocol_version),
            vector_id=str(vector.id),
            vector_digest=str(vector.vector_digest),
            epoch=int(vector.epoch),
            revision=int(vector.revision),
            netuid=int(vector.netuid),
            chain_endpoint=str(vector.chain_endpoint or ""),
            uids=[int(uid) for uid in (vector.uids or [])],
            weights=[float(weight) for weight in (vector.weights or [])],
            hotkey_weights={
                str(hotkey): float(value)
                for hotkey, value in (vector.hotkey_weights or {}).items()
            },
            chain_domain_bytes=str(vector.chain_domain_bytes),
            computed_at=_aware(vector.computed_at),
            expires_at=_aware(vector.expires_at),
            source_challenges=source_challenges,
            source_snapshots=source_snapshots,
            source_outcomes=outcomes,
            emission_policy_version=str(vector.emission_policy_version),
            emission_shares={
                str(slug): float(share)
                for slug, share in (vector.emission_shares or {}).items()
            },
            burn_policy_version=str(vector.burn_policy_version),
            mapping_policy_version=str(vector.mapping_policy_version),
            metagraph_identity=dict(vector.metagraph_identity or {}),
            metagraph_hash=vector.metagraph_hash,
            metagraph_block=vector.metagraph_block,
            burn_outcome=(
                ZERO_MINER_BURN_UID in {int(uid) for uid in (vector.uids or [])}
            ),
            metagraph_updated_at=_aware(
                vector.metagraph_updated_at or vector.computed_at
            ),
        )


def _slug_for_source(vector: FinalWeightVector, idx: int) -> str:
    outcomes = vector.source_outcomes or []
    if idx < len(outcomes) and isinstance(outcomes[idx], dict):
        slug = outcomes[idx].get("challenge_slug")
        if slug:
            return str(slug)
    return ""


__all__ = [
    "AggregationError",
    "AggregationService",
    "BURN_POLICY_VERSION",
    "EMISSION_POLICY_VERSION",
    "EpochWithheldError",
    "EmissionPolicyError",
    "MAPPING_POLICY_VERSION",
    "SOURCE_OUTCOME_POLICY_VERSION",
    "VECTOR_PROTOCOL_VERSION",
    "VectorNotFoundError",
    "compute_vector_digest",
    "fractions_from_percent",
    "snapshot_to_challenge_result",
    "validate_emission_shares",
]
