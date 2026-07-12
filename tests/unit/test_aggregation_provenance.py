"""Durable aggregation, epoch sealing/withhold, and immutable vector provenance.

Covers VAL-WEIGHT-051/052/053/088/089/090/097/098/099/100 and VAL-CROSS-067/076:
master sole aggregation authority, durable raw-snapshot selection, withhold on
missing sources, immutable vector-by-id, no recompute on read.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from base.challenge_sdk.roles import Role, activate_role
from base.db import Base, RawWeightSnapshot, session_scope
from base.master.aggregation import (
    AggregationService,
    EmissionPolicyError,
    EpochWithheldError,
    VectorNotFoundError,
    compute_vector_digest,
    fractions_from_percent,
    validate_emission_shares,
)
from base.master.service import MasterWeightService
from base.schemas.challenge import ChallengeStatus, RegistryChallenge


class FakeClock:
    def __init__(self) -> None:
        self._now = datetime.now(UTC).replace(microsecond=0)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class StubMetagraphCache:
    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = dict(mapping)
        self._updated_at = 0.0

    def get(self, *, force: bool = False) -> dict[str, int]:
        return dict(self._mapping)


@pytest.fixture
async def session_factory(tmp_path: Any) -> AsyncIterator[async_sessionmaker]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'agg.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


async def _insert_selected_snapshot(
    session_factory: async_sessionmaker,
    *,
    slug: str,
    epoch: int,
    revision: int,
    weights: dict[str, float],
    digest: str | None = None,
) -> RawWeightSnapshot:
    payload_digest = digest or ("a" * 64)
    async with session_scope(session_factory) as session:
        snap = RawWeightSnapshot(
            id=uuid.uuid4(),
            challenge_slug=slug,
            epoch=epoch,
            revision=revision,
            protocol_version="1.0",
            computed_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            nonce=f"n-{slug}-{revision}-{uuid.uuid4().hex[:8]}",
            payload_digest=payload_digest,
            canonical_payload="{}",
            weights=weights,
            is_selected_source=True,
        )
        session.add(snap)
        await session.flush()
        await session.refresh(snap)
        return snap


def test_validate_emission_shares_rejects_over_allocation() -> None:
    with pytest.raises(EmissionPolicyError):
        validate_emission_shares({"a": 0.7, "b": 0.7})


def test_fractions_from_percent_absolute() -> None:
    shares = fractions_from_percent({"prism": 30, "agent-challenge": 15})
    assert shares == {"prism": 0.3, "agent-challenge": 0.15}


def test_vector_digest_is_stable() -> None:
    d1, c1, b1 = compute_vector_digest(
        protocol_version="1.0",
        epoch=1,
        revision=1,
        netuid=100,
        chain_endpoint="wss://x",
        uids=[0, 1],
        weights=[0.4, 0.6],
        emission_policy_version="emission-shares.absolute.v1",
        emission_shares={"prism": 0.3},
        burn_policy_version="burn-uid0.v1",
        mapping_policy_version="hotkey-to-uid.v1",
        source_snapshot_ids=["snap-1"],
        source_snapshot_digests=["d" * 64],
        metagraph_hash="m" * 64,
    )
    d2, c2, b2 = compute_vector_digest(
        protocol_version="1.0",
        epoch=1,
        revision=1,
        netuid=100,
        chain_endpoint="wss://x",
        uids=[0, 1],
        weights=[0.4, 0.6],
        emission_policy_version="emission-shares.absolute.v1",
        emission_shares={"prism": 0.3},
        burn_policy_version="burn-uid0.v1",
        mapping_policy_version="hotkey-to-uid.v1",
        source_snapshot_ids=["snap-1"],
        source_snapshot_digests=["d" * 64],
        metagraph_hash="m" * 64,
    )
    assert d1 == d2
    assert c1 == c2
    assert b1 == b2
    assert len(d1) == 64


@pytest.mark.asyncio
async def test_seal_epoch_from_durable_snapshots_only(
    session_factory: async_sessionmaker,
) -> None:
    clock = FakeClock()
    service = AggregationService(session_factory, now_fn=clock.now)
    epoch = 42
    s1 = await _insert_selected_snapshot(
        session_factory,
        slug="prism",
        epoch=epoch,
        revision=2,
        weights={"5CkeyAAA": 1.0},
        digest="1" * 64,
    )
    s2 = await _insert_selected_snapshot(
        session_factory,
        slug="agent-challenge",
        epoch=epoch,
        revision=1,
        weights={"5CkeyBBB": 1.0},
        digest="2" * 64,
    )
    with activate_role(Role.MASTER):
        await service.open_epoch(
            epoch,
            expected_challenges=["prism", "agent-challenge"],
            emission_shares={"prism": 0.3, "agent-challenge": 0.15},
        )
        vector = await service.seal_epoch(
            epoch,
            hotkey_to_uid={"5CkeyAAA": 10, "5CkeyBBB": 20},
            netuid=100,
            chain_endpoint="wss://chain.test",
        )
    assert vector.epoch == epoch
    assert str(s1.id) in (vector.source_snapshot_ids or [])
    assert str(s2.id) in (vector.source_snapshot_ids or [])
    assert vector.vector_digest
    assert sorted(vector.uids) == [0, 10, 20]  # remainder burns to 0
    # Re-seal is idempotent and immutable.
    with activate_role(Role.MASTER):
        again = await service.seal_epoch(
            epoch,
            hotkey_to_uid={"5CkeyAAA": 99},
            netuid=100,
        )
    assert again.id == vector.id
    assert again.vector_digest == vector.vector_digest
    assert again.uids == vector.uids


@pytest.mark.asyncio
async def test_missing_active_source_withholds_epoch(
    session_factory: async_sessionmaker,
) -> None:
    clock = FakeClock()
    service = AggregationService(session_factory, now_fn=clock.now)
    epoch = 7
    await _insert_selected_snapshot(
        session_factory,
        slug="prism",
        epoch=epoch,
        revision=1,
        weights={"5CkeyAAA": 1.0},
    )
    with activate_role(Role.MASTER):
        await service.open_epoch(
            epoch,
            expected_challenges=["prism", "agent-challenge"],
            emission_shares={"prism": 0.3, "agent-challenge": 0.15},
        )
        with pytest.raises(EpochWithheldError) as exc:
            await service.seal_epoch(
                epoch,
                hotkey_to_uid={"5CkeyAAA": 1},
                netuid=100,
            )
    assert "agent-challenge" in exc.value.reason
    with activate_role(Role.MASTER):
        latest = await service.get_latest_vector()
    assert latest is None


@pytest.mark.asyncio
async def test_vector_by_id_immutable_and_latest_serves_persisted(
    session_factory: async_sessionmaker,
) -> None:
    clock = FakeClock()
    agg = AggregationService(session_factory, now_fn=clock.now)
    epoch = 9
    await _insert_selected_snapshot(
        session_factory,
        slug="prism",
        epoch=epoch,
        revision=1,
        weights={"5CkeyAAA": 1.0},
        digest="f" * 64,
    )
    with activate_role(Role.MASTER):
        await agg.open_epoch(
            epoch,
            expected_challenges=["prism"],
            emission_shares={"prism": 0.5},
        )
        sealed = await agg.seal_epoch(
            epoch,
            hotkey_to_uid={"5CkeyAAA": 3},
            netuid=42,
            chain_endpoint="wss://c",
        )
    response = agg.vector_to_response(sealed)
    assert response.vector_id == str(sealed.id)
    assert response.vector_digest == sealed.vector_digest
    assert response.epoch == epoch
    assert response.protocol_version == "1.0"
    assert response.source_snapshots
    assert response.emission_policy_version
    assert response.chain_domain_bytes

    with activate_role(Role.MASTER):
        by_id = await agg.get_vector_by_id(str(sealed.id))
    assert by_id.canonical_payload == sealed.canonical_payload

    with activate_role(Role.MASTER):
        with pytest.raises(VectorNotFoundError):
            await agg.get_vector_by_id(str(uuid.uuid4()))

    # MasterWeightService durable path never pulls associates.
    master = MasterWeightService(
        metagraph_cache=StubMetagraphCache({"5CkeyAAA": 3}),  # type: ignore[arg-type]
        session_factory=session_factory,
        aggregation_service=agg,
    )
    with activate_role(Role.MASTER):
        latest = await master.compute_latest_response(
            [],
            {},
            netuid=42,
            chain_endpoint="wss://c",
        )
    assert latest.vector_id == str(sealed.id)
    assert latest.uids == sealed.uids
    assert latest.weights == [float(w) for w in sealed.weights]


@pytest.mark.asyncio
async def test_collect_durable_weights_requires_selected_snapshots(
    session_factory: async_sessionmaker,
) -> None:
    master = MasterWeightService(
        metagraph_cache=StubMetagraphCache({}),  # type: ignore[arg-type]
        session_factory=session_factory,
    )
    challenges = [_challenge("prism", "30")]
    with activate_role(Role.MASTER):
        with pytest.raises(EpochWithheldError):
            await master.collect_durable_weights(epoch=3, challenges=challenges)

    await _insert_selected_snapshot(
        session_factory,
        slug="prism",
        epoch=3,
        revision=1,
        weights={"5CkeyAAA": 1.0},
    )
    with activate_role(Role.MASTER):
        results = await master.collect_durable_weights(epoch=3, challenges=challenges)
    assert len(results) == 1
    assert results[0].slug == "prism"
    assert results[0].weights == {"5CkeyAAA": 1.0}


def test_compute_latest_without_durable_store_still_aggregates() -> None:
    """Unit-test diagnostic path without session remains functional."""

    class _Client:
        async def get_weights(self, **kwargs: Any) -> Any:
            from base.schemas.weights import ChallengeWeightsResult

            return ChallengeWeightsResult(
                slug=str(kwargs["slug"]),
                emission_percent=float(kwargs["emission_percent"]),
                weights={"hk1": 1.0},
                ok=True,
            )

    service = MasterWeightService(
        metagraph_cache=StubMetagraphCache({"hk1": 1}),  # type: ignore[arg-type]
        challenge_client=_Client(),  # type: ignore[arg-type]
    )
    challenges = [_challenge("prism", "100")]
    now = datetime.now(UTC)
    with activate_role(Role.MASTER):
        response = asyncio.run(
            service.compute_latest_response(
                challenges,
                {"prism": "tok"},
                netuid=1,
                chain_endpoint="",
                now_fn=lambda: now,
            )
        )
    assert response.uids
    assert response.vector_id is None


@pytest.mark.asyncio
async def test_run_epoch_with_session_factory_refuses_get_weights_fallback(
    session_factory: async_sessionmaker,
) -> None:
    """VAL-CROSS-067: durable session force-seal; missing ids fail closed."""

    pulls: list[str] = []

    class _Pull:
        async def get_weights(self, **kwargs: Any) -> Any:
            pulls.append(str(kwargs.get("slug")))
            from base.schemas.weights import ChallengeWeightsResult

            return ChallengeWeightsResult(
                slug=str(kwargs["slug"]),
                emission_percent=float(kwargs["emission_percent"]),
                weights={"5CkeyAAA": 1.0},
                ok=True,
            )

    master = MasterWeightService(
        metagraph_cache=StubMetagraphCache({"5CkeyAAA": 3}),  # type: ignore[arg-type]
        challenge_client=_Pull(),  # type: ignore[arg-type]
        session_factory=session_factory,
    )
    challenges = [_challenge("prism", "100")]
    with activate_role(Role.MASTER):
        with pytest.raises(RuntimeError, match="VAL-CROSS-067"):
            await master.run_epoch(challenges, {"prism": "tok"})
        with pytest.raises(RuntimeError, match="VAL-CROSS-067"):
            await master.run_epoch(challenges, {"prism": "tok"}, epoch=1)
        with pytest.raises(RuntimeError, match="VAL-CROSS-067"):
            await master.run_epoch(challenges, {"prism": "tok"}, netuid=42)
    assert pulls == []

    await _insert_selected_snapshot(
        session_factory,
        slug="prism",
        epoch=11,
        revision=1,
        weights={"5CkeyAAA": 1.0},
    )
    with activate_role(Role.MASTER):
        final = await master.run_epoch(
            challenges,
            {"prism": "tok"},
            epoch=11,
            netuid=42,
            chain_endpoint="wss://c",
        )
    assert final.uids
    assert pulls == []


def test_cli_run_master_weight_epoch_forwards_epoch_and_netuid() -> None:
    import inspect

    import base.cli_app.main as cli_main

    params = inspect.signature(cli_main._run_master_weight_epoch).parameters
    assert "epoch" in params
    assert "netuid" in params
    assert "chain_endpoint" in params


def _challenge(slug: str, emission: str) -> RegistryChallenge:
    return RegistryChallenge(
        slug=slug,
        name=slug.title(),
        image=f"ghcr.io/x/{slug}:1",
        version="1.0.0",
        emission_percent=Decimal(emission),
        status=ChallengeStatus.ACTIVE,
        internal_base_url=f"http://{slug}:8000",
        public_proxy_base_path=f"/challenges/{slug}",
        required_capabilities=["get_weights", "proxy_routes"],
        resources={},
        volumes={},
        env={},
        secrets=[],
    )
