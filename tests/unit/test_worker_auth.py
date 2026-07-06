from __future__ import annotations

import hashlib
from typing import Any

import pytest

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.security.validator_auth import canonical_validator_request
from base.security.worker_auth import (
    CoordinationReadEligibility,
    MetagraphMinerMembership,
    RegisteredWorkerEligibility,
    SqlAlchemyWorkerNonceStore,
    WorkerAuthError,
    WorkerEligibilityError,
    WorkerReplayError,
    WorkerSignedRequestVerifier,
    worker_binding_message,
)

MINER = "miner-1"
VALIDATOR = "validator-1"
WORKER = "worker-pubkey-1"


def _sign(hotkey: str, message: bytes) -> str:
    return hashlib.sha256(hotkey.encode() + b":" + message).hexdigest()


def _fake_verifier(hotkey: str, message: bytes, signature: str) -> bool:
    return signature == _sign(hotkey, message)


def test_worker_binding_message_matches_pinned_format() -> None:
    assert (
        worker_binding_message(worker_pubkey="wp", miner_hotkey="mh", nonce="n1")
        == b"worker-binding:wp:mh:n1"
    )


def test_metagraph_miner_membership_requires_only_graph_presence() -> None:
    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    # miner is present WITHOUT a validator permit; validator holds a permit.
    cache.update_from_metagraph(
        [MINER, VALIDATOR], validator_permits=[False, True], stakes=[0.0, 1.0]
    )
    membership = MetagraphMinerMembership(cache)
    assert membership.is_registered(MINER) is True
    assert membership.is_registered("off-graph") is False


async def _session_factory_with_worker(status: WorkerStatus) -> Any:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = create_session_factory(engine)
    async with session_scope(session_factory) as session:
        session.add(
            WorkerRegistration(
                worker_id="w-1",
                worker_pubkey=WORKER,
                miner_hotkey=MINER,
                binding_signature="sig",
                binding_nonce="n1",
                provider="local",
                provider_instance_ref=None,
                capabilities=["gpu"],
                status=status,
            )
        )
    return session_factory, engine


async def test_registered_worker_eligibility_matches_any_status() -> None:
    session_factory, engine = await _session_factory_with_worker(WorkerStatus.RETIRED)
    try:
        eligibility = RegisteredWorkerEligibility(session_factory)
        assert await eligibility.is_eligible(WORKER) is True
        assert await eligibility.is_eligible("unknown") is False
    finally:
        await engine.dispose()


async def test_coordination_read_eligibility_allows_worker_or_validator() -> None:
    session_factory, engine = await _session_factory_with_worker(WorkerStatus.ACTIVE)
    try:
        cache = MetagraphCache(netuid=1, ttl_seconds=300)
        cache.update_from_metagraph(
            [MINER, VALIDATOR], validator_permits=[False, True], stakes=[0.0, 1.0]
        )
        eligibility = CoordinationReadEligibility(session_factory, cache)
        assert await eligibility.is_eligible(WORKER) is True  # registered worker
        assert await eligibility.is_eligible(VALIDATOR) is True  # metagraph validator
        assert await eligibility.is_eligible(MINER) is False  # miner, not a worker
        assert await eligibility.is_eligible("stranger") is False
    finally:
        await engine.dispose()


async def _verify(
    verifier: WorkerSignedRequestVerifier,
    *,
    hotkey: str,
    nonce: str,
    signature: str | None = None,
    method: str = "GET",
    path: str = "/v1/workers",
    timestamp: str = "1000",
) -> Any:
    canonical = canonical_validator_request(
        method=method,
        path=path,
        query_string="",
        timestamp=timestamp,
        nonce=nonce,
        body=b"",
    )
    sig = signature if signature is not None else _sign(hotkey, canonical.encode())
    return await verifier.verify(
        method=method,
        path=path,
        query_string="",
        headers={
            "X-Hotkey": hotkey,
            "X-Signature": sig,
            "X-Nonce": nonce,
            "X-Timestamp": timestamp,
        },
        body=b"",
    )


async def test_signed_request_verifier_happy_path_and_replay() -> None:
    session_factory, engine = await _session_factory_with_worker(WorkerStatus.ACTIVE)
    try:
        cache = MetagraphCache(netuid=1, ttl_seconds=300)
        cache.update_from_metagraph([VALIDATOR], validator_permits=[True], stakes=[1.0])
        verifier = WorkerSignedRequestVerifier(
            nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
            eligibility=CoordinationReadEligibility(session_factory, cache),
            signature_verifier=_fake_verifier,
            ttl_seconds=300,
            now_fn=lambda: 1000.0,
        )
        identity = await _verify(verifier, hotkey=WORKER, nonce="rq-1")
        assert identity.hotkey == WORKER

        with pytest.raises(WorkerReplayError):
            await _verify(verifier, hotkey=WORKER, nonce="rq-1")
    finally:
        await engine.dispose()


async def test_signed_request_verifier_rejects_bad_signature_and_ineligible() -> None:
    session_factory, engine = await _session_factory_with_worker(WorkerStatus.ACTIVE)
    try:
        cache = MetagraphCache(netuid=1, ttl_seconds=300)
        cache.update_from_metagraph([VALIDATOR], validator_permits=[True], stakes=[1.0])
        verifier = WorkerSignedRequestVerifier(
            nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
            eligibility=CoordinationReadEligibility(session_factory, cache),
            signature_verifier=_fake_verifier,
            ttl_seconds=300,
            now_fn=lambda: 1000.0,
        )
        with pytest.raises(WorkerAuthError):
            await _verify(verifier, hotkey=WORKER, nonce="rq-2", signature="bad")

        with pytest.raises(WorkerEligibilityError):
            await _verify(verifier, hotkey="stranger", nonce="rq-3")
    finally:
        await engine.dispose()


async def test_signed_request_verifier_rejects_stale_timestamp() -> None:
    session_factory, engine = await _session_factory_with_worker(WorkerStatus.ACTIVE)
    try:
        cache = MetagraphCache(netuid=1, ttl_seconds=300)
        cache.update_from_metagraph([VALIDATOR], validator_permits=[True], stakes=[1.0])
        verifier = WorkerSignedRequestVerifier(
            nonce_store=SqlAlchemyWorkerNonceStore(session_factory),
            eligibility=CoordinationReadEligibility(session_factory, cache),
            signature_verifier=_fake_verifier,
            ttl_seconds=10,
            now_fn=lambda: 1_000_000.0,
        )
        with pytest.raises(WorkerAuthError):
            await _verify(verifier, hotkey=WORKER, nonce="rq-4", timestamp="1000")
    finally:
        await engine.dispose()
