"""Base-side cross-integration: R=1 at the master + carry-chain integrity.

Offline complements to the agent-challenge cross-integration suite
(``agent-challenge/tests/test_cross_integration_e2e_offline.py``). These cover
the two legs that live in the base repo:

* **VAL-CROSS-003** -- for an attested agent-challenge unit the master
  coordination plane (``MasterOrchestrationDriver.run_once``) assigns it as a
  SINGLE cpu work unit at replication factor 1 and never enters the worker-plane
  replicate+reconcile path: exactly one owner, no second replica, no
  reconciliation/audit/dispute row, across repeated passes. A sibling prism gpu
  unit in the SAME pass IS replicated to R=2, proving the worker plane is
  genuinely active (the R=1 result is not vacuous). (This is the VAL-CROSS
  framing of the behaviour ``test_master_r1_preserved`` asserts under
  VAL-VERIFY-020/021.)

* **VAL-CROSS-007 (base leg)** -- the TDX quote / ``report_data`` / measurement
  emitted alongside the ``BASE_BENCHMARK_RESULT=`` line by the in-CVM backend is
  mapped onto the BASE ``ExecutionProof`` Phala tier and carried to the master
  BYTE-FOR-BYTE across the JSON serialization boundary and the validator
  tier-0 signature rebind. The carried envelope still verifies, and a single
  flipped quote byte / re-encoded ``report_data`` breaks it (the carry is a real
  discriminator, not a constant pass). The challenge leg (envelope emitted
  alongside the result line and parsed by the host normalizer) is asserted in the
  agent-challenge repo.

* **VAL-CROSS-021 (base leg)** -- with the flag OFF the BASE validator adapter +
  master handle an agent-challenge unit exactly as legacy: no Phala tier is
  required on results, no attestation gate is applied, and master orchestration
  keeps the legacy reassign-on-failure (NEVER replicate) path for the cpu units.
  A sibling gpu unit replicating to R=2 (VAL-CROSS-003 above) proves the R=1
  reassign-only result here is not vacuous.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from base.bittensor.metagraph_cache import MetagraphCache
from base.db import (
    Base,
    Validator,
    ValidatorStatus,
    WorkAssignmentStatus,
    WorkerAssignment,
    WorkerFault,
    WorkerRegistration,
    WorkerStatus,
    create_engine,
    create_session_factory,
    session_scope,
)
from base.db.models import WorkAssignment
from base.master.assignment import CAPABILITY_GPU, AssignmentService
from base.master.assignment_coordination import AssignmentCoordinationService
from base.master.orchestration import (
    ChallengePendingWork,
    MasterOrchestrationDriver,
)
from base.master.validator_coordination import ValidatorCoordinationService
from base.master.worker_assignment import WorkerAssignmentService
from base.master.worker_assignment_engine import WorkerAssignmentEngine
from base.master.worker_coordination import WorkerCoordinationService
from base.master.worker_reconciliation import (
    AUDIT_WORK_UNIT_SUFFIX,
    WorkerReconciliationService,
)
from base.schemas.worker import (
    PHALA_TDX_TIER,
    ExecutionProof,
    PhalaAttestation,
    PhalaMeasurement,
    WorkerSignature,
)
from base.security.worker_auth import (
    MetagraphMinerMembership,
    SqlAlchemyWorkerNonceStore,
)
from base.validator.agent.adapters.agent_challenge import rebind_worker_signature
from base.worker.phala_quote import (
    StaticQuoteVerifier,
    build_rtmr3_event_log,
    build_tdx_quote,
)
from base.worker.phala_verify import (
    InMemoryNonceValidator,
    MeasurementAllowlist,
    PhalaBinding,
)
from base.worker.proof import (
    build_phala_execution_proof,
    phala_report_data_hex,
    verify_execution_proof,
)

NOW = datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC)
TTL = 120
_ACTIVE = (WorkAssignmentStatus.ASSIGNED, WorkAssignmentStatus.RUNNING)


# =========================================================================== #
# VAL-CROSS-003: master keeps attested agent-challenge units at R=1
# =========================================================================== #
@dataclass
class _FakeWorkSource:
    works: list[ChallengePendingWork] = field(default_factory=list)

    async def fetch_pending_work(self) -> list[ChallengePendingWork]:
        return list(self.works)


@dataclass
class _FakeFoldTrigger:
    calls: list[tuple[str, str, str, str]] = field(default_factory=list)

    async def fold(
        self, *, challenge_slug: str, job_id: str, task_id: str, reason: str
    ) -> None:
        self.calls.append((challenge_slug, job_id, task_id, reason))


class _FakeForwarder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def forward_result(
        self,
        *,
        challenge_slug: str,
        work_unit_id: str,
        submission_ref: str,
        result_payload: Any,
    ) -> None:
        self.calls.append(work_unit_id)


async def _setup() -> tuple[Any, Any]:
    engine = create_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, create_session_factory(engine)


def _build_flag_on_driver(
    factory: Any, works: list[ChallengePendingWork]
) -> MasterOrchestrationDriver:
    """A full flag-ON driver: worker assignment engine + reconciler both live."""

    cache = MetagraphCache(netuid=1, ttl_seconds=300)
    worker_service = WorkerCoordinationService(
        factory,
        miner_membership=MetagraphMinerMembership(cache),
        binding_nonce_store=SqlAlchemyWorkerNonceStore(factory),
        heartbeat_ttl_seconds=TTL,
        now_fn=lambda: NOW,
    )
    worker_assignment_service = WorkerAssignmentService(
        factory, worker_service=worker_service, now_fn=lambda: NOW
    )
    worker_engine = WorkerAssignmentEngine(
        factory,
        assignment_service=worker_assignment_service,
        worker_service=worker_service,
        replication_factor=2,
        now_fn=lambda: NOW,
    )
    reconciler = WorkerReconciliationService(
        factory, result_forwarder=_FakeForwarder(), now_fn=lambda: NOW
    )
    assignment_service = AssignmentService(
        factory,
        now_fn=lambda: NOW,
        default_max_attempts=3,
        worker_plane_capabilities=frozenset({CAPABILITY_GPU}),
    )
    return MasterOrchestrationDriver(
        assignment_service=assignment_service,
        validator_service=ValidatorCoordinationService(factory, now_fn=lambda: NOW),
        work_source=_FakeWorkSource(works=works),
        fold_trigger=_FakeFoldTrigger(),
        worker_assignment_engine=worker_engine,
        worker_reconciler=reconciler,
        seed=1,
    )


async def _add_worker(factory: Any, *, worker_pubkey: str, miner_hotkey: str) -> None:
    async with session_scope(factory) as session:
        session.add(
            WorkerRegistration(
                worker_id=f"wid-{worker_pubkey}",
                worker_pubkey=worker_pubkey,
                miner_hotkey=miner_hotkey,
                binding_signature="sig",
                binding_nonce=f"nonce-{worker_pubkey}",
                provider="local",
                provider_instance_ref="local-1",
                capabilities=["gpu"],
                status=WorkerStatus.ACTIVE,
                last_heartbeat_at=NOW,
                created_at=NOW,
                updated_at=NOW,
            )
        )


async def _add_validator(factory: Any, hotkey: str, capabilities: list[str]) -> None:
    async with session_scope(factory) as session:
        session.add(
            Validator(
                hotkey=hotkey,
                uid=None,
                status=ValidatorStatus.ONLINE,
                capabilities=list(capabilities),
                version="1.0.0",
                registered_at=NOW,
                last_heartbeat_at=NOW,
            )
        )


async def _units(factory: Any) -> dict[str, WorkAssignment]:
    async with factory() as session:
        rows = (await session.execute(select(WorkAssignment))).scalars().all()
        return {r.work_unit_id: r for r in rows}


async def _replica_count(factory: Any, work_unit_id: str) -> int:
    async with factory() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(WorkerAssignment)
                .where(
                    WorkerAssignment.work_unit_id == work_unit_id,
                    WorkerAssignment.status.in_(_ACTIVE),
                )
            )
        ).scalar_one()


async def _worker_fault_count(factory: Any) -> int:
    async with factory() as session:
        return (
            await session.execute(select(func.count()).select_from(WorkerFault))
        ).scalar_one()


def _attested_agent_work() -> ChallengePendingWork:
    # Attestation rides in the result payload; the unit itself is a plain cpu
    # unit as far as the master's assignment/reconciliation plane is concerned.
    return ChallengePendingWork(
        challenge_slug="agent-challenge",
        submission_id="sub",
        submission_ref="miner-C",
        task_ids=("a", "b", "c"),
        job_id="job-1",
        payload={"proof": {"tier": PHALA_TDX_TIER}},
    )


def _prism_work() -> ChallengePendingWork:
    return ChallengePendingWork(
        challenge_slug="prism", submission_id="psub", submission_ref="miner-P"
    )


async def test_val_cross_003_master_keeps_attested_units_at_r1_no_reconcile() -> None:
    engine, factory = await _setup()
    try:
        driver = _build_flag_on_driver(
            factory, works=[_attested_agent_work(), _prism_work()]
        )
        # Two distinct-owner gpu workers so the prism gpu primary genuinely
        # replicates to R=2 this pass (non-vacuous worker plane).
        await _add_worker(factory, worker_pubkey="wp-a", miner_hotkey="miner-A")
        await _add_worker(factory, worker_pubkey="wp-b", miner_hotkey="miner-B")
        await _add_validator(factory, "v1", ["cpu"])

        result = await driver.run_once()

        units = await _units(factory)
        cpu_ids = ["sub:a", "sub:b", "sub:c"]
        # Each selected task is exactly ONE cpu unit, one owner, one attempt, and
        # NO worker-plane replica (replication factor 1).
        for uid in cpu_ids:
            unit = units[uid]
            assert unit.required_capability == "cpu"
            assert unit.status == WorkAssignmentStatus.ASSIGNED
            assert unit.assigned_validator_hotkey == "v1"
            assert unit.attempt_count == 1
            assert await _replica_count(factory, uid) == 0

        # Sibling gpu unit replicated to R=2 -> the worker plane is genuinely on,
        # so the cpu R=1 result above is not vacuous.
        assert units["psub"].assigned_validator_hotkey is None
        assert await _replica_count(factory, "psub") == 2

        # The reconciler ran (flag ON) but produced NO artifacts for the attested
        # agent-challenge units: nothing disputed/audited/faulted.
        assert result.reconciliation is not None
        assert result.reconciliation.disputed == []
        assert result.reconciliation.audit_units == {}
        assert result.reconciliation.faults == []
        assert await _worker_fault_count(factory) == 0
        assert all(not uid.endswith(AUDIT_WORK_UNIT_SUFFIX) for uid in units)

        # Repeated passes never add a second replica/assignment nor a new unit
        # for a cpu submission (still R=1; no reconciliation/replica rows).
        for _ in range(3):
            again = await driver.run_once()
            assert again.folded == []
            units = await _units(factory)
            for uid in cpu_ids:
                assert units[uid].assigned_validator_hotkey == "v1"
                assert units[uid].attempt_count == 1
                assert await _replica_count(factory, uid) == 0
        assert set(units) == {"sub:a", "sub:b", "sub:c", "psub"}
    finally:
        await engine.dispose()


# =========================================================================== #
# VAL-CROSS-007 (base leg): the attestation envelope survives the challenge<->
# BASE boundary + the validator tier-0 rebind byte-for-byte.
# =========================================================================== #
MANIFEST = "a" * 64
UNIT_ID = "submission-carry-1"
MRTD = "a1" * 48
RTMR0 = "b0" * 48
RTMR1 = "b1" * 48
RTMR2 = "b2" * 48
COMPOSE_PAYLOAD = bytes.fromhex("c3" * 32)
AGENT_HASH = "f0" * 32
TASK_IDS = ("task-b", "task-a", "task-c")
SCORES_DIGEST = "9a" * 32


def _signer(hotkey: str = "carry-chain-worker") -> _FakeSigner:
    return _FakeSigner(hotkey=hotkey)


@dataclass(frozen=True)
class _FakeSigner:
    """A ``RequestSigner`` that signs deterministically without bittensor.

    The sr25519 primitive itself is covered by ``test_worker_proof_phala_verify``;
    this carry-chain test targets attestation-envelope integrity. Importing
    bittensor here would reconfigure process-wide logging at import time (it
    disables previously-imported loggers), which breaks a fragile logging test
    that sorts after this module. The signature is still a real function of
    ``(hotkey, message)`` so the tier-0 no-cross-unit-replay property is exercised.
    """

    hotkey: str = "carry-chain-worker"

    def sign(self, message: bytes) -> str:
        return "0x" + hashlib.sha256(self.hotkey.encode() + message).hexdigest()


def _fake_verify(pubkey: str, message: bytes, signature: str) -> bool:
    return signature == "0x" + hashlib.sha256(pubkey.encode() + message).hexdigest()


def _os_image_hash(mrtd: str, rtmr1: str, rtmr2: str) -> str:
    return hashlib.sha256(
        bytes.fromhex(mrtd) + bytes.fromhex(rtmr1) + bytes.fromhex(rtmr2)
    ).hexdigest()


def _cvm_emitted_attestation(
    *, validator_nonce: str
) -> tuple[PhalaAttestation, dict[str, str]]:
    """What the in-CVM backend emits: a self-consistent Phala attestation."""

    event_log, rtmr3 = build_rtmr3_event_log([("compose-hash", COMPOSE_PAYLOAD)])
    measurement = {
        "mrtd": MRTD,
        "rtmr0": RTMR0,
        "rtmr1": RTMR1,
        "rtmr2": RTMR2,
        "compose_hash": COMPOSE_PAYLOAD.hex(),
        "os_image_hash": _os_image_hash(MRTD, RTMR1, RTMR2),
    }
    report_data_hex = phala_report_data_hex(
        canonical_measurement=measurement,
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=validator_nonce,
    )
    quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=report_data_hex,
    )
    attestation = PhalaAttestation(
        tdx_quote=quote,
        event_log=event_log,
        report_data=report_data_hex,
        measurement=PhalaMeasurement(
            mrtd=MRTD,
            rtmr0=RTMR0,
            rtmr1=RTMR1,
            rtmr2=RTMR2,
            rtmr3=rtmr3,
            compose_hash=COMPOSE_PAYLOAD.hex(),
            os_image_hash=measurement["os_image_hash"],
        ),
        vm_config={"vcpu": 1, "memory_mb": 2048},
    )
    return attestation, measurement


def _binding(nonce: str) -> PhalaBinding:
    return PhalaBinding(
        agent_hash=AGENT_HASH,
        task_ids=TASK_IDS,
        scores_digest=SCORES_DIGEST,
        validator_nonce=nonce,
    )


def test_val_cross_007_base_leg_envelope_carried_byte_for_byte_and_verifies() -> None:
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    emitted, measurement = _cvm_emitted_attestation(validator_nonce=nonce)
    emitted_quote = emitted.tdx_quote
    emitted_report_data = emitted.report_data

    # 1. The lean CVM image maps the attestation onto the BASE ExecutionProof
    #    Phala tier with a PLACEHOLDER (empty) worker signature, and it rides the
    #    challenge<->BASE JSON boundary (model_dump -> model_validate).
    placeholder = ExecutionProof(
        version=1,
        tier=PHALA_TDX_TIER,
        manifest_sha256=MANIFEST,
        worker_signature=WorkerSignature(worker_pubkey="", sig=""),
        attestation=emitted.model_dump(mode="json"),
    )
    round_tripped = ExecutionProof.model_validate(placeholder.model_dump(mode="json"))
    carried = PhalaAttestation.model_validate(round_tripped.attestation)
    # Quote + report_data survive the boundary byte-for-byte (no truncation /
    # re-encode / field loss).
    assert carried.tdx_quote == emitted_quote
    assert carried.report_data == emitted_report_data
    assert carried.measurement.model_dump() == emitted.measurement.model_dump()

    # 2. The validator ingests the payload and rebinds ONLY the tier-0 worker
    #    signature to this unit; the attestation payload is carried through
    #    UNCHANGED (the quote is the trust root, never re-minted).
    bound = rebind_worker_signature(round_tripped, signer=_signer(), unit_id=UNIT_ID)
    assert bound.attestation == round_tripped.attestation
    assert bound.tier == PHALA_TDX_TIER

    # 3. What reaches the master (serialize the bound proof) still carries the
    #    exact bytes the CVM emitted.
    at_master = PhalaAttestation.model_validate(
        ExecutionProof.model_validate(bound.model_dump(mode="json")).attestation
    )
    assert at_master.tdx_quote == emitted_quote
    assert at_master.report_data == emitted_report_data

    # 4. The carried envelope still verifies against the validator's expectations.
    assert (
        verify_execution_proof(
            bound,
            unit_id=UNIT_ID,
            expected_binding=_binding(nonce),
            quote_verifier=StaticQuoteVerifier(),
            allowlist=MeasurementAllowlist.from_measurements([measurement]),
            nonce_validator=nonces,
            signature_verifier=_fake_verify,
        )
        is True
    )


def test_val_cross_007_base_leg_single_carried_byte_flip_breaks_verification() -> None:
    # Discriminator: the carry is NOT a constant pass. Flipping ONE quote byte
    # (last nibble -> perturbs the trailing report_data byte) breaks the chain.
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    emitted, measurement = _cvm_emitted_attestation(validator_nonce=nonce)

    last = emitted.tdx_quote[-1]
    tampered_quote = emitted.tdx_quote[:-1] + ("0" if last != "0" else "1")
    assert tampered_quote != emitted.tdx_quote

    tampered = emitted.model_copy(update={"tdx_quote": tampered_quote})
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=tampered,
    )
    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=_binding(nonce),
            quote_verifier=StaticQuoteVerifier(),
            allowlist=MeasurementAllowlist.from_measurements([measurement]),
            nonce_validator=nonces,
            signature_verifier=_fake_verify,
        )
        is False
    )


def test_val_cross_007_base_leg_report_data_reencode_breaks_verification() -> None:
    # A re-encoded (zero-truncated) report_data no longer matches the quote-bound
    # field, so the carried envelope is rejected.
    nonces = InMemoryNonceValidator()
    nonce = nonces.issue()
    emitted, measurement = _cvm_emitted_attestation(validator_nonce=nonce)

    # Rebuild the quote with a DIFFERENT report_data than the binding expects.
    event_log, rtmr3 = build_rtmr3_event_log([("compose-hash", COMPOSE_PAYLOAD)])
    wrong_report_data = ("00" * 32).ljust(128, "0")
    reencoded_quote = build_tdx_quote(
        mrtd=MRTD,
        rtmr0=RTMR0,
        rtmr1=RTMR1,
        rtmr2=RTMR2,
        rtmr3=rtmr3,
        report_data=wrong_report_data,
    )
    tampered = emitted.model_copy(
        update={"tdx_quote": reencoded_quote, "report_data": wrong_report_data}
    )
    proof = build_phala_execution_proof(
        signer=_signer(),
        manifest_sha256=MANIFEST,
        unit_id=UNIT_ID,
        attestation=tampered,
    )
    assert (
        verify_execution_proof(
            proof,
            unit_id=UNIT_ID,
            expected_binding=_binding(nonce),
            quote_verifier=StaticQuoteVerifier(),
            allowlist=MeasurementAllowlist.from_measurements([measurement]),
            nonce_validator=nonces,
            signature_verifier=_fake_verify,
        )
        is False
    )


# =========================================================================== #
# VAL-CROSS-021 (base leg): flag OFF => the BASE side treats agent-challenge
# units as legacy (no Phala tier, no attestation gate, reassign-never-replicate).
# =========================================================================== #
def _build_flag_off_driver(
    factory: Any, works: list[ChallengePendingWork], *, service: AssignmentService
) -> MasterOrchestrationDriver:
    """A flag-OFF driver: no worker engine, no reconciler (legacy routing)."""

    return MasterOrchestrationDriver(
        assignment_service=service,
        validator_service=ValidatorCoordinationService(factory, now_fn=lambda: NOW),
        work_source=_FakeWorkSource(works=works),
        fold_trigger=_FakeFoldTrigger(),
        worker_assignment_engine=None,
        worker_reconciler=None,
        seed=1,
    )


async def _set_validator_status(
    factory: Any, hotkey: str, status: ValidatorStatus
) -> None:
    async with session_scope(factory) as session:
        validator = (
            await session.execute(select(Validator).where(Validator.hotkey == hotkey))
        ).scalar_one()
        validator.status = status


async def test_val_cross_021_flag_off_base_treats_agent_challenge_units_as_legacy() -> (
    None
):
    engine, factory = await _setup()
    try:
        # Flag OFF: even an attestation-bearing agent-challenge submission is
        # bridged exactly like legacy -- no Phala tier required, no worker plane.
        service = AssignmentService(factory, now_fn=lambda: NOW, default_max_attempts=3)
        driver = _build_flag_off_driver(
            factory, works=[_attested_agent_work()], service=service
        )
        await _add_validator(factory, "v1", ["cpu"])

        result = await driver.run_once()
        assert result.worker is None  # no worker-plane engine ran (legacy)
        assert result.reconciliation is None  # no attestation/audit reconcile

        units = await _units(factory)
        cpu_ids = ["sub:a", "sub:b", "sub:c"]
        assert set(units) == set(cpu_ids)  # only the fanned cpu units
        for uid in cpu_ids:
            unit = units[uid]
            assert unit.required_capability == "cpu"
            assert unit.assigned_validator_hotkey == "v1"  # single executor (R=1)
            assert unit.attempt_count == 1
            assert await _replica_count(factory, uid) == 0  # never replicated
        assert all(not uid.endswith(AUDIT_WORK_UNIT_SUFFIX) for uid in units)

        # Legacy reassign-on-failure (NEVER replicate): the owner crashes, the cpu
        # units revert to pending and reassign to a DIFFERENT validator (exactly
        # one more attempt) -- still zero worker replicas. The gpu R=2 replica in
        # VAL-CROSS-003 above proves this R=1 reassign-only path is not vacuous.
        coordination = AssignmentCoordinationService(factory, now_fn=lambda: NOW)
        await coordination.pull(hotkey="v1")  # assigned -> running
        await _set_validator_status(factory, "v1", ValidatorStatus.OFFLINE)
        await _add_validator(factory, "v2", ["cpu"])

        outcome = await service.reclaim_stale_assignments()
        assert set(outcome.reverted) == set(cpu_ids)
        await service.assign_pending(seed=1)

        units = await _units(factory)
        for uid in cpu_ids:
            unit = units[uid]
            assert unit.assigned_validator_hotkey == "v2"  # reassigned, not replicated
            assert unit.attempt_count == 2  # exactly one more attempt
            assert await _replica_count(factory, uid) == 0  # NEVER a worker replica
    finally:
        await engine.dispose()
