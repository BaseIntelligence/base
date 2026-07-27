"""TDD: ProductionConstationOrchestrator — issue nonce → run → seal → put (B2).

B2: orchestrator MUST issue nonces and MUST NEVER consume. After seal, the
end-phase nonce remains first-consumable for prism ingest.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from base.compute.attestation_nonce import (
    AttestationNonceService,
    NonceBinding,
    NonceConsumeHit,
    NonceConsumeMiss,
    NonceConsumeReason,
)
from base.compute.constation_custody import LiumKeyCustody, generate_custody_master_key
from base.compute.constation_poller import PollerConfig
from base.compute.constation_runner import ConstationRunRequest
from base.compute.constation_types import (
    ConstationFailCode,
    ConstationRunRecord,
    CorroborationStatus,
    FaultClass,
)
from base.compute.digest_allowlist import ImageVariant
from base.master.constation.bundle_store import ConstationBundleStore
from base.master.constation.orchestrator import (
    ConstationOrchestrationRequest,
    ConstationOrchestrationResult,
    ProductionConstationOrchestrator,
)
from base.master.constation.pod_binding import MinerPodBinding

COMMIT = "a" * 40
TREE = "b" * 40
DIGEST = "sha256:" + ("c" * 64)
MANIFEST = {"src/harness.py": "d" * 64}
HOTKEY = "5MinerOrchTestHotkey000000000000000001"
POD = "pod-orch-001"
WORK_UNIT = "wu-orch-001"
T0 = datetime(2026, 7, 27, 5, 0, 0, tzinfo=UTC)
WIRE = {
    "payload": {
        "digest": DIGEST,
        "nonce": "will-be-overwritten-in-assert",
        "pod_id": POD,
        "variant": "cuda",
        "build_secret_response": "ab" * 32,
        "sealed_manifest_hashes": dict(MANIFEST),
    },
    "signature": "ef" * 32,
    "algorithm": "hmac-sha256",
    "schema_version": "prism_attestation_payload.v1",
    "phase": "end",
}


def _ok_record(
    *,
    work_unit_id: str = WORK_UNIT,
    miner_hotkey: str = HOTKEY,
    pod_id: str = POD,
) -> ConstationRunRecord:
    return ConstationRunRecord(
        ok=True,
        reason=ConstationFailCode.OK,
        fault_class=None,
        miner_hotkey=miner_hotkey,
        work_unit_id=work_unit_id,
        pod_id=pod_id,
        sidecar_digest=DIGEST,
        lium_declared_digest=DIGEST,
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=1.0,
        corroboration_status=CorroborationStatus.AGREE,
        samples=(),
    )


def _fail_record(
    *,
    reason: ConstationFailCode = ConstationFailCode.SIDECAR_ATTEST_FAILED,
) -> ConstationRunRecord:
    return ConstationRunRecord(
        ok=False,
        reason=reason,
        fault_class=FaultClass.INFRA,
        miner_hotkey=HOTKEY,
        work_unit_id=WORK_UNIT,
        pod_id=POD,
        sidecar_digest=None,
        lium_declared_digest=None,
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=0.0,
        corroboration_status=CorroborationStatus.NOT_EVALUATED,
        samples=(),
    )


@dataclass
class _FakeRunner:
    """Stand-in ConstationRunner: issues via poll_nonce_fn, sets last_signed_wire."""

    poll_nonce_fn: Callable[[], str] | None = None
    last_signed_wire: Mapping[str, Any] | None = field(default=None, init=False)
    outcome: ConstationRunRecord = field(default_factory=_ok_record)
    wire: Mapping[str, Any] | None = field(default_factory=lambda: dict(WIRE))
    run_calls: int = 0
    last_request: ConstationRunRequest | None = None
    phases_to_issue: tuple[str, ...] = ("start", "mid", "end")

    async def run(self, request: ConstationRunRequest) -> ConstationRunRecord:
        self.run_calls += 1
        self.last_request = request
        if self.outcome.ok and self.poll_nonce_fn is not None:
            for _phase in self.phases_to_issue:
                self.poll_nonce_fn()
            self.last_signed_wire = dict(self.wire) if self.wire is not None else None
        else:
            self.last_signed_wire = None
        # Align binding fields with the request the orchestrator built.
        if self.outcome.ok:
            return _ok_record(
                work_unit_id=request.work_unit_id,
                miner_hotkey=request.miner_hotkey,
                pod_id=request.pod_id,
            )
        return self.outcome


def _binding_store(*, hotkey: str = HOTKEY, pod_id: str = POD) -> MinerPodBinding:
    custody = LiumKeyCustody(master_key=generate_custody_master_key())
    custody.store_probed_key(miner_hotkey=hotkey, api_key="lium-test-key-not-logged")
    binding = MinerPodBinding(custody=custody)
    binding._instance_by_hotkey[hotkey.strip()] = pod_id  # noqa: SLF001 — test seam
    return binding


def _request(**overrides: Any) -> ConstationOrchestrationRequest:
    base: dict[str, Any] = {
        "work_unit_id": WORK_UNIT,
        "miner_hotkey": HOTKEY,
        "required_digest": DIGEST,
        "commit_sha": COMMIT,
        "tree_sha": TREE,
        "variant": ImageVariant.CUDA,
        "sealed_manifest_hashes": dict(MANIFEST),
        "duration_seconds": 5.0,
    }
    base.update(overrides)
    return ConstationOrchestrationRequest(**base)


def _orchestrator(
    *,
    pod_binding: MinerPodBinding | None = None,
    nonce_service: AttestationNonceService | None = None,
    bundle_store: ConstationBundleStore | None = None,
    fake: _FakeRunner | None = None,
) -> tuple[
    ProductionConstationOrchestrator,
    AttestationNonceService,
    ConstationBundleStore,
    _FakeRunner,
]:
    binding = pod_binding if pod_binding is not None else _binding_store()
    nonces = (
        nonce_service
        if nonce_service is not None
        else AttestationNonceService(ttl=timedelta(hours=1), now_fn=lambda: T0)
    )
    store = bundle_store if bundle_store is not None else ConstationBundleStore()
    runner = fake if fake is not None else _FakeRunner()

    def runner_factory(**kwargs: Any) -> _FakeRunner:
        runner.poll_nonce_fn = kwargs.get("poll_nonce_fn")
        # Preserve other kwargs for API compatibility with real ConstationRunner
        return runner

    orch = ProductionConstationOrchestrator(
        pod_binding=binding,
        nonce_service=nonces,
        bundle_store=store,
        poller_config=PollerConfig(),
        now_fn=lambda: 0.0,
        sleep_fn=_async_noop_sleep,
        rng_fn=lambda: 0.0,
        runner_factory=runner_factory,
    )
    return orch, nonces, store, runner


async def _async_noop_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_happy_path_seals_and_puts_bundle() -> None:
    """S1: binding ok → issue nonces → runner ok → seal → bundle_store.put."""
    orch, nonces, store, runner = _orchestrator()

    result = await orch.run(_request())

    assert isinstance(result, ConstationOrchestrationResult)
    assert result.ok is True
    assert result.reason is ConstationFailCode.OK
    assert runner.run_calls == 1
    assert runner.last_request is not None
    assert runner.last_request.required_digest == DIGEST
    assert runner.last_request.pod_id == POD
    assert runner.last_request.work_unit_id == WORK_UNIT

    stored = store.get(WORK_UNIT)
    assert stored is not None
    assert stored["work_unit_id"] == WORK_UNIT
    assert stored["miner_hotkey"] == HOTKEY
    assert stored["pod_id"] == POD
    assert stored["digest"] == DIGEST
    assert stored["commit_sha"] == COMMIT
    assert stored["tree_sha"] == TREE
    assert stored["variant"] == "cuda"
    assert stored["nonce"] == result.end_phase_nonce
    assert result.end_phase_nonce is not None
    assert stored["signed_attestation"] == dict(WIRE)
    # Three poll phases each issued a nonce; end-phase is the last.
    assert len(nonces.snapshot().records) == 3
    assert result.end_phase_nonce == nonces.snapshot().records[-1].nonce


@pytest.mark.asyncio
async def test_runner_fail_does_not_put_bundle() -> None:
    """S2: runner.ok False → do NOT put bundle (prism missing_constation_bundle)."""
    fake = _FakeRunner(outcome=_fail_record())
    orch, _nonces, store, runner = _orchestrator(fake=fake)

    result = await orch.run(_request())

    assert result.ok is False
    assert result.reason is ConstationFailCode.SIDECAR_ATTEST_FAILED
    assert store.get(WORK_UNIT) is None
    assert result.bundle is None
    assert runner.run_calls == 1


@pytest.mark.asyncio
async def test_b2_end_phase_nonce_first_consumable_after_seal() -> None:
    """B2: AFTER seal, consume once → Hit; second → already_consumed.

    If orchestrator had consumed, first post-seal consume would miss.
    """
    orch, nonces, store, _runner = _orchestrator()

    result = await orch.run(_request())

    assert result.ok is True
    assert result.end_phase_nonce is not None
    binding = NonceBinding(work_unit_id=WORK_UNIT, miner_hotkey=HOTKEY, pod_id=POD)
    stored = store.get(WORK_UNIT)
    assert stored is not None
    assert stored["nonce"] == result.end_phase_nonce

    first = nonces.consume(result.end_phase_nonce, binding)
    assert isinstance(first, NonceConsumeHit), (
        f"B2 violated: end-phase nonce not first-consumable after seal "
        f"(got {first!r}); orchestrator must not consume"
    )

    second = nonces.consume(result.end_phase_nonce, binding)
    assert isinstance(second, NonceConsumeMiss)
    assert second.reason is NonceConsumeReason.ALREADY_CONSUMED


@pytest.mark.asyncio
async def test_b2_orchestrator_source_never_calls_consume() -> None:
    """Static guard: orchestrator module must not invoke nonce consume."""
    from base.master.constation import orchestrator as orch_mod

    src = inspect.getsource(orch_mod)
    # Allow the word only in comments/docstrings about NEVER consume — ban call form.
    assert ".consume(" not in src
    assert "nonce_service.consume" not in src


@pytest.mark.asyncio
async def test_missing_binding_fail_closed_no_put() -> None:
    """S4: custody/binding missing → fail closed, no runner, no put."""
    custody = LiumKeyCustody(master_key=generate_custody_master_key())
    empty = MinerPodBinding(custody=custody)
    store = ConstationBundleStore()
    fake = _FakeRunner()
    orch, _n, store, runner = _orchestrator(
        pod_binding=empty, bundle_store=store, fake=fake
    )

    result = await orch.run(_request())

    assert result.ok is False
    assert result.reason is ConstationFailCode.KEY_NOT_REGISTERED
    assert store.get(WORK_UNIT) is None
    assert runner.run_calls == 0


@pytest.mark.asyncio
async def test_request_pod_id_override_and_instance_id_alias() -> None:
    """DTO accepts pod_id or instance_id; override beats binding lookup."""
    orch, _n, store, runner = _orchestrator()
    other_pod = "pod-override-99"

    result = await orch.run(_request(pod_id=other_pod))

    assert result.ok is True
    assert runner.last_request is not None
    assert runner.last_request.pod_id == other_pod
    stored = store.get(WORK_UNIT)
    assert stored is not None
    assert stored["pod_id"] == other_pod

    # instance_id alias when pod_id omitted
    fake2 = _FakeRunner()
    orch2, _n2, store2, runner2 = _orchestrator(fake=fake2)
    result2 = await orch2.run(_request(pod_id=None, instance_id="pod-via-instance"))
    assert result2.ok is True
    assert runner2.last_request is not None
    assert runner2.last_request.pod_id == "pod-via-instance"


@pytest.mark.asyncio
async def test_missing_signed_wire_fail_closed_no_put() -> None:
    """Fail-closed: runner ok but no last_signed_wire → no bundle put."""
    fake = _FakeRunner(wire=None)
    # Force ok path without wire: custom run
    original_run = fake.run

    async def run_ok_no_wire(request: ConstationRunRequest) -> ConstationRunRecord:
        fake.run_calls += 1
        fake.last_request = request
        if fake.poll_nonce_fn is not None:
            fake.poll_nonce_fn()
        fake.last_signed_wire = None
        return _ok_record()

    fake.run = run_ok_no_wire  # type: ignore[method-assign]
    del original_run

    orch, _n, store, _r = _orchestrator(fake=fake)
    result = await orch.run(_request())

    assert result.ok is False
    assert store.get(WORK_UNIT) is None


def test_orchestration_request_is_explicit_dto_no_hidden_globals() -> None:
    """ConstationOrchestrationRequest exposes all required fields explicitly."""
    req = _request(instance_id="pod-x", duration_seconds=12.5)
    assert req.work_unit_id == WORK_UNIT
    assert req.miner_hotkey == HOTKEY
    assert req.required_digest == DIGEST
    assert req.commit_sha == COMMIT
    assert req.tree_sha == TREE
    assert req.variant == ImageVariant.CUDA
    assert dict(req.sealed_manifest_hashes) == MANIFEST
    assert req.duration_seconds == 12.5
    assert req.instance_id == "pod-x"


@dataclass
class _AsyncNonceService:
    """Durable-shaped async issuer for B2 / async issue path."""

    inner: AttestationNonceService

    async def issue(self, binding: NonceBinding) -> Any:
        return self.inner.issue(binding)

    async def consume(self, nonce: str, binding: NonceBinding) -> Any:
        return self.inner.consume(nonce, binding)


@pytest.mark.asyncio
async def test_b2_async_nonce_service_end_phase_still_first_consumable() -> None:
    """Async durable-shaped issuer: still issue-only; consume Hit after seal."""
    inner = AttestationNonceService(ttl=timedelta(hours=1), now_fn=lambda: T0)
    async_svc = _AsyncNonceService(inner=inner)
    orch, _n, store, _r = _orchestrator(nonce_service=async_svc)  # type: ignore[arg-type]

    result = await orch.run(_request())

    assert result.ok is True
    assert result.end_phase_nonce is not None
    assert store.get(WORK_UNIT) is not None
    binding = NonceBinding(work_unit_id=WORK_UNIT, miner_hotkey=HOTKEY, pod_id=POD)
    first = await async_svc.consume(result.end_phase_nonce, binding)
    assert isinstance(first, NonceConsumeHit)
    second = await async_svc.consume(result.end_phase_nonce, binding)
    assert isinstance(second, NonceConsumeMiss)
    assert second.reason is NonceConsumeReason.ALREADY_CONSUMED
