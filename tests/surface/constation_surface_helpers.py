"""Shared fixtures/helpers for production constation surface tests (T12)."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from base.compute.attestation_nonce import AttestationNonceService
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
    ProductionConstationOrchestrator,
)
from base.master.constation.pod_binding import MinerPodBinding

COMMIT = "a" * 40
TREE = "b" * 40
DIGEST = "sha256:" + ("c" * 64)
DIGEST_BAD = "sha256:" + ("d" * 64)
MANIFEST = {"src/harness.py": "e" * 64}
HOTKEY = "5MinerSurfaceTestHotkey000000000000001"
POD = "pod-surface-001"
WORK_UNIT = "wu-surface-001"
T0 = datetime(2026, 7, 27, 5, 0, 0, tzinfo=UTC)
TOKEN = "surface-internal"
BUILD_SECRET = b"surface-build-secret-fixture"

WIRE: dict[str, Any] = {
    "payload": {
        "digest": DIGEST,
        "nonce": "will-be-overwritten",
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRISM_SRC = _REPO_ROOT / "packages" / "challenges" / "prism" / "src"


def ensure_prism_on_path() -> bool:
    """Make prism_challenge importable without a separate uv env."""
    if not _PRISM_SRC.is_dir():
        return False
    inserted = str(_PRISM_SRC)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    try:
        import prism_challenge  # noqa: F401

        return True
    except ImportError:
        return False


def ok_record(
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


def fail_record(
    *,
    reason: ConstationFailCode = ConstationFailCode.REQUIRED_DIGEST_MISMATCH,
) -> ConstationRunRecord:
    return ConstationRunRecord(
        ok=False,
        reason=reason,
        fault_class=FaultClass.MINER,
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
class FakeRunner:
    """Stand-in ConstationRunner for surface composition."""

    poll_nonce_fn: Callable[[], str] | None = None
    last_signed_wire: Mapping[str, Any] | None = field(default=None, init=False)
    outcome: ConstationRunRecord = field(default_factory=ok_record)
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
        if self.outcome.ok:
            return ok_record(
                work_unit_id=request.work_unit_id,
                miner_hotkey=request.miner_hotkey,
                pod_id=request.pod_id,
            )
        return self.outcome


def binding_store(*, hotkey: str = HOTKEY, pod_id: str = POD) -> MinerPodBinding:
    custody = LiumKeyCustody(master_key=generate_custody_master_key())
    custody.store_probed_key(miner_hotkey=hotkey, api_key="lium-surface-key-not-logged")
    binding = MinerPodBinding(custody=custody)
    binding._instance_by_hotkey[hotkey.strip()] = pod_id  # noqa: SLF001 — test seam
    return binding


def orchestration_request(**overrides: Any) -> ConstationOrchestrationRequest:
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


async def async_noop_sleep(_seconds: float) -> None:
    return None


def make_orchestrator(
    *,
    pod_binding: MinerPodBinding | None = None,
    nonce_service: AttestationNonceService | None = None,
    bundle_store: ConstationBundleStore | None = None,
    fake: FakeRunner | None = None,
) -> tuple[
    ProductionConstationOrchestrator,
    AttestationNonceService,
    ConstationBundleStore,
    FakeRunner,
]:
    binding = pod_binding if pod_binding is not None else binding_store()
    nonces = (
        nonce_service
        if nonce_service is not None
        else AttestationNonceService(ttl=timedelta(hours=1), now_fn=lambda: T0)
    )
    store = bundle_store if bundle_store is not None else ConstationBundleStore()
    runner = fake if fake is not None else FakeRunner()

    def runner_factory(**kwargs: Any) -> FakeRunner:
        runner.poll_nonce_fn = kwargs.get("poll_nonce_fn")
        return runner

    orch = ProductionConstationOrchestrator(
        pod_binding=binding,
        nonce_service=nonces,
        bundle_store=store,
        poller_config=PollerConfig(),
        now_fn=lambda: 0.0,
        sleep_fn=async_noop_sleep,
        rng_fn=lambda: 0.0,
        runner_factory=runner_factory,
    )
    return orch, nonces, store, runner


@contextmanager
def preserved_logging_levels() -> Iterator[None]:
    """Restore logger levels after bittensor import side effects."""
    manager = logging.root.manager
    prev_disable = manager.disable
    prev_root_level = logging.getLogger().level
    prev_levels = [
        (obj, obj.level, obj.disabled)
        for obj in list(manager.loggerDict.values())
        if isinstance(obj, logging.Logger)
    ]
    try:
        yield
    finally:
        for logger, level, disabled in prev_levels:
            logger.setLevel(level)
            logger.disabled = disabled
        logging.getLogger().setLevel(prev_root_level)
        manager.disable = prev_disable


class ChallengeRegistry:
    async def get(self, slug: str) -> Any:
        class R:
            internal_base_url = "http://prism.surface.test"

        return R()

    async def get_token(self, slug: str) -> str:
        return "tok"


class CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json

        self.bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"status": "accepted"})


def minimal_proof() -> dict[str, Any]:
    with preserved_logging_levels():
        import bittensor as bt

        from base.validator.agent.signing import KeypairRequestSigner
        from base.worker.proof import build_execution_proof

        signer = KeypairRequestSigner(bt.Keypair.create_from_uri("//WorkerSurface"))
        proof = build_execution_proof(
            signer=signer, manifest_sha256="a" * 64, unit_id=WORK_UNIT
        )
    return proof.model_dump(mode="json")
