"""Production constation orchestrator: issue → run → seal → put (never consume).

B2
--
Nonces are single-use and prism consumes at ingest. This orchestrator **issues**
nonces for each poll (via ``poll_nonce_fn``) and seals the end-phase nonce into
the bundle. It **never** calls ``consume`` — the end-phase nonce must remain
first-consumable after ``bundle_store.put``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from base.compute.attestation_nonce import NonceBinding, NonceRecord
from base.compute.constation_poller import PollerConfig
from base.compute.constation_runner import (
    AttestorFactory,
    ConstationRunner,
    ConstationRunRequest,
    NowFn,
    RngFn,
    SidecarAttestor,
    SleepFn,
)
from base.compute.constation_sidecar_client import DEFAULT_TIMEOUT_SECONDS
from base.compute.constation_types import (
    ConstationFailCode,
    ConstationRunRecord,
)
from base.compute.digest_allowlist import DigestRecord, ImageVariant
from base.master.constation.bundle_seal import seal_constation_bundle
from base.master.constation.bundle_store import ConstationBundleStore
from base.master.constation.pod_binding import MinerPodBinding

logger = logging.getLogger(__name__)

RunnerFactory = Callable[..., Any]


class NonceIssuer(Protocol):
    """Sync or async nonce issuer (in-memory or durable). Never consume here."""

    def issue(self, binding: NonceBinding) -> NonceRecord | Awaitable[NonceRecord]: ...


@dataclass(frozen=True, slots=True)
class ConstationOrchestrationRequest:
    """Explicit inputs for one production constation run (no hidden globals)."""

    work_unit_id: str
    miner_hotkey: str
    required_digest: str
    commit_sha: str
    tree_sha: str
    variant: ImageVariant | str
    sealed_manifest_hashes: Mapping[str, str]
    duration_seconds: float
    pod_id: str | None = None
    instance_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConstationOrchestrationResult:
    """Outcome of :meth:`ProductionConstationOrchestrator.run`."""

    ok: bool
    reason: ConstationFailCode
    run_record: ConstationRunRecord | None
    bundle: dict[str, object] | None
    end_phase_nonce: str | None


@dataclass
class ProductionConstationOrchestrator:
    """Issue nonces → ConstationRunner → seal_constation_bundle → bundle_store.put.

    Caller gates on settings (constation enabled). This class is fail-closed and
    unit-testable via ``runner_factory`` and fakes for nonce/binding/store.
    """

    pod_binding: MinerPodBinding
    nonce_service: NonceIssuer
    bundle_store: ConstationBundleStore
    poller_config: PollerConfig
    now_fn: NowFn
    sleep_fn: SleepFn
    rng_fn: RngFn
    sidecar: SidecarAttestor | None = None
    attestor_factory: AttestorFactory | None = None
    sidecar_internal_port: int | None = None
    sidecar_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    runner_factory: RunnerFactory | None = None

    async def run(
        self, request: ConstationOrchestrationRequest
    ) -> ConstationOrchestrationResult:
        """Execute one constation orchestration; never consume nonces."""
        hotkey = request.miner_hotkey.strip()
        work_unit_id = request.work_unit_id.strip()

        resolved = self._resolve_pod_id(request, hotkey)
        if resolved is None:
            logger.warning(
                "constation orchestrator: no binding for miner_hotkey=%s",
                hotkey,
            )
            return ConstationOrchestrationResult(
                ok=False,
                reason=ConstationFailCode.KEY_NOT_REGISTERED,
                run_record=None,
                bundle=None,
                end_phase_nonce=None,
            )
        pod_id = resolved

        if not self.pod_binding.custody.has_key(hotkey):
            return ConstationOrchestrationResult(
                ok=False,
                reason=ConstationFailCode.KEY_NOT_REGISTERED,
                run_record=None,
                bundle=None,
                end_phase_nonce=None,
            )

        binding = NonceBinding(
            work_unit_id=work_unit_id,
            miner_hotkey=hotkey,
            pod_id=pod_id,
        )
        issued: list[str] = []

        def poll_nonce_fn() -> str:
            """Issue-only poll nonce for ConstationRunner (sync NonceFn).

            Supports sync ``AttestationNonceService.issue`` and async
            ``DurableAttestationNonceService.issue`` (via a worker-thread loop
            so we never deadlock the running event loop). Never consumes.
            """
            record = _issue_blocking(self.nonce_service, binding)
            issued.append(record.nonce)
            return record.nonce

        runner = self._build_runner(poll_nonce_fn=poll_nonce_fn)
        run_req = ConstationRunRequest(
            miner_hotkey=hotkey,
            work_unit_id=work_unit_id,
            pod_id=pod_id,
            duration_seconds=request.duration_seconds,
            required_digest=request.required_digest.strip(),
        )
        run_record = await runner.run(run_req)

        if not run_record.ok:
            return ConstationOrchestrationResult(
                ok=False,
                reason=run_record.reason,
                run_record=run_record,
                bundle=None,
                end_phase_nonce=None,
            )

        end_nonce = issued[-1] if issued else None
        if end_nonce is None:
            # Runner paths that never dial HttpSidecarAttestor still need an
            # end-phase nonce for the sealed bundle (issue-only).
            try:
                end_record = await _maybe_await_issue(self.nonce_service.issue(binding))
            except Exception as exc:
                logger.warning(
                    "constation orchestrator: end-phase issue failed wu=%s err=%s",
                    work_unit_id,
                    type(exc).__name__,
                )
                return ConstationOrchestrationResult(
                    ok=False,
                    reason=ConstationFailCode.RUN_INCOMPLETE,
                    run_record=run_record,
                    bundle=None,
                    end_phase_nonce=None,
                )
            end_nonce = end_record.nonce
            issued.append(end_nonce)

        wire = getattr(runner, "last_signed_wire", None)
        if wire is None:
            logger.warning(
                "constation orchestrator: missing last_signed_wire wu=%s",
                work_unit_id,
            )
            return ConstationOrchestrationResult(
                ok=False,
                reason=ConstationFailCode.RUN_INCOMPLETE,
                run_record=run_record,
                bundle=None,
                end_phase_nonce=None,
            )

        try:
            variant = (
                request.variant
                if isinstance(request.variant, ImageVariant)
                else ImageVariant(str(request.variant).strip().lower())
            )
            allowlist_record = DigestRecord(
                commit_sha=request.commit_sha,
                tree_sha=request.tree_sha,
                variant=variant,
                digest=request.required_digest,
                sealed_manifest_hashes=dict(request.sealed_manifest_hashes),
            )
            bundle = seal_constation_bundle(
                allowlist_record=allowlist_record,
                run_record=run_record,
                nonce=end_nonce,
                signed_attestation=wire,
            )
        except ValueError as exc:
            logger.warning(
                "constation orchestrator: seal failed wu=%s err=%s",
                work_unit_id,
                exc,
            )
            return ConstationOrchestrationResult(
                ok=False,
                reason=ConstationFailCode.RUN_INCOMPLETE,
                run_record=run_record,
                bundle=None,
                end_phase_nonce=None,
            )

        self.bundle_store.put(work_unit_id, bundle)
        return ConstationOrchestrationResult(
            ok=True,
            reason=ConstationFailCode.OK,
            run_record=run_record,
            bundle=bundle,
            end_phase_nonce=end_nonce,
        )

    def _resolve_pod_id(
        self,
        request: ConstationOrchestrationRequest,
        hotkey: str,
    ) -> str | None:
        if request.pod_id is not None and str(request.pod_id).strip():
            return str(request.pod_id).strip()
        if request.instance_id is not None and str(request.instance_id).strip():
            return str(request.instance_id).strip()
        if not self.pod_binding.has_binding(hotkey):
            return None
        instance = self.pod_binding.get_instance_id(hotkey)
        if instance is None or not instance.strip():
            return None
        return instance.strip()

    def _build_runner(self, *, poll_nonce_fn: Callable[[], str]) -> Any:
        factory = self.runner_factory
        if factory is not None:
            return factory(
                custody=self.pod_binding.custody,
                sidecar=self.sidecar if self.sidecar is not None else _NullSidecar(),
                poller_config=self.poller_config,
                now_fn=self.now_fn,
                sleep_fn=self.sleep_fn,
                rng_fn=self.rng_fn,
                attestor_factory=self.attestor_factory,
                sidecar_internal_port=self.sidecar_internal_port,
                sidecar_timeout_seconds=self.sidecar_timeout_seconds,
                poll_nonce_fn=poll_nonce_fn,
            )
        sidecar: SidecarAttestor
        if self.sidecar is not None:
            sidecar = self.sidecar
        else:
            sidecar = _NullSidecar()
        return ConstationRunner(
            custody=self.pod_binding.custody,
            sidecar=sidecar,
            poller_config=self.poller_config,
            now_fn=self.now_fn,
            sleep_fn=self.sleep_fn,
            rng_fn=self.rng_fn,
            attestor_factory=self.attestor_factory,
            sidecar_internal_port=self.sidecar_internal_port,
            sidecar_timeout_seconds=self.sidecar_timeout_seconds,
            poll_nonce_fn=poll_nonce_fn,
        )


@dataclass
class _NullSidecar:
    """Placeholder when factory/port supplies the real attestor."""

    async def attest(self, *, pod_id: str, phase: str) -> str:
        del pod_id, phase
        raise RuntimeError("sidecar not configured")


async def _maybe_await_issue(
    value: NonceRecord | Awaitable[NonceRecord],
) -> NonceRecord:
    if inspect.isawaitable(value):
        return await value
    return value


def _issue_blocking(nonce_service: NonceIssuer, binding: NonceBinding) -> NonceRecord:
    """Call ``issue`` from a sync context (runner poll_nonce_fn)."""
    issue = nonce_service.issue
    # Coroutine function (DurableAttestationNonceService.issue)
    if inspect.iscoroutinefunction(issue):
        return _run_coro_in_worker(issue(binding))
    record = issue(binding)
    if inspect.isawaitable(record):
        return _run_coro_in_worker(record)
    return record


def _run_coro_in_worker(coro: Awaitable[NonceRecord]) -> NonceRecord:
    """Run ``coro`` on a fresh loop in a worker thread (no same-loop deadlock)."""
    import concurrent.futures

    async def _await_once() -> NonceRecord:
        return await coro

    def _thread_main() -> NonceRecord:
        return asyncio.run(_await_once())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_thread_main).result(timeout=60.0)


__all__ = [
    "ConstationOrchestrationRequest",
    "ConstationOrchestrationResult",
    "NonceIssuer",
    "ProductionConstationOrchestrator",
]
