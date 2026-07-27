"""Constation runner — master/worker-plane service entry (todos 15–17).

Orchestrates encrypted key custody, continuous polling, Lium declared-digest
reads, sidecar attestation, pod bind checks, and three-way digest triangle
(required / Lium-declared / sidecar-actual). Same-account corroboration is
subsumed by the triangle (negative only).

Callers
-------
Master and worker-plane code construct :class:`ConstationRunner` with a shared
:class:`~base.compute.constation_custody.LiumKeyCustody` and invoke
:meth:`ConstationRunner.run` per submission. This is **not** miner-CLI-only.

Outputs a :class:`~base.compute.constation_types.ConstationRunRecord` whose
fields map onto prism ``ConstationBundle`` gap / digest / corroboration inputs.
Agreement on the triangle never elevates by itself.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from base.compute.constation_custody import LiumKeyCustody
from base.compute.constation_pod import assert_pod_bound
from base.compute.constation_poller import ContinuousConstationPoller, PollerConfig
from base.compute.constation_sidecar_client import (
    DEFAULT_TIMEOUT_SECONDS,
    HttpSidecarAttestor,
    SidecarAttestHit,
    SidecarAttestMiss,
)
from base.compute.constation_sidecar_endpoint import (
    SidecarEndpointMiss,
    SidecarEndpointMissReason,
    resolve_sidecar_base_url,
)
from base.compute.constation_triangle import evaluate_digest_triangle
from base.compute.constation_types import (
    ConstationFailCode,
    ConstationRunRecord,
    ConstationVerdict,
    CorroborationStatus,
    FaultClass,
    PollSample,
    fault_class_for,
)
from base.compute.lium import (
    LiumAuthError,
    LiumClient,
    LiumError,
    LiumPodRead,
    LiumRateLimitError,
)

logger = logging.getLogger(__name__)


class SidecarAttestor(Protocol):
    """Fetch sidecar-reported digest / attestation for one poll."""

    async def attest(self, *, pod_id: str, phase: str) -> str:
        """Return sidecar-reported image digest (``sha256:...``)."""
        ...


@dataclass(frozen=True, slots=True)
class ConstationRunRequest:
    """One submission's continuous constation parameters."""

    miner_hotkey: str
    work_unit_id: str
    pod_id: str
    duration_seconds: float
    required_digest: str
    # Back-compat alias accepted by older call sites via keyword only if needed.
    # Prefer ``required_digest`` (allowlist / BASE required image digest).


NowFn = Callable[[], float]
SleepFn = Callable[[float], Awaitable[None]]
RngFn = Callable[[], float]
AttestorFactory = Callable[[Mapping[str, Any]], object]
NonceFn = Callable[[], str]


def _default_poll_nonce() -> str:
    """Ephemeral poll nonce — not durable-consumed (seal nonce is separate)."""
    return secrets.token_hex(16)


@dataclass
class ConstationRunner:
    """Service entry: custody + poller + triangle → run record."""

    custody: LiumKeyCustody
    sidecar: SidecarAttestor
    poller_config: PollerConfig
    now_fn: NowFn
    sleep_fn: SleepFn
    rng_fn: RngFn
    attestor_factory: AttestorFactory | None = None
    sidecar_internal_port: int | None = None
    sidecar_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    poll_nonce_fn: NonceFn | None = None
    last_signed_wire: Mapping[str, Any] | None = field(default=None, init=False)

    async def run(self, request: ConstationRunRequest) -> ConstationRunRecord:
        """Execute continuous constation for ``request``; fail closed on faults."""
        self.last_signed_wire = None
        hotkey = request.miner_hotkey.strip()
        if not self.custody.has_key(hotkey):
            return _record(
                request,
                ok=False,
                reason=ConstationFailCode.KEY_NOT_REGISTERED,
                gap_budget=self.poller_config.gap_budget_seconds,
                observed_gap=0.0,
                sidecar=None,
                lium=None,
                status=CorroborationStatus.NOT_EVALUATED,
                samples=(),
            )

        try:
            client = self.custody.build_client(hotkey)
        except (KeyError, ValueError) as exc:
            return _record(
                request,
                ok=False,
                reason=ConstationFailCode.KEY_NOT_REGISTERED,
                gap_budget=self.poller_config.gap_budget_seconds,
                observed_gap=0.0,
                sidecar=None,
                lium=None,
                status=CorroborationStatus.NOT_EVALUATED,
                samples=(),
                detail=type(exc).__name__,
            )

        poller = ContinuousConstationPoller(
            config=self.poller_config,
            now_fn=self.now_fn,
            sleep_fn=self.sleep_fn,
            rng_fn=self.rng_fn,
        )

        async def poll_once(phase: str) -> PollSample | ConstationVerdict:
            return await self._poll_once(client, request, phase)

        result = await poller.run(
            duration_seconds=request.duration_seconds,
            poll_once=poll_once,
        )

        if not result.ok:
            last_side = result.samples[-1].sidecar_digest if result.samples else None
            last_lium = (
                result.samples[-1].lium_declared_digest if result.samples else None
            )
            status = _status_for_fail(result.reason)
            return _record(
                request,
                ok=False,
                reason=result.reason,
                gap_budget=result.gap_budget_seconds,
                observed_gap=result.observed_max_gap_seconds,
                sidecar=last_side,
                lium=last_lium,
                status=status,
                samples=result.samples,
                detail=result.detail,
                fault_class=result.fault_class,
            )

        if not result.samples:
            return _record(
                request,
                ok=False,
                reason=ConstationFailCode.RUN_INCOMPLETE,
                gap_budget=result.gap_budget_seconds,
                observed_gap=result.observed_max_gap_seconds,
                sidecar=None,
                lium=None,
                status=CorroborationStatus.NOT_EVALUATED,
                samples=(),
            )

        # Final triangle across every sample (fail-closed; triangle > two-way).
        for sample in result.samples:
            tri = evaluate_digest_triangle(
                required=request.required_digest,
                lium_declared=sample.lium_declared_digest,
                sidecar=sample.sidecar_digest,
            )
            if not tri.ok:
                fail = tri.fail_code or ConstationFailCode.CORROBORATION_MISMATCH
                return _record(
                    request,
                    ok=False,
                    reason=fail,
                    gap_budget=result.gap_budget_seconds,
                    observed_gap=result.observed_max_gap_seconds,
                    sidecar=sample.sidecar_digest,
                    lium=sample.lium_declared_digest,
                    status=_status_for_fail(fail),
                    samples=result.samples,
                )

        last = result.samples[-1]
        return _record(
            request,
            ok=True,
            reason=ConstationFailCode.OK,
            gap_budget=result.gap_budget_seconds,
            observed_gap=result.observed_max_gap_seconds,
            sidecar=last.sidecar_digest,
            lium=last.lium_declared_digest,
            status=CorroborationStatus.AGREE,
            samples=result.samples,
        )

    async def _poll_once(
        self,
        client: LiumClient,
        request: ConstationRunRequest,
        phase: str,
    ) -> PollSample | ConstationVerdict:
        try:
            pod = await client.get_pod_raw(request.pod_id)
        except LiumAuthError:
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.LIUM_AUTH_REVOKED,
                detail=f"phase={phase}",
            )
        except LiumRateLimitError:
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.LIUM_RATE_LIMITED,
                detail=f"phase={phase}",
            )
        except LiumError:
            # Re-raise so poller applies bounded network retry
            raise

        bound = assert_pod_bound(
            pod_raw=pod.raw,
            expected_hotkey=request.miner_hotkey,
        )
        if not bound.ok:
            return ConstationVerdict(
                ok=False,
                reason=bound.reason,
                detail=f"phase={phase}",
                fault_class=bound.fault_class,
            )

        obtained = await self._obtain_sidecar_digest(
            pod=pod, request=request, phase=phase
        )
        if isinstance(obtained, ConstationVerdict):
            return obtained
        sidecar_digest, wire = obtained
        if wire is not None:
            self.last_signed_wire = wire

        tri = evaluate_digest_triangle(
            required=request.required_digest,
            lium_declared=pod.docker_image_digest,
            sidecar=sidecar_digest,
        )
        if not tri.ok:
            fail = tri.fail_code or ConstationFailCode.CORROBORATION_MISMATCH
            return ConstationVerdict(
                ok=False,
                reason=fail,
                detail=f"phase={phase}",
            )

        return PollSample(
            at_monotonic=self.now_fn(),
            phase=phase,
            sidecar_digest=sidecar_digest,
            lium_declared_digest=pod.docker_image_digest,
        )

    async def _obtain_sidecar_digest(
        self,
        *,
        pod: LiumPodRead,
        request: ConstationRunRequest,
        phase: str,
    ) -> tuple[str, Mapping[str, Any] | None] | ConstationVerdict:
        """Resolve attestor (factory / HTTP dial / injected) and fetch digest."""
        attestor: object
        if self.attestor_factory is not None:
            try:
                attestor = self.attestor_factory(pod.raw)
            except Exception as exc:
                logger.warning(
                    "attestor_factory failed pod=%s phase=%s err=%s",
                    request.pod_id,
                    phase,
                    type(exc).__name__,
                )
                return ConstationVerdict(
                    ok=False,
                    reason=ConstationFailCode.SIDECAR_ATTEST_FAILED,
                    detail=f"phase={phase} err={type(exc).__name__}",
                )
        elif self.sidecar_internal_port is not None:
            resolved = resolve_sidecar_base_url(
                pod.raw,
                internal_port=self.sidecar_internal_port,
            )
            if isinstance(resolved, SidecarEndpointMiss):
                return ConstationVerdict(
                    ok=False,
                    reason=_endpoint_miss_to_fail(resolved.reason),
                    detail=f"phase={phase} endpoint={resolved.reason.value}",
                )
            attestor = HttpSidecarAttestor(
                base_url=resolved.base_url,
                timeout_seconds=self.sidecar_timeout_seconds,
            )
        else:
            attestor = self.sidecar

        return await self._call_attestor(
            attestor,
            pod_id=request.pod_id,
            phase=phase,
        )

    async def _call_attestor(
        self,
        attestor: object,
        *,
        pod_id: str,
        phase: str,
    ) -> tuple[str, Mapping[str, Any] | None] | ConstationVerdict:
        try:
            if isinstance(attestor, HttpSidecarAttestor):
                nonce_fn = self.poll_nonce_fn or _default_poll_nonce
                result = await attestor.attest(nonce=nonce_fn(), phase=phase)
                if isinstance(result, SidecarAttestMiss):
                    return ConstationVerdict(
                        ok=False,
                        reason=result.reason,
                        detail=result.detail or f"phase={phase}",
                    )
                if isinstance(result, SidecarAttestHit):
                    return result.digest, result.wire
                return ConstationVerdict(
                    ok=False,
                    reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
                    detail=f"phase={phase}",
                )

            digest = await cast(SidecarAttestor, attestor).attest(
                pod_id=pod_id, phase=phase
            )
        except Exception as exc:
            logger.warning(
                "sidecar attest failed pod=%s phase=%s err=%s",
                pod_id,
                phase,
                type(exc).__name__,
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.SIDECAR_ATTEST_FAILED,
                detail=f"phase={phase} err={type(exc).__name__}",
            )

        if not isinstance(digest, str) or not digest.strip():
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.SIDECAR_RESPONSE_INVALID,
                detail=f"phase={phase}",
            )
        return digest, None


def _endpoint_miss_to_fail(reason: SidecarEndpointMissReason) -> ConstationFailCode:
    if reason is SidecarEndpointMissReason.SIDECAR_PORT_UNPUBLISHED:
        return ConstationFailCode.SIDECAR_PORT_UNPUBLISHED
    return ConstationFailCode.SIDECAR_ATTEST_FAILED


def _status_for_fail(reason: ConstationFailCode) -> CorroborationStatus:
    if reason is ConstationFailCode.CORROBORATION_MISMATCH:
        return CorroborationStatus.MISMATCH
    return CorroborationStatus.NOT_EVALUATED


def _record(
    request: ConstationRunRequest,
    *,
    ok: bool,
    reason: ConstationFailCode,
    gap_budget: float,
    observed_gap: float,
    sidecar: str | None,
    lium: str | None,
    status: CorroborationStatus,
    samples: tuple[PollSample, ...],
    detail: str | None = None,
    fault_class: FaultClass | None = None,
) -> ConstationRunRecord:
    return ConstationRunRecord(
        ok=ok,
        reason=reason,
        fault_class=fault_class if fault_class is not None else fault_class_for(reason),
        miner_hotkey=request.miner_hotkey.strip(),
        work_unit_id=request.work_unit_id.strip(),
        pod_id=request.pod_id.strip(),
        sidecar_digest=sidecar,
        lium_declared_digest=lium,
        constation_gap_budget_seconds=gap_budget,
        constation_observed_max_gap_seconds=observed_gap,
        corroboration_status=status,
        samples=samples,
        detail=detail,
    )


__all__ = [
    "AttestorFactory",
    "ConstationRunRequest",
    "ConstationRunner",
    "SidecarAttestor",
]
