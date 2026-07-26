"""Constation runner — master/worker-plane service entry (todos 15–17).

Orchestrates encrypted key custody, continuous polling, Lium declared-digest
reads, sidecar attestation, and same-account **corroboration** (negative only).

Callers
-------
Master and worker-plane code construct :class:`ConstationRunner` with a shared
:class:`~base.compute.constation_custody.LiumKeyCustody` and invoke
:meth:`ConstationRunner.run` per submission. This is **not** miner-CLI-only.

Outputs a :class:`~base.compute.constation_types.ConstationRunRecord` whose
fields map onto prism ``ConstationBundle`` gap / digest / corroboration inputs.
Agreement on corroboration never elevates by itself.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from base.compute.constation_corroboration import evaluate_corroboration
from base.compute.constation_custody import LiumKeyCustody
from base.compute.constation_poller import ContinuousConstationPoller, PollerConfig
from base.compute.constation_types import (
    ConstationFailCode,
    ConstationRunRecord,
    ConstationVerdict,
    CorroborationStatus,
    FaultClass,
    PollSample,
    fault_class_for,
)
from base.compute.lium import LiumAuthError, LiumClient, LiumError, LiumRateLimitError

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
    expected_sidecar_digest: str | None = None


NowFn = Callable[[], float]
SleepFn = Callable[[float], Awaitable[None]]
RngFn = Callable[[], float]


@dataclass
class ConstationRunner:
    """Service entry: custody + poller + corroboration → run record."""

    custody: LiumKeyCustody
    sidecar: SidecarAttestor
    poller_config: PollerConfig
    now_fn: NowFn
    sleep_fn: SleepFn
    rng_fn: RngFn

    async def run(self, request: ConstationRunRequest) -> ConstationRunRecord:
        """Execute continuous constation for ``request``; fail closed on faults."""
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
            status = CorroborationStatus.NOT_EVALUATED
            if result.reason is ConstationFailCode.CORROBORATION_MISMATCH:
                status = CorroborationStatus.MISMATCH
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

        # Final corroboration across last sample (and ensure no sample mismatched)
        for sample in result.samples:
            corr = evaluate_corroboration(
                lium_declared_digest=sample.lium_declared_digest,
                sidecar_digest=sample.sidecar_digest,
            )
            if not corr.ok:
                return _record(
                    request,
                    ok=False,
                    reason=ConstationFailCode.CORROBORATION_MISMATCH,
                    gap_budget=result.gap_budget_seconds,
                    observed_gap=result.observed_max_gap_seconds,
                    sidecar=corr.sidecar_digest,
                    lium=corr.lium_declared_digest,
                    status=CorroborationStatus.MISMATCH,
                    samples=result.samples,
                    detail=corr.verdict.detail,
                )

        last = result.samples[-1]
        corr = evaluate_corroboration(
            lium_declared_digest=last.lium_declared_digest,
            sidecar_digest=last.sidecar_digest,
        )
        return _record(
            request,
            ok=True,
            reason=ConstationFailCode.OK,
            gap_budget=result.gap_budget_seconds,
            observed_gap=result.observed_max_gap_seconds,
            sidecar=corr.sidecar_digest,
            lium=corr.lium_declared_digest,
            status=corr.status,
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

        try:
            sidecar_digest = await self.sidecar.attest(
                pod_id=request.pod_id, phase=phase
            )
        except Exception as exc:
            logger.warning(
                "sidecar attest failed pod=%s phase=%s err=%s",
                request.pod_id,
                phase,
                type(exc).__name__,
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.SIDECAR_ATTEST_FAILED,
                detail=f"phase={phase} err={type(exc).__name__}",
            )

        lium_digest = pod.docker_image_digest
        corr = evaluate_corroboration(
            lium_declared_digest=lium_digest,
            sidecar_digest=sidecar_digest,
        )
        if not corr.ok:
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.CORROBORATION_MISMATCH,
                detail=corr.verdict.detail,
            )

        return PollSample(
            at_monotonic=self.now_fn(),
            phase=phase,
            sidecar_digest=corr.sidecar_digest,
            lium_declared_digest=corr.lium_declared_digest,
        )


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
    "ConstationRunRequest",
    "ConstationRunner",
    "SidecarAttestor",
]
