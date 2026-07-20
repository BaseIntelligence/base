"""Tier verification + probabilistic audit sampling (architecture.md 3.4/3.5).

The proof tier a worker CLAIMS is an input to the audit rate, so it must never be trusted verbatim:
a claim is only worth its tier if the backing metadata is verifiable. :func:`effective_tier` maps a
proof's CLAIMED tier onto the EFFECTIVE tier the audit scheduler actually consumes:

* claimed tier >= 2 -> effective 0 always (Prism no longer elevates via TEE; claim/attestation
  fields may remain on the wire for compatibility but never authorize tier 2);
* claimed tier 1 -> effective 1 iff the proof's ``image_digest`` equals the configured pinned
  evaluator/worker digest AND provider pod binding is present, else effective 0 (IMAGE_PIN);
* claimed tier 0 (or any unknown tier) -> effective 0.

Max effective tier is **1**. :class:`AuditSampler` then samples finalized results at the per-tier
rate of their EFFECTIVE tier. Sampling is deterministic in the sampler's ``seed`` and the
per-result key, so the same seed reproduces the same sample set (VAL-PRISM-011) and a downgraded
claim is audited at its lower (effective) rate rather than the rate it dishonestly claimed
(VAL-PRISM-019).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from .config import WorkerPlaneConfig
from .proof import ExecutionProof

#: The three proof tiers the audit rate is keyed on (architecture.md 3.4).
#: Effective elevation stops at tier 1 (IMAGE_PIN); tier 2 rates remain for sampler config only.
TIER_0, TIER_1, TIER_2 = 0, 1, 2

#: Prefix that makes an audit work-unit id DISTINCT from the primary unit id (== submission_id),
#: which is reserved for the submission's own evaluation unit (VAL-PRISM-012).
AUDIT_UNIT_PREFIX = "audit:"

#: Audit-unit lifecycle (architecture.md 3.5). ``pending`` units are the only ones exposed on the
#: coordination plane (pending-only listing semantics); the rest are terminal EXCEPT ``pending`` set
#: again on a bounded re-audit after an inconclusive failure/timeout (VAL-PRISM-024).
AUDIT_STATUS_PENDING = "pending"
AUDIT_STATUS_PASSED = "passed"
AUDIT_STATUS_MISMATCH = "mismatch"
AUDIT_STATUS_FAILED = "failed"

#: Audit resolutions recorded against the unit (distinct from the lifecycle status).
AUDIT_RESOLUTION_PASS = "pass"
AUDIT_RESOLUTION_MISMATCH = "mismatch"
AUDIT_RESOLUTION_INCONCLUSIVE = "inconclusive"


def audit_unit_id_for(submission_id: str) -> str:
    """Return the audit work-unit id for ``submission_id`` (distinct from the primary unit id)."""

    return f"{AUDIT_UNIT_PREFIX}{submission_id}"


def is_audit_unit_id(work_unit_id: str) -> bool:
    """Whether ``work_unit_id`` names an audit unit rather than a primary evaluation unit."""

    return work_unit_id.startswith(AUDIT_UNIT_PREFIX)


def effective_tier(
    proof: ExecutionProof,
    *,
    pinned_image_digest: str | None = None,
) -> int:
    """Return the VERIFIED tier for ``proof`` (never higher than image-pin verifiable backing).

    Claimed tier never controls elevation. Max effective tier is 1 (IMAGE_PIN match + pod binding).
    Claimed tier 2 / attestation fields never elevate — Prism has no TEE verifier path.
    """

    claimed = int(getattr(proof, "tier", 0) or 0)

    # Any claim of tier 2+ without a TEE product path downgrades to 0 (never silent tier-2,
    # never fall through to tier-1 unless the independent pin policy below applies — and
    # claimed>=2 intentionally does NOT fall through: pin match alone cannot re-elevate a
    # hardware-attestation claim).
    if claimed >= 2:
        return 0

    if claimed == 1:
        provider = proof.provider
        matches_digest = bool(pinned_image_digest) and proof.image_digest == pinned_image_digest
        has_pod = provider is not None and bool(provider.pod_id)
        return 1 if matches_digest and has_pod else 0

    return 0


def is_tier_downgraded(
    proof: ExecutionProof,
    *,
    pinned_image_digest: str | None = None,
) -> bool:
    """Whether ``proof``'s claimed tier is higher than its verified effective tier."""

    return int(proof.tier) != effective_tier(proof, pinned_image_digest=pinned_image_digest)


@dataclass(frozen=True)
class AuditDecision:
    """The outcome of applying the audit sampler to one finalized result."""

    work_unit_id: str
    claimed_tier: int
    effective_tier: int
    sampled: bool

    @property
    def downgraded(self) -> bool:
        return self.claimed_tier != self.effective_tier


@dataclass(frozen=True)
class AuditSampler:
    """Deterministic, per-tier probabilistic sampler over finalized results (architecture.md 3.4).

    The sampled fraction of each tier converges to its configured rate; a rate of ``0.0`` yields
    exactly zero samples for that tier and ``>= 1.0`` samples every one. Sampling is a pure function
    of ``seed``, the server-side secret ``salt`` and the per-result key, so it is reproducible and
    order-insensitive. Mixing the secret ``salt`` in makes selection unpredictable from the public
    ``submission_id`` alone yet reproducible for a fixed salt (architecture.md 3.4; VAL-FINAL-006).
    """

    audit_rate_tier0: float = 0.10
    audit_rate_tier1: float = 0.05
    audit_rate_tier2: float = 0.02
    seed: int = 0
    salt: str = ""

    def rate_for_tier(self, tier: int) -> float:
        """The configured audit rate for an EFFECTIVE tier (unknown tiers fall back to tier 0)."""

        if tier == TIER_2:
            return self.audit_rate_tier2
        if tier == TIER_1:
            return self.audit_rate_tier1
        return self.audit_rate_tier0

    def should_sample(self, *, work_unit_id: str, effective_tier: int) -> bool:
        """Whether a result of ``effective_tier`` is sampled for audit (deterministic in seed)."""

        rate = self.rate_for_tier(effective_tier)
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return self._uniform(work_unit_id) < rate

    def decide(
        self,
        *,
        work_unit_id: str,
        proof: ExecutionProof,
        pinned_image_digest: str | None = None,
    ) -> AuditDecision:
        """Verify ``proof``'s tier and decide whether it is sampled at its EFFECTIVE rate."""

        tier = effective_tier(proof, pinned_image_digest=pinned_image_digest)
        return AuditDecision(
            work_unit_id=work_unit_id,
            claimed_tier=int(proof.tier),
            effective_tier=tier,
            sampled=self.should_sample(work_unit_id=work_unit_id, effective_tier=tier),
        )

    def _uniform(self, key: str) -> float:
        """A deterministic uniform draw in ``[0, 1)`` keyed on ``(seed, salt, key)``.

        The secret ``salt`` is folded into the hashed material so the draw for a given public
        ``submission_id`` cannot be reproduced without it (VAL-FINAL-006).
        """

        digest = hashlib.sha256(f"{self.seed}:{self.salt}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)


def audit_sampler_from_config(worker_plane: WorkerPlaneConfig, *, seed: int = 0) -> AuditSampler:
    """Build an :class:`AuditSampler` from the prism ``worker_plane`` audit-rate config.

    The server-side secret ``audit_salt`` is mixed into the sampler seed so audit selection is
    unpredictable from the public ``submission_id`` yet reproducible for a fixed salt
    (VAL-FINAL-006).
    """

    return AuditSampler(
        audit_rate_tier0=worker_plane.audit_rate_tier0,
        audit_rate_tier1=worker_plane.audit_rate_tier1,
        audit_rate_tier2=worker_plane.audit_rate_tier2,
        seed=seed,
        salt=worker_plane.audit_salt or "",
    )


class SupportsAuditResolution(Protocol):
    """The slice of :class:`~prism_challenge.repository.PrismRepository` audit resolution needs."""

    async def get_audit_unit(self, audit_unit_id: str) -> dict[str, object] | None: ...

    async def record_audit_resolution(
        self,
        *,
        audit_unit_id: str,
        status: str,
        attempts: int,
        resolution: str | None,
        resolved_manifest_sha256: str | None,
        error: str | None,
    ) -> None: ...

    async def invalidate_submission_score(self, submission_id: str, *, reason: str) -> bool: ...

    async def get_work_unit_result(self, work_unit_id: str) -> dict[str, object] | None: ...

    async def record_worker_fault(
        self,
        *,
        audit_unit_id: str,
        submission_id: str,
        worker_pubkey: str | None,
        audited_manifest_sha256: str,
        replay_manifest_sha256: str,
        reason: str,
    ) -> None: ...


@dataclass(frozen=True)
class AuditResolution:
    """The observable outcome of resolving one audit unit against a validator replay."""

    audit_unit_id: str
    submission_id: str
    status: str
    resolution: str | None
    attempts: int
    invalidated: bool
    terminal: bool

    def to_response(self) -> dict[str, object]:
        return {
            "audit_unit_id": self.audit_unit_id,
            "submission_id": self.submission_id,
            "status": self.status,
            "resolution": self.resolution,
            "attempts": self.attempts,
            "invalidated": self.invalidated,
            "terminal": self.terminal,
        }


async def resolve_audit_unit(
    repository: SupportsAuditResolution,
    *,
    audit_unit_id: str,
    replay_manifest_sha256: str | None = None,
    failed: bool = False,
    error: str | None = None,
) -> AuditResolution:
    """Resolve an audit unit from a validator's authoritative replay (architecture.md 3.5).

    The validator replay is authoritative. Outcomes:

    * an authoritative replay hash EQUAL to the audited worker hash -> ``passed`` (the audited score
      is left untouched);
    * an authoritative replay hash DIFFERENT from it -> ``mismatch``: the audited submission's score
      is invalidated (VAL-PRISM-013) and the crown/weights recomputed (VAL-PRISM-023);
    * a replay FAILURE or TIMEOUT (``failed`` / no ``replay_manifest_sha256``) NEVER confirms the
      audited result: the unit is re-audited (back to ``pending``) until ``max_attempts`` is
      exhausted, then reaches the terminal, observable ``failed`` state -- the audited submission is
      left unresolved (never silently reverted to accepted) and NO fault is attributed, because a
      fault requires an authoritative divergent manifest (VAL-PRISM-024).

    Resolving an already-terminal unit is an idempotent no-op.
    """

    unit = await repository.get_audit_unit(audit_unit_id)
    if unit is None:
        raise KeyError(f"audit unit {audit_unit_id!r} not found")

    submission_id = str(unit["submission_id"])
    status = str(unit["status"])
    attempts = int(unit["attempts"])  # type: ignore[call-overload]
    if status in (AUDIT_STATUS_PASSED, AUDIT_STATUS_MISMATCH, AUDIT_STATUS_FAILED):
        return AuditResolution(
            audit_unit_id=audit_unit_id,
            submission_id=submission_id,
            status=status,
            resolution=(str(unit["resolution"]) if unit["resolution"] is not None else None),
            attempts=attempts,
            invalidated=status == AUDIT_STATUS_MISMATCH,
            terminal=True,
        )

    max_attempts = int(unit["max_attempts"])  # type: ignore[call-overload]
    attempts += 1
    inconclusive = failed or not replay_manifest_sha256

    if inconclusive:
        exhausted = attempts >= max_attempts
        new_status = AUDIT_STATUS_FAILED if exhausted else AUDIT_STATUS_PENDING
        await repository.record_audit_resolution(
            audit_unit_id=audit_unit_id,
            status=new_status,
            attempts=attempts,
            resolution=AUDIT_RESOLUTION_INCONCLUSIVE if exhausted else None,
            resolved_manifest_sha256=None,
            error=error or "audit replay failed or timed out",
        )
        return AuditResolution(
            audit_unit_id=audit_unit_id,
            submission_id=submission_id,
            status=new_status,
            resolution=AUDIT_RESOLUTION_INCONCLUSIVE if exhausted else None,
            attempts=attempts,
            invalidated=False,
            terminal=exhausted,
        )

    audited_hash = str(unit["audited_manifest_sha256"])
    assert replay_manifest_sha256 is not None  # narrowed: inconclusive returned above
    matches = replay_manifest_sha256 == audited_hash
    invalidated = False
    if matches:
        new_status = AUDIT_STATUS_PASSED
        resolution = AUDIT_RESOLUTION_PASS
    else:
        new_status = AUDIT_STATUS_MISMATCH
        resolution = AUDIT_RESOLUTION_MISMATCH
        invalidated = await repository.invalidate_submission_score(
            submission_id,
            reason=f"audit invalidated: manifest mismatch (audit_unit={audit_unit_id})",
        )
        # The authoritative validator replay diverged from the audited worker manifest: the worker
        # that produced it lied. Record a worker_fault against it (architecture.md 4;
        # VAL-FINAL-005). The faulty worker's pubkey is the one recorded on the audited primary
        # result; a missing result row leaves it null but still records the fault.
        origin_work_unit_id = str(unit["origin_work_unit_id"])
        worker_result = await repository.get_work_unit_result(origin_work_unit_id)
        worker_pubkey = (
            str(worker_result["worker_pubkey"])
            if worker_result is not None and worker_result.get("worker_pubkey") is not None
            else None
        )
        await repository.record_worker_fault(
            audit_unit_id=audit_unit_id,
            submission_id=submission_id,
            worker_pubkey=worker_pubkey,
            audited_manifest_sha256=audited_hash,
            replay_manifest_sha256=replay_manifest_sha256,
            reason="audit manifest mismatch",
        )
    await repository.record_audit_resolution(
        audit_unit_id=audit_unit_id,
        status=new_status,
        attempts=attempts,
        resolution=resolution,
        resolved_manifest_sha256=replay_manifest_sha256,
        error=None,
    )
    return AuditResolution(
        audit_unit_id=audit_unit_id,
        submission_id=submission_id,
        status=new_status,
        resolution=resolution,
        attempts=attempts,
        invalidated=invalidated,
        terminal=True,
    )


__all__ = [
    "AUDIT_RESOLUTION_INCONCLUSIVE",
    "AUDIT_RESOLUTION_MISMATCH",
    "AUDIT_RESOLUTION_PASS",
    "AUDIT_STATUS_FAILED",
    "AUDIT_STATUS_MISMATCH",
    "AUDIT_STATUS_PASSED",
    "AUDIT_STATUS_PENDING",
    "AUDIT_UNIT_PREFIX",
    "TIER_0",
    "TIER_1",
    "TIER_2",
    "AuditDecision",
    "AuditResolution",
    "AuditSampler",
    "SupportsAuditResolution",
    "audit_sampler_from_config",
    "audit_unit_id_for",
    "effective_tier",
    "is_audit_unit_id",
    "is_tier_downgraded",
    "resolve_audit_unit",
]
