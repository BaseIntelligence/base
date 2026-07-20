"""Low-rate replay-audit sampler (architecture sec 4 C6 / sec 8, defense-in-depth).

An attested result is trusted on its own (the hardware-signed quote proves the
canonical image produced the bound score), so no redundant re-execution is
required. The replay audit is a *net*, not a trust requirement: with low
probability a validator re-runs a sampled submission on its OWN broker and flags
score mismatches. This module is the SAMPLER -- the pure, deterministic decision
of *which* attested submissions to replay; the re-run + score comparison layer on
top of the ids it returns.

Design (behavioral contract):

* **Attested-only population.** Only submissions on the Phala attested path
  (:attr:`AuditCandidate.attested`) enter the audit; legacy/non-attested runs are
  never drawn in (VAL-SCORE-026).
* **Tier-driven rate (higher trust => strictly lower rate).** A verified
  attestation is the high-trust :data:`AUDIT_TIER_ATTESTED` tier, audited at the
  low ``attested`` rate; an unverifiable/failed attestation is the low-trust
  :data:`AUDIT_TIER_UNVERIFIED` tier, audited at the higher ``unverified`` rate.
  An unverifiable claim can therefore never buy the reduced rate (VAL-SCORE-025).
* **Deterministic and seedable.** Selection is a pure function of the seed and
  the submission ids: the same seed reproduces the identical subset, a different
  seed selects a different subset at the same rate (VAL-SCORE-017). Each tier's
  population is ranked by a seeded hash and the top ``round(rate * N)`` ids are
  taken, so the sampled fraction tracks the configured rate exactly rather than
  drifting with statistical noise (VAL-SCORE-016).
* **Rate 0 disables.** A tier rate of ``0`` samples nothing from that tier; both
  rates ``0`` samples nothing at all (VAL-SCORE-018).
* **Flag-off inert.** When constructed with ``enabled=False`` (the Phala flag
  off) the sampler selects nothing, so legacy scoring/weights are untouched
  (VAL-SCORE-026).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.models import EvalRun, ReplayAuditDispute, TaskAttestation, TaskResult
from agent_challenge.evaluation.authorization import load_eval_run_plan
from agent_challenge.evaluation.own_runner.keep_policy import (
    DEFAULT_KEEP_POLICY,
    keep_good_job_score,
    normalize_keep_policy,
)
from agent_challenge.evaluation.own_runner.variance import (
    DEFAULT_PER_TASK_AGGREGATION,
    aggregate_task_scores,
    normalize_aggregation_mode,
)
from agent_challenge.evaluation.plan_scoring import (
    CanonicalPlanScoringError,
    load_canonical_score_record,
    replay_score_from_eval_plan,
)
from agent_challenge.review.authorization import verified_review_assignment_for_submission

if TYPE_CHECKING:
    from agent_challenge.sdk.config import ChallengeSettings

#: High-trust tier: a verified Phala-tdx attestation (audited at the LOW rate).
AUDIT_TIER_ATTESTED = "attested"
#: Low-trust tier: an unverifiable/failed attestation (audited at the HIGHER rate).
AUDIT_TIER_UNVERIFIED = "unverified"
#: All audit trust tiers, ordered high-trust first.
AUDIT_TIERS: tuple[str, ...] = (AUDIT_TIER_ATTESTED, AUDIT_TIER_UNVERIFIED)


class InvalidAuditRateError(ValueError):
    """Raised when an audit rate is outside ``[0, 1]`` or a tier is unknown.

    Fail-closed: a malformed rate/tier is rejected rather than silently coerced
    (which could enable/disable auditing undetected).
    """


@dataclass(frozen=True)
class AuditCandidate:
    """A submission considered for the replay audit.

    ``attested`` marks whether the submission is on the Phala attested path -- only
    attested submissions enter the audit population (a legacy run has
    ``attested=False`` and is never sampled). ``verified`` marks whether its
    attestation verified: a verified quote is the high-trust
    :data:`AUDIT_TIER_ATTESTED` tier, an unverifiable/failed one the low-trust
    :data:`AUDIT_TIER_UNVERIFIED` tier (so it never buys the reduced rate).

    ``attested_score`` is the accepted (attested) job score the replay is compared
    against, and ``n_attempts`` is the attested run's ``k`` -- the replay re-runs
    the SAME ``k`` trials per task so the comparison is apples-to-apples. Both
    default to a legacy-safe value (``0.0`` / ``k=1``) and are only consulted by
    the execution/compare layer, never by the sampler.
    """

    submission_id: str
    attested: bool = True
    verified: bool = True
    attested_score: float = 0.0
    n_attempts: int = 1
    # An accepted attested run carries the exact plan bytes.  Legacy callers
    # omit it and keep their historical aggregation behavior.
    eval_plan: Mapping[str, object] | None = None
    eval_run_id: str | None = None
    plan_sha256: str | None = None
    # ``None`` preserves the original pure-function test seam.  Production
    # candidates are always constructed by ``accepted_verified_replay_population``
    # with an explicit eligibility decision from durable Eval state.
    population_eligible: bool | None = None

    @property
    def in_population(self) -> bool:
        """Whether this submission is eligible for the audit.

        Synthetic callers from the pre-attestation replay API retain their
        historical ``attested`` behavior.  All production candidates carry an
        explicit ``population_eligible`` decision, which is fail-closed and
        derived from the durable accepted Eval predicate.
        """

        if self.population_eligible is not None:
            return self.population_eligible
        return self.attested

    @property
    def tier(self) -> str:
        """The audit trust tier this submission is audited under."""

        return AUDIT_TIER_ATTESTED if self.verified else AUDIT_TIER_UNVERIFIED


async def accepted_verified_replay_population(
    session: AsyncSession,
    *,
    enabled: bool,
) -> list[AuditCandidate]:
    """Load only accepted, verified, plan-backed Eval runs for replay.

    This is the production population boundary.  It intentionally does not
    represent failed or unverifiable runs as score-zero candidates and never
    falls back to a legacy job.  Every field used by comparison is reconstructed
    from the durable run/job plan and canonical score record.
    """

    if not enabled:
        return []
    runs = (
        await session.scalars(
            select(EvalRun)
            .options(selectinload(EvalRun.submission), selectinload(EvalRun.result_job))
            .where(EvalRun.phase == "eval_accepted")
            .where(EvalRun.verified.is_(True))
            .where(EvalRun.result_available.is_(True))
            .where(EvalRun.key_granted_at.is_not(None))
            .order_by(EvalRun.id)
        )
    ).all()
    candidates: list[AuditCandidate] = []
    for run in runs:
        submission = run.submission
        job = run.result_job
        if submission is None:
            continue
        if submission.version_number != run.submission_version:
            continue
        review = await verified_review_assignment_for_submission(session, submission)
        if review is None or review.review_digest != run.authorizing_review_digest:
            continue
        try:
            plan = load_eval_run_plan(run)
            record = (
                load_canonical_score_record(job)
                if job is not None
                else ew.validate_canonical_score_record(
                    json.loads(run.canonical_score_record_json or ""),
                    scoring_policy=plan["scoring_policy"],
                    expected_eval_run_id=plan["eval_run_id"],
                    expected_task_ids=[item["task_id"] for item in plan["selected_tasks"]],
                    expected_k=plan["k"],
                )
            )
        except (CanonicalPlanScoringError, ValueError, TypeError, KeyError):
            continue
        if record is None:
            continue
        if (
            plan["eval_run_id"] != run.eval_run_id
            or plan["submission_id"] != str(submission.id)
            or plan["submission_version"] != run.submission_version
            or plan["authorizing_review_digest"] != run.authorizing_review_digest
            or plan["agent_hash"] != submission.agent_hash
        ):
            continue
        expected_tasks = {item["task_id"] for item in plan["selected_tasks"]}
        if job is None:
            candidates.append(
                AuditCandidate(
                    submission_id=str(plan["submission_id"]),
                    attested=True,
                    verified=True,
                    attested_score=ew.decode_score_f64be(record["final"]["job_score_f64be"]),
                    n_attempts=plan["k"],
                    eval_plan=plan,
                    eval_run_id=run.eval_run_id,
                    plan_sha256=run.plan_sha256,
                    population_eligible=True,
                )
            )
            continue
        attestation_rows = (
            await session.scalars(select(TaskAttestation).where(TaskAttestation.job_id == job.id))
        ).all()
        attestation_tasks = {row.task_id for row in attestation_rows}
        verified_tasks = {row.task_id for row in attestation_rows if row.verified}
        if attestation_tasks != expected_tasks or verified_tasks != expected_tasks:
            continue
        task_rows = (
            await session.scalars(select(TaskResult).where(TaskResult.job_id == job.id))
        ).all()
        if {row.task_id for row in task_rows} != expected_tasks or any(
            row.status != "completed" for row in task_rows
        ):
            continue
        aggregate_scores = {
            item["task_id"]: item["aggregate_score_f64be"] for item in record["tasks"]
        }
        if any(
            ew.encode_score_f64be(row.score) != aggregate_scores.get(row.task_id)
            for row in task_rows
        ):
            continue
        candidates.append(
            AuditCandidate(
                submission_id=str(plan["submission_id"]),
                attested=True,
                verified=True,
                attested_score=ew.decode_score_f64be(record["final"]["job_score_f64be"]),
                n_attempts=plan["k"],
                eval_plan=plan,
                eval_run_id=run.eval_run_id,
                plan_sha256=run.plan_sha256,
                population_eligible=True,
            )
        )
    return candidates


# Name used by the internal replay scheduler and convenient for downstream
# integrations that should not need to know the storage model terminology.
load_replay_population = accepted_verified_replay_population


REPLAY_AUDIT_LABEL = "agent-challenge.replay-audit.v1"
REPLAY_AUDIT_REQUEST_KIND = "replay_audit_request"
REPLAY_AUDIT_RESULT_KIND = "replay_audit_result"


class ReplayAuditWireError(ValueError):
    """Raised when a labelled replay seam payload is malformed or mismatched."""


@dataclass(frozen=True)
class ReplayAuditRequest:
    """BASE-facing request carrying the exact immutable Eval plan."""

    audit_id: str
    submission_id: str
    eval_run_id: str
    replay_attempt: int
    plan_sha256: str
    eval_plan: Mapping[str, object]
    attested_score: float

    def to_dict(self) -> dict[str, object]:
        plan = dict(self.eval_plan)
        policy = plan.get("scoring_policy")
        selected_tasks = plan.get("selected_tasks")
        return {
            "schema_version": 1,
            "audit_label": REPLAY_AUDIT_LABEL,
            "kind": REPLAY_AUDIT_REQUEST_KIND,
            "audit_id": self.audit_id,
            "submission_id": self.submission_id,
            "eval_run_id": self.eval_run_id,
            "replay_attempt": self.replay_attempt,
            "plan_sha256": self.plan_sha256,
            "eval_plan": plan,
            "k": plan.get("k"),
            "selected_tasks": selected_tasks,
            "scoring_policy": policy,
            "scoring_policy_digest": plan.get("scoring_policy_digest"),
            "attested_score": self.attested_score,
        }

    @property
    def k(self) -> int:
        value = self.eval_plan.get("k")
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ReplayAuditWireError("immutable Eval plan has no valid k")
        return value


def replay_request_from_mapping(value: Mapping[str, object]) -> ReplayAuditRequest:
    """Validate one labelled replay request received from the challenge API."""

    from agent_challenge.canonical import eval_wire as ew

    expected = {
        "schema_version",
        "audit_label",
        "kind",
        "audit_id",
        "submission_id",
        "eval_run_id",
        "replay_attempt",
        "plan_sha256",
        "eval_plan",
        "k",
        "selected_tasks",
        "scoring_policy",
        "scoring_policy_digest",
        "attested_score",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ReplayAuditWireError("replay request has unknown or missing fields")
    if (
        value["schema_version"] != 1
        or value["audit_label"] != REPLAY_AUDIT_LABEL
        or value["kind"] != REPLAY_AUDIT_REQUEST_KIND
        or not isinstance(value["audit_id"], str)
        or not isinstance(value["submission_id"], str)
        or not isinstance(value["eval_run_id"], str)
        or not isinstance(value["replay_attempt"], int)
        or isinstance(value["replay_attempt"], bool)
        or value["replay_attempt"] < 1
        or not isinstance(value["plan_sha256"], str)
        or len(value["plan_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["plan_sha256"])
        or not isinstance(value["eval_plan"], Mapping)
        or not isinstance(value["attested_score"], (int, float))
        or isinstance(value["attested_score"], bool)
    ):
        raise ReplayAuditWireError("replay request has invalid fields")
    try:
        plan = ew.validate_eval_plan(value["eval_plan"])
    except ew.EvalWireError as exc:
        raise ReplayAuditWireError("replay request has invalid immutable Eval plan") from exc
    if (
        value["eval_run_id"] != plan["eval_run_id"]
        or value["submission_id"] != plan["submission_id"]
        or value["audit_id"] != replay_audit_id(value["eval_run_id"], value["replay_attempt"])
        or value["k"] != plan["k"]
        or value["selected_tasks"] != plan["selected_tasks"]
        or value["scoring_policy"] != plan["scoring_policy"]
        or value["scoring_policy_digest"] != plan["scoring_policy_digest"]
        or value["plan_sha256"] != _canonical_plan_digest(plan)
    ):
        raise ReplayAuditWireError("replay request does not match immutable Eval plan")
    score = float(value["attested_score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ReplayAuditWireError("replay request score is invalid")
    return ReplayAuditRequest(
        audit_id=value["audit_id"],
        submission_id=value["submission_id"],
        eval_run_id=value["eval_run_id"],
        replay_attempt=value["replay_attempt"],
        plan_sha256=value["plan_sha256"],
        eval_plan=plan,
        attested_score=score,
    )


@dataclass(frozen=True)
class ReplayAuditResult:
    """BASE-facing replay result carrying raw ordered trials, not an aggregate."""

    audit_id: str
    submission_id: str
    eval_run_id: str
    replay_attempt: int
    plan_sha256: str
    trial_scores_by_task: Mapping[str, Sequence[float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "audit_label": REPLAY_AUDIT_LABEL,
            "kind": REPLAY_AUDIT_RESULT_KIND,
            "audit_id": self.audit_id,
            "submission_id": self.submission_id,
            "eval_run_id": self.eval_run_id,
            "replay_attempt": self.replay_attempt,
            "plan_sha256": self.plan_sha256,
            "trial_scores_by_task": {
                task_id: list(scores) for task_id, scores in self.trial_scores_by_task.items()
            },
        }

    def validate_against(self, request: ReplayAuditRequest) -> None:
        """Require exact identity, selected-task order, and plan ``k``."""

        if (
            self.audit_id != request.audit_id
            or self.submission_id != request.submission_id
            or self.eval_run_id != request.eval_run_id
            or self.replay_attempt != request.replay_attempt
            or self.plan_sha256 != request.plan_sha256
        ):
            raise ReplayAuditWireError("replay result identity differs from request")
        expected_task_ids = [item["task_id"] for item in request.eval_plan["selected_tasks"]]
        if list(self.trial_scores_by_task) != expected_task_ids:
            raise ReplayAuditWireError("replay result task order differs from immutable plan")
        if any(len(scores) != request.k for scores in self.trial_scores_by_task.values()):
            raise ReplayAuditWireError("replay result trial count differs from immutable k")


def replay_audit_id(eval_run_id: str, replay_attempt: int = 1) -> str:
    """Return the stable id used by the challenge/BASE replay seam."""

    if not isinstance(eval_run_id, str) or not eval_run_id or replay_attempt < 1:
        raise ReplayAuditWireError("invalid replay audit identity")
    return f"replay:{eval_run_id}:{replay_attempt}"


def _canonical_plan_digest(plan: Mapping[str, object]) -> str:
    from agent_challenge.canonical import eval_wire as ew

    try:
        normalized = ew.validate_eval_plan(plan)
    except ew.EvalWireError as exc:
        raise ReplayAuditWireError(f"invalid immutable Eval plan: {exc}") from exc
    return sha256(ew.canonical_json_v1(normalized)).hexdigest()


def replay_request_for_candidate(
    candidate: AuditCandidate,
    *,
    replay_attempt: int = 1,
) -> ReplayAuditRequest:
    """Build the labelled BASE request from one accepted population member."""

    if not candidate.in_population or candidate.eval_plan is None or not candidate.eval_run_id:
        raise ReplayAuditWireError("replay request requires an eligible accepted Eval candidate")
    if candidate.population_eligible is True and not candidate.verified:
        raise ReplayAuditWireError("replay request requires a verified Eval candidate")
    plan_digest = _canonical_plan_digest(candidate.eval_plan)
    if candidate.plan_sha256 is not None and candidate.plan_sha256 != plan_digest:
        raise ReplayAuditWireError("replay request plan digest does not match immutable plan bytes")
    if candidate.eval_run_id != candidate.eval_plan.get("eval_run_id"):
        raise ReplayAuditWireError("replay request run id does not match immutable plan")
    return ReplayAuditRequest(
        audit_id=replay_audit_id(candidate.eval_run_id, replay_attempt),
        submission_id=candidate.submission_id,
        eval_run_id=candidate.eval_run_id,
        replay_attempt=replay_attempt,
        plan_sha256=plan_digest,
        eval_plan=candidate.eval_plan,
        attested_score=candidate.attested_score,
    )


def replay_result_from_mapping(value: Mapping[str, object]) -> ReplayAuditResult:
    """Validate a labelled BASE result without accepting an aggregate-only body."""

    if not isinstance(value, Mapping):
        raise ReplayAuditWireError("replay result must be an object")
    expected = {
        "schema_version",
        "audit_label",
        "kind",
        "audit_id",
        "submission_id",
        "eval_run_id",
        "replay_attempt",
        "plan_sha256",
        "trial_scores_by_task",
    }
    if set(value) != expected:
        raise ReplayAuditWireError("replay result has unknown or missing fields")
    if (
        value["schema_version"] != 1
        or value["audit_label"] != REPLAY_AUDIT_LABEL
        or value["kind"] != REPLAY_AUDIT_RESULT_KIND
        or not isinstance(value["audit_id"], str)
        or not isinstance(value["submission_id"], str)
        or not isinstance(value["eval_run_id"], str)
        or not isinstance(value["replay_attempt"], int)
        or isinstance(value["replay_attempt"], bool)
        or value["replay_attempt"] < 1
        or not isinstance(value["plan_sha256"], str)
        or len(value["plan_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["plan_sha256"])
        or not isinstance(value["trial_scores_by_task"], Mapping)
    ):
        raise ReplayAuditWireError("replay result has invalid fields")
    trial_scores: dict[str, list[float]] = {}
    for task_id, scores in value["trial_scores_by_task"].items():
        if (
            not isinstance(task_id, str)
            or not isinstance(scores, Sequence)
            or isinstance(scores, str)
            or not all(
                isinstance(score, (int, float)) and not isinstance(score, bool) for score in scores
            )
        ):
            raise ReplayAuditWireError("replay result trial scores are invalid")
        values = [float(score) for score in scores]
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in values):
            raise ReplayAuditWireError("replay result scores must be finite in [0, 1]")
        trial_scores[task_id] = values
    return ReplayAuditResult(
        audit_id=value["audit_id"],
        submission_id=value["submission_id"],
        eval_run_id=value["eval_run_id"],
        replay_attempt=value["replay_attempt"],
        plan_sha256=value["plan_sha256"],
        trial_scores_by_task=trial_scores,
    )


@dataclass(frozen=True)
class ReplayAuditSampler:
    """Deterministic, seedable, tier-driven replay-audit sampler.

    ``attested_rate`` / ``unverified_rate`` are the per-tier replay fractions in
    ``[0, 1]`` (higher trust => strictly lower rate is the intended, and default,
    configuration). ``seed`` makes the selection reproducible and seedable.
    ``enabled`` gates the whole sampler on the Phala flag: when off it selects
    nothing so legacy behavior is untouched.
    """

    attested_rate: float = 0.0
    unverified_rate: float = 0.0
    seed: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        for name, rate in (
            ("attested_rate", self.attested_rate),
            ("unverified_rate", self.unverified_rate),
        ):
            if not 0.0 <= float(rate) <= 1.0:
                raise InvalidAuditRateError(f"{name} must be between 0 and 1, got {rate!r}")

    def rate_for_tier(self, tier: str) -> float:
        """Return the configured replay rate for an audit ``tier``."""

        if tier == AUDIT_TIER_ATTESTED:
            return self.attested_rate
        if tier == AUDIT_TIER_UNVERIFIED:
            return self.unverified_rate
        raise InvalidAuditRateError(
            f"unknown audit tier {tier!r}; expected one of {list(AUDIT_TIERS)}"
        )

    def sample(self, candidates: Iterable[AuditCandidate]) -> list[str]:
        """Return the submission ids selected for a replay audit.

        The result is the attested-only population, sampled per tier at its
        configured rate with a seeded, deterministic hash rank, returned in the
        original population order. An empty list when the sampler is disabled, the
        population is empty, or every applicable rate is ``0``.
        """

        if not self.enabled:
            return []

        population = [
            c
            for c in candidates
            if c.in_population and (c.population_eligible is None or c.verified)
        ]
        if not population:
            return []

        by_tier: dict[str, list[str]] = {}
        for candidate in population:
            by_tier.setdefault(candidate.tier, []).append(candidate.submission_id)

        selected: set[str] = set()
        for tier, ids in by_tier.items():
            rate = self.rate_for_tier(tier)
            if rate <= 0.0:
                continue
            if rate >= 1.0:
                selected.update(ids)
                continue
            target = _target_count(rate, len(ids))
            if target <= 0:
                continue
            ranked = sorted(ids, key=lambda sid: _rank_key(self.seed, sid))
            selected.update(ranked[:target])

        return [c.submission_id for c in population if c.submission_id in selected]


def _rank_key(seed: int, submission_id: str) -> int:
    """A seeded, uniformly-distributed 256-bit rank for a submission id.

    Deterministic in ``(seed, submission_id)`` so the same seed reproduces the
    identical ordering and a different seed reshuffles it -- the source of the
    sampler's determinism/seedability.
    """

    digest = hashlib.sha256(f"{seed}:{submission_id}".encode()).digest()
    return int.from_bytes(digest, "big")


def _target_count(rate: float, population_size: int) -> int:
    """The number of submissions to sample from a tier of ``population_size``.

    Round-half-up of ``rate * population_size`` so the sampled fraction tracks the
    configured rate closely (no statistical drift) and never exceeds the tier.
    """

    if population_size <= 0:
        return 0
    return min(population_size, int(rate * population_size + 0.5))


def replay_audit_sampler_from_settings(settings: ChallengeSettings) -> ReplayAuditSampler:
    """Build a :class:`ReplayAuditSampler` from challenge settings.

    The sampler is enabled only when the Phala attestation flag is on, so a
    legacy (flag-off) deployment never audits.
    """

    return ReplayAuditSampler(
        attested_rate=settings.replay_audit_attested_rate,
        unverified_rate=settings.replay_audit_unverified_rate,
        seed=settings.replay_audit_seed,
        enabled=settings.phala_attestation_enabled,
    )


class InvalidReplayTrialsError(ValueError):
    """Raised when a broker replay does not return the attested ``k`` per task.

    Fail-closed: an apples-to-apples audit requires the replay to run the SAME
    number of trials per task as the attested run (:data:`VAL-SCORE-028`). A
    replay whose per-task trial count differs from the attested ``k`` is rejected
    rather than compared, so an attested ``k=3`` mean is never silently compared
    against a ``k=1`` single trial. A broker that returns ZERO tasks is likewise
    abnormal (it ran nothing) and is rejected rather than compared as a spurious
    ``0.0``-vs-attested mismatch.
    """


class BrokerReplay(Protocol):
    """The validator's OWN broker re-running a submission on the legacy path.

    Called with the submission id and the attested run's ``k`` (``n_attempts``);
    returns the per-task ordered per-trial scores the legacy own_runner broker
    produced (``k`` trials per task). The audit aggregates these itself -- the
    broker returns raw trial scores, never a pre-aggregated job score, and never
    the attested envelope's score.
    """

    def __call__(
        self, submission_id: str, *, k: int
    ) -> Mapping[str, Sequence[float]]:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class AggregationSpec:
    """The per-task aggregation + keep policy applied IDENTICALLY to both scores.

    The replay's job score must be computed with the SAME per-task aggregation
    mode and keep policy as the accepted attested score, so the comparison is
    apples-to-apples (:data:`VAL-SCORE-020`): given identical trial outcomes the
    two job scores are equal (zero delta) under every policy. This is the exact
    pipeline :func:`validator_executor.finalize_job_if_complete` uses -- per-task
    aggregation (:mod:`own_runner.variance`) then the keep-policy mean over the
    per-task scores (:mod:`own_runner.keep_policy`).
    """

    per_task_aggregation: str = DEFAULT_PER_TASK_AGGREGATION
    keep_policy: str = DEFAULT_KEEP_POLICY
    drop_lowest_n: int = 0
    threshold: float = 0.0

    def __post_init__(self) -> None:
        # Normalize/validate up front so a misconfigured spec fails closed rather
        # than at comparison time (which could skew or skip an audit).
        object.__setattr__(
            self, "per_task_aggregation", normalize_aggregation_mode(self.per_task_aggregation)
        )
        object.__setattr__(self, "keep_policy", normalize_keep_policy(self.keep_policy))

    def job_score(self, trial_scores_by_task: Mapping[str, Sequence[float]]) -> float:
        """Aggregate per-task trial scores into ONE job score under this spec.

        Deterministic and order-preserving (the epsilon=0 harbor mean), so
        identical trial inputs always yield an identical job score.
        """

        per_task = aggregate_task_scores(trial_scores_by_task, mode=self.per_task_aggregation)
        return keep_good_job_score(
            list(per_task.values()),
            policy=self.keep_policy,
            drop_lowest_n=self.drop_lowest_n,
            threshold=self.threshold,
        )

    @classmethod
    def from_settings(cls, settings: ChallengeSettings) -> AggregationSpec:
        """Build the spec from the same challenge settings finalize scores with."""

        return cls(
            per_task_aggregation=settings.per_task_aggregation,
            keep_policy=settings.keep_good_tasks_policy,
            drop_lowest_n=settings.keep_good_tasks_drop_lowest,
            threshold=settings.keep_good_tasks_threshold,
        )

    @classmethod
    def from_eval_plan(cls, eval_plan: Mapping[str, object]) -> AggregationSpec:
        """Build the comparison policy from validated immutable Eval-plan bytes."""

        from agent_challenge.canonical import eval_wire as ew

        try:
            plan = ew.validate_eval_plan(eval_plan)
            policy = plan["scoring_policy"]
            threshold = (
                ew.decode_score_f64be(policy["threshold_f64be"])
                if policy["threshold_f64be"] is not None
                else 0.0
            )
            aggregation = policy["per_task_aggregation"].replace("_", "-")
            keep_policy = policy["keep_policy"].replace("_", "-")
            return cls(
                per_task_aggregation=aggregation,
                keep_policy=keep_policy,
                drop_lowest_n=policy["drop_lowest_n"],
                threshold=threshold,
            )
        except (ew.EvalWireError, KeyError, TypeError, ValueError) as exc:
            raise InvalidReplayTrialsError(
                f"replay policy is not valid immutable Eval-plan data: {exc}"
            ) from exc


@dataclass(frozen=True)
class AuditMismatchFlag:
    """A dispute record for a replay whose score diverges beyond tolerance.

    Carries exactly the four identifying fields a dispute needs (VAL-SCORE-021):
    the ``submission_id``, the accepted ``attested_score``, the ``replay_score``,
    and their absolute ``delta``. It is a SEPARATE signal -- raising it never
    mutates the accepted score or the weight map (VAL-SCORE-024).
    """

    submission_id: str
    attested_score: float
    replay_score: float
    delta: float


@dataclass(frozen=True)
class ReplayComparison:
    """The outcome of auditing one submission: replay score, delta, and any flag.

    ``flagged`` is ``True`` iff ``delta`` is STRICTLY greater than the tolerance
    (the boundary is inclusive, VAL-SCORE-023); ``flag`` carries the dispute
    record when flagged, else ``None``.
    """

    submission_id: str
    attested_score: float
    replay_score: float
    delta: float
    flagged: bool
    flag: AuditMismatchFlag | None


def compare_replay_trials(
    candidate: AuditCandidate,
    replay_trials: Mapping[str, Sequence[float]],
    *,
    spec: AggregationSpec,
    tolerance: float,
) -> ReplayComparison:
    """Compare raw BASE replay trials against the accepted immutable score."""

    if candidate.eval_plan is not None:
        try:
            from agent_challenge.canonical import eval_wire as ew

            plan = ew.validate_eval_plan(candidate.eval_plan)
        except ew.EvalWireError as exc:
            raise InvalidReplayTrialsError(
                f"replay has an invalid immutable Eval plan: {exc}"
            ) from exc
        plan_digest = sha256(ew.canonical_json_v1(plan)).hexdigest()
        if candidate.plan_sha256 is not None and candidate.plan_sha256 != plan_digest:
            raise InvalidReplayTrialsError(
                "replay plan digest does not match immutable Eval plan bytes"
            )
        if (
            candidate.population_eligible is True
            and candidate.eval_run_id is not None
            and candidate.eval_run_id != plan["eval_run_id"]
        ):
            raise InvalidReplayTrialsError("replay identity does not match immutable Eval plan")
        k = plan["k"]
    else:
        plan = None
        k = candidate.n_attempts
    if tolerance < 0.0 or tolerance > 1.0:
        raise InvalidReplayTrialsError("replay tolerance must be between 0 and 1")
    validated_trials = _validated_replay_trials(
        replay_trials,
        k=k,
        submission_id=candidate.submission_id,
    )
    if plan is None:
        replay_score = spec.job_score(validated_trials)
    else:
        try:
            replay_score = replay_score_from_eval_plan(plan, validated_trials)
        except CanonicalPlanScoringError as exc:
            raise InvalidReplayTrialsError(
                f"replay does not match immutable Eval plan: {exc}"
            ) from exc
    delta = abs(candidate.attested_score - replay_score)
    flagged = delta > tolerance and not math.isclose(
        delta,
        tolerance,
        rel_tol=1e-9,
        abs_tol=1e-12,
    )
    flag = (
        AuditMismatchFlag(
            submission_id=candidate.submission_id,
            attested_score=candidate.attested_score,
            replay_score=replay_score,
            delta=delta,
        )
        if flagged
        else None
    )
    return ReplayComparison(
        submission_id=candidate.submission_id,
        attested_score=candidate.attested_score,
        replay_score=replay_score,
        delta=delta,
        flagged=flagged,
        flag=flag,
    )


async def persist_replay_dispute(
    session: AsyncSession,
    *,
    candidate: AuditCandidate,
    comparison: ReplayComparison,
    replay_attempt: int = 1,
    now: datetime | None = None,
) -> ReplayAuditDispute | None:
    """Persist one genuine mismatch, idempotently, without touching accepted state."""

    if not comparison.flagged or comparison.flag is None:
        return None
    request = replay_request_for_candidate(candidate, replay_attempt=replay_attempt)
    plan = request.eval_plan
    policy_digest = plan.get("scoring_policy_digest")
    if not isinstance(policy_digest, str):
        raise ReplayAuditWireError("immutable plan has no scoring-policy digest")
    existing = await session.scalar(
        select(ReplayAuditDispute).where(ReplayAuditDispute.audit_id == request.audit_id)
    )
    if existing is not None:
        return existing
    dispute = ReplayAuditDispute(
        audit_id=request.audit_id,
        submission_id=int(request.submission_id),
        eval_run_id=request.eval_run_id,
        replay_attempt=request.replay_attempt,
        plan_sha256=request.plan_sha256,
        scoring_policy_digest=policy_digest,
        attested_score=comparison.attested_score,
        replay_score=comparison.replay_score,
        delta=comparison.delta,
        created_at=now or datetime.now(UTC),
    )
    try:
        async with session.begin_nested():
            session.add(dispute)
            await session.flush()
    except IntegrityError:
        return await session.scalar(
            select(ReplayAuditDispute).where(ReplayAuditDispute.audit_id == request.audit_id)
        )
    return dispute


def _validated_replay_trials(
    trials: Mapping[str, Sequence[float]], *, k: int, submission_id: str
) -> Mapping[str, Sequence[float]]:
    """Reject a replay whose per-task trial count is not the attested ``k``.

    A broker that returns ZERO tasks is an abnormal/fail-closed condition: it ran
    nothing, so its ``0.0`` job score would spuriously flag a mismatch against the
    attested score. Such a return is rejected (raise) rather than compared, so no
    false flag is ever emitted.
    """

    if not trials:
        raise InvalidReplayTrialsError(
            f"replay of {submission_id!r} returned zero tasks; a broker that ran no "
            "tasks is abnormal and is rejected (fail-closed) rather than compared as "
            "a 0.0 score"
        )
    for task_name, scores in trials.items():
        if len(scores) != k:
            raise InvalidReplayTrialsError(
                f"replay of {submission_id!r} ran {len(scores)} trial(s) for task "
                f"{task_name!r}, expected the attested k={k}"
            )
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            for score in scores
        ):
            raise InvalidReplayTrialsError(
                f"replay of {submission_id!r} returned a non-canonical score for task {task_name!r}"
            )
    return trials


def audit_submission(
    candidate: AuditCandidate,
    broker: BrokerReplay,
    *,
    spec: AggregationSpec,
    tolerance: float,
) -> ReplayComparison:
    """Replay one sampled submission on the validator broker and compare scores.

    Re-runs ``candidate`` on the validator's OWN broker (the legacy own_runner
    path) with the attested run's ``k = candidate.n_attempts`` (VAL-SCORE-019,
    -028), aggregates the replay trials with the SAME ``spec`` the attested score
    used (VAL-SCORE-020), and compares to ``candidate.attested_score``. Flags a
    genuine mismatch only when ``|attested - replay|`` is STRICTLY greater than
    ``tolerance`` (inclusive boundary, VAL-SCORE-021/-022/-023). The result is a
    pure value -- it never mutates the accepted score or weights (VAL-SCORE-024).
    """

    k = candidate.n_attempts
    if candidate.eval_plan is not None:
        try:
            from agent_challenge.canonical import eval_wire as ew

            k = ew.validate_eval_plan(candidate.eval_plan)["k"]
        except ew.EvalWireError as exc:
            raise InvalidReplayTrialsError(
                f"replay has an invalid immutable Eval plan: {exc}"
            ) from exc
    replay_trials = _validated_replay_trials(
        broker(candidate.submission_id, k=k), k=k, submission_id=candidate.submission_id
    )
    comparison_spec = (
        AggregationSpec.from_eval_plan(candidate.eval_plan)
        if candidate.eval_plan is not None
        else spec
    )
    return compare_replay_trials(
        candidate,
        replay_trials,
        spec=comparison_spec,
        tolerance=tolerance,
    )


def run_replay_audit(
    candidates: Iterable[AuditCandidate],
    broker: BrokerReplay,
    *,
    sampler: ReplayAuditSampler,
    spec: AggregationSpec,
    tolerance: float,
) -> list[ReplayComparison]:
    """Sample the attested population, then replay + compare each sampled id.

    Only the ``sampler``-selected (attested) submissions are replayed, so a
    disabled sampler (Phala flag off) dispatches zero replays and returns an empty
    list, leaving legacy scoring/weights untouched.
    """

    candidate_list = list(candidates)
    by_id = {candidate.submission_id: candidate for candidate in candidate_list}
    selected = sampler.sample(candidate_list)
    return [
        audit_submission(by_id[submission_id], broker, spec=spec, tolerance=tolerance)
        for submission_id in selected
    ]


@dataclass(frozen=True)
class ReplayAudit:
    """Wiring bundle for the replay audit: sampler + aggregation spec + tolerance.

    Built from challenge settings so the audit uses the SAME sampling rates/seed,
    aggregation, and keep policy the rest of the scoring path does.
    """

    sampler: ReplayAuditSampler
    spec: AggregationSpec
    tolerance: float

    def run(
        self, candidates: Iterable[AuditCandidate], broker: BrokerReplay
    ) -> list[ReplayComparison]:
        """Sample + replay + compare the attested population via ``broker``."""

        return run_replay_audit(
            candidates,
            broker,
            sampler=self.sampler,
            spec=self.spec,
            tolerance=self.tolerance,
        )

    @classmethod
    def from_settings(cls, settings: ChallengeSettings) -> ReplayAudit:
        """Build the audit bundle from challenge settings."""

        return cls(
            sampler=replay_audit_sampler_from_settings(settings),
            spec=AggregationSpec.from_settings(settings),
            tolerance=settings.replay_audit_tolerance,
        )


__all__ = [
    "AUDIT_TIER_ATTESTED",
    "AUDIT_TIER_UNVERIFIED",
    "AUDIT_TIERS",
    "AggregationSpec",
    "AuditCandidate",
    "AuditMismatchFlag",
    "BrokerReplay",
    "InvalidAuditRateError",
    "InvalidReplayTrialsError",
    "ReplayAuditDispute",
    "ReplayAuditRequest",
    "ReplayAuditResult",
    "ReplayAuditWireError",
    "REPLAY_AUDIT_LABEL",
    "REPLAY_AUDIT_REQUEST_KIND",
    "REPLAY_AUDIT_RESULT_KIND",
    "ReplayAudit",
    "ReplayAuditSampler",
    "ReplayComparison",
    "accepted_verified_replay_population",
    "audit_submission",
    "compare_replay_trials",
    "load_replay_population",
    "persist_replay_dispute",
    "replay_audit_id",
    "replay_audit_sampler_from_settings",
    "replay_request_for_candidate",
    "replay_request_from_mapping",
    "replay_result_from_mapping",
    "run_replay_audit",
]
