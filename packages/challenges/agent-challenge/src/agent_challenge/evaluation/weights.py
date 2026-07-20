"""Challenge weight computation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.authorization import (
    EvalAuthorizationConflict,
    load_eval_run_plan,
)

from ..core.config import settings
from ..core.db import database
from ..core.models import AgentSubmission, EvalRun, EvaluationJob
from ..review.authorization import verified_review_assignment_for_submission
from ..sdk.config import effective_evaluation_task_count
from .plan_scoring import CanonicalPlanScoringError, load_eval_plan, plan_backed_job_is_consistent
from .validator_executor import job_attestation_verified

EFFECTIVE_VALID_STATUSES = frozenset({"valid", "overridden_valid", "completed"})
SCORING_RAW_STATUSES = frozenset({"tb_completed"})


@dataclass(frozen=True)
class _DirectEligibleScore:
    """One reconstructed, eligibility-checked direct EvalRun score."""

    hotkey: str
    score: float
    created_at: Any
    submission_id: int
    eval_run_id: str


def is_effective_valid_submission(submission: AgentSubmission) -> bool:
    return submission.effective_status in EFFECTIVE_VALID_STATUSES


def is_scoring_submission(submission: AgentSubmission) -> bool:
    return (
        submission.raw_status in SCORING_RAW_STATUSES
        and submission.effective_status in EFFECTIVE_VALID_STATUSES
    )


def is_reward_eligible_job(
    job: EvaluationJob,
    required_task_count: int,
    *,
    attestation_verified: bool | None = None,
    review_verified: bool | None = None,
) -> bool:
    """Whether a scoring job may earn emission weight.

    A job earns weight only when it was evaluated on the FULL configured task
    set (``total_tasks >= required_task_count``) AND passed at least one task
    (``passed_tasks >= 1``). This burns emissions for partial evaluations (e.g.
    a leftover perfect score from a temporary smaller task-count window) and for
    zero-pass evaluations, neither of which earned rewards on the full task set.

    When the Phala attestation flag is ON the weights path additionally requires
    the job's scores to be backed by verified attestations (``attestation_verified``)
    and a matching verified review allow (``review_verified``). Both arguments
    default to ``True`` so the flag-off path is byte-identical to legacy
    eligibility.
    """
    # The helper remains a pure threshold predicate for legacy callers. The
    # attested production weights path supplies both proof arguments explicitly.
    attestation_ok = attestation_verified if attestation_verified is not None else True
    review_ok = review_verified if review_verified is not None else True
    return (
        job.total_tasks >= required_task_count
        and job.passed_tasks >= 1
        and attestation_ok
        and review_ok
        and plan_backed_job_is_consistent(job)
    )


def scoring_evaluation_jobs_statement():
    return (
        select(EvaluationJob)
        .join(EvaluationJob.submission)
        .options(selectinload(EvaluationJob.submission))
        .where(EvaluationJob.status == "completed")
        .where(AgentSubmission.raw_status.in_(SCORING_RAW_STATUSES))
        .where(AgentSubmission.effective_status.in_(EFFECTIVE_VALID_STATUSES))
        .order_by(desc(EvaluationJob.score), desc(EvaluationJob.created_at))
    )


def _reject_duplicate_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CanonicalPlanScoringError(f"duplicate persisted JSON key: {key!r}")
        result[key] = value
    return result


def _reconstruct_direct_score(
    run: EvalRun,
    *,
    required_task_count: int,
) -> tuple[float, int, int] | None:
    """Rebuild (score, passed_tasks, total_tasks) from immutable plan-bound bytes.

    Mutable ``EvalRun.score`` / ``passed_tasks`` / ``total_tasks`` columns are
    deliberately unread.  Missing, non-canonical, or plan-mismatched score
    records contribute nothing.
    """

    raw = run.canonical_score_record_json
    if raw is None or run.canonical_score_record_sha256 is None:
        return None
    try:
        plan = load_eval_run_plan(run)
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        record = ew.validate_canonical_score_record(
            parsed,
            scoring_policy=plan["scoring_policy"],
            expected_eval_run_id=plan["eval_run_id"],
            expected_task_ids=[task["task_id"] for task in plan["selected_tasks"]],
            expected_k=plan["k"],
        )
        if ew.canonical_json_v1(record).decode("utf-8") != raw:
            return None
        if run.canonical_score_record_sha256 != ew.score_record_digest(record):
            return None
        score = ew.decode_score_f64be(record["final"]["job_score_f64be"])
        passed_tasks = int(record["final"]["passed_tasks"])
        total_tasks = int(record["final"]["total_tasks"])
    except (
        CanonicalPlanScoringError,
        EvalAuthorizationConflict,
        ew.EvalWireError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    if total_tasks < required_task_count or passed_tasks < 1:
        return None
    return score, passed_tasks, total_tasks


async def _eligible_direct_scores(
    session: AsyncSession,
    *,
    required_task_count: int,
) -> list[_DirectEligibleScore]:
    """Build the sole filtered population for direct weight derivation.

    Per-hotkey and winner-take-all both consume this list exactly.  The query
    still gates on durable verification markers, but every numeric value and
    identity check is re-validated against immutable plan/score/review state.
    """

    candidates = (
        await session.scalars(
            select(EvalRun)
            .join(EvalRun.submission)
            .options(selectinload(EvalRun.submission))
            .where(EvalRun.phase == "eval_accepted")
            .where(EvalRun.verified.is_(True))
            .where(EvalRun.reward_eligible.is_(True))
            .where(EvalRun.result_available.is_(True))
            .where(AgentSubmission.raw_status.in_(SCORING_RAW_STATUSES))
            .where(AgentSubmission.effective_status.in_(EFFECTIVE_VALID_STATUSES))
        )
    ).all()

    eligible: list[_DirectEligibleScore] = []
    for run in candidates:
        submission = run.submission
        if not is_scoring_submission(submission):
            continue
        if submission.version_number != run.submission_version:
            continue
        reconstructed = _reconstruct_direct_score(run, required_task_count=required_task_count)
        if reconstructed is None:
            continue
        score, _passed, _total = reconstructed
        try:
            plan = load_eval_run_plan(run)
        except EvalAuthorizationConflict:
            continue
        if plan["submission_version"] != run.submission_version:
            continue
        if plan.get("authorizing_review_digest") != run.authorizing_review_digest:
            continue
        review = await verified_review_assignment_for_submission(session, submission)
        if review is None or review.review_digest != run.authorizing_review_digest:
            continue
        if review.review_digest != plan["authorizing_review_digest"]:
            continue
        eligible.append(
            _DirectEligibleScore(
                hotkey=submission.miner_hotkey,
                score=float(score),
                created_at=submission.created_at,
                submission_id=submission.id,
                eval_run_id=run.eval_run_id,
            )
        )
    return eligible


def _weights_from_direct_population(
    eligible: list[_DirectEligibleScore],
    *,
    winner_take_all: bool,
) -> dict[str, float]:
    """Collapse one filtered population into the emission weight map."""

    if not eligible:
        return {}

    if not winner_take_all:
        best: dict[str, float] = {}
        for item in sorted(
            eligible,
            key=lambda row: (-row.score, row.created_at, row.submission_id),
        ):
            best.setdefault(item.hotkey, item.score)
        return best

    winner = min(
        eligible,
        key=lambda row: (-row.score, row.created_at, row.submission_id),
    )
    if winner.score <= 0:
        return {}
    return {winner.hotkey: winner.score}


async def get_weights() -> dict[str, float]:
    """Return raw miner weights for the BASE master to normalize."""

    require_attestation = settings.phala_attestation_enabled
    required_task_count = effective_evaluation_task_count(settings.evaluation_task_count)
    async with database.session() as session:
        rows = (await session.execute(scoring_evaluation_jobs_statement())).scalars().all()
        if require_attestation:
            direct_eligible = await _eligible_direct_scores(
                session, required_task_count=required_task_count
            )
            # Full-attested production mode persists scores only on EvalRun.
            # When the challenge-owned population is non-empty and there are no
            # legacy EvaluationJob rows, emit weights from that single filtered
            # population for both per-hotkey and winner-take-all modes.
            if direct_eligible and not rows:
                return _weights_from_direct_population(
                    direct_eligible,
                    winner_take_all=settings.weights_winner_take_all,
                )
        attestation_verified: dict[int, bool] = {}
        review_status: dict[int, bool] = {}
        if require_attestation:
            for job in rows:
                task_attested = await job_attestation_verified(session, job)
                try:
                    plan = load_eval_plan(job)
                except CanonicalPlanScoringError:
                    attestation_verified[job.id] = False
                    continue
                if plan is None:
                    # Planless rows are the legacy validator path.  Keep its
                    # existing flag-on attestation behavior for compatibility;
                    # direct miner-funded results are always plan-backed.
                    attestation_verified[job.id] = task_attested
                    continue
                review = await verified_review_assignment_for_submission(session, job.submission)
                attestation_verified[job.id] = task_attested
                review_verified = bool(
                    review is not None
                    and review.review_digest == plan["authorizing_review_digest"]
                    and job.submission.version_number == plan["submission_version"]
                )
                review_status[job.id] = review_verified

    qualifying = [
        job
        for job in rows
        if is_scoring_submission(job.submission)
        and is_reward_eligible_job(
            job,
            required_task_count,
            attestation_verified=attestation_verified.get(job.id, True),
            review_verified=review_status.get(job.id, True),
        )
    ]

    if not settings.weights_winner_take_all:
        best: dict[str, float] = {}
        for job in qualifying:
            best.setdefault(job.submission.miner_hotkey, job.score)
        return best

    if not qualifying:
        return {}

    # Winner-take-all: a single hotkey collects the whole emission. Equal top
    # scores resolve to the earliest-arrived submission so re-submitting a tying
    # score can never displace the original winner; a non-positive top score has
    # no winner and burns (empty map).
    winner = min(
        qualifying,
        key=lambda job: (-job.score, job.submission.created_at, job.submission.id),
    )
    if winner.score <= 0:
        return {}
    return {winner.submission.miner_hotkey: winner.score}
