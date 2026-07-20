"""Replay-audit execution + score comparison (architecture sec 4 C6 / sec 8).

Behavioral contract for the ``replay-audit-execution-compare`` feature: a sampled
attested submission is RE-RUN on the validator's OWN broker (the legacy own_runner
path) with the SAME ``k = n_attempts`` as the attested run, aggregated with the
IDENTICAL per-task aggregation + keep policy, then the replay job score is compared
against the accepted attested score. A genuine mismatch beyond the variance
tolerance is flagged; ordinary LLM/agent variance within tolerance is not; the
tolerance boundary is inclusive; and the audit is a separate signal that NEVER
overwrites the accepted score or the computed weights. Anchored to:

* VAL-SCORE-019 -- replay runs on the validator broker (legacy path), never
  re-using the attested envelope as its own replay score.
* VAL-SCORE-020 -- replay applies the identical aggregation/keep policy: identical
  trial outcomes => zero delta under every policy.
* VAL-SCORE-021 -- a genuine mismatch beyond tolerance is flagged (one record with
  submission id, attested score, replay score, delta).
* VAL-SCORE-022 -- a within-tolerance difference is NOT flagged.
* VAL-SCORE-023 -- the tolerance boundary is inclusive (== tolerance not flagged;
  strictly greater flagged).
* VAL-SCORE-024 -- an audit flag never overwrites the accepted score/weights.
* VAL-SCORE-028 -- the replay re-runs the SAME k trials as the attested run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

from agent_challenge.core.config import settings as core_settings
from agent_challenge.evaluation.own_runner.keep_policy import keep_good_job_score
from agent_challenge.evaluation.own_runner.variance import aggregate_task_scores
from agent_challenge.evaluation.replay_audit import (
    AggregationSpec,
    AuditCandidate,
    AuditMismatchFlag,
    InvalidReplayTrialsError,
    ReplayAudit,
    ReplayAuditSampler,
    ReplayComparison,
    audit_submission,
    run_replay_audit,
)
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.config import (
    ChallengeSettings,
    effective_evaluation_task_count,
)
from agent_challenge.weights import get_weights

OFF = AggregationSpec(per_task_aggregation="mean", keep_policy="off")
BEST_OF_K = AggregationSpec(per_task_aggregation="mean", keep_policy="best-of-k")
DROP_LOWEST = AggregationSpec(
    per_task_aggregation="mean", keep_policy="drop-lowest-n", drop_lowest_n=1
)
THRESHOLD = AggregationSpec(
    per_task_aggregation="mean", keep_policy="threshold-band", threshold=0.5
)
ALL_SPECS = (OFF, BEST_OF_K, DROP_LOWEST, THRESHOLD)


def _trials(k: int, *tasks: float) -> dict[str, list[float]]:
    """A per-task trial mapping where each task has ``k`` identical trial scores."""

    return {f"task-{i}": [score] * k for i, score in enumerate(tasks)}


class RecordingBroker:
    """Stub validator broker (legacy own_runner path) that records its dispatch.

    Produces ``k`` trials per task on demand so the audit can prove it re-runs the
    attested ``k`` rather than re-using the attested envelope.
    """

    def __init__(self, task_scores: Sequence[float]) -> None:
        self._task_scores = list(task_scores)
        self.calls: list[tuple[str, int]] = []

    def __call__(self, submission_id: str, *, k: int) -> Mapping[str, Sequence[float]]:
        self.calls.append((submission_id, k))
        return _trials(k, *self._task_scores)


# --------------------------------------------------------------------------- #
# VAL-SCORE-019: replay runs on the validator broker (legacy path), not the envelope.
# --------------------------------------------------------------------------- #
def test_replay_dispatches_to_the_validator_broker() -> None:
    broker = RecordingBroker([0.2, 0.2])
    candidate = AuditCandidate("sub-1", attested_score=0.9, n_attempts=1)

    audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert broker.calls == [("sub-1", 1)]  # legacy broker actually invoked


def test_replay_score_comes_from_the_broker_not_the_attested_envelope() -> None:
    # The broker replays a genuinely different outcome (0.2) than the attested
    # score (0.9); the replay score MUST reflect the broker run, never echo back
    # the attested value.
    broker = RecordingBroker([0.2, 0.2])
    candidate = AuditCandidate("sub-1", attested_score=0.9, n_attempts=1)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert result.replay_score == pytest.approx(0.2)
    assert result.replay_score != candidate.attested_score


def test_zero_task_broker_return_fails_closed_without_flag() -> None:
    # A broker that returns NO tasks is abnormal (it ran nothing): its 0.0 job
    # score compared against the attested score would spuriously flag a mismatch,
    # so the audit fails CLOSED (raises) rather than emitting that false flag.
    def empty_broker(submission_id: str, *, k: int) -> Mapping[str, Sequence[float]]:
        return {}

    candidate = AuditCandidate("sub-1", attested_score=0.75, n_attempts=1)
    with pytest.raises(InvalidReplayTrialsError):
        audit_submission(candidate, empty_broker, spec=OFF, tolerance=0.2)


# --------------------------------------------------------------------------- #
# VAL-SCORE-020: identical trial outcomes => zero delta under EVERY policy.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", ALL_SPECS)
def test_identical_outcomes_yield_zero_delta_for_every_policy(spec: AggregationSpec) -> None:
    trials = _trials(3, 0.9, 0.4, 0.6, 1.0)
    attested = spec.job_score(trials)

    def broker(submission_id: str, *, k: int) -> Mapping[str, Sequence[float]]:
        return _trials(k, 0.9, 0.4, 0.6, 1.0)

    candidate = AuditCandidate("sub-1", attested_score=attested, n_attempts=3)
    result = audit_submission(candidate, broker, spec=spec, tolerance=0.2)

    assert result.replay_score == pytest.approx(attested)
    assert result.delta == 0.0
    assert result.flagged is False
    assert result.flag is None


@pytest.mark.parametrize("spec", ALL_SPECS)
def test_replay_uses_the_same_aggregation_as_finalize(spec: AggregationSpec) -> None:
    # The replay score is the SAME pipeline finalize uses: per-task aggregation
    # then the keep-policy mean over per-task scores.
    trials = _trials(2, 0.8, 0.3, 0.55)
    per_task = aggregate_task_scores(trials, mode=spec.per_task_aggregation)
    expected = keep_good_job_score(
        list(per_task.values()),
        policy=spec.keep_policy,
        drop_lowest_n=spec.drop_lowest_n,
        threshold=spec.threshold,
    )

    def broker(submission_id: str, *, k: int) -> Mapping[str, Sequence[float]]:
        return _trials(k, 0.8, 0.3, 0.55)

    candidate = AuditCandidate("sub-1", attested_score=expected, n_attempts=2)
    result = audit_submission(candidate, broker, spec=spec, tolerance=0.2)

    assert result.replay_score == expected


# --------------------------------------------------------------------------- #
# VAL-SCORE-021: a genuine mismatch beyond tolerance is flagged.
# --------------------------------------------------------------------------- #
def test_beyond_tolerance_is_flagged_with_all_four_fields() -> None:
    broker = RecordingBroker([0.2, 0.2])
    candidate = AuditCandidate("sub-42", attested_score=0.9, n_attempts=1)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert result.flagged is True
    flag = result.flag
    assert isinstance(flag, AuditMismatchFlag)
    assert flag.submission_id == "sub-42"
    assert flag.attested_score == pytest.approx(0.9)
    assert flag.replay_score == pytest.approx(0.2)
    assert flag.delta == pytest.approx(0.7)


def test_batch_audit_produces_exactly_one_flag_per_mismatch_no_duplicates() -> None:
    candidates = [
        AuditCandidate("good", attested_score=0.5, n_attempts=1),
        AuditCandidate("bad", attested_score=0.95, n_attempts=1),
    ]

    def broker(submission_id: str, *, k: int) -> Mapping[str, Sequence[float]]:
        return _trials(k, 0.5, 0.5)  # replay == 0.5 for both

    sampler = ReplayAuditSampler(attested_rate=1.0, unverified_rate=1.0, seed=0)
    results = run_replay_audit(candidates, broker, sampler=sampler, spec=OFF, tolerance=0.2)

    flags = [r.flag for r in results if r.flagged]
    assert len(flags) == 1
    assert flags[0].submission_id == "bad"
    # No duplicate flag for the same submission.
    assert len({f.submission_id for f in flags}) == len(flags)


# --------------------------------------------------------------------------- #
# VAL-SCORE-022: a within-tolerance difference (LLM variance) is NOT flagged.
# --------------------------------------------------------------------------- #
def test_within_tolerance_is_not_flagged() -> None:
    broker = RecordingBroker([0.8, 0.8])
    candidate = AuditCandidate("sub-1", attested_score=0.9, n_attempts=1)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert result.delta == pytest.approx(0.1)
    assert result.flagged is False
    assert result.flag is None


# --------------------------------------------------------------------------- #
# VAL-SCORE-023: the tolerance boundary is inclusive.
# --------------------------------------------------------------------------- #
def test_exact_tolerance_boundary_is_not_flagged() -> None:
    broker = RecordingBroker([0.7, 0.7])
    candidate = AuditCandidate("sub-1", attested_score=0.9, n_attempts=1)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert result.delta == pytest.approx(0.2)
    assert result.flagged is False  # |delta| == tolerance is inclusive


def test_smallest_delta_above_tolerance_is_flagged() -> None:
    tolerance = 0.2
    attested = 0.9
    replay = attested - tolerance - 1e-9  # just over the boundary
    broker = RecordingBroker([replay, replay])
    candidate = AuditCandidate("sub-1", attested_score=attested, n_attempts=1)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=tolerance)

    assert result.delta > tolerance
    assert result.flagged is True


def test_delta_is_symmetric_absolute_value() -> None:
    # A replay HIGHER than attested beyond tolerance flags too (abs value).
    broker = RecordingBroker([0.9, 0.9])
    candidate = AuditCandidate("sub-1", attested_score=0.2, n_attempts=1)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert result.delta == pytest.approx(0.7)
    assert result.flagged is True


# --------------------------------------------------------------------------- #
# VAL-SCORE-028: the replay re-runs the SAME k trials as the attested run.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [1, 3])
def test_replay_uses_the_attested_k(k: int) -> None:
    broker = RecordingBroker([0.6, 0.6])
    candidate = AuditCandidate("sub-1", attested_score=0.6, n_attempts=k)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert broker.calls == [("sub-1", k)]  # same k handed to the broker
    assert result.delta == 0.0  # identical outcomes => zero delta at any k


def test_replay_with_wrong_trial_count_is_rejected_not_skewed() -> None:
    # A broker that returns fewer trials than the attested k must NOT be compared
    # (an attested k=3 mean vs a replay k=1 single trial would be apples-to-
    # oranges); the audit fails closed rather than producing a skewed delta.
    def short_broker(submission_id: str, *, k: int) -> Mapping[str, Sequence[float]]:
        return {"task-0": [1.0]}  # only 1 trial regardless of k

    candidate = AuditCandidate("sub-1", attested_score=0.6, n_attempts=3)
    with pytest.raises(InvalidReplayTrialsError):
        audit_submission(candidate, short_broker, spec=OFF, tolerance=0.2)


# --------------------------------------------------------------------------- #
# VAL-SCORE-024: an audit flag never overwrites the accepted score/weights.
# --------------------------------------------------------------------------- #
async def _add_completed_job(
    session,
    *,
    hotkey: str,
    agent_hash: str,
    score: float,
    passed_tasks: int,
    total_tasks: int,
) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    submission = AgentSubmission(
        miner_hotkey=hotkey,
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        status="tb_completed",
        raw_status="tb_completed",
        effective_status="valid",
        submitted_at=now,
        created_at=now,
    )
    session.add(submission)
    await session.flush()
    job = EvaluationJob(
        job_id=f"job-{agent_hash}",
        submission_id=submission.id,
        status="completed",
        selected_tasks_json="[]",
        score=score,
        passed_tasks=passed_tasks,
        total_tasks=total_tasks,
        verdict="valid",
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id


async def test_audit_flag_does_not_overwrite_score_or_weights(database_session) -> None:
    required = effective_evaluation_task_count(core_settings.evaluation_task_count)
    async with database_session() as session:
        await _add_completed_job(
            session,
            hotkey="hk-1",
            agent_hash="a1",
            score=0.9,
            passed_tasks=1,
            total_tasks=required,
        )
        await session.commit()

    weights_before = await get_weights()
    assert weights_before == {"hk-1": 0.9}

    # The audit replays and flags a genuine mismatch (attested 0.9 vs replay 0.1).
    broker = RecordingBroker([0.1, 0.1])
    candidate = AuditCandidate("a1", attested_score=0.9, n_attempts=1)
    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)
    assert result.flagged is True

    # The flag is a separate signal: the accepted score and the weight map are
    # byte-identical before and after.
    weights_after = await get_weights()
    assert weights_after == weights_before
    async with database_session() as session:
        job = (
            await session.execute(
                EvaluationJob.__table__.select().where(EvaluationJob.job_id == "job-a1")
            )
        ).first()
    assert job is not None
    assert job.score == 0.9  # accepted score untouched by the audit


# --------------------------------------------------------------------------- #
# Wiring: sampler + compare orchestration and flag-off inertness.
# --------------------------------------------------------------------------- #
def test_run_replay_audit_only_audits_sampled_submissions() -> None:
    candidates = [AuditCandidate(f"sub-{i}", attested_score=0.9, n_attempts=1) for i in range(4)]
    broker = RecordingBroker([0.9, 0.9])
    # attested-rate 1.0 => every attested submission sampled.
    sampler = ReplayAuditSampler(attested_rate=1.0, unverified_rate=1.0, seed=0)

    results = run_replay_audit(candidates, broker, sampler=sampler, spec=OFF, tolerance=0.2)

    assert {r.submission_id for r in results} == {c.submission_id for c in candidates}
    assert len(broker.calls) == len(candidates)


def test_run_replay_audit_is_inert_when_sampler_disabled() -> None:
    candidates = [AuditCandidate(f"sub-{i}", attested_score=0.9, n_attempts=1) for i in range(4)]
    broker = RecordingBroker([0.1, 0.1])
    sampler = ReplayAuditSampler(attested_rate=1.0, unverified_rate=1.0, seed=0, enabled=False)

    results = run_replay_audit(candidates, broker, sampler=sampler, spec=OFF, tolerance=0.2)

    assert results == []
    assert broker.calls == []  # no replay dispatched while the flag is off


def test_replay_audit_from_settings_wires_sampler_spec_and_tolerance() -> None:
    settings = ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        per_task_aggregation="best-of-k",
        keep_good_tasks_policy="drop-lowest-n",
        keep_good_tasks_drop_lowest=2,
        replay_audit_tolerance=0.15,
    )
    audit = ReplayAudit.from_settings(settings)

    assert audit.tolerance == pytest.approx(0.15)
    assert audit.spec.per_task_aggregation == "best-of-k"
    assert audit.spec.keep_policy == "drop-lowest-n"
    assert audit.spec.drop_lowest_n == 2
    assert audit.sampler.enabled is True


def test_replay_audit_run_flags_beyond_tolerance_end_to_end() -> None:
    settings = ChallengeSettings(
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        replay_audit_attested_rate=1.0,
        replay_audit_unverified_rate=0.0,
        replay_audit_tolerance=0.2,
    )
    audit = ReplayAudit.from_settings(settings)
    broker = RecordingBroker([0.1, 0.1])
    candidates = [AuditCandidate("sub-1", attested_score=0.9, n_attempts=1)]

    results = audit.run(candidates, broker)

    assert len(results) == 1
    assert results[0].flagged is True


# --------------------------------------------------------------------------- #
# AggregationSpec fail-closed configuration.
# --------------------------------------------------------------------------- #
def test_aggregation_spec_from_settings_matches_finalize_knobs() -> None:
    settings = ChallengeSettings(
        per_task_aggregation="mean",
        keep_good_tasks_policy="threshold-band",
        keep_good_tasks_threshold=0.4,
    )
    spec = AggregationSpec.from_settings(settings)
    assert spec.per_task_aggregation == "mean"
    assert spec.keep_policy == "threshold-band"
    assert spec.threshold == pytest.approx(0.4)


@pytest.mark.parametrize("bad_tolerance", [-0.01, 1.01, math.inf])
def test_settings_reject_out_of_range_tolerance(bad_tolerance: float) -> None:
    with pytest.raises(ValueError):
        ChallengeSettings(replay_audit_tolerance=bad_tolerance)


def test_default_tolerance_is_a_sane_positive_fraction() -> None:
    settings = ChallengeSettings()
    assert 0.0 < settings.replay_audit_tolerance <= 1.0


def test_comparison_carries_submission_id_and_scores() -> None:
    broker = RecordingBroker([0.9, 0.9])
    candidate = AuditCandidate("sub-xyz", attested_score=0.9, n_attempts=1)

    result = audit_submission(candidate, broker, spec=OFF, tolerance=0.2)

    assert isinstance(result, ReplayComparison)
    assert result.submission_id == "sub-xyz"
    assert result.attested_score == pytest.approx(0.9)
