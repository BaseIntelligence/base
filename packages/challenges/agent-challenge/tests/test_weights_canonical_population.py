"""Direct-weight path: one canonical eligible EvalRun population.

VAL-SCORE-012 / VAL-VERIFY-019 / VAL-CROSS-006:

* Weights and winner-take-all must consume exactly the same filtered
  population reconstructed from immutable plan-bound score bytes plus exact
  submission/review state.
* Mutable ``EvalRun.score`` / ``passed_tasks`` / ``total_tasks`` columns cannot
  alter the output.
* Invalid, suspicious, overridden-invalid, stale-version, mismatched-review,
  incomplete, or malformed runs earn no weight.
* Canonical ``final.total_tasks`` remains the complete selected set even when a
  keep-policy excludes tasks from the scoring mean.
* Fully legacy (flag-off) weight calculation remains unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.core.config import settings
from agent_challenge.core.models import (
    AgentSubmission,
    EvalRun,
    EvaluationJob,
    ReviewAssignment,
    ReviewSession,
)
from agent_challenge.evaluation.plan_scoring import (
    build_score_record_from_eval_plan,
    canonical_eval_plan_json,
)
from agent_challenge.evaluation.weights import get_weights
from agent_challenge.sdk.config import effective_evaluation_task_count

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
REQUIRED_TASKS = effective_evaluation_task_count(settings.evaluation_task_count)
MEASUREMENT = {
    "mrtd": "a" * 96,
    "rtmr0": "b" * 96,
    "rtmr1": "c" * 96,
    "rtmr2": "d" * 96,
    "compose_hash": "e" * 64,
    "os_image_hash": "f" * 64,
}
AGENT_HASH = "1" * 64
REVIEW_DIGEST = "2" * 64


def _policy(
    *,
    keep_policy: str = "off",
    drop_lowest_n: int = 0,
    threshold_f64be: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": keep_policy,
        "drop_lowest_n": drop_lowest_n,
        "threshold_f64be": threshold_f64be,
    }


def _plan(
    *,
    eval_run_id: str,
    policy: dict[str, object] | None = None,
    k: int = 1,
    task_count: int | None = None,
    authorizing_review_digest: str = REVIEW_DIGEST,
    submission_version: int = 1,
) -> dict[str, object]:
    count = task_count if task_count is not None else REQUIRED_TASKS
    policy = policy or _policy()
    task_ids = [f"task-{index:03d}" for index in range(count)]
    return {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": f"submission-{eval_run_id}",
        "submission_version": submission_version,
        "authorizing_review_digest": authorizing_review_digest,
        "agent_hash": AGENT_HASH,
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "selected_tasks": [
            {
                "task_id": task_id,
                "image_ref": "registry.example/task@sha256:" + "3" * 64,
                "task_config_sha256": "4" * 64,
            }
            for task_id in task_ids
        ],
        "k": k,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "5" * 64,
            "compose_hash": MEASUREMENT["compose_hash"],
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "6" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("6" * 64)).hexdigest(),
            "measurement": {
                "mrtd": MEASUREMENT["mrtd"],
                "rtmr0": MEASUREMENT["rtmr0"],
                "rtmr1": MEASUREMENT["rtmr1"],
                "rtmr2": MEASUREMENT["rtmr2"],
                "os_image_hash": MEASUREMENT["os_image_hash"],
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
        "key_release_nonce": f"key-nonce-{eval_run_id}",
        "score_nonce": f"score-nonce-{eval_run_id}",
        "run_token_sha256": "7" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }


def _trials_for(
    plan: dict[str, object],
    *,
    perfect_count: int,
) -> dict[str, list[float]]:
    task_ids = [item["task_id"] for item in plan["selected_tasks"]]
    trials: dict[str, list[float]] = {}
    for index, task_id in enumerate(task_ids):
        value = 1.0 if index < perfect_count else 0.0
        trials[task_id] = [value] * int(plan["k"])
    return trials


async def _attach_verified_review(
    session, submission: AgentSubmission, *, review_digest: str
) -> None:
    assignment_id = f"ra-{submission.id}-{review_digest[:8]}"
    artifact_sha = submission.agent_hash or ("8" * 64)
    review_session = ReviewSession(
        session_id=f"review-session-{submission.id}",
        submission_id=submission.id,
        artifact_sha256=artifact_sha,
        artifact_size_bytes=1,
        manifest_sha256="11" * 32,
        manifest_entries_sha256="12" * 32,
        current_assignment_id=assignment_id,
        authorizing_assignment_id=assignment_id,
    )
    session.add(review_session)
    await session.flush()
    assignment = ReviewAssignment(
        session_id=review_session.id,
        assignment_id=assignment_id,
        attempt=1,
        assignment_bytes="{}",
        assignment_digest="13" * 32,
        artifact_sha256=artifact_sha,
        rules_snapshot_sha256="14" * 32,
        rules_revision_id="rules-v1",
        review_nonce=f"review-nonce-{submission.id}",
        session_token_sha256="15" * 32,
        capability_state="revoked",
        phase="review_allowed",
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        review_report_envelope_json='{"schema_version":1}',
        review_digest=review_digest,
        review_verification_outcome_json=json.dumps(
            {
                "status": "verified_allow",
                "terminal": True,
                "retryable": False,
                "reason_code": "policy_allowed",
                "nonce_consumed": True,
                "measurement_allowlisted": True,
                "report_data_matched": True,
                "verified_at_ms": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    session.add(assignment)
    await session.flush()


async def _add_direct_run(
    session,
    *,
    hotkey: str,
    eval_run_id: str,
    plan: dict[str, object],
    trial_scores_by_task: dict[str, list[float]],
    created_at: datetime = NOW,
    raw_status: str = "tb_completed",
    effective_status: str = "valid",
    submission_version: int = 1,
    run_submission_version: int | None = None,
    authorizing_review_digest: str = REVIEW_DIGEST,
    attach_matching_review: bool = True,
    phase: str = "eval_accepted",
    verified: bool = True,
    reward_eligible: bool = True,
    result_available: bool = True,
    mutate_score: float | None = None,
    mutate_passed: int | None = None,
    mutate_total: int | None = None,
    corrupt_score_record: bool = False,
    omit_score_record: bool = False,
) -> EvalRun:
    unique_agent_hash = hashlib.sha256(eval_run_id.encode("utf-8")).hexdigest()
    submission = AgentSubmission(
        miner_hotkey=hotkey,
        name=f"agent-{eval_run_id}",
        agent_hash=unique_agent_hash,
        artifact_uri=f"/tmp/{eval_run_id}.zip",
        status="tb_completed",
        raw_status=raw_status,
        effective_status=effective_status,
        version_number=submission_version,
        submitted_at=created_at,
        created_at=created_at,
    )
    session.add(submission)
    await session.flush()
    if attach_matching_review:
        await _attach_verified_review(session, submission, review_digest=authorizing_review_digest)

    record = build_score_record_from_eval_plan(plan, trial_scores_by_task)
    plan_json = canonical_eval_plan_json(plan)
    score_json = ew.canonical_json_v1(record).decode("utf-8")
    if corrupt_score_record:
        # Structural malformed JSON for the score record surface.
        score_json = '{"schema_version":1,"not":"a-score-record"}'
    run = EvalRun(
        eval_run_id=eval_run_id,
        submission_id=submission.id,
        submission_version=(
            run_submission_version if run_submission_version is not None else submission_version
        ),
        authorizing_review_digest=authorizing_review_digest,
        plan_json=plan_json,
        plan_sha256=hashlib.sha256(plan_json.encode("utf-8")).hexdigest(),
        token_sha256=hashlib.sha256(f"token-{eval_run_id}".encode()).hexdigest(),
        phase=phase,
        verified=verified,
        reward_eligible=reward_eligible,
        result_available=result_available,
        score=(
            mutate_score
            if mutate_score is not None
            else ew.decode_score_f64be(record["final"]["job_score_f64be"])
        ),
        passed_tasks=(
            mutate_passed if mutate_passed is not None else record["final"]["passed_tasks"]
        ),
        total_tasks=(mutate_total if mutate_total is not None else record["final"]["total_tasks"]),
        canonical_score_record_json=None if omit_score_record else score_json,
        canonical_score_record_sha256=(
            None if omit_score_record else ew.score_record_digest(record)
        ),
        issued_at=created_at,
        expires_at=created_at + timedelta(hours=6),
        finalized_at=created_at + timedelta(minutes=5),
    )
    session.add(run)
    await session.flush()
    return run


@pytest.fixture
def enable_attestation(monkeypatch):
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.phala_attestation_enabled",
        True,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.attested_review_enabled",
        True,
    )


async def test_mutable_score_columns_cannot_change_weights(
    database_session, enable_attestation, monkeypatch
):
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    plan = _plan(eval_run_id="eval-mutable")
    trials = _trials_for(plan, perfect_count=max(1, REQUIRED_TASKS // 2))
    true_record = build_score_record_from_eval_plan(plan, trials)
    true_score = ew.decode_score_f64be(true_record["final"]["job_score_f64be"])

    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-mutable",
            eval_run_id="eval-mutable",
            plan=plan,
            trial_scores_by_task=trials,
            mutate_score=0.999,
            mutate_passed=REQUIRED_TASKS,
            mutate_total=1,  # under-count on the mutable column
        )
        await session.commit()

    assert await get_weights() == {"hk-mutable": true_score}


async def test_keep_policy_exclusion_preserves_full_selected_total_tasks(
    database_session, enable_attestation, monkeypatch
):
    """VAL-SCORE-012: eligibility gates on full selected set, not kept-set size."""

    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    # One perfect, rest zeros; drop-lowest keeps only the perfect task for the mean
    # but final.total_tasks / eligibility still cover the FULL selected set.
    policy = _policy(keep_policy="drop_lowest_n", drop_lowest_n=REQUIRED_TASKS - 1)
    plan = _plan(eval_run_id="eval-keep", policy=policy)
    trials = _trials_for(plan, perfect_count=1)
    record = build_score_record_from_eval_plan(plan, trials)
    assert record["final"]["total_tasks"] == REQUIRED_TASKS
    assert record["final"]["passed_tasks"] == 1
    true_score = ew.decode_score_f64be(record["final"]["job_score_f64be"])

    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-keep",
            eval_run_id="eval-keep",
            plan=plan,
            trial_scores_by_task=trials,
            # Attacker shrinks the mutable total_tasks gate.
            mutate_total=1,
            mutate_passed=1,
            mutate_score=1.0,
        )
        await session.commit()

    assert await get_weights() == {"hk-keep": true_score}


@pytest.mark.parametrize(
    "effective_status",
    ["invalid", "suspicious", "overridden_invalid"],
)
async def test_non_valid_effective_status_earns_no_weight(
    database_session, enable_attestation, effective_status
):
    plan = _plan(eval_run_id=f"eval-{effective_status}")
    trials = _trials_for(plan, perfect_count=REQUIRED_TASKS)
    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey=f"hk-{effective_status}",
            eval_run_id=f"eval-{effective_status}",
            plan=plan,
            trial_scores_by_task=trials,
            effective_status=effective_status,
        )
        await session.commit()

    assert await get_weights() == {}


async def test_stale_submission_version_earns_no_weight(database_session, enable_attestation):
    plan = _plan(eval_run_id="eval-stale-version", submission_version=1)
    trials = _trials_for(plan, perfect_count=REQUIRED_TASKS)
    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-stale",
            eval_run_id="eval-stale-version",
            plan=plan,
            trial_scores_by_task=trials,
            submission_version=2,
            run_submission_version=1,
        )
        await session.commit()

    assert await get_weights() == {}


async def test_mismatched_review_digest_earns_no_weight(database_session, enable_attestation):
    plan = _plan(eval_run_id="eval-mismatch", authorizing_review_digest=REVIEW_DIGEST)
    trials = _trials_for(plan, perfect_count=REQUIRED_TASKS)
    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-mismatch",
            eval_run_id="eval-mismatch",
            plan=plan,
            trial_scores_by_task=trials,
            authorizing_review_digest=REVIEW_DIGEST,
            attach_matching_review=True,
        )
        # Point the run at a different digest than the authorizing assignment.
        run = await session.scalar(select(EvalRun).where(EvalRun.eval_run_id == "eval-mismatch"))
        assert run is not None
        run.authorizing_review_digest = "c" * 64
        await session.commit()

    assert await get_weights() == {}


async def test_incomplete_and_malformed_runs_earn_no_weight(
    database_session, enable_attestation, monkeypatch
):
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    good_plan = _plan(eval_run_id="eval-good")
    incomplete_plan = _plan(eval_run_id="eval-incomplete")
    malformed_plan = _plan(eval_run_id="eval-malformed")
    good_trials = _trials_for(good_plan, perfect_count=REQUIRED_TASKS)
    incomplete_trials = _trials_for(incomplete_plan, perfect_count=REQUIRED_TASKS)
    malformed_trials = _trials_for(malformed_plan, perfect_count=REQUIRED_TASKS)
    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-good",
            eval_run_id="eval-good",
            plan=good_plan,
            trial_scores_by_task=good_trials,
        )
        await _add_direct_run(
            session,
            hotkey="hk-incomplete",
            eval_run_id="eval-incomplete",
            plan=incomplete_plan,
            trial_scores_by_task=incomplete_trials,
            omit_score_record=True,
        )
        await _add_direct_run(
            session,
            hotkey="hk-malformed",
            eval_run_id="eval-malformed",
            plan=malformed_plan,
            trial_scores_by_task=malformed_trials,
            corrupt_score_record=True,
        )
        await session.commit()

    assert await get_weights() == {"hk-good": 1.0}


async def test_per_hotkey_and_wta_use_identical_filtered_population(
    database_session, enable_attestation, monkeypatch
):
    good_plan_a = _plan(eval_run_id="eval-a")
    good_plan_b = _plan(eval_run_id="eval-b")
    bad_plan = _plan(eval_run_id="eval-bad-status")
    trials_pass = _trials_for(good_plan_a, perfect_count=REQUIRED_TASKS)
    trials_half = _trials_for(good_plan_b, perfect_count=max(1, REQUIRED_TASKS // 2))
    trials_bad = _trials_for(bad_plan, perfect_count=REQUIRED_TASKS)
    score_half = ew.decode_score_f64be(
        build_score_record_from_eval_plan(good_plan_b, trials_half)["final"]["job_score_f64be"]
    )

    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-a",
            eval_run_id="eval-a",
            plan=good_plan_a,
            trial_scores_by_task=trials_pass,
            created_at=NOW,
        )
        await _add_direct_run(
            session,
            hotkey="hk-b",
            eval_run_id="eval-b",
            plan=good_plan_b,
            trial_scores_by_task=trials_half,
            created_at=NOW + timedelta(minutes=1),
        )
        await _add_direct_run(
            session,
            hotkey="hk-bad",
            eval_run_id="eval-bad-status",
            plan=bad_plan,
            trial_scores_by_task=trials_bad,
            effective_status="suspicious",
            # Mutable columns claim a perfect score that must still be excluded.
            mutate_score=1.0,
            mutate_passed=REQUIRED_TASKS,
            mutate_total=REQUIRED_TASKS,
        )
        await session.commit()

    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    per_hotkey = await get_weights()
    assert per_hotkey == {"hk-a": 1.0, "hk-b": score_half}
    assert "hk-bad" not in per_hotkey

    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        True,
    )
    wta = await get_weights()
    assert wta == {"hk-a": 1.0}
    assert "hk-bad" not in wta


async def test_wta_tie_break_uses_filtered_population_only(
    database_session, enable_attestation, monkeypatch
):
    """Winner-take-all must not consult toxic unfiltered EvalRun rows."""

    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        True,
    )
    eligible_plan = _plan(eval_run_id="eval-eligible-tie")
    ineligible_plan = _plan(eval_run_id="eval-ineligible-higher")
    trials = _trials_for(eligible_plan, perfect_count=max(1, REQUIRED_TASKS // 3))
    high_trials = _trials_for(ineligible_plan, perfect_count=REQUIRED_TASKS)
    eligible_score = ew.decode_score_f64be(
        build_score_record_from_eval_plan(eligible_plan, trials)["final"]["job_score_f64be"]
    )

    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-eligible",
            eval_run_id="eval-eligible-tie",
            plan=eligible_plan,
            trial_scores_by_task=trials,
            created_at=NOW,
        )
        # Higher mutable score + earlier arrival, but invalid status: must not win.
        await _add_direct_run(
            session,
            hotkey="hk-toxic",
            eval_run_id="eval-ineligible-higher",
            plan=ineligible_plan,
            trial_scores_by_task=high_trials,
            created_at=NOW - timedelta(hours=1),
            effective_status="invalid",
            mutate_score=1.0,
        )
        await session.commit()

    assert await get_weights() == {"hk-eligible": eligible_score}


async def test_flag_off_legacy_weights_ignore_eval_runs(database_session, monkeypatch):
    """Fully legacy mode is byte-identical: only EvaluationJob rows feed weights."""

    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.phala_attestation_enabled",
        False,
    )
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    plan = _plan(eval_run_id="eval-legacy-ignore")
    trials = _trials_for(plan, perfect_count=REQUIRED_TASKS)
    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-eval-only",
            eval_run_id="eval-legacy-ignore",
            plan=plan,
            trial_scores_by_task=trials,
        )
        submission = AgentSubmission(
            miner_hotkey="hk-legacy-job",
            name="agent-legacy-job",
            agent_hash="legacy-hash",
            artifact_uri="/tmp/legacy.zip",
            status="tb_completed",
            raw_status="tb_completed",
            effective_status="valid",
            submitted_at=NOW,
            created_at=NOW,
        )
        session.add(submission)
        await session.flush()
        job = EvaluationJob(
            job_id="job-legacy",
            submission_id=submission.id,
            status="completed",
            selected_tasks_json="[]",
            score=0.42,
            passed_tasks=1,
            total_tasks=REQUIRED_TASKS,
            verdict="valid",
        )
        session.add(job)
        await session.flush()
        submission.latest_evaluation_job_id = job.id
        await session.commit()

    assert await get_weights() == {"hk-legacy-job": 0.42}


async def test_reward_eligible_flag_alone_cannot_admit_ineligible_run(
    database_session, enable_attestation
):
    """Mutable reward_eligible=True cannot override a missing verified review."""

    plan = _plan(eval_run_id="eval-no-review")
    trials = _trials_for(plan, perfect_count=REQUIRED_TASKS)
    async with database_session() as session:
        await _add_direct_run(
            session,
            hotkey="hk-no-review",
            eval_run_id="eval-no-review",
            plan=plan,
            trial_scores_by_task=trials,
            attach_matching_review=False,
            reward_eligible=True,
        )
        await session.commit()

    assert await get_weights() == {}
