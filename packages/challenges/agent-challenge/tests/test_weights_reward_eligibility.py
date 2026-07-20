"""Reward-eligibility gating for get_weights.

A completed, scoring EvaluationJob earns emission weight ONLY when it covered the
FULL configured task set (``total_tasks >= required``) AND passed at least one
task (``passed_tasks >= 1``). Partial-window scores and zero-pass evals must burn
(no emission). This mirrors the production incident where a leftover 2-of-30
perfect score captured 100% of emissions.
"""

from __future__ import annotations

from datetime import UTC, datetime

from agent_challenge.core.config import settings
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.config import effective_evaluation_task_count
from agent_challenge.weights import get_weights

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
REQUIRED_TASKS = effective_evaluation_task_count(settings.evaluation_task_count)
PARTIAL_TASKS = 2


async def _add_job(
    session,
    *,
    hotkey: str,
    agent_hash: str,
    score: float,
    passed_tasks: int,
    total_tasks: int,
    created_at: datetime = NOW,
) -> int:
    submission = AgentSubmission(
        miner_hotkey=hotkey,
        name=f"agent-{agent_hash}",
        agent_hash=agent_hash,
        artifact_uri=f"/tmp/{agent_hash}.zip",
        status="tb_completed",
        raw_status="tb_completed",
        effective_status="valid",
        submitted_at=created_at,
        created_at=created_at,
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
    return submission.id


def test_required_task_count_is_the_full_configured_set():
    # Sanity: the gate compares against the full configured eval size (30 by
    # default), so PARTIAL_TASKS must be a genuine partial window.
    assert REQUIRED_TASKS >= 2
    assert PARTIAL_TASKS < REQUIRED_TASKS


async def test_partial_task_set_excluded_even_when_passing(database_session):
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="hk-partial",
            agent_hash="partial",
            score=1.0,
            passed_tasks=PARTIAL_TASKS,
            total_tasks=PARTIAL_TASKS,
        )
        await session.commit()

    assert await get_weights() == {}


async def test_zero_pass_full_eval_excluded(database_session):
    # Even a positive score with the FULL task set earns nothing when no task
    # passed - proving the exclusion is the passed-tasks gate, not the burn.
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="hk-zero-pass",
            agent_hash="zero-pass",
            score=0.9,
            passed_tasks=0,
            total_tasks=REQUIRED_TASKS,
        )
        await session.commit()

    assert await get_weights() == {}


async def test_full_task_set_with_a_pass_is_included(database_session):
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="hk-eligible",
            agent_hash="eligible",
            score=0.5,
            passed_tasks=1,
            total_tasks=REQUIRED_TASKS,
        )
        await session.commit()

    assert await get_weights() == {"hk-eligible": 0.5}


async def test_boundary_exactly_below_required_is_excluded(database_session):
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="hk-almost",
            agent_hash="almost",
            score=1.0,
            passed_tasks=1,
            total_tasks=REQUIRED_TASKS - 1,
        )
        await session.commit()

    assert await get_weights() == {}


async def test_winner_take_all_ineligible_high_score_loses_to_eligible_low_score(database_session):
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="hk-high-partial",
            agent_hash="high-partial",
            score=1.0,
            passed_tasks=PARTIAL_TASKS,
            total_tasks=PARTIAL_TASKS,
        )
        await _add_job(
            session,
            hotkey="hk-low-full",
            agent_hash="low-full",
            score=0.4,
            passed_tasks=5,
            total_tasks=REQUIRED_TASKS,
        )
        await session.commit()

    assert await get_weights() == {"hk-low-full": 0.4}


async def test_winner_take_all_all_ineligible_burns(database_session):
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="hk-partial",
            agent_hash="partial",
            score=1.0,
            passed_tasks=PARTIAL_TASKS,
            total_tasks=PARTIAL_TASKS,
        )
        await _add_job(
            session,
            hotkey="hk-zero-pass",
            agent_hash="zero-pass",
            score=0.9,
            passed_tasks=0,
            total_tasks=REQUIRED_TASKS,
        )
        await session.commit()

    assert await get_weights() == {}


async def test_best_per_hotkey_path_uses_only_eligible_jobs(database_session, monkeypatch):
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="hk-eligible",
            agent_hash="eligible",
            score=0.7,
            passed_tasks=3,
            total_tasks=REQUIRED_TASKS,
        )
        await _add_job(
            session,
            hotkey="hk-partial",
            agent_hash="partial",
            score=1.0,
            passed_tasks=PARTIAL_TASKS,
            total_tasks=PARTIAL_TASKS,
        )
        await _add_job(
            session,
            hotkey="hk-zero-pass",
            agent_hash="zero-pass",
            score=0.9,
            passed_tasks=0,
            total_tasks=REQUIRED_TASKS,
        )
        await session.commit()

    assert await get_weights() == {"hk-eligible": 0.7}


async def test_production_incident_partial_perfect_and_full_zero_pass_both_burn(database_session):
    # The real incident: "kira" scored 1.0 on a leftover 2-task window while the
    # full 30-task baseline passed nothing. Neither may earn emissions.
    async with database_session() as session:
        await _add_job(
            session,
            hotkey="kira",
            agent_hash="kira",
            score=1.0,
            passed_tasks=2,
            total_tasks=2,
        )
        await _add_job(
            session,
            hotkey="base",
            agent_hash="base",
            score=0.0,
            passed_tasks=0,
            total_tasks=REQUIRED_TASKS,
        )
        await session.commit()

    assert await get_weights() == {}
