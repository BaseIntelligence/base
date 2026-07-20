"""Winner-take-all emission weights for get_weights.

WTA collapses the per-challenge map to a single winning hotkey (highest best
score). Equal top scores resolve to the earliest-arrived submission
(``created_at`` ASC, then ``id`` ASC). A non-positive top score or no qualifying
jobs burns (empty map). Toggling ``weights_winner_take_all`` off restores the
historical best-per-hotkey map.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent_challenge.core.config import settings
from agent_challenge.models import AgentSubmission, EvaluationJob
from agent_challenge.sdk.config import effective_evaluation_task_count
from agent_challenge.weights import get_weights

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
FULL_TASK_COUNT = effective_evaluation_task_count(settings.evaluation_task_count)


async def _add_scored_submission(
    session,
    *,
    hotkey: str,
    agent_hash: str,
    score: float,
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
        passed_tasks=1,
        total_tasks=FULL_TASK_COUNT,
        verdict="valid",
    )
    session.add(job)
    await session.flush()
    submission.latest_evaluation_job_id = job.id
    return submission.id


async def test_winner_take_all_single_winner_among_many(database_session):
    async with database_session() as session:
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a", score=0.4)
        await _add_scored_submission(session, hotkey="hk-b", agent_hash="b", score=0.9)
        await _add_scored_submission(session, hotkey="hk-c", agent_hash="c", score=0.7)
        await session.commit()

    assert await get_weights() == {"hk-b": 0.9}


async def test_winner_take_all_single_hotkey_with_multiple_submissions(database_session):
    # WTA-on when a SINGLE hotkey has SEVERAL qualifying submissions: the whole
    # emission collapses to that hotkey at its BEST score - never a lower
    # submission of the same hotkey, and never more than one entry for it, even
    # when it competes against another hotkey.
    async with database_session() as session:
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a-low", score=0.4)
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a-high", score=0.9)
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a-mid", score=0.7)
        await _add_scored_submission(session, hotkey="hk-b", agent_hash="b", score=0.5)
        await session.commit()

    assert await get_weights() == {"hk-a": 0.9}


async def test_winner_take_all_earliest_submission_breaks_score_tie(database_session):
    async with database_session() as session:
        await _add_scored_submission(
            session,
            hotkey="hk-late",
            agent_hash="late",
            score=0.8,
            created_at=NOW + timedelta(hours=1),
        )
        await _add_scored_submission(
            session,
            hotkey="hk-early",
            agent_hash="early",
            score=0.8,
            created_at=NOW,
        )
        await session.commit()

    assert await get_weights() == {"hk-early": 0.8}


async def test_winner_take_all_lower_id_breaks_equal_created_at_tie(database_session):
    async with database_session() as session:
        await _add_scored_submission(session, hotkey="hk-first", agent_hash="first", score=0.6)
        await _add_scored_submission(session, hotkey="hk-second", agent_hash="second", score=0.6)
        await session.commit()

    assert await get_weights() == {"hk-first": 0.6}


async def test_winner_take_all_all_zero_scores_burn(database_session):
    async with database_session() as session:
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a", score=0.0)
        await _add_scored_submission(session, hotkey="hk-b", agent_hash="b", score=0.0)
        await session.commit()

    assert await get_weights() == {}


async def test_winner_take_all_negative_top_score_burns(database_session):
    async with database_session() as session:
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a", score=-0.5)
        await _add_scored_submission(session, hotkey="hk-b", agent_hash="b", score=-0.1)
        await session.commit()

    assert await get_weights() == {}


async def test_winner_take_all_positive_winner_survives_non_positive_field(database_session):
    async with database_session() as session:
        await _add_scored_submission(session, hotkey="hk-zero", agent_hash="zero", score=0.0)
        await _add_scored_submission(session, hotkey="hk-win", agent_hash="win", score=0.5)
        await _add_scored_submission(session, hotkey="hk-neg", agent_hash="neg", score=-0.3)
        await session.commit()

    assert await get_weights() == {"hk-win": 0.5}


async def test_winner_take_all_no_qualifying_jobs_burn(database_session):
    assert await get_weights() == {}


async def test_toggle_off_restores_best_per_hotkey(database_session, monkeypatch):
    monkeypatch.setattr(
        "agent_challenge.evaluation.weights.settings.weights_winner_take_all",
        False,
    )
    async with database_session() as session:
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a-low", score=0.4)
        await _add_scored_submission(session, hotkey="hk-a", agent_hash="a-high", score=0.6)
        await _add_scored_submission(session, hotkey="hk-b", agent_hash="b", score=0.9)
        await session.commit()

    assert await get_weights() == {"hk-a": 0.6, "hk-b": 0.9}
