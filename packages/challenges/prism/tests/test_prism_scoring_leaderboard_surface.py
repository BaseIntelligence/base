from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import anyio

from prism_challenge.repository import epoch_id_for


def _seed_submission(
    client,
    *,
    submission_id: str,
    hotkey: str,
    epoch_id: int,
    status: str,
    created_at: str,
    final_score: float | None = None,
    metrics: dict | None = None,
) -> None:
    """Insert a submission (and, when ``final_score`` is given, a scores row) for the epoch."""
    repository = client.app.state.repository

    async def insert() -> None:
        async with repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
                (epoch_id, created_at, created_at, "open"),
            )
            await conn.execute(
                "INSERT INTO submissions("
                "id, hotkey, epoch_id, filename, code, code_hash, metadata, status, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    hotkey,
                    epoch_id,
                    "project.zip",
                    "code",
                    f"hash-{submission_id}",
                    "{}",
                    status,
                    created_at,
                    created_at,
                ),
            )
            if final_score is not None:
                await conn.execute(
                    "INSERT INTO scores("
                    "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, "
                    "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        submission_id,
                        final_score,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        final_score,
                        json.dumps(metrics or {}),
                        created_at,
                    ),
                )

    anyio.run(insert)


def _current_epoch_id(client) -> int:
    return epoch_id_for(datetime.now(UTC), client.app.state.settings.epoch_seconds)


def _scores_count(client, submission_id: str) -> int:
    db_path = str(client.app.state.settings.database_url).split("///", 1)[1]
    conn = sqlite3.connect(db_path)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM scores WHERE submission_id=?", (submission_id,)
            ).fetchone()[0]
        )
    finally:
        conn.close()


# --- VAL-SCORE-011: score appears on the leaderboard with the miner hotkey -----------------------


def test_scoring_leaderboard_surfaces_bpb_with_hotkey(client) -> None:
    epoch = _current_epoch_id(client) + 5000
    _seed_submission(
        client,
        submission_id="sub-scored",
        hotkey="miner-hotkey-1",
        epoch_id=epoch,
        status="completed",
        created_at="2024-01-01T00:00:00+00:00",
        final_score=0.625,
        metrics={"bits_per_byte": 0.6, "prequential_bpb": 0.6, "final_score": 0.625},
    )
    body = client.get("/v1/leaderboard", params={"epoch_id": epoch}).json()
    assert len(body["entries"]) == 1
    entry = body["entries"][0]
    assert entry["hotkey"] == "miner-hotkey-1"
    assert entry["submission_id"] == "sub-scored"
    assert isinstance(entry["score"], int | float)
    assert entry["score"] == 0.625


# --- VAL-SCORE-012: leaderboard ordering reflects bpb / learning ---------------------------------


def test_scoring_leaderboard_lower_bpb_ranks_above_higher_bpb(client) -> None:
    epoch = _current_epoch_id(client) + 5100
    # final_score is a monotone-decreasing transform of bpb -> lower bpb has the higher final_score.
    _seed_submission(
        client,
        submission_id="worse-bpb",
        hotkey="hk-worse",
        epoch_id=epoch,
        status="completed",
        created_at="2024-01-01T00:00:00+00:00",
        final_score=1.0 / (1.0 + 2.0),
        metrics={"bits_per_byte": 2.0},
    )
    _seed_submission(
        client,
        submission_id="better-bpb",
        hotkey="hk-better",
        epoch_id=epoch,
        status="completed",
        created_at="2024-01-02T00:00:00+00:00",
        final_score=1.0 / (1.0 + 0.5),
        metrics={"bits_per_byte": 0.5},
    )
    body = client.get("/v1/leaderboard", params={"epoch_id": epoch}).json()
    assert [e["submission_id"] for e in body["entries"]] == ["better-bpb", "worse-bpb"]
    assert [e["rank"] for e in body["entries"]] == [1, 2]


# --- VAL-SCORE-019: deterministic final tie-break on the live leaderboard ------------------------


def test_scoring_leaderboard_equal_score_tie_break_earliest_commit(client) -> None:
    epoch = _current_epoch_id(client) + 5200
    _seed_submission(
        client,
        submission_id="tie-late",
        hotkey="hk-late",
        epoch_id=epoch,
        status="completed",
        created_at="2024-03-02T00:00:00+00:00",
        final_score=0.5,
        metrics={"bits_per_byte": 1.0, "heldout_delta": 0.3},
    )
    _seed_submission(
        client,
        submission_id="tie-early",
        hotkey="hk-early",
        epoch_id=epoch,
        status="completed",
        created_at="2024-03-01T00:00:00+00:00",
        final_score=0.5,
        metrics={"bits_per_byte": 1.0, "heldout_delta": 0.3},
    )
    first = client.get("/v1/leaderboard", params={"epoch_id": epoch}).json()
    second = client.get("/v1/leaderboard", params={"epoch_id": epoch}).json()
    order_first = [e["submission_id"] for e in first["entries"]]
    order_second = [e["submission_id"] for e in second["entries"]]
    assert order_first == ["tie-early", "tie-late"]
    assert order_first == order_second  # reproducible across re-queries


# --- VAL-SCORE-016: a rejected / failed submission gets NO score (never a 0-that-ranks) ----------


def test_scoring_failed_and_rejected_get_no_score_and_absent_from_leaderboard(client) -> None:
    epoch = _current_epoch_id(client) + 5300
    _seed_submission(
        client,
        submission_id="ok-scored",
        hotkey="hk-ok",
        epoch_id=epoch,
        status="completed",
        created_at="2024-04-01T00:00:00+00:00",
        final_score=0.4,
        metrics={"bits_per_byte": 1.5},
    )
    _seed_submission(
        client,
        submission_id="failed-sub",
        hotkey="hk-failed",
        epoch_id=epoch,
        status="failed",
        created_at="2024-04-02T00:00:00+00:00",
    )
    _seed_submission(
        client,
        submission_id="rejected-sub",
        hotkey="hk-rejected",
        epoch_id=epoch,
        status="rejected",
        created_at="2024-04-03T00:00:00+00:00",
    )

    # No scores row exists for the failed/rejected submissions.
    assert _scores_count(client, "failed-sub") == 0
    assert _scores_count(client, "rejected-sub") == 0

    # final_score is null on the status surface for both.
    failed_status = client.get("/v1/submissions/failed-sub").json()
    rejected_status = client.get("/v1/submissions/rejected-sub").json()
    assert failed_status["status"] == "failed"
    assert failed_status["final_score"] is None
    assert rejected_status["status"] == "rejected"
    assert rejected_status["final_score"] is None

    # Neither ranks on the leaderboard; only the completed+scored submission appears.
    body = client.get("/v1/leaderboard", params={"epoch_id": epoch}).json()
    surfaced = {e["submission_id"] for e in body["entries"]}
    assert surfaced == {"ok-scored"}
    assert "failed-sub" not in surfaced
    assert "rejected-sub" not in surfaced
