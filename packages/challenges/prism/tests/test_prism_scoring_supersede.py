from __future__ import annotations

import pytest
from test_prism_scoring_leaderboard_determinism import _manifest

from prism_challenge.db import Database
from prism_challenge.evaluator.scoring import (
    LeaderboardRow,
    bpb_to_final_score,
    score_prequential_bpb,
)
from prism_challenge.repository import PrismRepository

EPOCH_SECONDS = 60


@pytest.fixture
async def repository(tmp_path) -> PrismRepository:
    database = Database(tmp_path / "supersede.sqlite3")
    await database.init()
    return PrismRepository(database, epoch_seconds=EPOCH_SECONDS)


async def _insert_completed(
    repository: PrismRepository,
    *,
    submission_id: str,
    hotkey: str,
    final_score: float,
    created_at: str,
    epoch_id: int = 1,
) -> None:
    async with repository.database.connect() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
            (epoch_id, created_at, created_at, "open"),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO miners(hotkey, first_seen, last_seen) VALUES (?, ?, ?)",
            (hotkey, created_at, created_at),
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
                "x",
                submission_id,
                "{}",
                "completed",
                created_at,
                created_at,
            ),
        )
        await conn.execute(
            "INSERT INTO scores("
            "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, "
            "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (submission_id, final_score, 0.0, 1.0, 0.0, 0.0, final_score, "{}", created_at),
        )


# --- repository.score_rows(): exactly one surviving submission per hotkey (the leaderboard-best) --


async def test_score_rows_dedups_to_one_row_per_hotkey(repository: PrismRepository) -> None:
    await _insert_completed(
        repository,
        submission_id="alice-worse",
        hotkey="alice",
        final_score=0.2,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _insert_completed(
        repository,
        submission_id="alice-better",
        hotkey="alice",
        final_score=0.8,
        created_at="2024-01-01T00:01:00+00:00",
    )
    await _insert_completed(
        repository,
        submission_id="bob-only",
        hotkey="bob",
        final_score=1.0,
        created_at="2024-01-01T00:00:00+00:00",
    )

    rows = await repository.score_rows(1)

    by_hotkey = {str(row["hotkey"]): row for row in rows}
    assert len(rows) == 2
    assert set(by_hotkey) == {"alice", "bob"}
    # The surviving alice row is her BEST submission.
    assert by_hotkey["alice"]["id"] == "alice-better"
    assert float(by_hotkey["alice"]["final_score"]) == pytest.approx(0.8)


# --- repository.leaderboard(): a hotkey appears at most ONCE (the best) --------------------------


async def test_leaderboard_shows_each_hotkey_once(repository: PrismRepository) -> None:
    await _insert_completed(
        repository,
        submission_id="alice-worse",
        hotkey="alice",
        final_score=0.2,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _insert_completed(
        repository,
        submission_id="alice-better",
        hotkey="alice",
        final_score=0.8,
        created_at="2024-01-01T00:01:00+00:00",
    )
    await _insert_completed(
        repository,
        submission_id="bob-only",
        hotkey="bob",
        final_score=1.0,
        created_at="2024-01-01T00:00:00+00:00",
    )

    board = await repository.leaderboard(1)

    hotkeys = [str(row["hotkey"]) for row in board]
    assert hotkeys.count("alice") == 1
    assert hotkeys.count("bob") == 1
    alice_row = next(row for row in board if row["hotkey"] == "alice")
    assert alice_row["id"] == "alice-better"
    assert float(alice_row["final_score"]) == pytest.approx(0.8)
    # Ordering is still by the canonical total-order: bob (1.0) ranks above alice (0.8).
    assert hotkeys == ["bob", "alice"]


# --- repository.leaderboard(): dedupe per hotkey BEFORE applying the display LIMIT ---------------


async def test_leaderboard_dedupes_per_hotkey_before_display_limit(
    repository: PrismRepository,
) -> None:
    # Hotkey "alice" owns the TWO highest scores; both fall inside a limit=2 display window. If the
    # SQL LIMIT were applied BEFORE the per-hotkey dedupe, the window would collapse to a single
    # distinct hotkey. Deduping first must keep a FULL window of 2 DISTINCT hotkeys.
    await _insert_completed(
        repository,
        submission_id="alice-best",
        hotkey="alice",
        final_score=0.9,
        created_at="2024-01-01T00:00:00+00:00",
    )
    await _insert_completed(
        repository,
        submission_id="alice-second",
        hotkey="alice",
        final_score=0.8,
        created_at="2024-01-02T00:00:00+00:00",
    )
    await _insert_completed(
        repository,
        submission_id="bob-only",
        hotkey="bob",
        final_score=0.7,
        created_at="2024-01-03T00:00:00+00:00",
    )
    await _insert_completed(
        repository,
        submission_id="carol-only",
        hotkey="carol",
        final_score=0.6,
        created_at="2024-01-04T00:00:00+00:00",
    )

    board = await repository.leaderboard(1, limit=2)

    # A full window of 2 DISTINCT hotkeys (alice-best, bob), not 1 collapsed alice window.
    assert [str(row["hotkey"]) for row in board] == ["alice", "bob"]
    assert [str(row["id"]) for row in board] == ["alice-best", "bob-only"]
    assert len({str(row["hotkey"]) for row in board}) == len(board) == 2

    # score_rows()/weights remain unlimited + correct: best-per-hotkey across ALL hotkeys, with
    # alice driven by her BEST (0.9) submission (VAL-CROSS-017 unaffected by the display LIMIT).
    rows = await repository.score_rows(1)
    by_hotkey = {str(row["hotkey"]): row for row in rows}
    assert set(by_hotkey) == {"alice", "bob", "carol"}
    assert by_hotkey["alice"]["id"] == "alice-best"


# --- supersede reuses the canonical tie-break (delta > epsilon grid > earliest-commit > sub id) ---


def test_dedupe_best_per_hotkey_keeps_higher_final_score() -> None:
    from prism_challenge.evaluator.scoring import dedupe_best_per_hotkey

    worse = LeaderboardRow("worse", "hk", bpb_to_final_score(2.0), "2024-01-01T00:00:00+00:00")
    # The better (lower bpb => higher final_score) submission survives even when committed LATER.
    better = LeaderboardRow("better", "hk", bpb_to_final_score(0.5), "2024-01-02T00:00:00+00:00")
    survivors = dedupe_best_per_hotkey([worse, better])
    assert [r.submission_id for r in survivors] == ["better"]


def test_dedupe_best_per_hotkey_equal_score_breaks_by_earliest_commit() -> None:
    from prism_challenge.evaluator.scoring import dedupe_best_per_hotkey

    late = LeaderboardRow("late", "hk", 0.5, "2024-01-02T00:00:00+00:00")
    early = LeaderboardRow("early", "hk", 0.5, "2024-01-01T00:00:00+00:00")
    survivors = dedupe_best_per_hotkey([late, early])
    assert [r.submission_id for r in survivors] == ["early"]


def test_dedupe_best_per_hotkey_equal_score_and_commit_breaks_by_submission_id() -> None:
    from prism_challenge.evaluator.scoring import dedupe_best_per_hotkey

    row_b = LeaderboardRow("sub-b", "hk", 0.5, "2024-01-01T00:00:00+00:00")
    row_a = LeaderboardRow("sub-a", "hk", 0.5, "2024-01-01T00:00:00+00:00")
    survivors = dedupe_best_per_hotkey([row_b, row_a])
    assert [r.submission_id for r in survivors] == ["sub-a"]


def test_dedupe_best_per_hotkey_respects_heldout_delta_above_epsilon_grid() -> None:
    from prism_challenge.evaluator.scoring import dedupe_best_per_hotkey

    # Near-equal primary bpb but different held-out delta: the held-out-delta tie-break is folded
    # into final_score ABOVE the epsilon grid, so the larger-delta submission survives the supersede
    # even though it was committed later (canonical m3 tie-break preserved).
    big_delta = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.8))
    small_delta = score_prequential_bpb(_manifest(bpb=1.0, heldout_delta=0.1))
    big_late = LeaderboardRow(
        "big-delta-late", "hk", big_delta.final_score, "2024-01-02T00:00:00+00:00"
    )
    small_early = LeaderboardRow(
        "small-delta-early", "hk", small_delta.final_score, "2024-01-01T00:00:00+00:00"
    )
    survivors = dedupe_best_per_hotkey([big_late, small_early])
    assert [r.submission_id for r in survivors] == ["big-delta-late"]


def test_dedupe_best_per_hotkey_distinct_hotkeys_all_survive() -> None:
    from prism_challenge.evaluator.scoring import dedupe_best_per_hotkey

    rows = [
        LeaderboardRow("a", "alice", 0.3, "2024-01-01T00:00:00+00:00"),
        LeaderboardRow("b", "bob", 0.7, "2024-01-01T00:00:00+00:00"),
        LeaderboardRow("c", "carol", 1.0, "2024-01-01T00:00:00+00:00"),
    ]
    survivors = dedupe_best_per_hotkey(rows)
    assert {r.hotkey for r in survivors} == {"alice", "bob", "carol"}
    assert len(survivors) == 3
