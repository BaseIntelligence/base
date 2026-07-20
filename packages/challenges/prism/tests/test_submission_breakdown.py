from __future__ import annotations

import anyio

_NOW = "2024-01-01T00:00:00+00:00"

# All six breakdown components are dedicated REAL columns on the `scores` table
# (db.py scores DDL): q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus,
# penalty, final_score. None live inside the `metrics` JSON blob. Via the LEFT
# JOIN in repository.get_submission they surface as NULL when no scores row exists.
_BREAKDOWN_FIELDS = (
    "q_arch",
    "q_recipe",
    "final_score",
    "anti_cheat_multiplier",
    "diversity_bonus",
    "penalty",
)


def _seed_submission(client, *, submission_id: str, with_score: bool) -> None:
    repository = client.app.state.repository

    async def insert() -> None:
        async with repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
                (0, _NOW, _NOW, "open"),
            )
            await conn.execute(
                "INSERT INTO submissions("
                "id, hotkey, epoch_id, filename, code, code_hash, metadata, status, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    "hk-breakdown",
                    0,
                    "model.py",
                    "code",
                    "hash",
                    "{}",
                    "completed" if with_score else "pending",
                    _NOW,
                    _NOW,
                ),
            )
            if with_score:
                await conn.execute(
                    "INSERT INTO scores("
                    "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, "
                    "penalty, final_score, metrics, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (submission_id, 0.81, 0.72, 0.95, 0.13, 0.04, 0.88, "{}", _NOW),
                )

    anyio.run(insert)


def test_submission_detail_scored_exposes_full_breakdown(client):
    """A scored submission returns all six breakdown components populated."""
    _seed_submission(client, submission_id="sub-scored", with_score=True)

    response = client.get("/v1/submissions/sub-scored")
    assert response.status_code == 200, response.text
    body = response.json()

    # Existing behaviour preserved.
    assert body["status"] == "completed"
    assert body["hotkey"] == "hk-breakdown"

    for field in _BREAKDOWN_FIELDS:
        assert field in body, field
        assert body[field] is not None, field

    assert body["q_arch"] == 0.81
    assert body["q_recipe"] == 0.72
    assert body["anti_cheat_multiplier"] == 0.95
    assert body["diversity_bonus"] == 0.13
    assert body["penalty"] == 0.04
    assert body["final_score"] == 0.88


def test_submission_detail_unscored_returns_nulls_200(client):
    """An unscored submission (no scores row) returns nulls, not 404/500."""
    _seed_submission(client, submission_id="sub-unscored", with_score=False)

    response = client.get("/v1/submissions/sub-unscored")
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "pending"
    for field in _BREAKDOWN_FIELDS:
        assert field in body, field
        assert body[field] is None, field
