from __future__ import annotations

from datetime import UTC, datetime

import anyio

from prism_challenge.repository import epoch_id_for


def _seed_score(
    client,
    *,
    submission_id: str,
    hotkey: str,
    epoch_id: int,
    final_score: float,
) -> None:
    repository = client.app.state.repository
    now = "2024-01-01T00:00:00+00:00"

    async def insert() -> None:
        async with repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
                (epoch_id, now, now, "open"),
            )
            await conn.execute(
                "INSERT INTO submissions("
                "id, hotkey, epoch_id, filename, code, code_hash, metadata, status, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    hotkey,
                    epoch_id,
                    "model.py",
                    "code",
                    "hash",
                    "{}",
                    "completed",
                    now,
                    now,
                ),
            )
            await conn.execute(
                "INSERT INTO scores("
                "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, "
                "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (submission_id, 0.0, 0.0, 1.0, 0.0, 0.0, final_score, "{}", now),
            )

    anyio.run(insert)


def _current_epoch_id(client) -> int:
    epoch_seconds = client.app.state.settings.epoch_seconds
    return epoch_id_for(datetime.now(UTC), epoch_seconds)


def test_leaderboard_no_arg_unchanged(client):
    """No-arg request resolves to the current/default epoch with the prior shape."""
    current = _current_epoch_id(client)
    _seed_score(
        client,
        submission_id="sub-current",
        hotkey="hk-current",
        epoch_id=current,
        final_score=0.9,
    )

    response = client.get("/v1/leaderboard")
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body.keys()) == {"epoch_id", "entries"}
    assert body["epoch_id"] == current
    assert len(body["entries"]) == 1
    entry = body["entries"][0]
    assert set(entry.keys()) == {"rank", "hotkey", "score", "submission_id"}
    assert entry == {
        "rank": 1,
        "hotkey": "hk-current",
        "score": 0.9,
        "submission_id": "sub-current",
    }


def test_leaderboard_epoch_scoped_happy(client):
    """An explicit known epoch_id returns only that epoch's entries, ordered by score."""
    current = _current_epoch_id(client)
    target_epoch = current + 1000
    _seed_score(
        client,
        submission_id="sub-current",
        hotkey="hk-current",
        epoch_id=current,
        final_score=0.5,
    )
    _seed_score(
        client,
        submission_id="sub-target-low",
        hotkey="hk-low",
        epoch_id=target_epoch,
        final_score=0.2,
    )
    _seed_score(
        client,
        submission_id="sub-target-high",
        hotkey="hk-high",
        epoch_id=target_epoch,
        final_score=0.8,
    )

    response = client.get("/v1/leaderboard", params={"epoch_id": target_epoch})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["epoch_id"] == target_epoch
    assert [e["submission_id"] for e in body["entries"]] == [
        "sub-target-high",
        "sub-target-low",
    ]
    assert [e["rank"] for e in body["entries"]] == [1, 2]
    # Current-epoch submission is excluded from the scoped result.
    assert all(e["submission_id"] != "sub-current" for e in body["entries"])


def test_leaderboard_unknown_epoch_returns_empty_200(client):
    """An unknown epoch_id returns an empty leaderboard (200, not an error)."""
    response = client.get("/v1/leaderboard", params={"epoch_id": 424242})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"epoch_id": 424242, "entries": []}


def test_leaderboard_invalid_epoch_rejected(client):
    assert client.get("/v1/leaderboard", params={"epoch_id": -1}).status_code == 422
    assert client.get("/v1/leaderboard", params={"epoch_id": "abc"}).status_code == 422
