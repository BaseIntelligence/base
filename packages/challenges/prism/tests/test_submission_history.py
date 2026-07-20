from __future__ import annotations

from datetime import UTC, datetime, timedelta

import anyio


def _seed_submissions(client, rows: list[tuple[str, str]]) -> None:
    """rows = list of (submission_id, created_at ISO-8601 TEXT)."""
    repository = client.app.state.repository

    async def insert() -> None:
        async with repository.database.connect() as conn:
            for submission_id, created_at in rows:
                await conn.execute(
                    "INSERT INTO submissions("
                    "id, hotkey, epoch_id, filename, code, code_hash, metadata, "
                    "status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        submission_id,
                        "hk",
                        1,
                        "model.py",
                        "print(1)",
                        "hash",
                        "{}",
                        "completed",
                        created_at,
                        created_at,
                    ),
                )

    anyio.run(insert)


def test_submission_history_counts_per_day(client):
    now = datetime.now(UTC)
    today = now.date().isoformat()
    yesterday = (now - timedelta(days=1)).date().isoformat()
    two_days = (now - timedelta(days=2)).date().isoformat()

    _seed_submissions(
        client,
        [
            ("s1", f"{today}T01:00:00+00:00"),
            ("s2", f"{today}T09:30:00.500000+00:00"),
            ("s3", f"{yesterday}T23:59:59+00:00"),
            ("s4", f"{two_days}T12:00:00+00:00"),
        ],
    )

    response = client.get("/v1/submissions/history")
    assert response.status_code == 200, response.text
    body = response.json()

    # Ascending by date, one bucket per distinct day, correct counts.
    assert body == [
        {"date": two_days, "count": 1},
        {"date": yesterday, "count": 1},
        {"date": today, "count": 2},
    ]

    # Only date + count aggregates — no submission content leaks.
    leaked = repr(body)
    assert "hotkey" not in leaked
    assert "s1" not in leaked
    assert "print(1)" not in leaked
    assert "code" not in leaked


def test_submission_history_empty_returns_empty_list(client):
    response = client.get("/v1/submissions/history")
    assert response.status_code == 200, response.text
    assert response.json() == []


def test_submission_history_outside_range_excluded(client):
    now = datetime.now(UTC)
    old = (now - timedelta(days=120)).date().isoformat()
    recent = now.date().isoformat()

    _seed_submissions(
        client,
        [
            ("old", f"{old}T00:00:00+00:00"),
            ("new", f"{recent}T00:00:00+00:00"),
        ],
    )

    # Default days=90 excludes the 120-day-old submission.
    response = client.get("/v1/submissions/history")
    assert response.status_code == 200, response.text
    assert response.json() == [{"date": recent, "count": 1}]


def test_submission_history_invalid_days_returns_422(client):
    for bad in ("0", "367", "-1", "abc"):
        response = client.get("/v1/submissions/history", params={"days": bad})
        assert response.status_code == 422, (bad, response.text)


def test_submission_history_does_not_collide_with_submission_id(client):
    # GET /v1/submissions/history must resolve to the history endpoint,
    # NOT the /submissions/{submission_id} path param (which would 404).
    response = client.get("/v1/submissions/history")
    assert response.status_code == 200, response.text
    assert isinstance(response.json(), list)
    assert "submission not found" not in response.text
