from __future__ import annotations

import anyio


def _seed_eval_jobs(client) -> None:
    repository = client.app.state.repository

    async def insert() -> None:
        async with repository.database.connect() as conn:
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, attempts, metrics, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "job-1",
                    "sub-1",
                    "l1",
                    "completed",
                    1,
                    "{}",
                    "2024-01-01T00:00:00+00:00",
                    "2024-01-01T00:05:00+00:00",
                ),
            )
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, attempts, metrics, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "job-2",
                    "sub-2",
                    "l2",
                    "failed",
                    2,
                    "{}",
                    "2024-01-02T00:00:00+00:00",
                    "2024-01-02T00:05:00+00:00",
                ),
            )
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, attempts, metrics, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "job-3",
                    "sub-3",
                    "l1",
                    "running",
                    0,
                    "{}",
                    "2024-01-03T00:00:00+00:00",
                    "2024-01-03T00:05:00+00:00",
                ),
            )

    anyio.run(insert)


def test_eval_job_health_returns_recent_statuses_newest_first(client):
    _seed_eval_jobs(client)

    response = client.get("/v1/health/eval-jobs")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 3

    # Newest first: job-3 (latest created_at) precedes job-2 then job-1.
    assert [row["id"] for row in body] == ["job-3", "job-2", "job-1"]
    assert [row["status"] for row in body] == ["running", "failed", "completed"]

    first = body[0]
    assert set(first.keys()) == {
        "id",
        "submission_id",
        "level",
        "status",
        "attempts",
        "created_at",
        "updated_at",
    }
    assert first["submission_id"] == "sub-3"
    assert first["level"] == "l1"
    assert first["attempts"] == 0
    assert first["created_at"].startswith("2024-01-03")
    assert first["updated_at"].startswith("2024-01-03")
    # No sensitive infra fields (paths/hosts/lease ids) leak into the series.
    for row in body:
        assert "run_manifest_path" not in row
        assert "artifact_output_path" not in row
        assert "target_server" not in row
        assert "gpu_lease_id" not in row


def test_eval_job_health_empty_table_returns_empty_array(client):
    response = client.get("/v1/health/eval-jobs")
    assert response.status_code == 200, response.text
    assert response.json() == []


def test_eval_job_health_respects_limit(client):
    _seed_eval_jobs(client)

    response = client.get("/v1/health/eval-jobs", params={"limit": 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 2
    assert [row["id"] for row in body] == ["job-3", "job-2"]


def test_eval_job_health_invalid_limit_rejected(client):
    assert client.get("/v1/health/eval-jobs", params={"limit": 0}).status_code == 422
    assert client.get("/v1/health/eval-jobs", params={"limit": 201}).status_code == 422
    assert client.get("/v1/health/eval-jobs", params={"limit": "abc"}).status_code == 422
