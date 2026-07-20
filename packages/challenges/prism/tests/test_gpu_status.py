from __future__ import annotations

import anyio


def _seed_gpu_leases(client) -> None:
    repository = client.app.state.repository

    rows = [
        # id, submission_id, gpu_count, max_gpu_count, mode, tier, status
        ("lease-1", "sub-1", 4, 8, "official", "a100", "active"),
        ("lease-2", "sub-2", 2, 8, "official", "a100", "active"),
        ("lease-3", "sub-3", 0, 8, "official", "h100", "pending"),
        ("lease-4", "sub-4", 1, 8, "official", "h100", "released"),
    ]

    async def insert() -> None:
        async with repository.database.connect() as conn:
            for lease_id, submission_id, gpu_count, max_gpu, mode, tier, status in rows:
                await conn.execute(
                    "INSERT INTO gpu_leases("
                    "id, submission_id, gpu_count, min_gpu_count, max_gpu_count, "
                    "requested_gpu_count, mode, tier, score_eligible, status, "
                    "created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        lease_id,
                        submission_id,
                        gpu_count,
                        1,
                        max_gpu,
                        gpu_count,
                        mode,
                        tier,
                        1,
                        status,
                        "2024-01-01T00:00:00+00:00",
                        "2024-01-01T00:05:00+00:00",
                    ),
                )

    anyio.run(insert)


def test_gpu_status_summarizes_leases(client):
    _seed_gpu_leases(client)

    response = client.get("/v1/gpu/status")
    assert response.status_code == 200, response.text
    body = response.json()

    assert set(body.keys()) == {"total_gpus", "active_leases", "by_status", "by_tier"}
    # Two active leases hold 4 + 2 = 6 GPUs in use.
    assert body["total_gpus"] == 6
    assert body["active_leases"] == 2
    assert body["by_status"] == {"active": 2, "pending": 1, "released": 1}
    assert body["by_tier"] == {"a100": 2, "h100": 2}

    # Only harmless aggregates/enums — no host/path/lease identifiers leak.
    leaked = repr(body)
    assert "target_server" not in leaked
    assert "submission_id" not in leaked
    assert "lease-1" not in leaked
    assert "device_ids" not in leaked


def test_gpu_status_empty_table_returns_safe_defaults(client):
    response = client.get("/v1/gpu/status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "total_gpus": 0,
        "active_leases": 0,
        "by_status": {},
        "by_tier": {},
    }
