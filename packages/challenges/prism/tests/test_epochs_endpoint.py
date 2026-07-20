from __future__ import annotations

import anyio


def _seed_epochs(client) -> None:
    repository = client.app.state.repository

    async def insert() -> None:
        async with repository.database.connect() as conn:
            await conn.execute(
                "INSERT INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
                (1, "2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00", "closed"),
            )
            await conn.execute(
                "INSERT INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
                (2, "2024-01-02T00:00:00+00:00", "2024-01-02T01:00:00+00:00", "open"),
            )

    anyio.run(insert)


def test_list_epochs_returns_rows_newest_first(client):
    _seed_epochs(client)

    response = client.get("/v1/epochs")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 2

    # Newest first: epoch 2 (later starts_at) precedes epoch 1.
    assert [row["id"] for row in body] == [2, 1]
    first = body[0]
    assert set(first.keys()) == {"id", "starts_at", "ends_at", "status"}
    assert first["id"] == 2
    assert first["status"] == "open"
    assert first["starts_at"].startswith("2024-01-02")
    assert first["ends_at"].startswith("2024-01-02")


def test_list_epochs_empty_table_returns_empty_array(client):
    response = client.get("/v1/epochs")
    assert response.status_code == 200, response.text
    assert response.json() == []


def test_list_epochs_respects_limit(client):
    _seed_epochs(client)

    response = client.get("/v1/epochs", params={"limit": 1})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == 2


def test_list_epochs_invalid_limit_rejected(client):
    assert client.get("/v1/epochs", params={"limit": 0}).status_code == 422
    assert client.get("/v1/epochs", params={"limit": 201}).status_code == 422
    assert client.get("/v1/epochs", params={"limit": "abc"}).status_code == 422
