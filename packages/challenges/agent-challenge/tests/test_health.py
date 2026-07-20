from __future__ import annotations


async def test_health(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "slug": "agent-challenge",
        "version": "1.0.1",
    }


async def test_version(client):
    response = await client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "1.0"
    assert payload["challenge_version"] == "1.0.1"
    assert payload["sdk_version"] == "1.0.1"
    assert "get_weights" in payload["capabilities"]
    assert "proxy_routes" in payload["capabilities"]
    assert "swe_forge" in payload["capabilities"]
