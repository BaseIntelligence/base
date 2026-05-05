from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_network.master.registry import (
    ChallengeAlreadyExistsError,
    ChallengeNotFoundError,
    ChallengeRegistry,
    FileChallengeRegistry,
    default_internal_base_url,
    default_public_proxy_base_path,
    default_sqlite_volume_name,
    record_to_admin_view,
    record_to_registry_view,
)
from platform_network.schemas.challenge import (
    ChallengeCreate,
    ChallengeStatus,
    ChallengeUpdate,
)


def payload(slug: str = "demo") -> ChallengeCreate:
    return ChallengeCreate(
        slug=slug,
        name="Demo",
        image="ghcr.io/platformnetwork/demo:1.0.0",
        version="1.0.0",
        emission_percent=10,
    )


def test_registry_update_views_and_errors() -> None:
    registry = ChallengeRegistry(network="net", api_version="2", master_uid=7)
    record, token = registry.create(payload("demo-case"))
    assert record.slug == "demo-case"
    assert token
    assert default_internal_base_url("x") == "http://challenge-x:8000"
    assert default_public_proxy_base_path("x") == "/challenges/x"
    assert default_sqlite_volume_name("a-b") == "platform_a_b_sqlite"

    with pytest.raises(ChallengeAlreadyExistsError):
        registry.create(payload("demo-case"))
    with pytest.raises(ChallengeNotFoundError):
        registry.get("missing")

    updated = registry.update(
        "demo-case",
        ChallengeUpdate(name="Updated", metadata={"k": "v"}, env={"A": "B"}),
    )
    assert updated.name == "Updated"
    assert updated.metadata == {"k": "v"}
    assert registry.list(active_only=True) == []

    registry.set_status("demo-case", ChallengeStatus.ACTIVE)
    response = registry.registry_response()
    assert response.network == "net"
    assert response.api_version == "2"
    assert response.master_uid == 7
    assert response.challenges[0].slug == "demo-case"

    admin = record_to_admin_view(registry.get("demo-case"))
    public = record_to_registry_view(registry.get("demo-case"))
    assert admin.token_hint
    assert public.public_proxy_base_path == "/challenges/demo-case"


def test_file_registry_handles_missing_and_invalid_state(tmp_path: Path) -> None:
    state = tmp_path / "registry.json"
    registry = FileChallengeRegistry(state)
    assert registry.list() == []

    state.write_text(json.dumps({"records": []}), encoding="utf-8")
    assert FileChallengeRegistry(state).list() == []

    registry = FileChallengeRegistry(state)
    record, token = registry.create(payload())
    assert record.slug == "demo"
    assert registry.get_token("demo") == token
    assert (tmp_path / "demo_challenge_token").is_file()

    reloaded = FileChallengeRegistry(state)
    assert reloaded.get("demo").slug == "demo"
    assert reloaded.get_token("missing") == ""
