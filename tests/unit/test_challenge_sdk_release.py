from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping, Sequence

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.version import (
    API_VERSION,
    ARTIFACT_VERSION,
    DISTRIBUTION_NAME,
    RELEASE_MANIFEST,
    SDK_CONTRACT_VERSION,
)


class _Database:
    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None


def test_release_manifest_matches_distribution_and_public_imports() -> None:
    assert RELEASE_MANIFEST.distribution_name == DISTRIBUTION_NAME
    assert RELEASE_MANIFEST.artifact_version == ARTIFACT_VERSION
    assert RELEASE_MANIFEST.sdk_contract_version == SDK_CONTRACT_VERSION
    assert RELEASE_MANIFEST.api_version == API_VERSION
    assert RELEASE_MANIFEST.release_id == f"v{ARTIFACT_VERSION}"
    assert importlib.metadata.version(DISTRIBUTION_NAME) == ARTIFACT_VERSION
    with pytest.raises(TypeError):
        RELEASE_MANIFEST.public_imports["unexpected"] = ()  # type: ignore[index]

    public_imports: Mapping[str, Sequence[str]] = RELEASE_MANIFEST.public_imports
    for module_name, symbol_names in public_imports.items():
        module = importlib.import_module(module_name)
        for symbol_name in symbol_names:
            symbol = getattr(module, symbol_name)
            assert symbol is not None
            defining_module = getattr(
                symbol, "__module__", "base.challenge_sdk.version"
            )
            assert defining_module.startswith("base.")


def test_challenge_app_projects_release_identity() -> None:
    settings = ChallengeSettings(
        slug="release-test",
        name="Release Test",
        version="2.3.4",
        shared_token="secret",
        shared_token_file=None,
    )

    async def get_weights() -> dict[str, float]:
        return {"hotkey": 1.0}

    app = create_challenge_app(
        settings=settings,
        database=_Database(),
        public_router=APIRouter(),
        get_weights_fn=get_weights,
    )

    with TestClient(app) as client:
        response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "distribution_name": DISTRIBUTION_NAME,
        "artifact_version": ARTIFACT_VERSION,
        "release_id": f"v{ARTIFACT_VERSION}",
        "api_version": API_VERSION,
        "challenge_slug": "release-test",
        "challenge_version": "2.3.4",
        "sdk_contract_version": SDK_CONTRACT_VERSION,
        "sdk_version": SDK_CONTRACT_VERSION,
        "role": "challenge",
        "capabilities": [
            "challenge.scoring",
            "challenge.ordinary_proof",
            "challenge.state",
        ],
    }
