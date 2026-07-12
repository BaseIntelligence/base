from __future__ import annotations

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from base.challenge_sdk.api_manifest import API_MANIFEST
from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.compatibility import is_compatible
from base.challenge_sdk.config import ChallengeSettings


class _Database:
    async def init(self) -> None:
        return None

    async def close(self) -> None:
        return None


def test_semver_ranges_are_conservative_and_deterministic() -> None:
    assert is_compatible("1.0.0", "^1.0.0")
    assert is_compatible("1.9.4", "^1.0.0")
    assert not is_compatible("2.0.0", "^1.0.0")
    assert is_compatible("1.2.3", ">=1.0 <2.0")
    assert not is_compatible("1.0.0", "not-a-range")


def test_manifest_exposes_immutable_surface_helpers() -> None:
    assert ("GET", "/health") in API_MANIFEST.route_keys()
    assert "base validator run" in API_MANIFEST.cli_names()
    assert "validator.own_set_weights" in API_MANIFEST.capability_tokens()


def test_invalid_compatibility_range_fails_settings_validation() -> None:
    with pytest.raises(ValueError, match="Incompatible SDK version"):
        ChallengeSettings(
            sdk_compatibility_range="<1.0.0",
            shared_token="test-token",
            shared_token_file=None,
        )


def test_missing_secret_refuses_app_start_before_database_init() -> None:
    database = _Database()
    app = create_challenge_app(
        settings=ChallengeSettings(
            shared_token=None,
            shared_token_file="/tmp/base-sdk-test-secret-that-does-not-exist",
        ),
        database=database,
        public_router=APIRouter(),
        get_weights_fn=_weights,
    )
    with pytest.raises(RuntimeError, match="secret is missing"):
        with TestClient(app):
            pass


async def _weights() -> dict[str, float]:
    return {"5CtestHotkey": 1.0}
