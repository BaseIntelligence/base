from __future__ import annotations

import pytest
from base.challenge_sdk.auth import build_internal_auth_dependency, load_shared_token
from base.challenge_sdk.config import ChallengeSettings
from fastapi import HTTPException
from pydantic import ValidationError


def _settings(**overrides) -> ChallengeSettings:
    base = {"shared_token": None, "shared_token_file": None}
    base.update(overrides)
    return ChallengeSettings(**base)


def test_load_shared_token_returns_inline_token():
    assert load_shared_token(_settings(shared_token="inline-secret")) == "inline-secret"


def test_load_shared_token_none_when_nothing_configured():
    with pytest.raises(ValidationError, match="challenge authentication"):
        _settings()


def test_load_shared_token_none_when_file_missing(tmp_path):
    missing = tmp_path / "absent"
    assert load_shared_token(_settings(shared_token_file=str(missing))) is None


def test_load_shared_token_reads_file(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("  file-secret\n", encoding="utf-8")
    assert load_shared_token(_settings(shared_token_file=str(token_file))) == "file-secret"


def test_load_shared_token_empty_file_is_none(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("   \n", encoding="utf-8")
    assert load_shared_token(_settings(shared_token_file=str(token_file))) is None


async def test_dependency_503_when_no_token_configured():
    verify = build_internal_auth_dependency(
        _settings(shared_token_file="/tmp/missing-prism-test-token")
    )
    with pytest.raises(HTTPException) as exc:
        await verify(authorization="Bearer anything", challenge_slug="prism")
    assert exc.value.status_code == 503


async def test_dependency_401_when_authorization_missing():
    verify = build_internal_auth_dependency(_settings(shared_token="t"))
    with pytest.raises(HTTPException) as exc:
        await verify(authorization=None, challenge_slug="prism")
    assert exc.value.status_code == 401


async def test_dependency_401_when_authorization_wrong():
    verify = build_internal_auth_dependency(_settings(shared_token="t"))
    with pytest.raises(HTTPException) as exc:
        await verify(authorization="Bearer wrong", challenge_slug="prism")
    assert exc.value.status_code == 401


async def test_dependency_400_when_slug_mismatch():
    verify = build_internal_auth_dependency(_settings(shared_token="t", slug="prism"))
    with pytest.raises(HTTPException) as exc:
        await verify(authorization="Bearer t", challenge_slug="other")
    assert exc.value.status_code == 400


async def test_dependency_passes_with_valid_token_and_slug():
    verify = build_internal_auth_dependency(_settings(shared_token="t", slug="prism"))
    assert await verify(authorization="Bearer t", challenge_slug="prism") is None
