"""Anti-cheat: REVIEW_API_BASE_URL hard-pin to joinbase agent-challenge.

Covers VAL-ACURL-001..007,012..014: constant pin, prod refuse matrix, deploy
encrypt refuse, default joinbase, honest accept, allowed_envs force, dev flag.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from unittest import mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from agent_challenge.review import compose as review_compose
from agent_challenge.review.canonical import canonical_sha256
from agent_challenge.review.schemas import ReviewInputConfig, build_review_assignment
from agent_challenge.review.urls import (
    ALLOW_DEV_REVIEW_URLS_ENV,
    DEFAULT_REVIEW_API_BASE_URL,
    PINNED_REVIEW_API_BASE_URL,
    ReviewApiBaseUrlError,
    assert_pinned_review_api_base_url,
    is_pinned_review_api_base_url,
    resolve_review_api_base_url,
)
from agent_challenge.selfdeploy.review import (
    REVIEW_ALLOWED_ENVS,
    ReviewDeploymentError,
    build_review_deployment_plan,
    encrypt_review_secrets,
)

JOINBASE = "https://chain.joinbase.ai/challenges/agent-challenge"
REVIEW_IMAGE = "docker.io/example/agent-challenge-review@sha256:" + ("a" * 64)
MEASUREMENT = {
    "mrtd": "01" * 48,
    "rtmr0": "02" * 48,
    "rtmr1": "03" * 48,
    "rtmr2": "04" * 48,
    "os_image_hash": "05" * 32,
    "key_provider": "phala",
    "vm_shape": "tdx.small",
}

REFUSE_MATRIX: tuple[str, ...] = (
    "http://chain.joinbase.ai/challenges/agent-challenge",
    "https://evil.example/challenges/agent-challenge",
    "https://chain.platform.network/challenges/agent-challenge",
    "https://joinbase.ai.evil.example/challenges/agent-challenge",
    "https://chain.joinbase.ai.evil/challenges/agent-challenge",
    "https://127.0.0.1/challenges/agent-challenge",
    "https://localhost/challenges/agent-challenge",
    "https://86.38.238.235/challenges/agent-challenge",
    "https://chain.joinbase.ai/challenges/prism",
    "https://chain.joinbase.ai/",
    "//chain.joinbase.ai/challenges/agent-challenge",
    "chain.joinbase.ai/challenges/agent-challenge",
)


def _load_review_runtime():
    path = Path(__file__).resolve().parents[1] / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_hardcode_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _review_compose() -> dict[str, object]:
    return review_compose.generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity="agent-challenge-review-v1",
    )


def _allowlisted(compose_hash: str | None = None) -> dict[str, str]:
    return {
        "mrtd": MEASUREMENT["mrtd"],
        "rtmr0": MEASUREMENT["rtmr0"],
        "rtmr1": MEASUREMENT["rtmr1"],
        "rtmr2": MEASUREMENT["rtmr2"],
        "compose_hash": compose_hash or review_compose.review_app_compose_hash(_review_compose()),
        "os_image_hash": MEASUREMENT["os_image_hash"],
    }


def _assignment(*, public_key_hex: str) -> tuple[dict[str, object], str]:
    compose_hash = review_compose.review_app_compose_hash(_review_compose())
    entries = (_allowlisted(compose_hash),)
    config = ReviewInputConfig(
        image_ref=REVIEW_IMAGE,
        compose_hash=compose_hash,
        app_identity="agent-challenge-review-v1",
        kms_public_key_hex=public_key_hex,
        measurement=MEASUREMENT,
        measurement_allowlist=entries,
        measurement_allowlist_sha256=canonical_sha256({"entries": list(entries)}),
    )
    token = "review-session-token-sentinel"
    assignment, _bytes, _digest = build_review_assignment(
        session_id="rs-review-url",
        assignment_id="ra-review-url",
        attempt=1,
        submission_id="42",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 1,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": "/review/v1/assignments/ra-review-url/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce="rn-review-url",
        issued_at_ms=1,
        expires_at_ms=2,
        session_token_sha256=__import__("hashlib").sha256(token.encode()).hexdigest(),
        config=config,
    )
    return assignment, token


def test_pinned_constant_is_joinbase_agent_challenge() -> None:
    """VAL-ACURL-001/004: product constant is exactly joinbase challenge path."""

    assert PINNED_REVIEW_API_BASE_URL == JOINBASE
    assert DEFAULT_REVIEW_API_BASE_URL == JOINBASE
    runtime = _load_review_runtime()
    assert runtime.DEFAULT_REVIEW_API_BASE_URL == JOINBASE
    assert runtime.PINNED_REVIEW_API_BASE_URL == JOINBASE
    assert "platform.network" not in PINNED_REVIEW_API_BASE_URL


def test_default_resolve_without_env_is_joinbase(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAL-ACURL-004: unset env → joinbase pin."""

    monkeypatch.delenv("REVIEW_API_BASE_URL", raising=False)
    monkeypatch.delenv(ALLOW_DEV_REVIEW_URLS_ENV, raising=False)
    assert resolve_review_api_base_url() == JOINBASE
    assert resolve_review_api_base_url(explicit=None, environ={}) == JOINBASE


def test_honest_joinbase_accepted_with_trailing_slash() -> None:
    """VAL-ACURL-005: exact pin and trailing-slash form accepted."""

    assert assert_pinned_review_api_base_url(JOINBASE) == JOINBASE
    assert assert_pinned_review_api_base_url(JOINBASE + "/") == JOINBASE
    assert is_pinned_review_api_base_url(JOINBASE + "/")
    assert resolve_review_api_base_url(environ={"REVIEW_API_BASE_URL": JOINBASE + "/"}) == JOINBASE


@pytest.mark.parametrize("bad", REFUSE_MATRIX)
def test_prod_assert_refuses_non_joinbase_matrix(bad: str) -> None:
    """VAL-ACURL-002/006/013: refuse host/scheme/path cheat classes in prod."""

    with pytest.raises(ReviewApiBaseUrlError, match="exactly"):
        assert_pinned_review_api_base_url(bad, allow_dev=False)
    with pytest.raises(ReviewApiBaseUrlError):
        resolve_review_api_base_url(
            explicit=bad,
            environ={ALLOW_DEV_REVIEW_URLS_ENV: "0"},
        )


@pytest.mark.parametrize("bad", REFUSE_MATRIX)
def test_encrypt_review_secrets_refuses_non_joinbase(bad: str) -> None:
    """VAL-ACURL-003/013: selfdeploy encrypt fails closed on non-joinbase."""

    private_key = X25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    assignment, token = _assignment(public_key_hex=public_key_hex)
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    with pytest.raises(ReviewDeploymentError, match="exactly|joinbase|REVIEW_API_BASE_URL"):
        encrypt_review_secrets(
            plan,
            {
                "OPENROUTER_API_KEY": "or-key",
                "REVIEW_API_BASE_URL": bad,
                "REVIEW_SESSION_TOKEN": token,
            },
        )


def test_encrypt_review_secrets_accepts_honest_joinbase() -> None:
    """VAL-ACURL-005/014: honest joinbase encrypt succeeds and stores pin."""

    private_key = X25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    assignment, token = _assignment(public_key_hex=public_key_hex)
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    encrypted = encrypt_review_secrets(
        plan,
        {
            "OPENROUTER_API_KEY": "or-key",
            "REVIEW_API_BASE_URL": JOINBASE + "/",
            "REVIEW_SESSION_TOKEN": token,
        },
    )
    assert encrypted.env_keys == REVIEW_ALLOWED_ENVS
    assert "REVIEW_API_BASE_URL" in REVIEW_ALLOWED_ENVS
    # Ciphertext opaque; force path already validated joinbase.


def test_review_allowed_envs_still_lists_review_api_but_authority_forced() -> None:
    """VAL-ACURL-007: REVIEW_API_BASE_URL stays in allowlist for compose_hash,
    but cannot change authority (force/refuse in encrypt)."""

    assert "REVIEW_API_BASE_URL" in review_compose.REVIEW_ALLOWED_ENVS
    assert review_compose.REVIEW_ALLOWED_ENVS == (
        "OPENROUTER_API_KEY",
        "REVIEW_API_BASE_URL",
        "REVIEW_SESSION_TOKEN",
    )


def test_runtime_main_refuses_non_joinbase_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAL-ACURL-002: measured runtime main exits refuse on evil env."""

    runtime = _load_review_runtime()
    env = {
        "REVIEW_SESSION_TOKEN": "tok",
        "OPENROUTER_API_KEY": "or",
        "REVIEW_API_BASE_URL": "https://evil.example/callback",
    }
    stderr = io.StringIO()
    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch.object(runtime.sys, "stderr", new=stderr),
        mock.patch.object(runtime, "run_assignment") as run_assignment,
    ):
        rc = runtime.main(["--run-assignment"])
    assert rc == 2
    run_assignment.assert_not_called()
    payload = json.loads(stderr.getvalue())
    assert payload["error"] == "review_api_base_url_refused"
    assert payload["pinned"] == JOINBASE


def test_runtime_main_defaults_to_joinbase_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAL-ACURL-004/014: unset REVIEW_API_BASE_URL → joinbase run_assignment."""

    runtime = _load_review_runtime()
    monkeypatch.delenv("REVIEW_API_BASE_URL", raising=False)
    monkeypatch.delenv(ALLOW_DEV_REVIEW_URLS_ENV, raising=False)
    env = {
        "REVIEW_SESSION_TOKEN": "tok",
        "OPENROUTER_API_KEY": "or",
    }
    captured: dict[str, str] = {}

    def _fake_run(*, api_base_url: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        captured["api_base_url"] = api_base_url
        return {"report_status": 200}

    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch.object(runtime, "run_assignment", side_effect=_fake_run),
        mock.patch.object(runtime.sys, "stdout", new=io.StringIO()),
    ):
        # Ensure env map does not reintroduce REVIEW_API_BASE_URL.
        monkeypatch.delenv("REVIEW_API_BASE_URL", raising=False)
        rc = runtime.main(["--run-assignment"])
    assert rc == 0
    assert captured["api_base_url"] == JOINBASE


def test_runtime_main_accepts_honest_joinbase_env() -> None:
    """VAL-ACURL-005: honest joinbase env accepted into run_assignment."""

    runtime = _load_review_runtime()
    env = {
        "REVIEW_SESSION_TOKEN": "tok",
        "OPENROUTER_API_KEY": "or",
        "REVIEW_API_BASE_URL": JOINBASE,
    }
    captured: dict[str, str] = {}

    def _fake_run(*, api_base_url: str, **kwargs: object) -> dict[str, object]:
        del kwargs
        captured["api_base_url"] = api_base_url
        return {"report_status": 202}

    with (
        mock.patch.dict("os.environ", env, clear=False),
        mock.patch.object(runtime, "run_assignment", side_effect=_fake_run),
        mock.patch.object(runtime.sys, "stdout", new=io.StringIO()),
    ):
        rc = runtime.main(["--run-assignment"])
    assert rc == 0
    assert captured["api_base_url"] == JOINBASE


def test_dev_flag_allows_https_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAL-ACURL-012: only explicit CHALLENGE_ALLOW_DEV_URLS=1 unlocks override."""

    monkeypatch.setenv(ALLOW_DEV_REVIEW_URLS_ENV, "1")
    assert (
        resolve_review_api_base_url(explicit="https://review.dev.local/ac")
        == "https://review.dev.local/ac"
    )
    monkeypatch.setenv(ALLOW_DEV_REVIEW_URLS_ENV, "0")
    with pytest.raises(ReviewApiBaseUrlError):
        resolve_review_api_base_url(explicit="https://review.dev.local/ac")


def test_dev_flag_still_rejects_http() -> None:
    """Dev override still requires https (no plaintext callback)."""

    with pytest.raises(ReviewApiBaseUrlError, match="https"):
        assert_pinned_review_api_base_url(
            "http://review.dev.local/ac",
            allow_dev=True,
        )
