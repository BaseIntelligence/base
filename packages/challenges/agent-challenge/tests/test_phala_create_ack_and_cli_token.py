"""TDD for tee-live-proof-v5 progressive residuals:

1. PhalaCloudClient sends CLI-equivalent User-Agent so Cloudflare 1010 does
   not block product urllib (keep X-API-Key / X-Phala-Version).
2. HttpReviewPhalaDeployment create ack: accept id/cvm_id as str or int, plus
   alternate fields and safe GET /cvms list fallback by app_id before fail-
   closed product_create_response_missing_cvm_id_field.
3. CLI review deploy consumes a review_retry one-time token path without
   fail-closing solely because prepare already delivered (null re-prepare).

Never invent TEE measurements. Secrets never appear in errors or logs.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.request import Request

import pytest

from agent_challenge.review.canonical import canonical_sha256
from agent_challenge.review.compose import (
    generate_review_app_compose,
    review_app_compose_hash,
)
from agent_challenge.review.schemas import ReviewInputConfig, build_review_assignment
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy import review as review_mod
from agent_challenge.selfdeploy.phala import (
    DEFAULT_PHALA_USER_AGENT,
    PhalaCloudClient,
    extract_cvm_id_from_create_response,
    resolve_cvm_id_from_list,
)
from agent_challenge.selfdeploy.plan import PHALA_API_KEY_ENV
from agent_challenge.selfdeploy.review import (
    ReviewDeploymentError,
    ReviewPhalaDeployment,
    build_review_deployment_plan,
    encrypt_review_secrets,
)

REVIEW_IMAGE = "registry.example/review@sha256:" + "a" * 64
PUBLIC_KEY = "c" * 64
MEASUREMENT = {
    "mrtd": "01" * 48,
    "rtmr0": "02" * 48,
    "rtmr1": "03" * 48,
    "rtmr2": "04" * 48,
    "os_image_hash": "05" * 32,
    "key_provider": "phala",
    "vm_shape": "tdx.small",
}
TOKEN = "review-token-sentinel-create-ack"


class _CapturingOpener:
    def __init__(self, payload: dict | list | None = None) -> None:
        self.requests: list[Request] = []
        self.payload = payload if payload is not None else {"ok": True}

    def __call__(self, request: Request, timeout: float = 0.0):  # noqa: ARG002
        self.requests.append(request)

        class _Resp:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self, n: int = -1) -> bytes:  # noqa: ARG002
                return self._body

        return _Resp(json.dumps(self.payload).encode())


def _assignment_and_plan() -> tuple[dict[str, Any], Any, Any]:
    compose = generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity="agent-challenge-review-v1",
    )
    compose_hash = review_app_compose_hash(compose)
    allowlist_entry = {
        "mrtd": MEASUREMENT["mrtd"],
        "rtmr0": MEASUREMENT["rtmr0"],
        "rtmr1": MEASUREMENT["rtmr1"],
        "rtmr2": MEASUREMENT["rtmr2"],
        "compose_hash": compose_hash,
        "os_image_hash": MEASUREMENT["os_image_hash"],
    }
    config = ReviewInputConfig(
        image_ref=REVIEW_IMAGE,
        compose_hash=compose_hash,
        app_identity="agent-challenge-review-v1",
        kms_public_key_hex=PUBLIC_KEY,
        measurement=MEASUREMENT,
        measurement_allowlist=(allowlist_entry,),
        measurement_allowlist_sha256=canonical_sha256({"entries": [allowlist_entry]}),
    )
    assignment, _, _ = build_review_assignment(
        session_id="session-1",
        assignment_id="assignment-1",
        attempt=1,
        submission_id="1",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 1,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": "/review/v1/assignments/assignment-1/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce="nonce-review",
        issued_at_ms=1,
        expires_at_ms=2,
        session_token_sha256=__import__("hashlib").sha256(TOKEN.encode()).hexdigest(),
        config=config,
    )
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": TOKEN})
    encrypted = encrypt_review_secrets(
        plan,
        {
            "OPENROUTER_API_KEY": "or-test-key-never-print",
            "REVIEW_API_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
            "REVIEW_SESSION_TOKEN": TOKEN,
        },
    )
    return assignment, plan, encrypted


# --------------------------------------------------------------------------- #
# 1) User-Agent header
# --------------------------------------------------------------------------- #


def test_phala_client_sends_cli_equivalent_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test_key")
    opener = _CapturingOpener({"ok": True})
    client = PhalaCloudClient(api_key="phak_test_key", opener=opener)

    client.post("/cvms/provision", {"app_id": "x", "name": "x"})

    headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
    assert headers.get("user-agent") == DEFAULT_PHALA_USER_AGENT
    assert DEFAULT_PHALA_USER_AGENT.startswith("phala-cli/")
    # Keep auth contract: X-API-Key, never Bearer; no Python-urllib bare agent.
    assert headers.get("x-api-key") == "phak_test_key"
    assert "authorization" not in headers
    assert "Python-urllib" not in (headers.get("user-agent") or "")


def test_phala_client_get_sends_same_auth_and_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test_key")
    opener = _CapturingOpener({"items": []})
    client = PhalaCloudClient(api_key="phak_test_key", opener=opener)

    client.get("/cvms")

    assert opener.requests[0].get_method() == "GET"
    headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
    assert headers.get("user-agent") == DEFAULT_PHALA_USER_AGENT
    assert headers.get("x-api-key") == "phak_test_key"
    assert headers.get("x-phala-version")


# --------------------------------------------------------------------------- #
# 2) Create response id mapping + list fallback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "created,expected",
    [
        ({"id": 4242, "app_id": "app-x", "status": "starting"}, "4242"),
        ({"id": "cvm-str-1", "request_id": "req-1"}, "cvm-str-1"),
        ({"cvm_id": "cvm-alt-9", "app_id": "app-x"}, "cvm-alt-9"),
        ({"vm_uuid": "vm-uuid-7", "app_id": "app-x"}, "vm-uuid-7"),
        ({"instance_id": "inst-3", "app_id": "app-x"}, "inst-3"),
    ],
)
def test_extract_cvm_id_accepts_alternate_create_shapes(
    created: dict[str, Any], expected: str
) -> None:
    assert extract_cvm_id_from_create_response(created) == expected


def test_extract_cvm_id_numeric_id_wins_over_app_id_string() -> None:
    # Live schema: id is int; app_id is the Phala app pin and is NOT the CVM id.
    created = {"id": 99, "app_id": "f024ea2315052843d0afd775b2b82b2d2455c798"}
    assert extract_cvm_id_from_create_response(created) == "99"


def test_extract_cvm_id_rejects_app_id_as_cvm_identity() -> None:
    with pytest.raises(ValueError, match="does not identify"):
        extract_cvm_id_from_create_response(
            {"app_id": "f024ea2315052843d0afd775b2b82b2d2455c798", "status": "starting"}
        )


def test_resolve_cvm_id_from_list_by_app_id() -> None:
    listing = {
        "items": [
            {"id": 1, "app_id": "other", "status": "running"},
            {"id": 77, "app_id": "f024ea2315052843d0afd775b2b82b2d2455c798", "status": "starting"},
        ]
    }
    assert (
        resolve_cvm_id_from_list(listing, app_id="f024ea2315052843d0afd775b2b82b2d2455c798") == "77"
    )


def test_review_deploy_accepts_numeric_create_id() -> None:
    _assignment, plan, encrypted = _assignment_and_plan()
    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": plan.app_identity,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": PUBLIC_KEY,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
        # Live Phala create schema uses numeric id (CLI zod schema).
        create_response={
            "id": 9175,
            "name": plan.compose_name,
            "status": "starting",
            "app_id": plan.app_identity,
            "created_at": "2026-07-15T00:00:00Z",
        },
    )
    acknowledgement = deployment.deploy(plan, encrypted)
    assert acknowledgement["cvm_id"] == "9175"
    assert acknowledgement["phala_create_receipt"]["cvm_id"] == "9175"


def test_review_deploy_list_fallback_by_app_id_when_create_lacks_id() -> None:
    _assignment, plan, encrypted = _assignment_and_plan()

    class _Api:
        def __init__(self) -> None:
            self.provision_requests: list[dict[str, Any]] = []
            self.create_requests: list[dict[str, Any]] = []
            self.get_paths: list[str] = []

        def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if path == "/cvms/provision":
                self.provision_requests.append(dict(payload))
                return {
                    "app_id": plan.app_identity,
                    "compose_hash": plan.compose_hash,
                    "app_env_encrypt_pubkey": PUBLIC_KEY,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                }
            if path == "/cvms":
                self.create_requests.append(dict(payload))
                # Residual shape: create 200 but no usable id field.
                return {
                    "app_id": plan.app_identity,
                    "status": "processing",
                    "name": plan.compose_name,
                }
            raise AssertionError(path)

        def get(self, path: str) -> dict[str, Any]:
            self.get_paths.append(path)
            assert path == "/cvms"
            return {
                "items": [
                    {
                        "id": 5511,
                        "app_id": plan.app_identity,
                        "status": "starting",
                        "name": plan.compose_name,
                    }
                ]
            }

    api = _Api()
    acknowledgement = review_mod.HttpReviewPhalaDeployment(api).deploy(plan, encrypted)
    assert api.get_paths == ["/cvms"]
    assert acknowledgement["cvm_id"] == "5511"


def test_review_deploy_still_fail_closed_when_list_has_no_matching_cvm() -> None:
    _assignment, plan, encrypted = _assignment_and_plan()

    class _Api:
        def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if path == "/cvms/provision":
                return {
                    "app_id": plan.app_identity,
                    "compose_hash": plan.compose_hash,
                    "app_env_encrypt_pubkey": PUBLIC_KEY,
                    "os_image_hash": MEASUREMENT["os_image_hash"],
                }
            return {"app_id": plan.app_identity, "status": "processing"}

        def get(self, path: str) -> dict[str, Any]:  # noqa: ARG002
            return {"items": [{"id": 1, "app_id": "someone-else", "status": "running"}]}

    with pytest.raises(ReviewDeploymentError, match="does not identify the review CVM"):
        review_mod.HttpReviewPhalaDeployment(_Api()).deploy(plan, encrypted)


def test_extract_helpers_never_echo_secrets() -> None:
    payload = {
        "id": 1,
        "encrypted_env": "CIPHERTEXT_SECRET_MUST_NOT_LEAK",
        "app_id": "app",
    }
    assert extract_cvm_id_from_create_response(payload) == "1"
    message = ""
    try:
        extract_cvm_id_from_create_response({"encrypted_env": "CIPHERTEXT_SECRET_MUST_NOT_LEAK"})
    except ValueError as exc:
        message = str(exc)
    assert "CIPHERTEXT_SECRET_MUST_NOT_LEAK" not in message


# --------------------------------------------------------------------------- #
# 3) CLI review deploy consumes retry-delivered token (no null re-prepare trap)
# --------------------------------------------------------------------------- #


def _fresh_assignment(*, assignment_id: str, attempt: int, token: str) -> dict[str, Any]:
    compose = generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity="agent-challenge-review-v1",
    )
    compose_hash = review_app_compose_hash(compose)
    allowlist_entry = {
        "mrtd": MEASUREMENT["mrtd"],
        "rtmr0": MEASUREMENT["rtmr0"],
        "rtmr1": MEASUREMENT["rtmr1"],
        "rtmr2": MEASUREMENT["rtmr2"],
        "compose_hash": compose_hash,
        "os_image_hash": MEASUREMENT["os_image_hash"],
    }
    config = ReviewInputConfig(
        image_ref=REVIEW_IMAGE,
        compose_hash=compose_hash,
        app_identity="agent-challenge-review-v1",
        kms_public_key_hex=PUBLIC_KEY,
        measurement=MEASUREMENT,
        measurement_allowlist=(allowlist_entry,),
        measurement_allowlist_sha256=canonical_sha256({"entries": [allowlist_entry]}),
    )
    import hashlib

    assignment, _, _ = build_review_assignment(
        session_id="session-1",
        assignment_id=assignment_id,
        attempt=attempt,
        submission_id="1",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 1,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": f"/review/v1/assignments/{assignment_id}/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce=f"nonce-{assignment_id}",
        issued_at_ms=1,
        expires_at_ms=2,
        session_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        config=config,
    )
    return assignment


def test_cli_review_deploy_uses_retry_when_prepare_token_already_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reused submission: prepare returns null token; product cancel+retry recovers.

    Discriminator residual: CLI always called review_prepare only, received
    null token on re-prepare, and fail-closed before Phala create.
    """
    assignment, plan, _encrypted = _assignment_and_plan()
    retry_token = "review-retry-token-fresh"
    retry_assignment = _fresh_assignment(
        assignment_id="assignment-2",
        attempt=2,
        token=retry_token,
    )

    fake_client = MagicMock()
    fake_client.review_prepare.return_value = {
        "session_id": "session-1",
        "assignment_id": "assignment-1",
        "attempt": 1,
        "assignment": assignment,
        "review_session_token": None,  # already delivered (one-time)
    }
    fake_client.review_cancel.return_value = {
        "assignment_id": "assignment-1",
        "phase": "review_cancelled",
    }
    fake_client.review_retry.return_value = {
        "session_id": "session-1",
        "assignment_id": "assignment-2",
        "attempt": 2,
        "assignment": retry_assignment,
        "review_session_token": retry_token,
    }
    fake_client.review_deployed.return_value = {"ok": True}

    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    class _FixedDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            assert plan_obj.review_session_token == retry_token
            assert encrypted_obj.assignment_id == "assignment-2"
            return {
                "schema_version": 1,
                "assignment_id": "assignment-2",
                "cvm_id": "cvm-from-deploy",
                "phala_create_receipt": {
                    "request_id": "req",
                    "app_id": plan.app_identity,
                    "cvm_id": "cvm-from-deploy",
                    "receipt_sha256": "a" * 64,
                    "created_at_ms": 1,
                },
                "compose_identity": {
                    "image_ref": plan.image_ref,
                    "compose_hash": plan.compose_hash,
                    "app_kms_public_key_sha256": plan.kms_public_key_sha256,
                },
            }

    monkeypatch.setattr(review_mod, "HttpReviewPhalaDeployment", _FixedDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    printed: list[Any] = []
    monkeypatch.setattr(cli, "_print", printed.append)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=1,
        dry_run=False,
        auto_sign=True,
        signature=None,
        nonce=None,
        prepare_response=None,
        review_instance_type=plan.instance_type,
        eval_instance_type="tdx.small",
        review_runtime_hours=1.0,
        eval_runtime_hours=1.0,
        money_cap_usd=20.0,
        openrouter_key_env="OPENROUTER_API_KEY",
        phala_api=None,
        base_url="https://challenge.example",
        hotkey="hk",
        timestamp=None,
    )
    code = cli._ordered_review_command(args)
    assert code == 0, printed
    fake_client.review_prepare.assert_called_once_with(1)
    fake_client.review_cancel.assert_called_once_with(1, "assignment-1")
    fake_client.review_retry.assert_called_once()
    assert fake_client.review_retry.call_args.args[0] == 1
    assert fake_client.review_retry.call_args.args[1] == "assignment-1"
    fake_client.review_deployed.assert_called_once()
    # Capability bytes never printed.
    assert retry_token not in json.dumps(printed)


def test_cli_review_deploy_uses_prepare_token_when_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assignment, plan, _encrypted = _assignment_and_plan()
    fake_client = MagicMock()
    fake_client.review_prepare.return_value = {
        "session_id": "session-1",
        "assignment_id": "assignment-1",
        "attempt": 1,
        "assignment": assignment,
        "review_session_token": TOKEN,
    }
    fake_client.review_deployed.return_value = {"ok": True}
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    class _FixedDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            assert plan_obj.review_session_token == TOKEN
            return {
                "schema_version": 1,
                "assignment_id": "assignment-1",
                "cvm_id": "cvm-1",
                "phala_create_receipt": {
                    "request_id": "req",
                    "app_id": plan.app_identity,
                    "cvm_id": "cvm-1",
                    "receipt_sha256": "b" * 64,
                    "created_at_ms": 1,
                },
                "compose_identity": {
                    "image_ref": plan.image_ref,
                    "compose_hash": plan.compose_hash,
                    "app_kms_public_key_sha256": plan.kms_public_key_sha256,
                },
            }

    monkeypatch.setattr(review_mod, "HttpReviewPhalaDeployment", _FixedDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    monkeypatch.setattr(cli, "_print", lambda _p: None)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=9,
        dry_run=False,
        auto_sign=True,
        signature=None,
        nonce=None,
        prepare_response=None,
        review_instance_type=plan.instance_type,
        eval_instance_type="tdx.small",
        review_runtime_hours=1.0,
        eval_runtime_hours=1.0,
        money_cap_usd=20.0,
        openrouter_key_env="OPENROUTER_API_KEY",
        phala_api=None,
        base_url="https://challenge.example",
        hotkey="hk",
        timestamp=None,
    )
    assert cli._ordered_review_command(args) == 0
    fake_client.review_cancel.assert_not_called()
    fake_client.review_retry.assert_not_called()


def test_cli_review_deploy_skips_cancel_when_prepare_redelivers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dry-run prepare then live deploy must reuse redelivered token (no cancel burn).

    Residual review-v9 token path: previous CLI always cancel+retried whenever
    prepare returned null and burned undepoyed active attempts 1-2. With product
    redelivery, prepare after dry-run yields the same token → zero cancel/retry.
    """

    assignment, plan, _encrypted = _assignment_and_plan()
    fake_client = MagicMock()
    # Simulate: first prepare (dry) already delivered; second prepare redelivers.
    fake_client.review_prepare.return_value = {
        "session_id": "session-1",
        "assignment_id": "assignment-1",
        "attempt": 1,
        "assignment": assignment,
        "review_session_token": TOKEN,
    }
    fake_client.review_deployed.return_value = {"ok": True}
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

    class _FixedDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            assert plan_obj.review_session_token == TOKEN
            assert encrypted_obj.assignment_id == "assignment-1"
            return {
                "schema_version": 1,
                "assignment_id": "assignment-1",
                "cvm_id": "cvm-reuse",
                "phala_create_receipt": {
                    "request_id": "req",
                    "app_id": plan.app_identity,
                    "cvm_id": "cvm-reuse",
                    "receipt_sha256": "c" * 64,
                    "created_at_ms": 1,
                },
                "compose_identity": {
                    "image_ref": plan.image_ref,
                    "compose_hash": plan.compose_hash,
                    "app_kms_public_key_sha256": plan.kms_public_key_sha256,
                },
            }

    monkeypatch.setattr(review_mod, "HttpReviewPhalaDeployment", _FixedDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    printed: list[Any] = []
    monkeypatch.setattr(cli, "_print", printed.append)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=3,
        dry_run=False,
        auto_sign=True,
        signature=None,
        nonce=None,
        prepare_response=None,
        review_instance_type=plan.instance_type,
        eval_instance_type="tdx.small",
        review_runtime_hours=1.0,
        eval_runtime_hours=1.0,
        money_cap_usd=12.0,
        openrouter_key_env="OPENROUTER_API_KEY",
        phala_api=None,
        base_url="https://chain.joinbase.ai/challenges/agent-challenge",
        hotkey="hk",
        timestamp=None,
    )
    assert cli._ordered_review_command(args) == 0
    fake_client.review_prepare.assert_called_once_with(3)
    fake_client.review_cancel.assert_not_called()
    fake_client.review_retry.assert_not_called()
    fake_client.review_deployed.assert_called_once()


def test_obtain_review_prepare_only_cancel_retries_when_token_null_and_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel+retry remains for sticky-null (post-deploy / terminal) only."""

    assignment, _plan, _enc = _assignment_and_plan()
    retry_token = "review-retry-after-sticky-null"
    retry_assignment = _fresh_assignment(
        assignment_id="assignment-3",
        attempt=3,
        token=retry_token,
    )
    fake_client = MagicMock()
    fake_client.review_prepare.return_value = {
        "session_id": "session-1",
        "assignment_id": "assignment-1",
        "attempt": 1,
        "assignment": assignment,
        "review_session_token": None,  # sticky after deploy or terminal
        "phase": "review_cvm_running",
    }
    fake_client.review_cancel.return_value = {
        "assignment_id": "assignment-1",
        "phase": "review_cancelled",
    }
    fake_client.review_retry.return_value = {
        "session_id": "session-1",
        "assignment_id": "assignment-3",
        "attempt": 3,
        "assignment": retry_assignment,
        "review_session_token": retry_token,
    }
    out = cli._obtain_review_prepare_with_token(fake_client, 7)
    assert out["review_session_token"] == retry_token
    fake_client.review_cancel.assert_called_once_with(7, "assignment-1")
    fake_client.review_retry.assert_called_once()
