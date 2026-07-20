"""PhalaCloudClient auth headers + DEFAULT_REGION selection (ERR-02-002 residual).

Live tee-proof-v4 residuals:

* ``Authorization: Bearer <api_key>`` → HTTP 401 Invalid/expired token
* bare region ``us-west`` → ERR-02-002 No teepod found (inventory is US-WEST-1 only)
* ``X-API-Key`` (+ ``X-Phala-Version``) matches Phala CLI and authenticates
* ``us-west-1`` / omitted region provisions successfully

These unit tests pin the product client contract without live spend or secret logging.
"""

from __future__ import annotations

import json
from urllib.request import Request

import pytest

from agent_challenge.selfdeploy import eval as eval_mod
from agent_challenge.selfdeploy import plan as plan_mod
from agent_challenge.selfdeploy import review as review_mod
from agent_challenge.selfdeploy.phala import (
    DEFAULT_PHALA_API_VERSION,
    PhalaApiError,
    PhalaCloudClient,
    normalize_phala_region,
    select_phala_region,
)
from agent_challenge.selfdeploy.plan import (
    DEFAULT_REGION,
    PHALA_API_KEY_ENV,
    build_deploy_plan,
)

DIGEST = "registry.example/agent-challenge@sha256:" + ("a" * 64)
KEY_RELEASE = "https://validator.example/v1/key-release"


class _CapturingOpener:
    """Capture the outbound Request and return a fixed JSON body."""

    def __init__(self, payload: dict | None = None) -> None:
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


# --------------------------------------------------------------------------- #
# Auth headers: X-API-Key, never Bearer
# --------------------------------------------------------------------------- #


def test_phala_client_sends_x_api_key_not_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test_key_never_print_me")
    opener = _CapturingOpener({"compose_hash": "abc", "app_id": "x"})
    client = PhalaCloudClient(api_key="phak_test_key_never_print_me", opener=opener)

    client.post("/cvms/provision", {"app_id": "x", "name": "x"})

    assert len(opener.requests) == 1
    headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
    assert headers.get("x-api-key") == "phak_test_key_never_print_me"
    assert "authorization" not in headers
    assert not any(
        (v or "").lower().startswith("bearer ") for _k, v in opener.requests[0].header_items()
    )


def test_phala_client_sends_x_phala_version_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test_key")
    opener = _CapturingOpener({"ok": True})
    client = PhalaCloudClient(api_key="phak_test_key", opener=opener)

    client.post("/cvms", {"app_id": "x", "compose_hash": "h", "encrypted_env": "c", "env_keys": []})

    headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
    assert headers.get("x-phala-version") == DEFAULT_PHALA_API_VERSION
    assert DEFAULT_PHALA_API_VERSION  # non-empty pin matching CLI Lo default


def test_phala_client_never_logs_or_echoes_api_key_in_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "phak_super_secret_should_never_appear_in_errors"
    monkeypatch.setenv(PHALA_API_KEY_ENV, sentinel)

    class _BoomOpener:
        def __call__(self, request: Request, timeout: float = 0.0):  # noqa: ARG002
            from urllib.error import HTTPError

            raise HTTPError(
                url=request.full_url,
                code=401,
                msg="Unauthorized",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )

    client = PhalaCloudClient(api_key=sentinel, opener=_BoomOpener())
    with pytest.raises(PhalaApiError) as excinfo:
        client.post("/cvms/provision", {"app_id": "x"})
    message = str(excinfo.value)
    assert sentinel not in message
    assert "401" in message
    assert "Bearer" not in message


def test_phala_client_rejects_http_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test")
    with pytest.raises(PhalaApiError, match="https"):
        PhalaCloudClient(api_key="phak_test", base_url="http://insecure.example/api/v1")


# --------------------------------------------------------------------------- #
# Region selection: us-west must not hard-fail; prefer us-west-1 / auto
# --------------------------------------------------------------------------- #


def test_default_region_is_not_bare_us_west_alias() -> None:
    """Bare ``us-west`` hard-fails live with ERR-02-002 (no teepod)."""

    assert DEFAULT_REGION != "us-west"
    assert DEFAULT_REGION in {"us-west-1", ""}
    # Plan / Review / Eval keep the same product default (no drift).
    assert review_mod.DEFAULT_REGION == DEFAULT_REGION
    assert eval_mod.DEFAULT_REGION == DEFAULT_REGION
    assert plan_mod.DEFAULT_REGION == DEFAULT_REGION


def test_normalize_phala_region_maps_us_west_to_us_west_1() -> None:
    assert normalize_phala_region("us-west") == "us-west-1"
    assert normalize_phala_region("US-WEST") == "us-west-1"
    assert normalize_phala_region("us-west-1") == "us-west-1"
    assert normalize_phala_region("US-WEST-1") == "us-west-1"
    # Empty → auto (omit sentinel for callers that want no region key)
    assert normalize_phala_region("") == ""
    assert normalize_phala_region(None) == ""


def test_select_phala_region_prefers_available_capacity() -> None:
    # When inventory lists only us-west-1, never pick bare us-west.
    assert (
        select_phala_region("us-west", available_regions=["us-west-1", "eu-central-1"])
        == "us-west-1"
    )
    # Explicit available region wins when present.
    assert select_phala_region(None, available_regions=["eu-central-1"]) == "eu-central-1"
    # Preferred fallback when inventory empty: us-west-1 (or empty auto).
    assert select_phala_region(None, available_regions=[]) in {"us-west-1", ""}
    # Explicit non-alias region passes through.
    assert select_phala_region("eu-central-1", available_regions=["us-west-1"]) == "eu-central-1"


def test_build_deploy_plan_default_region_avoids_us_west_hard_fail() -> None:
    plan = build_deploy_plan(image=DIGEST, key_release_url=KEY_RELEASE)
    assert plan.region != "us-west"
    assert plan.region == DEFAULT_REGION or plan.region == ""


def test_build_deploy_plan_normalizes_us_west_alias() -> None:
    plan = build_deploy_plan(image=DIGEST, key_release_url=KEY_RELEASE, region="us-west")
    assert plan.region != "us-west"
    assert plan.region == "us-west-1"


def test_review_deployment_plan_default_region() -> None:
    assert review_mod.DEFAULT_REGION != "us-west"


def test_phala_client_normalizes_region_in_provision_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test_key")
    opener = _CapturingOpener({"ok": True})
    client = PhalaCloudClient(api_key="phak_test_key", opener=opener)

    client.post(
        "/cvms/provision",
        {
            "app_id": "x",
            "name": "x",
            "instance_type": "tdx.small",
            "region": "us-west",
            "env_keys": [],
        },
    )

    body = json.loads(opener.requests[0].data.decode())
    assert body["region"] == "us-west-1"
    # Empty region is omitted (auto from teepods).
    opener2 = _CapturingOpener({"ok": True})
    client2 = PhalaCloudClient(api_key="phak_test_key", opener=opener2)
    client2.post(
        "/cvms/provision",
        {"app_id": "x", "name": "x", "region": "", "env_keys": []},
    )
    body2 = json.loads(opener2.requests[0].data.decode())
    assert "region" not in body2


def test_review_provision_sends_image_and_accepts_matching_os_hash() -> None:
    """Live teepod OS pin must be sent and checked (no auto-dev de9c drift)."""
    from agent_challenge.selfdeploy.review import (
        DEFAULT_OS_IMAGE,
        HttpReviewPhalaDeployment,
        ReviewDeploymentError,
        ReviewDeploymentPlan,
    )

    measurement = {
        "mrtd": "01" * 48,
        "rtmr0": "02" * 48,
        "rtmr1": "03" * 48,
        "rtmr2": "04" * 48,
        "os_image_hash": "bd" + "0" * 62,
        "key_provider": "phala",
        "vm_shape": "tdx.small",
    }
    plan = ReviewDeploymentPlan(
        assignment={"assignment_core": {"assignment_id": "a1"}},
        compose={"name": "agent-challenge-review-v1"},
        compose_text="{}",
        compose_hash="ab" * 32,
        app_identity="f024ea2315052843d0afd775b2b82b2d2455c798",
        image_ref="registry.example/r@sha256:" + "a" * 64,
        kms_public_key_hex="c" * 64,
        kms_public_key_sha256="d" * 64,
        measurement=measurement,
        measurement_allowlist_sha256="e" * 64,
        review_session_token="tok",
        compose_name="agent-challenge-review-v1",
        phala_app_nonce=0,
        os_image=DEFAULT_OS_IMAGE,
    )

    HttpReviewPhalaDeployment._verify_provision_response(
        plan,
        {
            "app_id": plan.app_identity,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": plan.kms_public_key_hex,
            "os_image_hash": measurement["os_image_hash"],
        },
    )
    with pytest.raises(ReviewDeploymentError, match="os_image_hash"):
        HttpReviewPhalaDeployment._verify_provision_response(
            plan,
            {
                "app_id": plan.app_identity,
                "compose_hash": plan.compose_hash,
                "app_env_encrypt_pubkey": plan.kms_public_key_hex,
                "os_image_hash": "de" + "9" * 62,  # live-auto dev mismatch
            },
        )
    with pytest.raises(ReviewDeploymentError, match="app identity"):
        HttpReviewPhalaDeployment._verify_provision_response(
            plan,
            {
                "app_id": "random-minted-hex",
                "compose_hash": plan.compose_hash,
                "app_env_encrypt_pubkey": plan.kms_public_key_hex,
                "os_image_hash": measurement["os_image_hash"],
            },
        )
    assert DEFAULT_OS_IMAGE == "dstack-0.5.9"


def test_default_os_image_is_live_available_non_dev() -> None:
    from agent_challenge.selfdeploy.shapes import DEFAULT_OS_IMAGE

    assert DEFAULT_OS_IMAGE == "dstack-0.5.9"
    assert "nvidia" not in DEFAULT_OS_IMAGE
    assert "dev" not in DEFAULT_OS_IMAGE
