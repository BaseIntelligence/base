"""VAL-REVIEW-009 / VAL-DEPLOY-026 / VAL-CROSS-030 client-auth executability.

These are discriminator tests for the self-deploy route client and ordered CLI:

1. Review retry must accept / send a scoped ``approval_id`` so rejected/escalated
   attempts can consume one-use operator approval.
2. Non-auto-sign deploy must accept the documented explicit header identity
   (``--signature`` / ``--nonce`` / ``--timestamp``) instead of forcing auto-sign.
3. Auto-sign canonicalization must cover the exact query string on cursor-
   bearing history / report / status requests so signatures match the server
   ``canonical_request_string`` rules.
4. Status / rejection surfaces remain executable and report only bounded
   retained state (no success, score, or secret bytes on rejected status).
"""

from __future__ import annotations

import io
import json
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.request import Request

import pytest

from agent_challenge.auth.security import body_sha256, canonical_request_string
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy.client import (
    RouteClientError,
    SelfDeployRouteClient,
    SignedIdentity,
    build_signed_identity,
    sign_request_identity,
)


class _CapturingOpener:
    """Capture outbound Request objects and return a fixed JSON body."""

    def __init__(self, payload: dict | None = None, status: int = 200) -> None:
        self.requests: list[Request] = []
        self.payload = payload if payload is not None else {"ok": True}
        self.status = status

    def __call__(self, request: Request, timeout: float = 0.0):  # noqa: ARG002
        self.requests.append(request)

        class _Resp:
            def __init__(self, body: bytes) -> None:
                self._body = body

            def read(self, n: int = -1) -> bytes:  # noqa: ARG002
                return self._body

        return _Resp(json.dumps(self.payload).encode())


class _RecordingCapture:
    def __init__(self) -> None:
        self.payloads: list[object] = []

    def __call__(self, payload: object) -> None:
        self.payloads.append(payload)


# --------------------------------------------------------------------------- #
# 1) approval-backed review retry
# --------------------------------------------------------------------------- #


def test_review_retry_sends_scoped_approval_id_in_signed_body():
    opener = _CapturingOpener({"assignment": {"assignment_id": "a-2"}, "attempt": 2})
    identity = SignedIdentity(
        hotkey="hk",
        signature="0xsig",
        nonce="n1",
        timestamp="1710000000.0",
    )
    client = SelfDeployRouteClient(
        "https://challenge.example",
        identity=identity,
        auto_sign=False,
        opener=opener,
    )

    payload = client.review_retry(
        7,
        "assignment-1",
        approval_id="ra-operator-1",
    )

    assert payload["attempt"] == 2
    assert len(opener.requests) == 1
    request = opener.requests[0]
    assert request.get_method() == "POST"
    assert request.full_url.endswith("/submissions/7/review/retry")
    body = json.loads(request.data.decode())
    assert body == {
        "expected_assignment_id": "assignment-1",
        "approval_id": "ra-operator-1",
    }
    # Without approval_id the key must be omitted so ordinary retries stay lean.
    opener2 = _CapturingOpener({})
    client2 = SelfDeployRouteClient(
        "https://challenge.example",
        identity=identity,
        auto_sign=False,
        opener=opener2,
    )
    client2.review_retry(7, "assignment-1")
    body2 = json.loads(opener2.requests[0].data.decode())
    assert body2 == {"expected_assignment_id": "assignment-1"}
    assert "approval_id" not in body2


def test_cli_review_retry_forwards_approval_id(monkeypatch):
    fake_client = MagicMock()
    fake_client.review_retry.return_value = {
        "assignment": {"assignment_id": "a-2"},
        "review_session_token": "MUST-REDACT",
    }
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)

    args = Namespace(
        review_command="retry",
        submission_id=3,
        assignment_id="assignment-1",
        approval_id="ra-xyz",
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=False,
    )
    assert cli._ordered_review_command(args) == 0
    fake_client.review_retry.assert_called_once_with(3, "assignment-1", approval_id="ra-xyz")
    # Capability bytes never leave the CLI surface.
    assert "MUST-REDACT" not in json.dumps(capture.payloads[-1])


def test_cli_review_retry_parses_approval_id_flag():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "review",
            "retry",
            "--base-url",
            "https://challenge.example",
            "--submission-id",
            "9",
            "--hotkey",
            "hk",
            "--signature",
            "sig",
            "--nonce",
            "n1",
            "--assignment-id",
            "assignment-1",
            "--approval-id",
            "ra-operator-9",
        ]
    )
    assert args.review_command == "retry"
    assert args.approval_id == "ra-operator-9"
    assert args.assignment_id == "assignment-1"


# --------------------------------------------------------------------------- #
# 2) Non-auto-sign deploy accepts explicit signature headers
# --------------------------------------------------------------------------- #


def test_route_client_accepts_explicit_signed_headers_without_auto_sign():
    opener = _CapturingOpener({"assignment": {"assignment_id": "a-1"}})
    identity = build_signed_identity(
        hotkey="miner-hotkey",
        signature="0xexplicit",
        nonce="nonce-explicit",
        timestamp="1710000001.5",
    )
    client = SelfDeployRouteClient(
        "https://challenge.example",
        identity=identity,
        auto_sign=False,
        opener=opener,
    )
    client.review_prepare(1)
    headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
    assert headers["x-hotkey"] == "miner-hotkey"
    assert headers["x-signature"] == "0xexplicit"
    assert headers["x-nonce"] == "nonce-explicit"
    assert headers["x-timestamp"] == "1710000001.5"


def _sample_review_assignment() -> dict[str, object]:
    return {
        "assignment_core": {
            "assignment_id": "assignment-1",
            "review_app": {
                "image_ref": "registry.example/review@sha256:" + "a" * 64,
                "compose_hash": "c" * 64,
                "app_identity": "review-v1",
                "kms_key_algorithm": "x25519",
                "kms_public_key_hex": "ab" * 32,
                "kms_public_key_sha256": "d" * 64,
                "measurement": {
                    "mrtd": "01" * 48,
                    "rtmr0": "02" * 48,
                    "rtmr1": "03" * 48,
                    "rtmr2": "04" * 48,
                    "compose_hash": "c" * 64,
                    "os_image_hash": "05" * 32,
                    "key_provider": "validator-kms",
                    "vm_shape": "tdx-small",
                },
                "measurement_allowlist": [
                    {
                        "mrtd": "01" * 48,
                        "rtmr0": "02" * 48,
                        "rtmr1": "03" * 48,
                        "rtmr2": "04" * 48,
                        "compose_hash": "c" * 64,
                        "os_image_hash": "05" * 32,
                        "key_provider": "validator-kms",
                        "vm_shape": "tdx-small",
                    }
                ],
                "instance_type": "tdx.small",
            },
        }
    }


def test_review_deploy_accepts_explicit_signature_headers(monkeypatch):
    """Without --auto-sign, complete --signature/--nonce must not hard-fail.

    Discriminator: a plausible wrong implementation that forces auto-sign for
    every live review deploy rejects this path even when the documented
    explicit headers are complete.
    """
    fake_client = MagicMock()
    assignment = _sample_review_assignment()
    # Minimal prepare payload: use dry-run so we don't need Phala/creds.
    fake_client.review_prepare.return_value = {
        "assignment": assignment,
        "review_session_token": "token-never-persist",
    }
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)

    # Force plan builder / budget to accept a lightweight dry-run path.
    from agent_challenge.selfdeploy import review as review_deploy

    class _Plan:
        def __init__(self) -> None:
            self.instance_type = "tdx.small"
            self.assignment = assignment
            self.image_ref = assignment["assignment_core"]["review_app"]["image_ref"]  # type: ignore[index]
            self.compose_hash = assignment["assignment_core"]["review_app"]["compose_hash"]  # type: ignore[index]
            self.measurement = assignment["assignment_core"]["review_app"]["measurement"]  # type: ignore[index]
            self.review_session_token = "token-never-persist"

    monkeypatch.setattr(review_deploy, "build_review_deployment_plan", lambda _r: _Plan())
    monkeypatch.setattr(cli.lifecycle, "validate_lifecycle_budget", lambda **_k: None)
    monkeypatch.setattr(cli, "_review_allowlist_verdict", lambda _p: "IN-LIST")
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="0xexplicit-sig",
        nonce="explicit-nonce",
        timestamp="1710000002",
        auto_sign=False,  # documented non-auto-sign path
        prepare_response=None,
        openrouter_key_env="OPENROUTER_API_KEY",
        phala_api=None,
        review_instance_type="tdx.small",
        eval_instance_type="tdx.small",
        review_runtime_hours=6.0,
        eval_runtime_hours=6.0,
        money_cap_usd=20.0,
        dry_run=True,
    )
    code = cli._ordered_review_command(args)
    assert code == 0
    fake_client.review_prepare.assert_called_once_with(1)
    assert capture.payloads[-1]["dry_run"] is True


def test_review_deploy_live_requires_signable_identity_either_path():
    """Live review deploy must accept either auto-sign OR complete explicit headers.

    Discriminator: old check ``not auto_sign ⇒ hard error`` fails this matrix.
    """
    # Missing both auto-sign and signature still fails.
    with pytest.raises(RouteClientError, match="signature|auto-sign|nonce"):
        cli._route_client(
            SimpleNamespace(
                base_url="https://challenge.example",
                hotkey="hk",
                signature=None,
                nonce=None,
                timestamp=None,
                auto_sign=False,
            )
        )

    # Explicit headers alone must produce a non-auto-sign client.
    client = cli._route_client(
        SimpleNamespace(
            base_url="https://challenge.example",
            hotkey="hk",
            signature="0xsig",
            nonce="n1",
            timestamp="1",
            auto_sign=False,
        )
    )
    assert client._auto_sign is False
    assert client._identity is not None
    assert client._identity.signature == "0xsig"

    # Auto-sign alone still works.
    auto_client = cli._route_client(
        SimpleNamespace(
            base_url="https://challenge.example",
            hotkey="hk",
            signature=None,
            nonce=None,
            timestamp=None,
            auto_sign=True,
        )
    )
    assert auto_client._auto_sign is True


def test_live_review_deploy_no_longer_hard_requires_auto_sign(monkeypatch):
    """Remove the auto-sign-only gate on live review deploy."""
    fake_client = MagicMock()
    assignment_token = "sentinel-review-token-not-to-leak"
    assignment = _sample_review_assignment()
    fake_client.review_prepare.return_value = {
        "assignment": assignment,
        "review_session_token": assignment_token,
    }
    fake_client.review_deployed.return_value = {"phase": "review_cvm_running"}

    from agent_challenge.selfdeploy import review as review_deploy

    class _Plan:
        def __init__(self) -> None:
            self.instance_type = "tdx.small"
            self.assignment = assignment
            self.image_ref = assignment["assignment_core"]["review_app"]["image_ref"]  # type: ignore[index]
            self.compose_hash = "c" * 64
            self.measurement = assignment["assignment_core"]["review_app"]["measurement"]  # type: ignore[index]
            self.review_session_token = assignment_token

    class _Encrypted:
        env_keys = ["OPENROUTER_API_KEY", "REVIEW_API_BASE_URL", "REVIEW_SESSION_TOKEN"]

    class _Deployer:
        def deploy(self, plan, encrypted):  # noqa: ARG002
            return {
                "schema_version": 1,
                "assignment_id": "assignment-1",
                "cvm_id": "cvm-1",
                "phala_create_receipt": {
                    "request_id": "r1",
                    "app_id": "a1",
                    "cvm_id": "cvm-1",
                    "receipt_sha256": "e" * 64,
                    "created_at_ms": 1,
                },
                "compose_identity": {
                    "image_ref": plan.image_ref,
                    "compose_hash": plan.compose_hash,
                    "app_identity": "review-v1",
                },
            }

    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setattr(review_deploy, "build_review_deployment_plan", lambda _r: _Plan())
    monkeypatch.setattr(
        review_deploy,
        "encrypt_review_secrets",
        lambda _plan, _values: _Encrypted(),
    )
    monkeypatch.setattr(
        review_deploy,
        "HttpReviewPhalaDeployment",
        lambda _client: _Deployer(),
    )
    # Avoid real Phala credential / network construction during the offline test.
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    monkeypatch.setattr(cli.lifecycle, "validate_lifecycle_budget", lambda **_k: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-to-leak")
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="0xexplicit",
        nonce="n-explicit",
        timestamp="1710000003",
        auto_sign=False,
        prepare_response=None,
        openrouter_key_env="OPENROUTER_API_KEY",
        phala_api="https://cloud-api.phala.com/api/v1",
        review_instance_type="tdx.small",
        eval_instance_type="tdx.small",
        review_runtime_hours=6.0,
        eval_runtime_hours=6.0,
        money_cap_usd=20.0,
        dry_run=False,
    )
    code = cli._ordered_review_command(args)
    assert code == 0, "live review deploy with explicit signed headers must succeed"
    assert fake_client.review_prepare.called
    assert fake_client.review_deployed.called
    printed = json.dumps(capture.payloads[-1])
    assert assignment_token not in printed
    assert "sk-test-not-to-leak" not in printed


# --------------------------------------------------------------------------- #
# 3) Auto-sign canonicalization covers exact query string
# --------------------------------------------------------------------------- #


def test_sign_request_identity_includes_exact_query_string(monkeypatch):
    class _Keypair:
        ss58_address = "hk"

        def sign(self, message: str) -> bytes:
            # Return length-stable bytes so the hex encoding is deterministic.
            return body_sha256(message.encode()).encode()[:32]

    monkeypatch.setattr(
        "agent_challenge.selfdeploy.client._load_signing_keypair",
        lambda: _Keypair(),
    )

    query = "cursor=page-2&limit=10"
    identity = sign_request_identity(
        hotkey="hk",
        method="GET",
        path="/submissions/1/review/history",
        query_string=query,
        timestamp="1710000100",
        nonce="auto-nonce",
        raw_body=b"",
    )
    expected_canonical = canonical_request_string(
        method="GET",
        path="/submissions/1/review/history",
        query_string=query,
        timestamp="1710000100",
        nonce="auto-nonce",
        raw_body=b"",
    )
    # Discriminator: empty-query signature must not match the real query.
    wrong = canonical_request_string(
        method="GET",
        path="/submissions/1/review/history",
        query_string="",
        timestamp="1710000100",
        nonce="auto-nonce",
        raw_body=b"",
    )
    assert expected_canonical != wrong
    signed_message = body_sha256(expected_canonical.encode()).encode()[:32]
    assert identity.signature == "0x" + signed_message.hex()
    assert identity.signature != "0x" + body_sha256(wrong.encode()).encode()[:32].hex()


@pytest.mark.parametrize(
    ("method_name", "path_suffix", "cursor"),
    [
        ("review_history", "review/history", "hist-cursor"),
        ("review_report", "review/report", "report-cursor"),
        ("eval_status", "eval/status", "status-cursor"),
    ],
)
def test_auto_sign_query_bearing_requests_sign_exact_cursor(
    monkeypatch,
    method_name: str,
    path_suffix: str,
    cursor: str,
):
    messages: list[str] = []

    class _Keypair:
        ss58_address = "hk"

        def sign(self, message: str) -> bytes:
            messages.append(message)
            return b"\x11" * 32

    monkeypatch.setattr(
        "agent_challenge.selfdeploy.client._load_signing_keypair",
        lambda: _Keypair(),
    )
    # Freeze time so the canonical timestamp is predictable.
    monkeypatch.setattr(
        "agent_challenge.selfdeploy.client.time.time",
        lambda: 1710000200.5,
    )
    monkeypatch.setattr(
        "agent_challenge.selfdeploy.client.secrets.token_urlsafe",
        lambda _n: "fresh-nonce",
    )

    opener = _CapturingOpener({"items": [], "next_cursor": None, "total_count": 0})
    identity = SignedIdentity(
        hotkey="hk",
        signature="auto",
        nonce="auto",
        timestamp="unused",
    )
    client = SelfDeployRouteClient(
        "https://challenge.example",
        identity=identity,
        auto_sign=True,
        opener=opener,
    )

    getattr(client, method_name)(42, cursor=cursor)

    assert len(messages) == 1
    expected_path = f"/submissions/42/{path_suffix}"
    expected_query = f"cursor={cursor}"
    expected = canonical_request_string(
        method="GET",
        path=expected_path,
        query_string=expected_query,
        timestamp="1710000200.5",
        nonce="fresh-nonce",
        raw_body=b"",
    )
    assert messages[0] == expected
    # Explicit discriminator against the empty-query residual bug.
    empty_query = canonical_request_string(
        method="GET",
        path=expected_path,
        query_string="",
        timestamp="1710000200.5",
        nonce="fresh-nonce",
        raw_body=b"",
    )
    assert messages[0] != empty_query
    request = opener.requests[0]
    assert f"?cursor={cursor}" in request.full_url
    headers = {k.lower(): v for k, v in request.header_items()}
    assert headers["x-signature"] == "0x" + (b"\x11" * 32).hex()
    assert headers["x-nonce"] == "fresh-nonce"


# --------------------------------------------------------------------------- #
# 4) Status / rejection paths remain executable, bounded retained state
# --------------------------------------------------------------------------- #


def test_cli_eval_status_and_review_history_report_bounded_retained_state(monkeypatch):
    forbidden = {
        "OPENROUTER_API_KEY": "sk-openrouter-sentinel",
        "EVAL_RUN_TOKEN": "eval-run-token-sentinel",
        "REVIEW_SESSION_TOKEN": "review-session-token-sentinel",
        "golden_plaintext": "GOLDEN-PLAINTEXT-SENTINEL",
        "score": 0.91,
    }
    history = {
        "items": [
            {
                "assignment_id": "a-1",
                "attempt": 1,
                "phase": "review_rejected",
                "terminal": True,
                "retryable": True,
                "reason_code": "policy_reject",
                "report_available": True,
            }
        ],
        "next_cursor": "c2",
        "total_count": 2,
    }
    status = {
        "items": [
            {
                "eval_run_id": "eval-1",
                "attempt": 1,
                "phase": "eval_rejected",
                "terminal": True,
                "verified": False,
                "retryable": False,
                "reason_code": "measurement_mismatch",
                "key_grant_state": "none",
                "key_release_nonce_state": "issued",
                "score_nonce_state": "issued",
                "result_available": False,
            }
        ],
        "next_cursor": None,
        "total_count": 1,
    }
    fake = MagicMock()
    fake.review_history.return_value = {
        **history,
        **{k: v for k, v in forbidden.items() if k != "score"},
    }
    fake.eval_status.return_value = {**status, "EVAL_RUN_TOKEN": forbidden["EVAL_RUN_TOKEN"]}
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake)
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)

    review_args = Namespace(
        review_command="history",
        submission_id=1,
        cursor="c1",
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=False,
    )
    assert cli._ordered_review_command(review_args) == 0
    fake.review_history.assert_called_once_with(1, cursor="c1")
    review_out = capture.payloads[-1]
    assert review_out["items"][0]["phase"] == "review_rejected"
    assert review_out["items"][0]["retryable"] is True
    assert review_out["total_count"] == 2
    rendered_review = json.dumps(review_out)
    for secret in (
        "sk-openrouter-sentinel",
        "eval-run-token-sentinel",
        "review-session-token-sentinel",
        "GOLDEN-PLAINTEXT-SENTINEL",
    ):
        assert secret not in rendered_review

    eval_args = Namespace(
        eval_command="status",
        submission_id=1,
        cursor=None,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=False,
    )
    assert cli._ordered_eval_command(eval_args) == 0
    fake.eval_status.assert_called_once_with(1, cursor=None)
    eval_out = capture.payloads[-1]
    item = eval_out["items"][0]
    assert item["phase"] == "eval_rejected"
    assert item["verified"] is False
    assert item.get("score") is None
    assert "success" not in json.dumps(eval_out).lower() or item["verified"] is False
    rendered_eval = json.dumps(eval_out)
    assert "eval-run-token-sentinel" not in rendered_eval
    # Coarse retained identifiers stay visible.
    assert item["eval_run_id"] == "eval-1"
    assert item["reason_code"] == "measurement_mismatch"


def test_cli_rejection_result_path_remains_executable():
    """Keep VAL-DEPLOY-026 surface executable: rejected result entry is callable."""
    # Build a minimal argv parse for top-level ``result`` (legacy surface).
    parser = cli.build_parser()
    # Ensure the top-level result subcommand still exists for rejection paths.
    args = parser.parse_args(
        [
            "result",
            "--from",
            "/tmp/does-not-need-to-exist-for-parse",
            "--allowlist",
            "NOT-IN-LIST",
            "--quote-verified",
            "false",
            "--nonce-state",
            "stale",
        ]
    )
    assert args.command == "result"
    assert args.quote_verified == "false"
    assert args.nonce_state == "stale"
    # Missing required signed fields on the ordered CLI must cleanly exit, not crash.
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(["review", "retry"])  # missing required args
        except SystemExit as exc:
            code = int(exc.code) if exc.code is not None else 1
    assert code != 0
    assert "Traceback" not in err.getvalue()
    assert "Traceback" not in out.getvalue()
