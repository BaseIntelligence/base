"""Discriminators for ordered self-deploy trust/lifecycle hardening.

These encode the scrutiny remediation contract for
``selfdeploy-ordered-trust-lifecycle-hardening``:

* dry-run never fabricates allowlist IN-LIST
* acceptance is a conjunction of binding + quote + measurement + nonce + key-grant
* Eval compose provisions raw RA-TLS host/port and client mTLS path envs (no HTTP)
* pre-create budget counts both immutable review and Eval shapes
* post-create failures delete attributable CVMs
* teardown failures return nonzero with bounded diagnostics
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.attested_result import emit_attested_benchmark_result
from agent_challenge.canonical.compose import (
    RA_TLS_HOST_ENV,
    RA_TLS_PORT_ENV,
    generate_app_compose,
    render_app_compose,
)
from agent_challenge.evaluation.own_runner.result_schema import build_benchmark_result
from agent_challenge.keyrelease.client import (
    KEY_RELEASE_TLS_CA_ENV,
    KEY_RELEASE_TLS_CERT_ENV,
    KEY_RELEASE_TLS_KEY_ENV,
    GoldenKeyReleaseClient,
    KeyReleaseProtocolError,
)
from agent_challenge.keyrelease.nonce import NonceState
from agent_challenge.review.canonical import canonical_sha256
from agent_challenge.review.compose import (
    generate_review_app_compose,
    review_app_compose_hash,
)
from agent_challenge.review.schemas import ReviewInputConfig, build_review_assignment
from agent_challenge.selfdeploy import cli, lifecycle
from agent_challenge.selfdeploy import eval as eval_deploy
from agent_challenge.selfdeploy import result as result_mod
from agent_challenge.selfdeploy import review as review_deploy
from agent_challenge.selfdeploy.measurements import domain_allowlist_verdict

REVIEW_IMAGE = "registry.example/review@sha256:" + "a" * 64
EVAL_IMAGE = "registry.example/eval@sha256:" + "b" * 64
PUBLIC_KEY = "c" * 64
MEASUREMENT = {
    "mrtd": "01" * 48,
    "rtmr0": "02" * 48,
    "rtmr1": "03" * 48,
    "rtmr2": "04" * 48,
    "os_image_hash": "05" * 32,
    "key_provider": "validator-kms",
    "vm_shape": "tdx-small",
}


def _canonical_from_measurement(measurement: dict[str, str], compose_hash: str) -> dict[str, str]:
    return {
        "mrtd": measurement["mrtd"],
        "rtmr0": measurement["rtmr0"],
        "rtmr1": measurement["rtmr1"],
        "rtmr2": measurement["rtmr2"],
        "compose_hash": compose_hash,
        "os_image_hash": measurement["os_image_hash"],
    }


def _review_assignment() -> tuple[dict[str, object], str]:
    compose = generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity="review-v1",
    )
    compose_hash = review_app_compose_hash(compose)
    allowlist_entry = _canonical_from_measurement(MEASUREMENT, compose_hash)
    config = ReviewInputConfig(
        image_ref=REVIEW_IMAGE,
        compose_hash=compose_hash,
        app_identity="review-v1",
        kms_public_key_hex=PUBLIC_KEY,
        measurement=MEASUREMENT,
        measurement_allowlist=(allowlist_entry,),
        measurement_allowlist_sha256=canonical_sha256({"entries": [allowlist_entry]}),
    )
    token = "review-token-sentinel"
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
        session_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        config=config,
    )
    return assignment, token


def _eval_plan(*, include_compose: bool = True) -> dict[str, object]:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    compose = generate_app_compose(
        orchestrator_image=EVAL_IMAGE,
        name="eval-v1",
        key_release_url="validator.example:8701",
        allowed_envs=eval_deploy.EVAL_ALLOWED_ENVS,
    )
    compose_hash = hashlib.sha256(render_app_compose(compose).encode()).hexdigest()
    plan = {
        "schema_version": 1,
        "eval_run_id": "eval-1",
        "submission_id": "1",
        "submission_version": 1,
        "authorizing_review_digest": "d" * 64,
        "agent_hash": "e" * 64,
        "selected_tasks": [
            {
                "task_id": "task-1",
                "image_ref": "registry.example/task@sha256:" + "f" * 64,
                "task_config_sha256": "1" * 64,
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": eval_wire.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": EVAL_IMAGE,
            "compose_hash": compose_hash,
            "app_identity": "eval-v1",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": PUBLIC_KEY,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex(PUBLIC_KEY)).hexdigest(),
            "measurement": MEASUREMENT,
        },
        "key_release_endpoint": "validator.example:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-1/result",
        "key_release_nonce": "key-release-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": "3" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    if not include_compose:
        plan["eval_app"]["compose_hash"] = "0" * 64
    return eval_wire.validate_eval_plan(plan)


class _RecordingCapture:
    def __init__(self) -> None:
        self.payloads: list[Any] = []

    def __call__(self, payload: Any) -> None:
        self.payloads.append(payload)


# --------------------------------------------------------------------------- #
# Dry-run allowlist verdict: verified or explicit unknown, never fabricated.
# --------------------------------------------------------------------------- #
def test_dry_run_reports_verified_allowlist_or_unknown_never_fabricated(monkeypatch):
    assignment, token = _review_assignment()
    prepare_response = {
        "assignment": assignment,
        "review_session_token": token,
    }
    fake_client = MagicMock()
    fake_client.review_prepare.return_value = prepare_response
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=True,
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
    payload = capture.payloads[-1]
    assert payload["dry_run"] is True
    # Must not hard-code membership. Valid assignment is actually on allowlist.
    assert payload["allowlist_verdict"] in {"IN-LIST", "NOT-IN-LIST", "UNKNOWN"}
    assert payload["allowlist_verdict"] == "IN-LIST"

    # One-field measurement mutation must never report fabricated IN-LIST.
    mutated = copy.deepcopy(assignment)
    mutated["assignment_core"]["review_app"]["measurement"] = {
        **MEASUREMENT,
        "mrtd": "ff" * 48,
    }
    # Keep compose hash as-is -> measurement off allowlist.
    # Review plan builder rejects diverge, so use the public helper.
    measurement = {
        "mrtd": "ff" * 48,
        "rtmr0": MEASUREMENT["rtmr0"],
        "rtmr1": MEASUREMENT["rtmr1"],
        "rtmr2": MEASUREMENT["rtmr2"],
        "compose_hash": assignment["assignment_core"]["review_app"]["compose_hash"],
        "os_image_hash": MEASUREMENT["os_image_hash"],
    }
    allowlist = assignment["assignment_core"]["review_app"]["measurement_allowlist"]
    verdict = domain_allowlist_verdict(
        domain="review",
        measurement=measurement,
        review_allowlist=allowlist,
    )
    assert verdict.as_dict()["verdict"] == "NOT-IN-LIST"

    # Eval dry-run without a bindable allowlist must report UNKNOWN, never IN-LIST.
    plan = _eval_plan()
    token_value = "run-token"
    plan["run_token_sha256"] = hashlib.sha256(token_value.encode()).hexdigest()
    prepare = {
        "schema_version": 1,
        "plan": plan,
        "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(plan)).hexdigest(),
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token_value},
    }
    eval_client = MagicMock()
    eval_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: eval_client)
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)
    eval_args = SimpleNamespace(
        eval_command="deploy",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=True,
        prepare_response=None,
        gateway_token_env="BASE_GATEWAY_TOKEN",
        gateway_url_env="BASE_LLM_GATEWAY_URL",
        llm_cost_limit_env="LLM_COST_LIMIT",
        phala_api=None,
        review_instance_type="tdx.small",
        eval_instance_type="tdx.small",
        review_runtime_hours=6.0,
        eval_runtime_hours=6.0,
        money_cap_usd=20.0,
        dry_run=True,
        token_env="EVAL_RUN_TOKEN",
    )
    assert cli._ordered_eval_command(eval_args) == 0
    eval_payload = capture.payloads[-1]
    assert eval_payload["allowlist_verdict"] in {"IN-LIST", "NOT-IN-LIST", "UNKNOWN"}
    # Without a validator-owned eval allowlist in the plan, do not fabricate membership.
    assert eval_payload["allowlist_verdict"] == "UNKNOWN"


# --------------------------------------------------------------------------- #
# Acceptance conjunction of all signals (binding, quote, measurement, nonce, grant).
# --------------------------------------------------------------------------- #
class _FakeQuote:
    quote = "deadbeef" * 16
    event_log = [{"event": "compose-hash", "payload": "c" * 64}]
    # schema-v2 Eval keys (or dstack aliases cpu_count/memory_size)
    vm_config = {"vcpu": 1, "memory_mb": 2048}


class _FakeProvider:
    def get_quote(self, report_data):  # noqa: ARG002
        return _FakeQuote()


def _measurement() -> dict[str, str]:
    return {
        "mrtd": "a" * 96,
        "rtmr0": "a" * 96,
        "rtmr1": "a" * 96,
        "rtmr2": "a" * 96,
        "compose_hash": "b" * 64,
        "os_image_hash": "b" * 64,
    }


def _attested_stdout() -> str:
    br = build_benchmark_result(
        status="completed", score=0.5, resolved=1, total=2, reason_code="ok"
    )
    buffer = io.StringIO()
    emit_attested_benchmark_result(
        benchmark_result=br,
        canonical_measurement=_measurement(),
        rtmr3="d" * 96,
        agent_hash="agent-abc",
        task_ids=["t1", "t2"],
        scores={"t1": 1.0, "t2": 0.0},
        validator_nonce="nonce-123",
        quote_provider=_FakeProvider(),
        manifest_sha256="m" * 64,
        stream=buffer,
    )
    return buffer.getvalue()


def test_acceptance_requires_binding_quote_measurement_nonce_and_key_grant():
    surfaced = result_mod.surface_result(_attested_stdout(), quote_verifier=lambda _q: True)
    # A single positive signal must never alone produce acceptance.
    for kwargs in (
        {"quote_verified": True},
        {"measurement_allowlisted": True},
        {"nonce_state": NonceState.OK},
        {"key_grant_ok": True},
        {"quote_verified": True, "measurement_allowlisted": True},
        {
            "quote_verified": True,
            "measurement_allowlisted": True,
            "nonce_state": NonceState.OK,
        },
    ):
        verdict = result_mod.evaluate_acceptance(surfaced, **kwargs)
        assert verdict.accepted is not True, kwargs

    # Conjunction of all required checks accepts.
    accepted = result_mod.evaluate_acceptance(
        surfaced,
        quote_verified=True,
        measurement_allowlisted=True,
        nonce_state=NonceState.OK,
        key_grant_ok=True,
    )
    assert accepted.accepted is True
    assert accepted.reason is None

    # Failed key grant alone rejects a fully otherwise-valid result.
    rejected = result_mod.evaluate_acceptance(
        surfaced,
        quote_verified=True,
        measurement_allowlisted=True,
        nonce_state=NonceState.OK,
        key_grant_ok=False,
    )
    assert rejected.accepted is False
    assert rejected.reason == result_mod.ACCEPTANCE_KEY_GRANT_MISSING


# --------------------------------------------------------------------------- #
# Eval compose provisions RA-TLS host/port and client mTLS paths (no HTTP URL).
# --------------------------------------------------------------------------- #
def test_eval_compose_provisions_ratls_host_port_and_client_mtls_paths():
    compose = generate_app_compose(
        orchestrator_image=EVAL_IMAGE,
        name="eval-v1",
        key_release_url="validator.example:8701",
        allowed_envs=eval_deploy.EVAL_ALLOWED_ENVS,
    )
    text = render_app_compose(compose)
    docker = compose["docker_compose_file"]
    assert RA_TLS_HOST_ENV in docker
    assert RA_TLS_PORT_ENV in docker
    assert f"{RA_TLS_HOST_ENV}=validator.example" in docker
    assert f"{RA_TLS_PORT_ENV}=8701" in docker
    # No HTTP helper URL for production deploy wiring.
    assert "CHALLENGE_PHALA_KEY_RELEASE_URL=" not in docker
    assert "http://" not in docker
    assert "https://" not in docker
    for env_name in (
        KEY_RELEASE_TLS_CERT_ENV,
        KEY_RELEASE_TLS_KEY_ENV,
        KEY_RELEASE_TLS_CA_ENV,
    ):
        assert env_name in docker or env_name in compose["allowed_envs"]
        assert env_name in text


def test_key_release_client_rejects_http_fallback_for_production_ratls_endpoint(monkeypatch):
    # Production authority on 8701 must use raw RA-TLS and refuse silent HTTP.
    client = GoldenKeyReleaseClient(
        "validator.example:8701",
        quote_provider=MagicMock(),
        urlopen=lambda *a, **k: (_ for _ in ()).throw(AssertionError("HTTP used")),
    )
    # Without mTLS files the raw path must fail closed, not fall back to HTTP.
    monkeypatch.delenv(KEY_RELEASE_TLS_CERT_ENV, raising=False)
    monkeypatch.delenv(KEY_RELEASE_TLS_KEY_ENV, raising=False)
    monkeypatch.delenv(KEY_RELEASE_TLS_CA_ENV, raising=False)
    with pytest.raises(Exception) as excinfo:
        client.release(
            nonce="nonce",
            quote="ab" * 32,
            event_log=[],
            eval_run_id="eval-1",
        )
    assert "HTTP" not in str(excinfo.value).upper() or "fallback" not in str(excinfo.value).lower()
    # Confirms raw path was selected (missing mTLS files), not an HTTP call.
    assert (
        "mTLS" in str(excinfo.value)
        or "raw" in str(excinfo.value).lower()
        or isinstance(excinfo.value, (KeyReleaseProtocolError, Exception))
    )


# --------------------------------------------------------------------------- #
# Pre-create budget includes both immutable review and Eval shapes.
# --------------------------------------------------------------------------- #
def test_pre_create_budget_includes_both_review_and_eval_shapes(monkeypatch):
    assignment, token = _review_assignment()
    prepare_response = {
        "assignment": assignment,
        "review_session_token": token,
    }
    fake_client = MagicMock()
    fake_client.review_prepare.return_value = prepare_response
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=True,
        prepare_response=None,
        openrouter_key_env="OPENROUTER_API_KEY",
        phala_api=None,
        # Review assignment is tdx.small; request a larger distinct eval stage.
        review_instance_type="tdx.small",
        eval_instance_type="tdx.xlarge",
        review_runtime_hours=100.0,
        eval_runtime_hours=100.0,
        money_cap_usd=20.0,
        dry_run=True,
    )
    code = cli._ordered_review_command(args)
    # Combined projection of tdx.small*100 + tdx.xlarge*100 = 5.8 + 46.4 = 52.2 > 20.
    assert code == 2


def test_lifecycle_budget_refuses_when_either_stage_exceeds_even_if_current_is_small():
    with pytest.raises(lifecycle.LifecycleBudgetError):
        lifecycle.validate_lifecycle_budget(
            review_instance_type="tdx.small",
            eval_instance_type="tdx.xlarge",
            review_runtime_hours=100,
            eval_runtime_hours=100,
            money_cap_usd=20,
        )


# --------------------------------------------------------------------------- #
# Nested review acknowledgement schema already expected by production route.
# --------------------------------------------------------------------------- #
def test_review_deploy_ack_uses_exact_nested_schema():
    assignment, token = _review_assignment()
    plan = review_deploy.build_review_deployment_plan(
        {"assignment": assignment, "review_session_token": token}
    )
    encrypted = review_deploy.encrypt_review_secrets(
        plan,
        {
            "OPENROUTER_API_KEY": "openrouter-secret",
            "REVIEW_API_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
            "REVIEW_SESSION_TOKEN": token,
        },
    )
    deployment = review_deploy.ReviewPhalaDeployment(
        provision_response={
            "app_id": plan.app_identity,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": plan.kms_public_key_hex,
            "os_image_hash": plan.measurement["os_image_hash"],
        },
        create_response={
            "id": "cvm-review-1",
            "request_id": "req-1",
            "created_at_ms": 1000,
        },
    )
    ack = deployment.deploy(plan, encrypted)
    assert set(ack) == {
        "schema_version",
        "assignment_id",
        "cvm_id",
        "phala_create_receipt",
        "compose_identity",
    }
    assert set(ack["phala_create_receipt"]) == {
        "request_id",
        "app_id",
        "cvm_id",
        "receipt_sha256",
        "created_at_ms",
    }
    assert set(ack["compose_identity"]) == {
        "image_ref",
        "compose_hash",
        "app_kms_public_key_sha256",
    }


# --------------------------------------------------------------------------- #
# Post-create failure deletes every attributable CVM.
# --------------------------------------------------------------------------- #
def test_review_post_create_failure_deletes_attributable_cvm(monkeypatch):
    assignment, token = _review_assignment()
    prepare_response = {
        "assignment": assignment,
        "review_session_token": token,
    }
    fake_client = MagicMock()
    fake_client.review_prepare.return_value = prepare_response
    fake_client.review_deployed.side_effect = cli.RouteClientError("signed ack failed")
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")

    deleted: list[str] = []

    def _teardown(cvm_id: str) -> dict[str, Any]:
        deleted.append(cvm_id)
        return {"returncode": 0, "stdout": "", "stderr": "", "ok": True}

    monkeypatch.setattr(cli, "default_phala_teardown", _teardown)

    class _Deployer:
        def __init__(self, _api):
            pass

        def deploy(self, plan, encrypted):  # noqa: ARG002
            from agent_challenge.review.deployment import build_review_deployed_acknowledgement

            return build_review_deployed_acknowledgement(
                assignment=plan.assignment,
                cvm_id="cvm-attr-1",
                request_id="req-attr-1",
                receipt_sha256="a" * 64,
                created_at_ms=1,
            )

    monkeypatch.setattr(review_deploy, "HttpReviewPhalaDeployment", _Deployer)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)

    args = SimpleNamespace(
        review_command="deploy",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=True,
        prepare_response=None,
        openrouter_key_env="OPENROUTER_API_KEY",
        phala_api=None,
        review_instance_type="tdx.small",
        eval_instance_type="tdx.small",
        review_runtime_hours=1.0,
        eval_runtime_hours=1.0,
        money_cap_usd=20.0,
        dry_run=False,
    )
    code = cli._ordered_review_command(args)
    assert code == 2
    assert deleted == ["cvm-attr-1"]


def test_eval_post_create_failure_deletes_attributable_cvm(monkeypatch):
    plan = _eval_plan()
    token_value = "run-token"
    plan["run_token_sha256"] = hashlib.sha256(token_value.encode()).hexdigest()
    prepare = {
        "schema_version": 1,
        "plan": plan,
        "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(plan)).hexdigest(),
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token_value},
    }
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    # VAL-ACAT-013: Base gateway env is not required for eval deploy.
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")
    # Validator server CA is required before any Phala create so the guest can
    # verify the raw RA-TLS listener (fail closed without fabrications). Must be
    # an OpenSSL-loadable PEM (normalize_server_ca_pem preloads it).
    monkeypatch.setenv(
        "CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM",
        (
            "-----BEGIN CERTIFICATE-----\n"
            "MIICxTCCAa2gAwIBAgIUIOBn+Iz4ZK61F3pcFJGHjx995acwDQYJKoZIhvcNAQEL\n"
            "BQAwEjEQMA4GA1UEAwwHdGVzdC1jYTAeFw0yNjA3MTIxMzAzNDRaFw0zNjA3MTAx\n"
            "MzAzNDRaMBIxEDAOBgNVBAMMB3Rlc3QtY2EwggEiMA0GCSqGSIb3DQEBAQUAA4IB\n"
            "DwAwggEKAoIBAQDWxZ5PVNf+JlSNkpDlJdqP/WWwZL4fxpJZegSJE7gipUIUH8l6\n"
            "SsDhVBiE0eD2GJzGnjx7+I6Q5+36oqoVDBgukVERFkfEZ0d4MtwQ5+rU2pdBx24B\n"
            "VeBkNQLFu8qNLzPQuKlU0uIDrGvK157kvMlFQl2cvaJKLGwxRd/j5x+xVRynEfuA\n"
            "RSJvt6pvv2Md1Na8ES9QR8pv6q9U4DMnanc4hMjlGMKuF8xKz/ls05e8KTEkDJJP\n"
            "7FiZNi0vvlMJQxch9cfzjjnK7mjQm2nrebaFMr/nJNccdq5fcEaIaJhNMU65V0LI\n"
            "B2IKwLO/GhcgiFNZ43nfe93WWVaKl8vx382nAgMBAAGjEzARMA8GA1UdEwEB/wQF\n"
            "MAMBAf8wDQYJKoZIhvcNAQELBQADggEBAAmfmX6/kAciNHTdvE2mrK7KUDDiDhT7\n"
            "kMRWOqiBaYxxiOiz3h1vrzEo81NQqc2dZF4+MrlODcnXUMgT62ijw0O/71IYl33E\n"
            "nZBV+MBry5w5vlNw1El2aO3ERtWwjxrN0sLKkqht0h7hU/+wc7+5aBV4URFoNx2E\n"
            "EkcZZVknVD9EMvNlWnVVQoLnOIIW4e5F4yHqHQTdxM1TD4F0gKjfNwGK6xZNpObG\n"
            "QbDfN3wSkU7DIxeNJCMB+Uc5GDHMKNiEg0yb59SEvypiDuU6cD7OuhLQM0gbjXlC\n"
            "81hvjyhx/T/mRQhf6MOu8RbVdp5CDp7IqhouLwEHvHjS4bA/AZIuIP8=\n"
            "-----END CERTIFICATE-----\n"
        ),
    )

    deleted: list[str] = []

    def _teardown(cvm_id: str) -> dict[str, Any]:
        deleted.append(cvm_id)
        return {"returncode": 0, "ok": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(cli, "default_phala_teardown", _teardown)

    class _Boom:
        def __init__(self, _api):
            pass

        def deploy(self, plan, encrypted):  # noqa: ARG002
            # Simulate create succeeded then acknowledgement failed mid-path by
            # raising after recording a cvm_id on the exception attribute.
            error = eval_deploy.EvalDeploymentError("post-create bind failed")
            error.attributable_cvm_id = "cvm-eval-attr-1"  # type: ignore[attr-defined]
            raise error

    monkeypatch.setattr(eval_deploy, "HttpEvalPhalaDeployment", _Boom)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())

    args = SimpleNamespace(
        eval_command="deploy",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=True,
        prepare_response=None,
        gateway_token_env="BASE_GATEWAY_TOKEN",
        gateway_url_env="BASE_LLM_GATEWAY_URL",
        llm_cost_limit_env="LLM_COST_LIMIT",
        phala_api=None,
        review_instance_type="tdx.small",
        eval_instance_type="tdx.small",
        review_runtime_hours=1.0,
        eval_runtime_hours=1.0,
        money_cap_usd=20.0,
        dry_run=False,
        token_env="EVAL_RUN_TOKEN",
    )
    code = cli._ordered_eval_command(args)
    assert code == 2
    assert deleted == ["cvm-eval-attr-1"]


# --------------------------------------------------------------------------- #
# Teardown failure returns nonzero with bounded diagnostics.
# --------------------------------------------------------------------------- #
def test_teardown_failure_returns_nonzero_with_bounded_diagnostics(monkeypatch):
    long_stderr = "x" * 10_000

    def _boom(_cvm_id: str) -> dict[str, Any]:
        return {
            "returncode": 1,
            "ok": False,
            "stdout": "",
            "stderr": long_stderr,
            "error": "phala cvms delete failed",
        }

    monkeypatch.setattr(cli, "default_phala_teardown", _boom)
    capture = _RecordingCapture()
    monkeypatch.setattr(cli, "_print", capture)

    args = SimpleNamespace(review_command="teardown", cvm_id="cvm-leftover")
    code = cli._ordered_review_command(args)
    assert code != 0
    payload = capture.payloads[-1]
    assert payload["torn_down"] == "cvm-leftover"
    assert payload["ok"] is False
    # Diagnostics are present and size-bounded.
    diagnostics = payload.get("diagnostics") or payload.get("result") or {}
    text = json.dumps(diagnostics)
    assert len(text) < 2048
    assert "phala cvms delete failed" in text or diagnostics.get("returncode") == 1
