"""TDD: eval deploy must fail LOUDLY on shape / measurement-pin mismatch.

Production residual: eval CVM shape moved tdx.small → tdx.xlarge while the
validator allowlist still pinned the small-shape rtmr0. Symptom was a generic
key-release denial deep in the TEE flow — nothing named vm_shape or rtmr0.

Miner-side guard must abort BEFORE Phala create and name:
- plan shape vs CLI --eval-instance-type
- field names vm_shape / instance_type
- that a shape change requires a matching rtmr0 pin + re-prepare
- that a stale pin surfaces only as a generic key-release denial
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import generate_app_compose, render_app_compose
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy import eval as eval_deploy

PUBLIC_KEY = "c" * 64
RUN_TOKEN = "eval-run-token-shape-pin-sentinel-7a1b2c3d"
EVAL_RUN_ID = "eval-run-shape-pin-1"

_SERVER_CA_PEM = (
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
)

# Distinct rtmr0 values (96 hex chars = 48 bytes) so prefix diffs are meaningful.
PLAN_RTMR0_SMALL = "68102e7b" + "aa" * 44  # 8 + 88 = 96
PLAN_RTMR0_XLARGE = "ec216f1d" + "bb" * 44
EXPECTED_RTMR0_MISMATCH = "deadbeef" + "cc" * 44


def _measurement(*, vm_shape: str, rtmr0: str) -> dict[str, str]:
    return {
        "mrtd": "01" * 48,
        "rtmr0": rtmr0,
        "rtmr1": "03" * 48,
        "rtmr2": "04" * 48,
        "os_image_hash": "05" * 32,
        "key_provider": "phala",
        "vm_shape": vm_shape,
    }


def _eval_prepare_wrapper(
    *,
    vm_shape: str = "tdx.small",
    rtmr0: str = PLAN_RTMR0_SMALL,
    token: str = RUN_TOKEN,
    eval_run_id: str = EVAL_RUN_ID,
) -> dict[str, Any]:
    eval_image = "registry.example/eval@sha256:" + "b" * 64
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    compose = generate_app_compose(
        orchestrator_image=eval_image,
        name="eval-v1",
        key_release_url="validator.example:8701",
        allowed_envs=eval_deploy.EVAL_ALLOWED_ENVS,
    )
    compose_hash = hashlib.sha256(render_app_compose(compose).encode()).hexdigest()
    plan = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": "1",
        "submission_version": 1,
        "authorizing_review_digest": "d" * 64,
        "agent_hash": "e" * 64,
        "package_tree_sha": "b" * 64,
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
            "image_ref": eval_image,
            "compose_hash": compose_hash,
            "app_identity": "eval-v1",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": PUBLIC_KEY,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex(PUBLIC_KEY)).hexdigest(),
            "measurement": _measurement(vm_shape=vm_shape, rtmr0=rtmr0),
        },
        "key_release_endpoint": "validator.example:8701",
        "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
        "key_release_nonce": "key-release-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    validated = eval_wire.validate_eval_plan(plan)
    return {
        "schema_version": 1,
        "plan": validated,
        "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(validated)).hexdigest(),
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token},
    }


def _deploy_args(**overrides: Any) -> SimpleNamespace:
    base = dict(
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
        eval_instance_type="tdx.xlarge",
        review_runtime_hours=1.0,
        eval_runtime_hours=1.0,
        money_cap_usd=20.0,
        dry_run=False,
        token_output=None,
        emit_run_token=True,
        output=None,
        expected_measurement=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _wire_prepare_only(
    monkeypatch: pytest.MonkeyPatch,
    prepare: dict[str, Any],
) -> tuple[MagicMock, list[Any]]:
    """Fake prepare; capture any Phala deploy attempts (must stay empty on mismatch)."""

    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM", _SERVER_CA_PEM)

    deploy_calls: list[Any] = []

    class _MustNotDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            deploy_calls.append((plan_obj, encrypted_obj))
            raise AssertionError("Phala deploy must not run on shape/pin mismatch")

    monkeypatch.setattr(eval_deploy, "HttpEvalPhalaDeployment", _MustNotDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    return fake_client, deploy_calls


# --------------------------------------------------------------------------- #
# S1 — shape mismatch names both shapes + field names + pin warning
# --------------------------------------------------------------------------- #


def test_shape_mismatch_stderr_names_shapes_fields_and_pin_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given plan vm_shape=tdx.small and CLI --eval-instance-type=tdx.xlarge,
    When eval deploy runs, Then exit 2, no Phala deploy, stderr is loud and specific.
    """

    prepare = _eval_prepare_wrapper(vm_shape="tdx.small", rtmr0=PLAN_RTMR0_SMALL)
    _client, deploy_calls = _wire_prepare_only(monkeypatch, prepare)

    args = _deploy_args(eval_instance_type="tdx.xlarge", emit_run_token=True)
    code = cli._ordered_eval_command(args)
    err = capsys.readouterr().err

    assert code == 2
    assert deploy_calls == []
    # Both shapes named.
    assert "tdx.small" in err
    assert "tdx.xlarge" in err
    # Field names the operator greps for.
    assert "vm_shape" in err
    assert "instance_type" in err
    # Stale pin / key-release footgun.
    assert "rtmr0" in err
    assert "key-release" in err.lower() or "key release" in err.lower()
    assert "allowlist" in err.lower()
    # Re-prepare guidance (prepare already spent the one-shot delivery).
    assert "re-prepare" in err.lower() or "reprepare" in err.lower() or "retry" in err.lower()
    # Never dump full measurement secrets.
    assert PLAN_RTMR0_SMALL not in err
    assert RUN_TOKEN not in err


def test_shape_mismatch_aborts_before_phala_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given shape mismatch, When deploy invoked, Then Phala deploy is never called."""

    prepare = _eval_prepare_wrapper(vm_shape="tdx.small")
    _client, deploy_calls = _wire_prepare_only(monkeypatch, prepare)
    args = _deploy_args(eval_instance_type="tdx.xlarge", emit_run_token=True)
    code = cli._ordered_eval_command(args)
    assert code == 2
    assert deploy_calls == []


def test_shape_mismatch_includes_truncated_plan_rtmr0_prefix_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given plan carries rtmr0, When shape mismatches, Then stderr shows a short prefix only."""

    prepare = _eval_prepare_wrapper(vm_shape="tdx.small", rtmr0=PLAN_RTMR0_SMALL)
    _wire_prepare_only(monkeypatch, prepare)
    args = _deploy_args(eval_instance_type="tdx.xlarge", emit_run_token=True)
    code = cli._ordered_eval_command(args)
    err = capsys.readouterr().err
    assert code == 2
    prefix = PLAN_RTMR0_SMALL[:12].lower()
    assert prefix in err.lower()
    assert PLAN_RTMR0_SMALL not in err
    assert PLAN_RTMR0_SMALL[12:] not in err


# --------------------------------------------------------------------------- #
# S2 — matching shapes still deploy (no false positive)
# --------------------------------------------------------------------------- #


def test_matching_shape_still_reaches_phala_deploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given plan shape matches CLI eval_instance_type, When deploy runs, Then Phala is called."""

    prepare = _eval_prepare_wrapper(vm_shape="tdx.xlarge", rtmr0=PLAN_RTMR0_XLARGE)
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM", _SERVER_CA_PEM)

    deploy_calls: list[Any] = []

    class _OkDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            deploy_calls.append(plan_obj.instance_type)
            return {
                "schema_version": 1,
                "eval_run_id": plan_obj.eval_run_id,
                "cvm_id": "cvm-ok-1",
                "phala_create_receipt": {
                    "request_id": "req",
                    "app_id": plan_obj.app_identity,
                    "cvm_id": "cvm-ok-1",
                    "receipt_sha256": "a" * 64,
                    "created_at_ms": 1,
                },
            }

    monkeypatch.setattr(eval_deploy, "HttpEvalPhalaDeployment", _OkDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    printed: list[Any] = []
    monkeypatch.setattr(cli, "_print", printed.append)

    args = _deploy_args(eval_instance_type="tdx.xlarge", emit_run_token=True)
    code = cli._ordered_eval_command(args)
    assert code == 0, printed
    assert deploy_calls == ["tdx.xlarge"]


# --------------------------------------------------------------------------- #
# S3 — optional --expected-measurement rtmr0 pin check
# --------------------------------------------------------------------------- #


def test_expected_measurement_rtmr0_mismatch_fails_before_phala(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given --expected-measurement with different rtmr0, When shapes match,
    Then abort before Phala with truncated prefix diff naming rtmr0.
    """

    prepare = _eval_prepare_wrapper(vm_shape="tdx.xlarge", rtmr0=PLAN_RTMR0_XLARGE)
    _client, deploy_calls = _wire_prepare_only(monkeypatch, prepare)

    expected_path = tmp_path / "expected-measurement.json"
    expected_path.write_text(
        json.dumps({"rtmr0": EXPECTED_RTMR0_MISMATCH, "vm_shape": "tdx.xlarge"}),
        encoding="utf-8",
    )

    args = _deploy_args(
        eval_instance_type="tdx.xlarge",
        emit_run_token=True,
        expected_measurement=str(expected_path),
    )
    code = cli._ordered_eval_command(args)
    err = capsys.readouterr().err

    assert code == 2
    assert deploy_calls == []
    assert "rtmr0" in err
    plan_prefix = PLAN_RTMR0_XLARGE[:12].lower()
    exp_prefix = EXPECTED_RTMR0_MISMATCH[:12].lower()
    assert plan_prefix in err.lower()
    assert exp_prefix in err.lower()
    # Full digests must not appear.
    assert PLAN_RTMR0_XLARGE not in err
    assert EXPECTED_RTMR0_MISMATCH not in err
    assert RUN_TOKEN not in err


def test_expected_measurement_rtmr0_match_allows_deploy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Given matching expected-measurement rtmr0, When deploy runs, Then Phala is called."""

    prepare = _eval_prepare_wrapper(vm_shape="tdx.xlarge", rtmr0=PLAN_RTMR0_XLARGE)
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM", _SERVER_CA_PEM)

    deploy_calls: list[Any] = []

    class _OkDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            deploy_calls.append(True)
            return {
                "schema_version": 1,
                "eval_run_id": plan_obj.eval_run_id,
                "cvm_id": "cvm-ok-2",
                "phala_create_receipt": {
                    "request_id": "req",
                    "app_id": plan_obj.app_identity,
                    "cvm_id": "cvm-ok-2",
                    "receipt_sha256": "a" * 64,
                    "created_at_ms": 1,
                },
            }

    monkeypatch.setattr(eval_deploy, "HttpEvalPhalaDeployment", _OkDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    monkeypatch.setattr(cli, "_print", lambda _p: None)

    expected_path = tmp_path / "expected-measurement.json"
    expected_path.write_text(
        json.dumps({"rtmr0": PLAN_RTMR0_XLARGE}),
        encoding="utf-8",
    )
    args = _deploy_args(
        eval_instance_type="tdx.xlarge",
        emit_run_token=True,
        expected_measurement=str(expected_path),
    )
    code = cli._ordered_eval_command(args)
    assert code == 0
    assert deploy_calls == [True]


def test_expected_measurement_flag_is_on_eval_deploy_parser() -> None:
    """Given CLI parser, When eval deploy --help is built, Then --expected-measurement exists."""

    parser = cli.build_parser()
    ns = parser.parse_args(
        [
            "eval",
            "deploy",
            "--base-url",
            "https://x",
            "--submission-id",
            "1",
            "--hotkey",
            "hk",
            "--auto-sign",
            "--expected-measurement",
            "/tmp/m.json",
            "--emit-run-token",
        ]
    )
    assert ns.expected_measurement == "/tmp/m.json"
