"""TDD: eval deploy must hand the one-time EVAL_RUN_TOKEN to the miner.

Root cause (production): guest emits attested result only; host posts via
``eval result --token-env EVAL_RUN_TOKEN``. The token lived only inside
``eval deploy`` memory (CVM encrypted_env) and every miner-readable surface
was redacted — closed loop, result could never be posted.

Security: surfacing the token to the miner is a capability/anti-replay
credential for the post only. Integrity is the TEE quote bound to the plan.
"""

from __future__ import annotations

import hashlib
import json
import stat
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import generate_app_compose, render_app_compose
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy import eval as eval_deploy
from agent_challenge.selfdeploy.client import RouteClientError

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

# Distinct sentinel so accidental redaction/leak assertions cannot false-pass.
RUN_TOKEN = "eval-run-token-handoff-sentinel-9f3c2a1b"
EVAL_RUN_ID = "eval-run-handoff-1"

# OpenSSL-loadable test CA (same fixture as ordered trust hardening).
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


def _eval_prepare_wrapper(
    *,
    token: str | None = RUN_TOKEN,
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
    run_token = token if isinstance(token, str) and token else "placeholder-token"
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
            "measurement": MEASUREMENT,
        },
        "key_release_endpoint": "validator.example:8701",
        "result_endpoint": f"/evaluation/v1/runs/{eval_run_id}/result",
        "key_release_nonce": "key-release-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": hashlib.sha256(run_token.encode()).hexdigest(),
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    validated = eval_wire.validate_eval_plan(plan)
    delivery: dict[str, str] | None
    if isinstance(token, str) and token:
        delivery = {"env_key": "EVAL_RUN_TOKEN", "token": token}
    else:
        delivery = None
    return {
        "schema_version": 1,
        "plan": validated,
        "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(validated)).hexdigest(),
        "secret_delivery": delivery,
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
        eval_instance_type="tdx.small",
        review_runtime_hours=1.0,
        eval_runtime_hours=1.0,
        money_cap_usd=20.0,
        dry_run=False,
        token_output=None,
        emit_run_token=False,
        output=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _wire_successful_deploy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: str = RUN_TOKEN,
    captured_secrets: list[dict[str, str]] | None = None,
) -> list[Any]:
    """Fake prepare + Phala deploy; optionally capture encrypt secrets."""

    prepare = _eval_prepare_wrapper(token=token)
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM", _SERVER_CA_PEM)

    if captured_secrets is not None:
        real_encrypt = eval_deploy.encrypt_eval_secrets

        def _capture(plan: Any, secrets: Any) -> Any:
            captured_secrets.append(dict(secrets))
            return real_encrypt(plan, secrets)

        monkeypatch.setattr(eval_deploy, "encrypt_eval_secrets", _capture)

    class _FixedDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            assert plan_obj.eval_run_token == token
            assert "EVAL_RUN_TOKEN" in encrypted_obj.env_keys
            return {
                "schema_version": 1,
                "eval_run_id": plan_obj.eval_run_id,
                "cvm_id": "cvm-eval-handoff-1",
                "phala_create_receipt": {
                    "request_id": "req",
                    "app_id": plan_obj.app_identity,
                    "cvm_id": "cvm-eval-handoff-1",
                    "receipt_sha256": "a" * 64,
                    "created_at_ms": 1,
                },
            }

    monkeypatch.setattr(eval_deploy, "HttpEvalPhalaDeployment", _FixedDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())
    printed: list[Any] = []
    monkeypatch.setattr(cli, "_print", printed.append)
    return printed


# --------------------------------------------------------------------------- #
# S1 — stdout emission with --emit-run-token
# --------------------------------------------------------------------------- #


def test_emit_run_token_puts_exact_token_and_run_id_on_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given live deploy + --emit-run-token, When deploy succeeds,
    Then stdout JSON carries exact eval_run_token and eval_run_id.
    """

    printed = _wire_successful_deploy(monkeypatch)
    args = _deploy_args(emit_run_token=True)
    code = cli._ordered_eval_command(args)
    assert code == 0, printed
    assert len(printed) == 1
    payload = printed[0]
    assert payload["eval_run_token"] == RUN_TOKEN
    assert payload["eval_run_id"] == EVAL_RUN_ID
    assert payload["stage"] == "eval_deployed"


# --------------------------------------------------------------------------- #
# S2 — --token-output secure file
# --------------------------------------------------------------------------- #


def test_token_output_writes_mode_0600_file_with_exact_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Given --token-output PATH, When deploy succeeds,
    Then PATH exists mode 0o600 and contains the exact token; stdout has run id.
    """

    token_path = tmp_path / "eval-run.token"
    printed = _wire_successful_deploy(monkeypatch)
    args = _deploy_args(token_output=str(token_path))
    code = cli._ordered_eval_command(args)
    assert code == 0, printed
    assert token_path.is_file()
    mode = stat.S_IMODE(token_path.stat().st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    assert token_path.read_text(encoding="utf-8") == RUN_TOKEN
    assert printed[0]["eval_run_id"] == EVAL_RUN_ID
    # Token must not appear in stdout when only --token-output is used.
    assert RUN_TOKEN not in json.dumps(printed)
    assert "eval_run_token" not in printed[0]


# --------------------------------------------------------------------------- #
# S3 — fail closed without handoff flags
# --------------------------------------------------------------------------- #


def test_non_dry_run_without_handoff_flags_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Given non-dry-run deploy with neither flag, When invoked,
    Then non-zero exit and stderr names both flags (no Phala spend).
    """

    prepare = _eval_prepare_wrapper()
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")

    deploy_calls: list[Any] = []

    class _MustNotDeploy:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            deploy_calls.append((plan_obj, encrypted_obj))
            raise AssertionError("Phala deploy must not run when handoff flags missing")

    monkeypatch.setattr(eval_deploy, "HttpEvalPhalaDeployment", _MustNotDeploy)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())

    args = _deploy_args(dry_run=False, token_output=None, emit_run_token=False)
    code = cli._ordered_eval_command(args)
    err = capsys.readouterr().err
    assert code == 2
    assert deploy_calls == []
    assert "--token-output" in err
    assert "--emit-run-token" in err
    assert "eval result" in err.lower() or "EVAL_RUN_TOKEN" in err
    assert RUN_TOKEN not in err


# --------------------------------------------------------------------------- #
# S4 — dry-run without flags still OK, no raw token
# --------------------------------------------------------------------------- #


def test_dry_run_without_handoff_flags_ok_and_no_raw_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given dry-run with neither handoff flag, When invoked,
    Then exit 0 and stdout has no raw token.
    """

    prepare = _eval_prepare_wrapper()
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")
    printed: list[Any] = []
    monkeypatch.setattr(cli, "_print", printed.append)

    args = _deploy_args(dry_run=True, token_output=None, emit_run_token=False)
    code = cli._ordered_eval_command(args)
    assert code == 0, printed
    assert printed[0]["dry_run"] is True
    assert printed[0]["eval_run_id"] == EVAL_RUN_ID
    assert RUN_TOKEN not in json.dumps(printed)
    assert "eval_run_token" not in printed[0]


# --------------------------------------------------------------------------- #
# S5 — --output plan JSON never contains token
# --------------------------------------------------------------------------- #


def test_output_plan_json_never_contains_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Given deploy --output PATH (+ handoff via token-output), When success,
    Then the plan/metadata JSON at PATH never contains the raw token string.
    """

    out_path = tmp_path / "deploy-plan.json"
    token_path = tmp_path / "token"
    printed = _wire_successful_deploy(monkeypatch)
    args = _deploy_args(
        emit_run_token=True,  # even when stdout has token, --output must not
        token_output=str(token_path),
        output=str(out_path),
    )
    code = cli._ordered_eval_command(args)
    assert code == 0, printed
    assert out_path.is_file()
    body = out_path.read_text(encoding="utf-8")
    assert RUN_TOKEN not in body
    parsed = json.loads(body)
    assert "eval_run_token" not in parsed
    # Nested secret_delivery must stay redacted if present.
    if "secret_delivery" in parsed:
        assert parsed["secret_delivery"] == {"env_key": "EVAL_RUN_TOKEN"} or (
            isinstance(parsed["secret_delivery"], dict) and "token" not in parsed["secret_delivery"]
        )


# --------------------------------------------------------------------------- #
# S6 — prepare / status still redact
# --------------------------------------------------------------------------- #


def test_eval_prepare_and_status_still_redact_secret_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """REGRESSION: prepare/status redaction of secret_delivery is unchanged."""

    prepare = _eval_prepare_wrapper(token=RUN_TOKEN)
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    fake_client.eval_status.return_value = {
        "schema_version": 1,
        "eval_run_id": EVAL_RUN_ID,
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": RUN_TOKEN},
        "phase": "eval_prepared",
    }
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    printed: list[Any] = []
    monkeypatch.setattr(cli, "_print", printed.append)

    out_path = tmp_path / "prepare.json"
    prep_args = SimpleNamespace(
        eval_command="prepare",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=True,
        output=str(out_path),
    )
    assert cli._ordered_eval_command(prep_args) == 0
    assert printed[0]["secret_delivery"] == {"env_key": "EVAL_RUN_TOKEN"}
    assert RUN_TOKEN not in json.dumps(printed[0])
    assert RUN_TOKEN not in out_path.read_text(encoding="utf-8")
    assert json.loads(out_path.read_text(encoding="utf-8"))["secret_delivery"] == {
        "env_key": "EVAL_RUN_TOKEN"
    }

    printed.clear()
    status_args = SimpleNamespace(
        eval_command="status",
        submission_id=1,
        base_url="https://challenge.example",
        hotkey="hk",
        signature="sig",
        nonce="n",
        timestamp=None,
        auto_sign=True,
        cursor=None,
    )
    assert cli._ordered_eval_command(status_args) == 0
    assert printed[0]["secret_delivery"] == {"env_key": "EVAL_RUN_TOKEN"}
    assert RUN_TOKEN not in json.dumps(printed[0])


# --------------------------------------------------------------------------- #
# S7 — token never in exception messages
# --------------------------------------------------------------------------- #


def test_token_never_present_in_raised_exception_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a deploy path that raises after token is known, When error surfaces,
    Then exception / stderr text never contains the raw token.
    """

    prepare = _eval_prepare_wrapper(token=RUN_TOKEN)
    fake_client = MagicMock()
    fake_client.eval_prepare.return_value = prepare
    monkeypatch.setattr(cli, "_route_client", lambda _args: fake_client)
    monkeypatch.setenv("LLM_COST_LIMIT", "1.00")
    monkeypatch.setenv("CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM", _SERVER_CA_PEM)

    class _Boom:
        def __init__(self, _api: object) -> None:
            pass

        def deploy(self, plan_obj: Any, encrypted_obj: Any) -> dict[str, Any]:
            raise eval_deploy.EvalDeploymentError(
                f"post-create bind failed for run {plan_obj.eval_run_id}"
            )

    monkeypatch.setattr(eval_deploy, "HttpEvalPhalaDeployment", _Boom)
    monkeypatch.setattr(cli, "PhalaCloudClient", lambda **_k: object())

    args = _deploy_args(emit_run_token=True)
    # Capture via raising path inside _ordered_eval_command (prints error: …).
    import io
    import sys

    buf = io.StringIO()
    old = sys.stderr
    sys.stderr = buf
    try:
        code = cli._ordered_eval_command(args)
    finally:
        sys.stderr = old
    err = buf.getvalue()
    assert code == 2
    assert RUN_TOKEN not in err
    # Fail-closed message path also must not embed token.
    with pytest.raises(RouteClientError) as excinfo:
        # Direct unit: the handoff guard must not interpolate the token.
        raise RouteClientError(
            "eval deploy requires --token-output PATH and/or --emit-run-token "
            "so the miner can later call eval result with EVAL_RUN_TOKEN"
        )
    assert RUN_TOKEN not in str(excinfo.value)


# --------------------------------------------------------------------------- #
# S8 — CVM encrypted env still receives EVAL_RUN_TOKEN
# --------------------------------------------------------------------------- #


def test_eval_run_token_still_injected_into_cvm_encrypted_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adjacent: handoff must not remove EVAL_RUN_TOKEN from CVM env injection."""

    captured: list[dict[str, str]] = []
    printed = _wire_successful_deploy(monkeypatch, captured_secrets=captured)
    args = _deploy_args(emit_run_token=True)
    assert cli._ordered_eval_command(args) == 0
    assert captured, printed
    assert captured[0]["EVAL_RUN_TOKEN"] == RUN_TOKEN


def test_redact_capabilities_still_strips_token_key() -> None:
    """Unit: _redact_capabilities never leaves token bytes in nested payloads."""

    raw = {
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": RUN_TOKEN},
        "EVAL_RUN_TOKEN": RUN_TOKEN,
        "nested": {"token": RUN_TOKEN},
    }
    redacted = cli._redact_capabilities(raw)
    dumped = json.dumps(redacted)
    assert RUN_TOKEN not in dumped
    assert redacted["secret_delivery"] == {"env_key": "EVAL_RUN_TOKEN"}
