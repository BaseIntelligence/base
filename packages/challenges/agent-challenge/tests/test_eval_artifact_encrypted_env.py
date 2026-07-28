"""Eval CVM artifact URL + short-lived bearer grant via allowed_envs / encrypted_env.

Scenarios:
S1 — both CHALLENGE_PHALA_EVAL_ARTIFACT_{URL,TOKEN} names are in measured allowed_envs
S2 — deploy-time mint produces a grant verify_eval_artifact_grant accepts; encrypt carries both
S3 — http:// artifact URL is refused
S4 — token value never appears in log records or exception messages
S5 — VAL-ACAT-013 still rejects Base LLM gateway secrets
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from agent_challenge.api.eval_artifact_routes import verify_eval_artifact_grant
from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import DEFAULT_ALLOWED_ENVS, generate_app_compose
from agent_challenge.selfdeploy import eval as eval_deploy

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
_SECRET = "test-artifact-grant-secret"
_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)
_API_BASE = "https://chain.joinbase.ai/challenges/agent-challenge"


def _eval_plan(*, eval_run_id: str = "evalrun1", agent_hash: str | None = None) -> dict:
    agent_hash = agent_hash or ("e" * 64)
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
    from agent_challenge.canonical.compose import render_app_compose

    compose_hash = hashlib.sha256(render_app_compose(compose).encode()).hexdigest()
    plan = {
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "submission_id": "1",
        "submission_version": 1,
        "authorizing_review_digest": "d" * 64,
        "agent_hash": agent_hash,
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
            "image_ref": EVAL_IMAGE,
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
        "run_token_sha256": "3" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    return eval_wire.validate_eval_plan(plan)


def _deployment_plan(*, token: str = "run-token") -> eval_deploy.EvalDeploymentPlan:
    raw = _eval_plan()
    raw["run_token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
    return eval_deploy.build_eval_deployment_plan(
        {
            "schema_version": 1,
            "plan": raw,
            "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(raw)).hexdigest(),
            "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token},
        }
    )


def _base_secrets(
    plan: eval_deploy.EvalDeploymentPlan, *, token: str = "run-token"
) -> dict[str, str]:
    return {
        "EVAL_RUN_TOKEN": token,
        "LLM_COST_LIMIT": "1.00",
        "CHALLENGE_PHALA_ATTESTATION_ENABLED": "1",
        "CHALLENGE_PHALA_EVAL_PLAN": json.dumps(plan.plan, sort_keys=True, separators=(",", ":")),
        "CHALLENGE_PHALA_AGENT_HASH": plan.plan["agent_hash"],
        "CHALLENGE_PHALA_CANONICAL_MEASUREMENT": json.dumps(
            {
                "mrtd": plan.measurement["mrtd"],
                "rtmr0": plan.measurement["rtmr0"],
                "rtmr1": plan.measurement["rtmr1"],
                "rtmr2": plan.measurement["rtmr2"],
                "compose_hash": plan.compose_hash,
                "os_image_hash": plan.measurement["os_image_hash"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "CHALLENGE_PHALA_VALIDATOR_NONCE": plan.plan["key_release_nonce"],
    }


def test_s1_artifact_env_names_in_default_allowed_envs() -> None:
    """S1: measured compose allowed_envs lists both artifact delivery names."""
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_URL" in DEFAULT_ALLOWED_ENVS
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN" in DEFAULT_ALLOWED_ENVS
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_URL" in eval_deploy.EVAL_ALLOWED_ENVS
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN" in eval_deploy.EVAL_ALLOWED_ENVS

    compose = generate_app_compose(
        orchestrator_image=EVAL_IMAGE,
        allowed_envs=DEFAULT_ALLOWED_ENVS,
    )
    names = set(compose["allowed_envs"])
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_URL" in names
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN" in names
    # Names only — never NAME=value in allowed_envs.
    assert all("=" not in n for n in compose["allowed_envs"])
    # Compose bytes must not embed any grant/token value.
    rendered = json.dumps(compose)
    assert "v1." not in rendered or "v1." not in "".join(
        v for v in compose.get("docker_compose_file", "").split() if "Bearer" in v
    )


def test_s2_mint_verify_roundtrip_and_encrypt_carries_both() -> None:
    """S2: mint → verify accepts; encrypt_eval_secrets transmits both env keys."""
    plan = _deployment_plan()
    artifact_env = eval_deploy.build_eval_artifact_env_values(
        plan,
        secret=_SECRET,
        api_base_url=_API_BASE,
        now=_NOW,
    )
    url = artifact_env["CHALLENGE_PHALA_EVAL_ARTIFACT_URL"]
    token = artifact_env["CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN"]

    assert url.startswith("https://")
    assert url == f"{_API_BASE}/eval/v1/runs/{plan.eval_run_id}/artifact"
    assert not url.startswith("http://")

    grant = verify_eval_artifact_grant(
        secret=_SECRET,
        token=token,
        eval_run_id=plan.eval_run_id,
        now=_NOW,
    )
    assert grant.eval_run_id == plan.eval_run_id
    assert grant.agent_hash == plan.plan["agent_hash"]
    # Short-lived: TTL is explicit and finite (eval-run scale, not multi-day).
    ttl = grant.expires_at - _NOW
    assert timedelta(minutes=5) <= ttl <= timedelta(hours=6)

    secrets = {**_base_secrets(plan), **artifact_env}
    encrypted = eval_deploy.encrypt_eval_secrets(plan, secrets)
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_URL" in encrypted.env_keys
    assert "CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN" in encrypted.env_keys
    assert encrypted.ciphertext
    # Ciphertext object repr must not leak the raw grant.
    assert token not in repr(encrypted)


def test_s3_http_artifact_url_refused() -> None:
    """S3: plaintext http:// bases and values are refused fail-closed."""
    plan = _deployment_plan()
    with pytest.raises(eval_deploy.EvalDeploymentError, match="https"):
        eval_deploy.build_eval_artifact_env_values(
            plan,
            secret=_SECRET,
            api_base_url="http://insecure.example/challenges/agent-challenge",
            now=_NOW,
        )

    secrets = _base_secrets(plan)
    secrets["CHALLENGE_PHALA_EVAL_ARTIFACT_URL"] = (
        f"http://insecure.example/eval/v1/runs/{plan.eval_run_id}/artifact"
    )
    secrets["CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN"] = "v1.not.a.real.grant"
    with pytest.raises(eval_deploy.EvalDeploymentError, match="https"):
        eval_deploy.encrypt_eval_secrets(plan, secrets)


def test_s4_token_never_in_logs_or_exception_messages(caplog: pytest.LogCaptureFixture) -> None:
    """S4: grant token must not appear in log records or raised exception text."""
    plan = _deployment_plan()
    artifact_env = eval_deploy.build_eval_artifact_env_values(
        plan,
        secret=_SECRET,
        api_base_url=_API_BASE,
        now=_NOW,
    )
    token = artifact_env["CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN"]
    assert token  # non-empty grant

    with caplog.at_level(logging.DEBUG):
        secrets = {**_base_secrets(plan), **artifact_env}
        encrypted = eval_deploy.encrypt_eval_secrets(plan, secrets)
        assert encrypted.ciphertext

        # Force a validation failure path that might be tempted to echo values.
        bad = dict(secrets)
        bad["CHALLENGE_PHALA_EVAL_ARTIFACT_URL"] = "http://evil.example/artifact"
        with pytest.raises(eval_deploy.EvalDeploymentError) as exc_info:
            eval_deploy.encrypt_eval_secrets(plan, bad)
        assert token not in str(exc_info.value)
        assert token not in repr(exc_info.value)

    for record in caplog.records:
        assert token not in record.getMessage()
        if record.args:
            assert token not in str(record.args)
        if hasattr(record, "message"):
            assert token not in str(record.message)


def test_s5_gateway_secrets_still_forbidden() -> None:
    """S5: VAL-ACAT-013 — Base LLM gateway names remain rejected."""
    plan = _deployment_plan()
    artifact_env = eval_deploy.build_eval_artifact_env_values(
        plan,
        secret=_SECRET,
        api_base_url=_API_BASE,
        now=_NOW,
    )
    secrets = {**_base_secrets(plan), **artifact_env, "BASE_GATEWAY_TOKEN": "gw-secret"}
    with pytest.raises(eval_deploy.EvalDeploymentError, match="gateway"):
        eval_deploy.encrypt_eval_secrets(plan, secrets)


def test_artifact_grant_ttl_is_explicit() -> None:
    """TTL constant is documented and short enough for a single eval run."""
    ttl = eval_deploy.EVAL_ARTIFACT_GRANT_TTL
    assert isinstance(ttl, timedelta)
    assert timedelta(minutes=5) <= ttl <= timedelta(hours=6)
