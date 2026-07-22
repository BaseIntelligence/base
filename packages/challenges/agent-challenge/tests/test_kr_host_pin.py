"""VAL-ACLOCK-008/009/010: pin key_release_endpoint; legacy free KR URL not miner trust."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.canonical.compose import (
    DEFAULT_ALLOWED_ENVS,
    DEFAULT_KEY_RELEASE_RA_TLS_PORT,
    is_validator_key_release_authority,
    parse_key_release_authority,
)
from agent_challenge.keyrelease.client import KEY_RELEASE_URL_ENV
from agent_challenge.selfdeploy import eval as eval_deploy
from agent_challenge.selfdeploy.eval import (
    EVAL_ALLOWED_ENVS,
    EVAL_REQUIRED_SECRET_ENVS,
    EvalDeploymentError,
    EvalDeploymentPlan,
    encrypt_eval_secrets,
)


def _eval_plan(*, key_release_endpoint: str) -> dict[str, Any]:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    return {
        "schema_version": 1,
        "eval_run_id": "eval-run-kr-pin",
        "submission_id": "submission-001",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": "a" * 64,
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "selected_tasks": [
            {
                "task_id": "task-a",
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "2" * 64,
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "d" * 64,
            "compose_hash": "c" * 64,
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "3" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("3" * 64)).hexdigest(),
            "measurement": {
                "mrtd": "a1" * 48,
                "rtmr0": "a2" * 48,
                "rtmr1": "a3" * 48,
                "rtmr2": "a4" * 48,
                "os_image_hash": "a5" * 32,
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": key_release_endpoint,
        "result_endpoint": "/evaluation/v1/runs/eval-run-kr-pin/result",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "run_token_sha256": "5" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-008 — plan wire rejects evil free KR hosts / non-authority forms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://evil.example/key-release",
        "http://evil.example:8701/",
        "https://validator-kr.example.invalid:8701",
        "https://86.38.238.235:8701/release",
        "evil.example",  # missing port
        "ftp://evil.example:8701",
        "host/with/path:8701",
        "user@evil.example:8701",
        "ratls://evil.example:8701/extra",
        "",
        "   ",
    ],
)
def test_validate_eval_plan_rejects_evil_or_free_kr_endpoints(endpoint: str) -> None:
    """VAL-ACLOCK-008: free HTTP(S)/junk KR endpoints fail closed on the plan wire."""

    plan = _eval_plan(key_release_endpoint=endpoint)
    with pytest.raises(ew.EvalWireError, match="key_release_endpoint"):
        ew.validate_eval_plan(plan)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://evil.example/kr",
        "http://127.0.0.1:8701",
        "evil.example",
        "not a host",
    ],
)
def test_is_validator_key_release_authority_rejects_evil(endpoint: str) -> None:
    assert is_validator_key_release_authority(endpoint) is False
    assert parse_key_release_authority(endpoint) is None


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-010 — honest validator RA-TLS / authority forms still validate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "endpoint",
    [
        "validator.example:8701",
        "keyrelease.example:8701",
        "86.38.238.235:8701",
        "ratls://84.32.70.61:8701",
        "tls://validator.example:8701",
        "tcp://127.0.0.1:8701",
        "validator.example:8700",  # authority form (non-default port still host:port)
        "ratls://validator.example:9443",
    ],
)
def test_validate_eval_plan_accepts_honest_ra_tls_authority(endpoint: str) -> None:
    """VAL-ACLOCK-010: honest validator KR / RA-TLS authority forms remain green."""

    plan = _eval_plan(key_release_endpoint=endpoint)
    validated = ew.validate_eval_plan(plan)
    assert validated["key_release_endpoint"] == endpoint.strip()
    parsed = parse_key_release_authority(endpoint)
    assert parsed is not None
    host, port = parsed
    assert host
    assert 1 <= port <= 65535


def test_parse_key_release_authority_default_port_constant() -> None:
    assert DEFAULT_KEY_RELEASE_RA_TLS_PORT == 8701
    assert parse_key_release_authority("validator.example:8701") == (
        "validator.example",
        8701,
    )


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-009 — legacy free KR URL is not miner-authoritative in prod
# --------------------------------------------------------------------------- #
def test_legacy_free_key_release_url_name_not_miner_trust_root() -> None:
    """VAL-ACLOCK-009: free KR URL name may remain for pin stability but is not trust root.

    Encrypt refuses free HTTP(S) / mismatched authorities; plan wire already
    pins RA-TLS form. Prefer KEY_RELEASE_RA_TLS_HOST/PORT + plan endpoint.
    """

    assert KEY_RELEASE_URL_ENV == "CHALLENGE_PHALA_KEY_RELEASE_URL"
    # Name retained so existing measure-time pin compose hashes stay stable.
    assert KEY_RELEASE_URL_ENV in DEFAULT_ALLOWED_ENVS
    assert KEY_RELEASE_URL_ENV in EVAL_ALLOWED_ENVS
    assert "CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM" in DEFAULT_ALLOWED_ENVS
    assert "EVAL_RUN_TOKEN" in EVAL_REQUIRED_SECRET_ENVS


def _deployment_plan_stub(
    *, key_release_endpoint: str = "validator.example:8701"
) -> EvalDeploymentPlan:
    private = X25519PrivateKey.generate()
    public_hex = private.public_key().public_bytes_raw().hex()
    plan = ew.validate_eval_plan(_eval_plan(key_release_endpoint=key_release_endpoint))
    return EvalDeploymentPlan(
        plan=plan,
        plan_sha256="b" * 64,
        compose={"name": "stub"},
        compose_text="{}",
        compose_hash="c" * 64,
        app_identity="agent-challenge-eval",
        image_ref=plan["eval_app"]["image_ref"],
        kms_public_key_hex=public_hex,
        kms_public_key_sha256=hashlib.sha256(bytes.fromhex(public_hex)).hexdigest(),
        measurement=dict(plan["eval_app"]["measurement"]),
        eval_run_id=plan["eval_run_id"],
        eval_run_token="eval-run-token-honest",
        instance_type="tdx.small",
    )


def test_encrypt_eval_secrets_refuses_legacy_free_http_key_release_url() -> None:
    """VAL-ACLOCK-009: encrypt refuses miner free HTTP(S) KR URL values."""

    dep = _deployment_plan_stub()
    secrets = {
        "CHALLENGE_PHALA_ATTESTATION_ENABLED": "1",
        "CHALLENGE_PHALA_EVAL_PLAN": "{}",
        "EVAL_RUN_TOKEN": dep.eval_run_token,
        "LLM_COST_LIMIT": "1.0",
        KEY_RELEASE_URL_ENV: "https://evil.example/key-release",
    }
    with pytest.raises(EvalDeploymentError, match="KEY_RELEASE|key_release|not miner"):
        encrypt_eval_secrets(dep, secrets)


def test_encrypt_eval_secrets_refuses_free_url_mismatching_plan_authority() -> None:
    """VAL-ACLOCK-009: free URL cannot override plan RA-TLS authority."""

    dep = _deployment_plan_stub(key_release_endpoint="validator.example:8701")
    secrets = {
        "CHALLENGE_PHALA_ATTESTATION_ENABLED": "1",
        "CHALLENGE_PHALA_EVAL_PLAN": "{}",
        "EVAL_RUN_TOKEN": dep.eval_run_token,
        "LLM_COST_LIMIT": "1.0",
        KEY_RELEASE_URL_ENV: "evil.example:8701",
    }
    with pytest.raises(EvalDeploymentError, match="KEY_RELEASE|key_release|not miner"):
        encrypt_eval_secrets(dep, secrets)

    # Matching authority form (redundant with plan) is accepted; not a free web URL.
    secrets[KEY_RELEASE_URL_ENV] = "validator.example:8701"
    encrypted = encrypt_eval_secrets(dep, secrets)
    assert KEY_RELEASE_URL_ENV in encrypted.env_keys


def test_encrypt_eval_secrets_honest_path_without_free_kr_url() -> None:
    """VAL-ACLOCK-010: honest encrypt path green without free KR URL env."""

    dep = _deployment_plan_stub(key_release_endpoint="ratls://84.32.70.61:8701")
    secrets = {
        "CHALLENGE_PHALA_ATTESTATION_ENABLED": "1",
        "CHALLENGE_PHALA_EVAL_PLAN": '{"k":1}',
        "EVAL_RUN_TOKEN": dep.eval_run_token,
        "LLM_COST_LIMIT": "2.5",
        "OPENROUTER_API_KEY": "sk-or-v1-test-not-real",
        "CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM": (
            "-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----"
        ),
    }
    encrypted = encrypt_eval_secrets(dep, secrets)
    assert set(encrypted.env_keys) <= set(EVAL_ALLOWED_ENVS)
    assert KEY_RELEASE_URL_ENV not in encrypted.env_keys
    assert encrypted.ciphertext


def test_measure_time_placeholder_not_accepted_as_signed_plan_endpoint() -> None:
    """Measure-time HTTPS placeholder is for compose pin only, not plan trust root."""

    plan = _eval_plan(
        key_release_endpoint=eval_deploy.MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER
    )
    with pytest.raises(ew.EvalWireError, match="key_release_endpoint"):
        ew.validate_eval_plan(plan)


def test_eval_plan_normalizes_stripped_authority() -> None:
    plan = _eval_plan(key_release_endpoint="  validator.example:8701  ")
    validated = ew.validate_eval_plan(plan)
    assert validated["key_release_endpoint"] == "validator.example:8701"


def test_validate_eval_plan_fixture_baseline_still_closed() -> None:
    """Regression: closed plan shape with honest KR remains round-trip stable."""

    plan = _eval_plan(key_release_endpoint="keyrelease.example:8701")
    assert ew.validate_eval_plan(plan) == {
        **plan,
        "key_release_endpoint": "keyrelease.example:8701",
    }
    crossed = copy.deepcopy(plan)
    crossed["score_nonce"] = crossed["key_release_nonce"]
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_plan(crossed)
