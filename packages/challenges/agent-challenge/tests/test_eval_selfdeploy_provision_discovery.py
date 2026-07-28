"""Eval self-deploy: discover Phala app_id from provision (handle, not pin).

Trust anchors remain compose_hash + OS/measurement. Assignment 40-hex
app_identity is advisory only and must never gate deploy when Phala returns a
different CREATE-style app_id for the deployer account.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import (
    app_compose_hash,
    generate_app_compose,
    render_app_compose,
)
from agent_challenge.selfdeploy import eval as eval_deploy

EVAL_IMAGE = "registry.example/eval@sha256:" + "b" * 64
# Plan/assignment pin (operator-minted) — must NOT gate discovery.
_ASSIGNMENT_PIN_APP_ID = "f024ea23" + ("ab" * 16)
# Live Phala CREATE app_id for a different deployer account.
_DISCOVERED_APP_ID = "1850aa11" + ("cd" * 16)
# Distinct encrypt material returned by provision (not the plan pin key).
_PLAN_PUBKEY = "aa" * 32
_DISCOVERED_PUBKEY = "bb" * 32
_OS_HASH = "05" * 32
MEASUREMENT = {
    "mrtd": "01" * 48,
    "rtmr0": "02" * 48,
    "rtmr1": "03" * 48,
    "rtmr2": "04" * 48,
    "os_image_hash": _OS_HASH,
    "key_provider": "validator-kms",
    "vm_shape": "tdx-small",
}
_TOKEN = "run-token-discovery"
_API_BASE = "https://chain.joinbase.ai/challenges/agent-challenge"
_GRANT_SECRET = "test-artifact-grant-secret-discovery"


def _compose_for_name(name: str) -> tuple[dict[str, Any], str]:
    compose = generate_app_compose(
        orchestrator_image=EVAL_IMAGE,
        name=name,
        key_release_url="validator.example:8701",
        allowed_envs=eval_deploy.EVAL_ALLOWED_ENVS,
    )
    compose_hash = hashlib.sha256(render_app_compose(compose).encode()).hexdigest()
    return compose, compose_hash


def _raw_plan(
    *,
    app_identity: str | None,
    compose_hash: str,
    pubkey: str = _PLAN_PUBKEY,
) -> dict[str, Any]:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    eval_app: dict[str, Any] = {
        "image_ref": EVAL_IMAGE,
        "compose_hash": compose_hash,
        "kms_key_algorithm": "x25519",
        "kms_public_key_hex": pubkey,
        "kms_public_key_sha256": hashlib.sha256(bytes.fromhex(pubkey)).hexdigest(),
        "measurement": MEASUREMENT,
    }
    if app_identity is not None:
        eval_app["app_identity"] = app_identity
    plan = {
        "schema_version": 1,
        "eval_run_id": "evaldisc1",
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
        "eval_app": eval_app,
        "key_release_endpoint": "validator.example:8701",
        "result_endpoint": "/evaluation/v1/runs/evaldisc1/result",
        "key_release_nonce": "key-release-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": hashlib.sha256(_TOKEN.encode()).hexdigest(),
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    return eval_wire.validate_eval_plan(plan)


def _deployment_plan(
    *,
    app_identity: str | None,
    compose_name: str,
) -> eval_deploy.EvalDeploymentPlan:
    _compose, compose_hash = _compose_for_name(compose_name)
    raw = _raw_plan(app_identity=app_identity, compose_hash=compose_hash)
    return eval_deploy.build_eval_deployment_plan(
        {
            "schema_version": 1,
            "plan": raw,
            "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(raw)).hexdigest(),
            "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": _TOKEN},
        }
    )


def _secrets(plan: eval_deploy.EvalDeploymentPlan) -> dict[str, str]:
    artifact_env = eval_deploy.build_eval_artifact_env_values(
        plan,
        secret=_GRANT_SECRET,
        api_base_url=_API_BASE,
    )
    return {
        "EVAL_RUN_TOKEN": _TOKEN,
        "LLM_COST_LIMIT": "1.00",
        "CHALLENGE_PHALA_ATTESTATION_ENABLED": "1",
        "CHALLENGE_PHALA_EVAL_PLAN": json.dumps(plan.plan),
        "CHALLENGE_PHALA_AGENT_HASH": plan.plan["agent_hash"],
        "CHALLENGE_PHALA_CANONICAL_MEASUREMENT": json.dumps(
            {
                "mrtd": plan.measurement["mrtd"],
                "rtmr0": plan.measurement["rtmr0"],
                "rtmr1": plan.measurement["rtmr1"],
                "rtmr2": plan.measurement["rtmr2"],
                "compose_hash": plan.compose_hash,
                "os_image_hash": plan.measurement["os_image_hash"],
            }
        ),
        "CHALLENGE_PHALA_VALIDATOR_NONCE": plan.plan["score_nonce"],
        **artifact_env,
    }


def _deploy(
    plan: eval_deploy.EvalDeploymentPlan,
    *,
    provision_app_id: str,
    provision_pubkey: str,
    provision_compose_hash: str | None = None,
    provision_os: str | None = None,
    create_id: str = "cvm-eval-discovered-1",
) -> tuple[dict[str, str], eval_deploy.EvalPhalaDeployment]:
    encrypted = eval_deploy.encrypt_eval_secrets(plan, _secrets(plan))
    deployment = eval_deploy.EvalPhalaDeployment(
        provision_response={
            "app_id": provision_app_id,
            "compose_hash": (
                plan.compose_hash if provision_compose_hash is None else provision_compose_hash
            ),
            "app_env_encrypt_pubkey": provision_pubkey,
            "os_image_hash": (
                plan.measurement["os_image_hash"] if provision_os is None else provision_os
            ),
        },
        create_response={"id": create_id},
    )
    ack = deployment.deploy(plan, encrypted)
    return ack, deployment


def test_s1_provision_app_id_different_from_assignment_pin_proceeds() -> None:
    """S1: Phala returns a different 40-hex app_id than the assignment pin → deploy OK."""

    assert _DISCOVERED_APP_ID != _ASSIGNMENT_PIN_APP_ID
    plan = _deployment_plan(
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
    )
    assert plan.app_identity == _ASSIGNMENT_PIN_APP_ID

    ack, deployment = _deploy(
        plan,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )

    assert ack["app_identity"] == _DISCOVERED_APP_ID
    assert ack["cvm_id"] == "cvm-eval-discovered-1"
    assert len(deployment.provision_requests) == 1
    assert len(deployment.create_requests) == 1
    # Provision sends env names only (no ciphertext) and must not require pin match.
    prov = deployment.provision_requests[0]
    assert "env_keys" in prov
    assert "encrypted_env" not in prov
    assert isinstance(prov["env_keys"], list)
    assert all(isinstance(name, str) for name in prov["env_keys"])
    # Create uses discovered handle, not the assignment pin.
    created = deployment.create_requests[0]
    assert created["app_id"] == _DISCOVERED_APP_ID
    assert created["app_id"] != _ASSIGNMENT_PIN_APP_ID
    assert created["compose_hash"] == plan.compose_hash
    assert created["encrypted_env"]
    assert set(created["env_keys"]) <= set(eval_deploy.EVAL_ALLOWED_ENVS)


def test_s2_compose_hash_mismatch_still_hard_fails() -> None:
    """S2: provision compose_hash ≠ plan → fail closed (trust anchor)."""

    plan = _deployment_plan(
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
    )
    encrypted = eval_deploy.encrypt_eval_secrets(plan, _secrets(plan))
    deployment = eval_deploy.EvalPhalaDeployment(
        provision_response={
            "app_id": _DISCOVERED_APP_ID,
            "compose_hash": "cc" * 32,
            "app_env_encrypt_pubkey": _DISCOVERED_PUBKEY,
            "os_image_hash": plan.measurement["os_image_hash"],
        },
        create_response={"id": "cvm-should-not-create"},
    )
    with pytest.raises(eval_deploy.EvalDeploymentError, match="compose"):
        deployment.deploy(plan, encrypted)
    assert deployment.create_requests == []


def test_s3_os_measurement_mismatch_still_hard_fails() -> None:
    """S3: provision os_image_hash ≠ plan measurement → fail closed."""

    plan = _deployment_plan(
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
    )
    encrypted = eval_deploy.encrypt_eval_secrets(plan, _secrets(plan))
    deployment = eval_deploy.EvalPhalaDeployment(
        provision_response={
            "app_id": _DISCOVERED_APP_ID,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": _DISCOVERED_PUBKEY,
            "os_image_hash": "ff" * 32,
        },
        create_response={"id": "cvm-should-not-create"},
    )
    with pytest.raises(eval_deploy.EvalDeploymentError, match="os_image_hash|measurement"):
        deployment.deploy(plan, encrypted)
    assert deployment.create_requests == []


def test_s4_create_and_ack_use_discovered_app_id_and_kms_digest() -> None:
    """S4: create payload + ack bind to discovered app_id and encrypt-key digest."""

    plan = _deployment_plan(
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
    )
    expected_kms_sha = hashlib.sha256(bytes.fromhex(_DISCOVERED_PUBKEY)).hexdigest()
    ack, deployment = _deploy(
        plan,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )
    assert deployment.create_requests[0]["app_id"] == _DISCOVERED_APP_ID
    assert ack["app_identity"] == _DISCOVERED_APP_ID
    assert ack["kms_public_key_sha256"] == expected_kms_sha
    assert ack["kms_public_key_sha256"] != plan.kms_public_key_sha256


def test_s5_moniker_path_compose_hash_byte_identical() -> None:
    """S5: non-hex moniker app_identity still seeds compose name / compose_hash."""

    moniker = "eval-v1"
    compose, expected_hash = _compose_for_name(moniker)
    assert app_compose_hash(compose) == expected_hash

    plan = _deployment_plan(app_identity=moniker, compose_name=moniker)
    assert plan.compose_name == moniker
    assert plan.compose_hash == expected_hash
    assert plan.app_identity == moniker
    # Deploy still discovers a real Phala handle; moniker is not the create app_id.
    ack, deployment = _deploy(
        plan,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )
    assert plan.compose_hash == expected_hash  # unchanged through deploy
    assert deployment.create_requests[0]["app_id"] == _DISCOVERED_APP_ID
    assert ack["compose_hash"] == expected_hash


def test_s5b_default_compose_name_hex_pin_matches_live_generator_hash() -> None:
    """S5b: 40-hex pin keeps DEFAULT_EVAL_COMPOSE_NAME (compose_hash path stable)."""

    name = eval_deploy.DEFAULT_EVAL_COMPOSE_NAME
    _compose, expected_hash = _compose_for_name(name)
    plan = _deployment_plan(app_identity=_ASSIGNMENT_PIN_APP_ID, compose_name=name)
    assert plan.compose_name == name
    assert plan.compose_hash == expected_hash


def test_s6_artifact_url_and_token_still_transmitted() -> None:
    """S6: artifact URL/token injection remains on the encrypt/deploy env_keys path."""

    plan = _deployment_plan(
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
    )
    secrets = _secrets(plan)
    assert eval_deploy.EVAL_ARTIFACT_URL_ENV in secrets
    assert eval_deploy.EVAL_ARTIFACT_TOKEN_ENV in secrets
    assert secrets[eval_deploy.EVAL_ARTIFACT_URL_ENV].startswith("https://")
    encrypted = eval_deploy.encrypt_eval_secrets(plan, secrets)
    assert eval_deploy.EVAL_ARTIFACT_URL_ENV in encrypted.env_keys
    assert eval_deploy.EVAL_ARTIFACT_TOKEN_ENV in encrypted.env_keys

    _ack, deployment = _deploy(
        plan,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )
    prov_keys = deployment.provision_requests[0]["env_keys"]
    create_keys = deployment.create_requests[0]["env_keys"]
    assert eval_deploy.EVAL_ARTIFACT_URL_ENV in prov_keys
    assert eval_deploy.EVAL_ARTIFACT_TOKEN_ENV in prov_keys
    assert eval_deploy.EVAL_ARTIFACT_URL_ENV in create_keys
    assert eval_deploy.EVAL_ARTIFACT_TOKEN_ENV in create_keys
    # Never leak grant token into request dumps beyond the ciphertext blob.
    token = secrets[eval_deploy.EVAL_ARTIFACT_TOKEN_ENV]
    assert token not in json.dumps(deployment.provision_requests, default=str)
    assert token not in repr(deployment.create_requests[0].get("env_keys"))


def test_s7_eval_wire_hex_app_identity_optional_moniker_still_accepted() -> None:
    """S7: missing app_identity is valid; moniker still validates as compose seed."""

    _compose, compose_hash = _compose_for_name(eval_deploy.DEFAULT_EVAL_COMPOSE_NAME)
    # Absent pin — advisory Phala handle not required on the wire.
    absent = _raw_plan(app_identity=None, compose_hash=compose_hash)
    assert "app_identity" not in absent["eval_app"] or absent["eval_app"].get("app_identity") in (
        None,
        "",
    )

    moniker_plan = _raw_plan(app_identity="agent-challenge-eval-v1", compose_hash=compose_hash)
    assert moniker_plan["eval_app"]["app_identity"] == "agent-challenge-eval-v1"

    hex_plan = _raw_plan(app_identity=_ASSIGNMENT_PIN_APP_ID, compose_hash=compose_hash)
    assert hex_plan["eval_app"]["app_identity"] == _ASSIGNMENT_PIN_APP_ID


def test_s7b_build_plan_without_app_identity_uses_default_compose_name() -> None:
    """Absent wire app_identity → DEFAULT_EVAL_COMPOSE_NAME compose path."""

    name = eval_deploy.DEFAULT_EVAL_COMPOSE_NAME
    _compose, compose_hash = _compose_for_name(name)
    raw = _raw_plan(app_identity=None, compose_hash=compose_hash)
    plan = eval_deploy.build_eval_deployment_plan(
        {
            "schema_version": 1,
            "plan": raw,
            "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(raw)).hexdigest(),
            "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": _TOKEN},
        }
    )
    assert plan.compose_name == name
    assert plan.compose_hash == compose_hash


def test_discovery_provision_request_omits_nonce_and_app_id() -> None:
    """S-shape: hex/absent discovery path must not send nonce or app_id (live 422)."""

    plan = _deployment_plan(
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
    )
    _ack, deployment = _deploy(
        plan,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )
    prov = deployment.provision_requests[0]
    assert "nonce" not in prov, f"discovery must omit nonce; got keys={sorted(prov)}"
    assert "app_id" not in prov, f"discovery must omit app_id; got keys={sorted(prov)}"
    assert deployment.create_requests[0]["app_id"] == _DISCOVERED_APP_ID


def test_absent_app_identity_discovery_omits_nonce_and_app_id() -> None:
    """S-shape: missing wire app_identity is discovery — neither nonce nor app_id."""

    name = eval_deploy.DEFAULT_EVAL_COMPOSE_NAME
    _compose, compose_hash = _compose_for_name(name)
    raw = _raw_plan(app_identity=None, compose_hash=compose_hash)
    plan = eval_deploy.build_eval_deployment_plan(
        {
            "schema_version": 1,
            "plan": raw,
            "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(raw)).hexdigest(),
            "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": _TOKEN},
        }
    )
    _ack, deployment = _deploy(
        plan,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )
    prov = deployment.provision_requests[0]
    assert "nonce" not in prov
    assert "app_id" not in prov
    assert deployment.create_requests[0]["app_id"] == _DISCOVERED_APP_ID


def test_moniker_provision_legal_shape() -> None:
    """S-legacy: moniker path is legal (app_id alone, or neither) — never nonce alone."""

    moniker = "eval-v1"
    plan = _deployment_plan(app_identity=moniker, compose_name=moniker)
    _ack, deployment = _deploy(
        plan,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )
    prov = deployment.provision_requests[0]
    has_nonce = "nonce" in prov
    has_app_id = "app_id" in prov
    assert not (has_nonce and not has_app_id)
    if has_app_id:
        assert prov["app_id"] == moniker
    assert "nonce" not in prov
    assert deployment.create_requests[0]["app_id"] == _DISCOVERED_APP_ID


def test_public_api_cannot_emit_nonce_without_app_id() -> None:
    """S-guard: hand-built plan with nonce must not emit illegal provision shape."""

    plan = _deployment_plan(
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
    )
    poisoned = eval_deploy.EvalDeploymentPlan(
        plan=plan.plan,
        plan_sha256=plan.plan_sha256,
        compose=plan.compose,
        compose_text=plan.compose_text,
        compose_hash=plan.compose_hash,
        app_identity=plan.app_identity,
        image_ref=plan.image_ref,
        kms_public_key_hex=plan.kms_public_key_hex,
        kms_public_key_sha256=plan.kms_public_key_sha256,
        measurement=plan.measurement,
        eval_run_id=plan.eval_run_id,
        eval_run_token=plan.eval_run_token,
        instance_type=plan.instance_type,
        os_image=plan.os_image,
        compose_name=plan.compose_name,
        phala_app_nonce=1,
    )
    _ack, deployment = _deploy(
        poisoned,
        provision_app_id=_DISCOVERED_APP_ID,
        provision_pubkey=_DISCOVERED_PUBKEY,
    )
    prov = deployment.provision_requests[0]
    assert not ("nonce" in prov and "app_id" not in prov), (
        f"illegal Phala shape nonce-without-app_id: keys={sorted(prov)}"
    )
