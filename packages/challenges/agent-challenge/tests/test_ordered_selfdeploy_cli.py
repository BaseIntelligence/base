"""Discriminator tests for the ordered review/eval self-deploy CLI."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from agent_challenge.canonical import eval_wire
from agent_challenge.selfdeploy import eval as eval_deploy
from agent_challenge.selfdeploy import lifecycle

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


def _review_assignment() -> tuple[dict[str, object], str]:
    from agent_challenge.review.compose import (
        generate_review_app_compose,
        review_app_compose_hash,
    )
    from agent_challenge.review.schemas import ReviewInputConfig, build_review_assignment

    compose = generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity="review-v1",
    )
    config = ReviewInputConfig(
        image_ref=REVIEW_IMAGE,
        compose_hash=review_app_compose_hash(compose),
        app_identity="review-v1",
        kms_public_key_hex=PUBLIC_KEY,
        measurement=MEASUREMENT,
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


def _eval_plan() -> dict[str, object]:
    from agent_challenge.canonical.compose import (
        generate_app_compose,
        render_app_compose,
    )

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
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
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
    return eval_wire.validate_eval_plan(plan)


def test_eval_prepare_requires_validator_review_authorization_before_deployment():
    plan = _eval_plan()
    plan["authorizing_review_digest"] = ""
    with pytest.raises(eval_deploy.EvalDeploymentError, match="Eval plan"):
        eval_deploy.build_eval_deployment_plan(
            {
                "schema_version": 1,
                "plan": plan,
                "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(plan)).hexdigest(),
                "secret_delivery": None,
            }
        )


def test_eval_prepare_rejects_plan_digest_or_app_mutation_before_phala_create():
    plan = _eval_plan()
    token = "run-token"
    plan["run_token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
    wrapper = {
        "schema_version": 1,
        "plan": plan,
        "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(plan)).hexdigest(),
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token},
    }
    deployment = eval_deploy.build_eval_deployment_plan(wrapper)
    assert deployment.image_ref == EVAL_IMAGE
    tampered = copy.deepcopy(wrapper)
    tampered["plan"]["eval_app"]["image_ref"] = REVIEW_IMAGE
    with pytest.raises(eval_deploy.EvalDeploymentError):
        eval_deploy.build_eval_deployment_plan(tampered)


def test_eval_encrypted_env_contains_only_scoped_capabilities_and_is_transmitted():
    raw_plan = _eval_plan()
    token = "run-token"
    raw_plan["run_token_sha256"] = hashlib.sha256(token.encode()).hexdigest()
    plan = eval_deploy.build_eval_deployment_plan(
        {
            "schema_version": 1,
            "plan": raw_plan,
            "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(raw_plan)).hexdigest(),
            "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token},
        },
    )
    encrypted = eval_deploy.encrypt_eval_secrets(
        plan,
        {
            "EVAL_RUN_TOKEN": "run-token",
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
        },
    )
    assert encrypted.ciphertext
    assert "run-token" not in repr(encrypted)
    assert set(encrypted.env_keys) == {
        "EVAL_RUN_TOKEN",
        "CHALLENGE_PHALA_ATTESTATION_ENABLED",
        "CHALLENGE_PHALA_AGENT_HASH",
        "CHALLENGE_PHALA_CANONICAL_MEASUREMENT",
        "CHALLENGE_PHALA_EVAL_PLAN",
        "CHALLENGE_PHALA_VALIDATOR_NONCE",
        "LLM_COST_LIMIT",
    }
    # VAL-ACAT-013: Base gateway secrets are neither required nor allowed.
    assert "BASE_GATEWAY_TOKEN" not in encrypted.env_keys
    assert "BASE_LLM_GATEWAY_URL" not in encrypted.env_keys
    # Production RA-TLS host/port/mTLS path names are provisioned in measured
    # compose static env, not as encrypted capability bytes.
    compose_text = plan.compose_text
    assert "KEY_RELEASE_RA_TLS_HOST=validator.example" in compose_text
    assert "KEY_RELEASE_RA_TLS_PORT=8701" in compose_text


def test_lifecycle_budget_counts_review_and_eval_together():
    estimate = lifecycle.projected_lifecycle_cost_usd(
        review_instance_type="tdx.small",
        eval_instance_type="tdx.small",
        review_runtime_hours=100,
        eval_runtime_hours=100,
    )
    assert estimate == pytest.approx(11.6)
    with pytest.raises(lifecycle.LifecycleBudgetError):
        lifecycle.validate_lifecycle_budget(
            review_instance_type="tdx.small",
            eval_instance_type="tdx.xlarge",
            review_runtime_hours=100,
            eval_runtime_hours=100,
            money_cap_usd=20,
        )
