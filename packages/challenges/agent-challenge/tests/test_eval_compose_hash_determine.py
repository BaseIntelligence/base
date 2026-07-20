"""Hash-determine realignment: miner dry/generate must match operator eval compose pin.

Live residual (e2e-live-v7/eval-1task-after-kr): prepare delivered signed plan with
``compose_hash=04011776…`` while miner ``generate_app_compose`` candidates baked
the live ``key_release_endpoint`` into static env and therefore never matched.
The operator pin was measured with a non-routable HTTPS placeholder so the
compose_hash stays unstable only to image+envelope factors, not the live KR IP.

Guest still resolves the real endpoint from the signed plan /
``CHALLENGE_PHALA_EVAL_PLAN`` (see own_runner_backend._resolve_key_release_endpoint)
and never invents MRTD/KR materials.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import (
    app_compose_hash,
    generate_app_compose,
    render_app_compose,
)
from agent_challenge.selfdeploy import eval as eval_deploy

#: Live dual-flag joinbase eval pin (tee-pin-pack + residual after KR).
LIVE_PIN_COMPOSE_HASH = "0401177601f46160c8127c007019401c1a7e6fb3cf8a0850c54a0b96fbbe67d2"
LIVE_PIN_IMAGE = (
    "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:"
    "753e2296635bcd3a30703dc706509f0f8c0e7dd2f82bef730ad7f1cc9443933c"
)
#: Residual Path A plan endpoint (raw RA-TLS authority).
LIVE_PLAN_KEY_RELEASE = "86.38.238.235:8701"
#: Mission-bundled pin pack composition (destroyable measure materials).
MISSION_PIN_COMPOSE = Path(
    "/root/.factory/missions/a43a16a7-2230-4853-ba8a-a6bfe993a90f"
    "/evidence/ac-attestation/tee-pin-pack/eval-app-compose.json"
)


def _signed_prepare(
    *,
    compose_hash: str,
    image_ref: str,
    app_identity: str,
    key_release_endpoint: str,
    token: str = "eval-run-token-hash-determine",
) -> dict:
    public_key = "22" * 32
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    plan = {
        "schema_version": 1,
        "eval_run_id": "eval_hash_determine",
        "submission_id": "30",
        "submission_version": 1,
        "authorizing_review_digest": "ab" * 32,
        "agent_hash": "cd" * 32,
        "selected_tasks": [
            {
                "task_id": "terminal-bench/financial-document-processor",
                "image_ref": "task-local/financial-document-processor@sha256:" + ("3" * 64),
                "task_config_sha256": "3" * 64,
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": eval_wire.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": image_ref,
            "compose_hash": compose_hash,
            "app_identity": app_identity,
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": public_key,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex(public_key)).hexdigest(),
            "measurement": {
                "mrtd": "a1" * 48,
                "rtmr0": "a2" * 48,
                "rtmr1": "a3" * 48,
                "rtmr2": "a4" * 48,
                "os_image_hash": "a5" * 32,
                "key_provider": "phala",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": key_release_endpoint,
        "result_endpoint": "/evaluation/v1/runs/eval_hash_determine/result",
        "key_release_nonce": "key-release-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    plan = eval_wire.validate_eval_plan(plan)
    return {
        "schema_version": 1,
        "plan": plan,
        "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(plan)).hexdigest(),
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token},
    }


def test_measure_time_placeholder_reproduces_live_pin_hash():
    """Product generator with pin-pack measure inputs must yield 040117…"""

    compose = generate_app_compose(
        orchestrator_image=LIVE_PIN_IMAGE,
        name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
        key_release_url=eval_deploy.MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER,
        allowed_envs=eval_deploy.EVAL_ALLOWED_ENVS,
    )
    assert app_compose_hash(compose) == LIVE_PIN_COMPOSE_HASH
    # Optional mission-only pin pack. Path may be absent or unreadable on CI
    # /sandbox runners (PermissionError on parent dirs); product hash assert above
    # already seals the live pin identity.
    try:
        present = MISSION_PIN_COMPOSE.is_file()
    except OSError:
        present = False
    if not present:
        return
    import json

    try:
        pin_doc = json.loads(MISSION_PIN_COMPOSE.read_text(encoding="utf-8"))
    except OSError:
        return
    assert render_app_compose(compose) == render_app_compose(pin_doc)
    assert app_compose_hash(pin_doc) == LIVE_PIN_COMPOSE_HASH


def test_build_eval_deployment_plan_matches_live_pin_with_raw_plan_endpoint():
    """Residual Path A: plan KR is raw host:port; pin used measure-time placeholder.

    Before realign this raised EvalDeploymentError (compose hash mismatches signed plan).
    """

    prepare = _signed_prepare(
        compose_hash=LIVE_PIN_COMPOSE_HASH,
        image_ref=LIVE_PIN_IMAGE,
        app_identity="bb35a8f627f0f8c991aa85c15742d352e658e0f7",
        key_release_endpoint=LIVE_PLAN_KEY_RELEASE,
    )
    dep = eval_deploy.build_eval_deployment_plan(prepare)
    assert dep.compose_hash == LIVE_PIN_COMPOSE_HASH
    assert dep.compose_name == eval_deploy.DEFAULT_EVAL_COMPOSE_NAME
    # Measured bytes keep the pin-stable placeholder, not the live IP (guest uses plan).
    assert (
        eval_deploy.MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER in dep.compose["docker_compose_file"]
    )
    assert LIVE_PLAN_KEY_RELEASE not in dep.compose["docker_compose_file"]
    # Plan still carries the real KR endpoint for runtime resolution.
    assert dep.plan["key_release_endpoint"] == LIVE_PLAN_KEY_RELEASE


def test_build_eval_deployment_plan_still_matches_when_plan_endpoint_is_placeholder():
    """When operator/plan already used the pin measure endpoint, still determine."""

    prepare = _signed_prepare(
        compose_hash=LIVE_PIN_COMPOSE_HASH,
        image_ref=LIVE_PIN_IMAGE,
        app_identity=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
        key_release_endpoint=eval_deploy.MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER,
    )
    dep = eval_deploy.build_eval_deployment_plan(prepare)
    assert dep.compose_hash == LIVE_PIN_COMPOSE_HASH


def test_build_eval_deployment_plan_matches_raw_baked_compose_when_pin_is_raw():
    """Future pin that bakes raw RA-TLS host/port must still determine without invent."""

    compose = generate_app_compose(
        orchestrator_image=LIVE_PIN_IMAGE,
        name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
        key_release_url=LIVE_PLAN_KEY_RELEASE,
        allowed_envs=eval_deploy.EVAL_ALLOWED_ENVS,
    )
    raw_hash = app_compose_hash(compose)
    assert raw_hash != LIVE_PIN_COMPOSE_HASH  # raw bake is a different pin
    prepare = _signed_prepare(
        compose_hash=raw_hash,
        image_ref=LIVE_PIN_IMAGE,
        app_identity=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
        key_release_endpoint=LIVE_PLAN_KEY_RELEASE,
    )
    dep = eval_deploy.build_eval_deployment_plan(prepare)
    assert dep.compose_hash == raw_hash
    assert (
        f"KEY_RELEASE_RA_TLS_HOST={LIVE_PLAN_KEY_RELEASE.split(':', 1)[0]}"
        in (dep.compose["docker_compose_file"])
    )


def test_build_eval_deployment_plan_fails_closed_on_unknown_compose_hash():
    prepare = _signed_prepare(
        compose_hash="0" * 64,
        image_ref=LIVE_PIN_IMAGE,
        app_identity=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
        key_release_endpoint=LIVE_PLAN_KEY_RELEASE,
    )
    with pytest.raises(eval_deploy.EvalDeploymentError, match="compose hash mismatches"):
        eval_deploy.build_eval_deployment_plan(prepare)
