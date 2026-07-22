"""Offline Phala provision envelope parity for the eval application compose.

Live self-deploy failed closed when the local ``generate_app_compose`` hash
(``e5814373...`` without the Phala rewrite factors) did not equal
``POST /cvms/provision``'s rewritten ``compose_hash`` (``27bf5...`` with
``pre_launch_script`` / ``features`` / tproxy / public_tcbinfo / storage_fs).

These tests prove the offline generator already emits the full Phala envelope so
local app-compose bytes === provision rewrite for the live 3-task terminal_bench
smoke inputs, ``build_eval_deployment_plan`` accepts identity, and review/eval
compose hashes remain disjoint.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

from agent_challenge.canonical import compose as eval_compose
from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.measurement import compose_hash, normalize_app_compose
from agent_challenge.review import compose as review_compose
from agent_challenge.selfdeploy import eval as eval_deploy

#: Digest-pinned canonical image used by the live 3-task terminal_bench smoke.
LIVE_SMOKE_EVAL_IMAGE = (
    "docker.io/mathiiss/agent-challenge-canonical@sha256:"
    "02331f0909f617e333f113be376d353770a673669946bcddaac3c53cbde7c9d8"
)
LIVE_SMOKE_APP_IDENTITY = "agent-challenge-eval-v1"
LIVE_SMOKE_KEY_RELEASE = "ratls://84.32.70.61:8701"

#: Synthetic review image pin (disjoint service inventory only).
REVIEW_IMAGE = "docker.io/mathiiss/agent-challenge-review@sha256:" + ("c" * 64)

#: Live residual local hash (pre-envelope, no guest golden/task bind mounts)
#: for the smoke inputs above.
# Residual pre-envelope hash of the live-smoke compose inputs when the allowed_envs
# list includes the validator server-CA injection names (RA_TLS_SERVER_CA_*). Updated
# when FAIL-CLOSED server-CA wiring lands so the discriminator still proves the
# envelope factors (not allowed_envs) are what Phala provision rewrites.
# Residual pre-envelope hash after RA-TLS server-CA + OPENROUTER_API_KEY allowed_envs lands.
LIVE_RESIDUAL_NO_ENVELOPE_HASH = "5e33c9be56dc518070045596f1c6d7b31c2d73f7508056f7db726f3ccd6179a3"


def _live_smoke_compose() -> dict:
    return eval_compose.generate_app_compose(
        orchestrator_image=LIVE_SMOKE_EVAL_IMAGE,
        name=LIVE_SMOKE_APP_IDENTITY,
        key_release_url=LIVE_SMOKE_KEY_RELEASE,
        allowed_envs=eval_deploy.EVAL_ALLOWED_ENVS,
    )


def _phala_provision_rewrite(local: dict) -> dict:
    """Re-apply the deterministic Phala Cloud envelope factors offline.

    Mirrors what ``POST /cvms/provision`` injects into AppCompose when the
    client omits them. When the local generator already emits the same
    factors, the rewritten document is byte-identical.
    """

    rewritten = copy.deepcopy(local)
    rewritten["tproxy_enabled"] = True
    rewritten["public_tcbinfo"] = True
    rewritten["secure_time"] = False
    rewritten["storage_fs"] = "zfs"
    rewritten["features"] = list(eval_compose.PHALA_DEFAULT_FEATURES)
    rewritten["pre_launch_script"] = eval_compose.phala_pre_launch_script()
    return rewritten


def test_generate_app_compose_emits_phala_envelope_keys():
    compose = _live_smoke_compose()
    assert set(compose) == eval_compose.PHALA_APP_COMPOSE_ENVELOPE_KEYS
    assert compose["features"] == list(eval_compose.PHALA_DEFAULT_FEATURES)
    assert compose["tproxy_enabled"] is True
    assert compose["public_tcbinfo"] is True
    assert compose["storage_fs"] == "zfs"
    assert compose["secure_time"] is False
    assert compose["pre_launch_script"] == eval_compose.phala_pre_launch_script()
    assert compose["pre_launch_script"].startswith("#!/bin/bash")
    assert "Phala Cloud Pre-Launch Script" in compose["pre_launch_script"]


def test_live_smoke_local_render_equals_phala_provision_rewrite_fixture():
    """Offline discriminator: local == provision-style rewrite for live smoke inputs.

    Would FAIL against a generator that still omits pre_launch/features (the
    ``e5814373...`` residual) because the rewritten hash would differ.
    """

    local = _live_smoke_compose()
    rewritten = _phala_provision_rewrite(local)
    local_text = eval_compose.render_app_compose(local)
    rewrite_text = normalize_app_compose(rewritten)
    assert local_text == rewrite_text
    assert eval_compose.app_compose_hash(local) == compose_hash(rewritten)
    stripped = copy.deepcopy(local)
    del stripped["pre_launch_script"]
    assert compose_hash(stripped) != eval_compose.app_compose_hash(local)
    stripped2 = copy.deepcopy(local)
    del stripped2["features"]
    assert compose_hash(stripped2) != eval_compose.app_compose_hash(local)


def test_build_eval_deployment_plan_accepts_parity_compose_identity():
    """Signed Eval prepare identity matches the local Phala-envelope generator."""

    compose = _live_smoke_compose()
    compose_text = eval_compose.render_app_compose(compose)
    compose_hash_hex = hashlib.sha256(compose_text.encode("utf-8")).hexdigest()
    public_key = "11" * 32
    token = "eval-run-token-parity-sentinel"
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    plan = {
        "schema_version": 1,
        "eval_run_id": "eval_parity_smoke",
        "submission_id": "1",
        "submission_version": 1,
        "authorizing_review_digest": "ab" * 32,
        "agent_hash": "cd" * 32,
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "selected_tasks": [
            {
                "task_id": "adaptive-rejection-sampler",
                "image_ref": "registry.example/task@sha256:" + ("1" * 64),
                "task_config_sha256": "2" * 64,
            },
            {
                "task_id": "bn-fit-modify",
                "image_ref": "registry.example/task@sha256:" + ("3" * 64),
                "task_config_sha256": "4" * 64,
            },
            {
                "task_id": "break-filter-js-from-html",
                "image_ref": "registry.example/task@sha256:" + ("5" * 64),
                "task_config_sha256": "6" * 64,
            },
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": eval_wire.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": LIVE_SMOKE_EVAL_IMAGE,
            "compose_hash": compose_hash_hex,
            "app_identity": LIVE_SMOKE_APP_IDENTITY,
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
        "key_release_endpoint": LIVE_SMOKE_KEY_RELEASE,
        "result_endpoint": "/evaluation/v1/runs/eval_parity_smoke/result",
        "key_release_nonce": "key-release-nonce",
        "score_nonce": "score-nonce",
        "run_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    plan = eval_wire.validate_eval_plan(plan)
    prepare_response = {
        "schema_version": 1,
        "plan": plan,
        "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(plan)).hexdigest(),
        "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token},
    }

    dep = eval_deploy.build_eval_deployment_plan(prepare_response)
    assert dep.compose_hash == compose_hash_hex
    assert dep.compose_text == compose_text
    assert set(dep.compose) == eval_compose.PHALA_APP_COMPOSE_ENVELOPE_KEYS
    assert dep.compose_hash == compose_hash(_phala_provision_rewrite(dep.compose))


def test_eval_and_review_compose_identities_remain_disjoint():
    local = _live_smoke_compose()
    eval_hash = eval_compose.app_compose_hash(local)
    review = review_compose.generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity=review_compose.DEFAULT_REVIEW_APP_IDENTITY,
    )
    review_hash = review_compose.review_app_compose_hash(review)
    assert eval_hash != review_hash
    assert local["name"] != review["name"]
    eval_services = set(yaml.safe_load(local["docker_compose_file"])["services"])
    review_services = set(yaml.safe_load(review["docker_compose_file"])["services"])
    assert eval_services == {eval_compose.ORCHESTRATOR_SERVICE}
    assert review_services == {review_compose.REVIEWER_SERVICE}
    assert eval_services.isdisjoint(review_services)
    assert eval_compose.phala_pre_launch_script() == review["pre_launch_script"]


def test_legacy_hash_without_envelope_is_rejected_as_identity_mismatch():
    """Positive discriminator: the pre-envelope residual hash is not accepted."""

    compose = _live_smoke_compose()
    parity = eval_compose.app_compose_hash(compose)
    without = {
        k: v
        for k, v in compose.items()
        if k
        not in {
            "pre_launch_script",
            "features",
            "tproxy_enabled",
            "public_tcbinfo",
            "secure_time",
            "storage_fs",
        }
    }
    residual = compose_hash(without)
    assert residual != parity
    assert residual != compose_hash(_phala_provision_rewrite(compose))
    assert residual == LIVE_RESIDUAL_NO_ENVELOPE_HASH


def test_phala_pre_launch_script_path_is_vendor_checked_in():
    path = eval_compose.PHALA_PRE_LAUNCH_SCRIPT_PATH
    assert path == Path(eval_compose.REPO_ROOT) / "docker" / "review" / "phala_pre_launch.sh"
    assert path.is_file()
    text = eval_compose.phala_pre_launch_script()
    assert "Phala Cloud Pre-Launch Script" in text
    assert "Script execution" in text
