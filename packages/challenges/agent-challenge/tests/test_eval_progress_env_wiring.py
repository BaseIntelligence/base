"""RED contract: selfdeploy env bundle carries progress + runner hotkey config.

ProgressReporter env must be deploy-injectable (mirror REVIEW_API_BASE_URL).
RUNNER_HOTKEY_MNEMONIC is the local-only signing secret name — its VALUE must
never appear in logs/redacted CLI output. The server receives hotkey_ss58 +
signature only.
"""

from __future__ import annotations

import hashlib
import json
import logging

from agent_challenge.canonical import eval_wire
from agent_challenge.canonical.compose import DEFAULT_ALLOWED_ENVS

PROGRESS_ENVS = (
    "EVAL_PROGRESS_BASE_URL",
    "EVAL_RUN_ID",
    "EVAL_SUBMISSION_ID",
    "EVAL_RUN_TOKEN",
)
# Local runner signing secret name (value never leaves the CVM / never hits master).
RUNNER_HOTKEY_ENV = "RUNNER_HOTKEY_MNEMONIC"
# Well-known 12-word test fixture — never a real secret; used only to prove redaction.
TEST_MNEMONIC = (
    "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
)


def test_default_allowed_envs_include_progress_and_runner_hotkey_names():
    allowed = set(DEFAULT_ALLOWED_ENVS)
    for name in PROGRESS_ENVS:
        assert name in allowed, f"{name} missing from DEFAULT_ALLOWED_ENVS"
    assert RUNNER_HOTKEY_ENV in allowed, f"{RUNNER_HOTKEY_ENV} missing from DEFAULT_ALLOWED_ENVS"


def test_build_eval_progress_env_helper_binds_ids_and_token():
    """CLI/deploy helper must bind base URL + ids from plan + token."""
    from agent_challenge.selfdeploy.eval import build_eval_progress_env

    values = build_eval_progress_env(
        base_url="https://chain.joinbase.ai/challenges/agent-challenge/",
        eval_run_id="eval-1",
        submission_id="7",
        eval_run_token="tok",
    )
    assert values == {
        "EVAL_PROGRESS_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
        "EVAL_RUN_ID": "eval-1",
        "EVAL_SUBMISSION_ID": "7",
        "EVAL_RUN_TOKEN": "tok",
    }
    # Helper must never require or embed a mnemonic — server never sees it.
    assert RUNNER_HOTKEY_ENV not in values
    assert "mnemonic" not in json.dumps(values).lower()


def test_encrypt_eval_secrets_accepts_progress_env_bundle():
    from agent_challenge.canonical.compose import generate_app_compose, render_app_compose
    from agent_challenge.selfdeploy import eval as eval_deploy

    try:
        from agent_challenge.evaluation.own_runner.progress_reporter import ProgressReporter
    except ImportError as exc:
        raise AssertionError(
            "progress_reporter.ProgressReporter missing — required for eval progress env wiring"
        ) from exc

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    img = "registry.example/eval@sha256:" + "b" * 64
    compose = generate_app_compose(
        orchestrator_image=img,
        name=eval_deploy.DEFAULT_EVAL_COMPOSE_NAME,
        key_release_url=eval_deploy.MEASURE_TIME_EVAL_KEY_RELEASE_PLACEHOLDER,
        allowed_envs=tuple(sorted(eval_deploy.EVAL_ALLOWED_ENVS)),
    )
    compose_hash = hashlib.sha256(render_app_compose(compose).encode()).hexdigest()
    token = "run-token-progress"
    public_key = "c" * 64
    plan = {
        "schema_version": 1,
        "eval_run_id": "eval-progress-1",
        "submission_id": "7",
        "submission_version": 1,
        "authorizing_review_digest": "d" * 64,
        "agent_hash": "e" * 64,
        "selected_tasks": [
            {
                "task_id": "terminal-bench/t",
                "image_ref": "task-local/t@sha256:" + "3" * 64,
                "task_config_sha256": "3" * 64,
            }
        ],
        "k": 1,
        "package_tree_sha": "a" * 64,
        "scoring_policy": policy,
        "scoring_policy_digest": eval_wire.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": img,
            "compose_hash": compose_hash,
            "app_identity": "bb35a8f627f0f8c991aa85c15742d352e658e0f7",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": public_key,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex(public_key)).hexdigest(),
            "measurement": {
                "mrtd": "01" * 48,
                "rtmr0": "02" * 48,
                "rtmr1": "03" * 48,
                "rtmr2": "04" * 48,
                "os_image_hash": "05" * 32,
                "key_provider": "phala",
                "vm_shape": "tdx.small",
            },
        },
        "key_release_endpoint": "86.38.238.235:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-progress-1/result",
        "key_release_nonce": "kr-n",
        "score_nonce": "sc-n",
        "run_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    plan = eval_wire.validate_eval_plan(plan)
    dep = eval_deploy.build_eval_deployment_plan(
        {
            "schema_version": 1,
            "plan": plan,
            "plan_sha256": hashlib.sha256(eval_wire.canonical_json_v1(plan)).hexdigest(),
            "secret_delivery": {"env_key": "EVAL_RUN_TOKEN", "token": token},
        }
    )
    secrets = {
        "EVAL_RUN_TOKEN": token,
        "LLM_COST_LIMIT": "1.00",
        "CHALLENGE_PHALA_ATTESTATION_ENABLED": "1",
        "CHALLENGE_PHALA_EVAL_PLAN": json.dumps(dep.plan, sort_keys=True, separators=(",", ":")),
        "CHALLENGE_PHALA_AGENT_HASH": dep.plan["agent_hash"],
        "CHALLENGE_PHALA_CANONICAL_MEASUREMENT": json.dumps(
            {
                "mrtd": dep.measurement["mrtd"],
                "rtmr0": dep.measurement["rtmr0"],
                "rtmr1": dep.measurement["rtmr1"],
                "rtmr2": dep.measurement["rtmr2"],
                "compose_hash": dep.compose_hash,
                "os_image_hash": dep.measurement["os_image_hash"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "CHALLENGE_PHALA_VALIDATOR_NONCE": dep.plan["key_release_nonce"],
        "EVAL_PROGRESS_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
        "EVAL_RUN_ID": dep.eval_run_id,
        "EVAL_SUBMISSION_ID": str(dep.plan["submission_id"]),
        # Local-only signing material name may be present in encrypted_env allow-list,
        # but encrypt path must accept the name without shipping the value to master APIs.
        RUNNER_HOTKEY_ENV: TEST_MNEMONIC,
    }
    encrypted = eval_deploy.encrypt_eval_secrets(dep, secrets)
    for name in PROGRESS_ENVS:
        assert name in encrypted.env_keys
    assert RUNNER_HOTKEY_ENV in encrypted.env_keys
    reporter = ProgressReporter.from_env(
        {
            "EVAL_PROGRESS_BASE_URL": secrets["EVAL_PROGRESS_BASE_URL"],
            "EVAL_RUN_ID": secrets["EVAL_RUN_ID"],
            "EVAL_SUBMISSION_ID": secrets["EVAL_SUBMISSION_ID"],
            "EVAL_RUN_TOKEN": token,
        }
    )
    assert reporter is not None
    assert reporter.eval_run_id == "eval-progress-1"
    assert "progress" in reporter.url
    # ProgressReporter must not require or hold the mnemonic.
    reporter_dump = json.dumps(reporter.__dict__, default=str)
    assert TEST_MNEMONIC not in reporter_dump
    assert "mnemonic" not in reporter_dump.lower()


def test_runner_hotkey_mnemonic_value_never_in_redacted_cli_output():
    """CLI redaction must strip RUNNER_HOTKEY_MNEMONIC values from operator output."""
    from agent_challenge.selfdeploy import cli as cli_mod

    assert hasattr(cli_mod, "_redact_capabilities")
    assert hasattr(cli_mod, "_REDACTED_CAPABILITY_KEYS")
    assert RUNNER_HOTKEY_ENV in cli_mod._REDACTED_CAPABILITY_KEYS

    payload = {
        "EVAL_RUN_ID": "eval-1",
        "EVAL_RUN_TOKEN": "tok-secret",
        RUNNER_HOTKEY_ENV: TEST_MNEMONIC,
        "nested": {RUNNER_HOTKEY_ENV: TEST_MNEMONIC, "ok": "visible"},
    }
    redacted = cli_mod._redact_capabilities(payload)
    dumped = json.dumps(redacted)
    assert TEST_MNEMONIC not in dumped
    assert RUNNER_HOTKEY_ENV not in redacted
    assert "tok-secret" not in dumped
    assert redacted["EVAL_RUN_ID"] == "eval-1"
    assert redacted["nested"]["ok"] == "visible"


def test_runner_hotkey_mnemonic_value_never_logged(caplog):
    """Logging helpers must not emit the mnemonic value."""
    from agent_challenge.selfdeploy import cli as cli_mod

    payload = {
        "stage": "eval-deploy",
        RUNNER_HOTKEY_ENV: TEST_MNEMONIC,
        "EVAL_RUN_TOKEN": "tok-secret",
    }
    with caplog.at_level(logging.DEBUG):
        redacted = cli_mod._redact_capabilities(payload)
        # Simulate what CLI would print.
        logging.getLogger("agent_challenge.selfdeploy.cli").info(
            "deploy payload %s", json.dumps(redacted)
        )
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert TEST_MNEMONIC not in joined
    assert "tok-secret" not in joined
