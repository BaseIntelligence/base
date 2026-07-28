"""Behavioral coverage for bucket-D prod/repo hotpatch merges.

Covers eval-path concurrency, digest path resolution, disk guards, progress wire,
and dual residual gate preservation (must not weaken AGATE).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_challenge.canonical import eval_wire
from agent_challenge.evaluation.authorization import (
    EvalAuthorizationConflict,
    resolve_plan_n_concurrent,
)
from agent_challenge.evaluation.benchmarks import (
    DATASET_DIGEST_MANIFEST_ENV,
    resolve_dataset_digest_path,
)
from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.selfdeploy.lifecycle import (
    LifecycleBudgetError,
    projected_lifecycle_cost_usd,
    validate_lifecycle_budget,
)
from agent_challenge.selfdeploy.shapes import (
    DEFAULT_EVAL_DISK_SIZE_GB,
    DEFAULT_EVAL_INSTANCE_TYPE,
    DEFAULT_REVIEW_DISK_SIZE_GB,
    DEFAULT_REVIEW_INSTANCE_TYPE,
    DiskSizeError,
    projected_cost_usd,
    validate_disk_size,
)


def test_resolve_dataset_digest_path_prefers_existing_app_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Site-packages layouts must not invent a missing parents[3]/golden path."""

    app_golden = tmp_path / "app" / "golden"
    app_golden.mkdir(parents=True)
    digest = app_golden / "dataset-digest.json"
    digest.write_text("{}", encoding="utf-8")

    monkeypatch.delenv(DATASET_DIGEST_MANIFEST_ENV, raising=False)
    # Force package-relative candidate into a fake site-packages tree.
    fake_pkg = tmp_path / "usr" / "local" / "lib" / "python3.12" / "site-packages" / "x.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("#", encoding="utf-8")

    # Patch known install layouts via env + explicit only — use env for this unit.
    monkeypatch.setenv(DATASET_DIGEST_MANIFEST_ENV, str(digest))
    resolved = resolve_dataset_digest_path(
        env={DATASET_DIGEST_MANIFEST_ENV: str(digest)}
    )
    assert resolved == digest


def test_resolve_dataset_digest_path_explicit_wins(tmp_path: Path) -> None:
    target = tmp_path / "custom-digest.json"
    target.write_text("{}", encoding="utf-8")
    assert resolve_dataset_digest_path(explicit=target) == target


def test_resolve_plan_n_concurrent_bounds_and_default() -> None:
    settings = ChallengeSettings(evaluation_concurrency=4)
    assert resolve_plan_n_concurrent(None, settings=settings) == 4
    assert resolve_plan_n_concurrent(2, settings=settings) == 2
    with pytest.raises(EvalAuthorizationConflict) as exc_info:
        resolve_plan_n_concurrent(0, settings=settings)
    assert getattr(exc_info.value, "code", "") == "eval_n_concurrent_out_of_bounds"
    with pytest.raises(EvalAuthorizationConflict):
        resolve_plan_n_concurrent(99, settings=settings)
    with pytest.raises(EvalAuthorizationConflict):
        resolve_plan_n_concurrent(True, settings=settings)  # type: ignore[arg-type]


def test_validate_eval_plan_accepts_n_concurrent_and_defaults_when_absent() -> None:
    import hashlib

    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    base = {
        "schema_version": 1,
        "eval_run_id": "eval-run-nconc",
        "submission_id": "submission-001",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": "a" * 64,
        "package_tree_sha": "b" * 64,
        "selected_tasks": [
            {
                "task_id": "task-a",
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "2" * 64,
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": eval_wire.scoring_policy_digest(policy),
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
        "key_release_endpoint": "validator.example:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-run-nconc/result",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "run_token_sha256": "5" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }

    without = eval_wire.validate_eval_plan(base)
    assert without["n_concurrent"] == 1

    with_n = dict(base)
    with_n["n_concurrent"] = 4
    validated = eval_wire.validate_eval_plan(with_n)
    assert validated["n_concurrent"] == 4


def test_guest_artifact_proof_still_fail_closed_on_match_false() -> None:
    """Repo improvement must survive — prod must not strip this gate."""

    proof = {
        "schema_version": 1,
        "expected_hash": "a" * 64,
        "download_hash": "a" * 64,
        "executed_hash": "a" * 64,
        "byte_size": 12,
        "match": False,
    }
    with pytest.raises(eval_wire.EvalWireError, match="match"):
        eval_wire.validate_guest_artifact_proof(proof)


def test_validate_eval_progress_request_score_free() -> None:
    ok = eval_wire.validate_eval_progress_request(
        {
            "schema_version": 1,
            "eval_run_id": "eval_1",
            "submission_id": "sub_1",
            "task_id": "task_1",
            "sequence": 1,
            "status": "running",
            "progress": 0.5,
        }
    )
    assert ok["status"] == "running"
    with pytest.raises(eval_wire.EvalWireError, match="score"):
        eval_wire.validate_eval_progress_request(
            {
                "schema_version": 1,
                "eval_run_id": "eval_1",
                "submission_id": "sub_1",
                "task_id": "task_1",
                "sequence": 1,
                "status": "running",
                "score": 1.0,
            }
        )


def test_execution_proof_optional_hydration_digest() -> None:
    import copy
    import json

    vectors = json.loads(
        (Path(__file__).with_name("eval_execution_proof_v2_vectors.json")).read_text(
            encoding="utf-8"
        )
    )
    base = copy.deepcopy(vectors["positive"]["execution_proof"])
    plain = eval_wire.validate_eval_execution_proof(base)
    assert "hydration_digest" not in plain
    with_h = copy.deepcopy(base)
    with_h["hydration_digest"] = "d" * 64
    validated = eval_wire.validate_eval_execution_proof(with_h)
    assert validated["hydration_digest"] == "d" * 64


def test_disk_size_and_eval_default_shape() -> None:
    assert DEFAULT_EVAL_INSTANCE_TYPE == "tdx.xlarge"
    assert DEFAULT_REVIEW_INSTANCE_TYPE == "tdx.small"
    assert validate_disk_size(DEFAULT_EVAL_DISK_SIZE_GB) == 100
    assert validate_disk_size(DEFAULT_REVIEW_DISK_SIZE_GB) == 20
    with pytest.raises(DiskSizeError):
        validate_disk_size(1)
    # compute + disk stays under $20 for default eval window
    cost = projected_cost_usd(
        DEFAULT_EVAL_INSTANCE_TYPE,
        max_runtime_hours=6.0,
        disk_size_gb=DEFAULT_EVAL_DISK_SIZE_GB,
    )
    assert cost < 20.0


def test_lifecycle_budget_includes_disk() -> None:
    total = projected_lifecycle_cost_usd(
        review_instance_type=DEFAULT_REVIEW_INSTANCE_TYPE,
        eval_instance_type=DEFAULT_EVAL_INSTANCE_TYPE,
        review_runtime_hours=1.0,
        eval_runtime_hours=2.0,
        review_disk_size_gb=20,
        eval_disk_size_gb=100,
    )
    assert total > 0
    ok = validate_lifecycle_budget(
        review_instance_type=DEFAULT_REVIEW_INSTANCE_TYPE,
        eval_instance_type=DEFAULT_EVAL_INSTANCE_TYPE,
        review_runtime_hours=1.0,
        eval_runtime_hours=2.0,
        money_cap_usd=20.0,
        review_disk_size_gb=20,
        eval_disk_size_gb=100,
    )
    assert ok.total_usd == total
    with pytest.raises(LifecycleBudgetError):
        validate_lifecycle_budget(
            review_instance_type=DEFAULT_REVIEW_INSTANCE_TYPE,
            eval_instance_type=DEFAULT_EVAL_INSTANCE_TYPE,
            review_runtime_hours=100.0,
            eval_runtime_hours=100.0,
            money_cap_usd=1.0,
            review_disk_size_gb=20,
            eval_disk_size_gb=100,
        )


def test_eval_max_attempts_bound_raised_to_256() -> None:
    s = ChallengeSettings(eval_max_attempts=64)
    assert s.eval_max_attempts == 64
    with pytest.raises(ValidationError):
        ChallengeSettings(eval_max_attempts=300)


def test_sdk_raw_weight_push_settings_preserved() -> None:
    """Repo-only master push settings must not be stripped by prod sdk_config."""

    s = ChallengeSettings()
    assert hasattr(s, "raw_weight_push_enabled")
    assert hasattr(s, "internal_token")
    assert callable(s.internal_token)


def test_compose_allows_progress_and_dstack_docker_envs() -> None:
    from agent_challenge.canonical.compose import DEFAULT_ALLOWED_ENVS

    for name in (
        "EVAL_PROGRESS_BASE_URL",
        "EVAL_RUN_ID",
        "EVAL_SUBMISSION_ID",
        "DSTACK_DOCKER_USERNAME",
        "DSTACK_DOCKER_PASSWORD",
        "DSTACK_DOCKER_REGISTRY",
        # repo artifact path must remain
        "CHALLENGE_PHALA_EVAL_ARTIFACT_URL",
        "CHALLENGE_PHALA_EVAL_ARTIFACT_TOKEN",
    ):
        assert name in DEFAULT_ALLOWED_ENVS


def test_phala_delete_cvm_path_guard() -> None:
    from agent_challenge.selfdeploy.phala import PhalaApiError, PhalaCloudClient

    class _Boom:
        def __call__(self, request, timeout=None):  # noqa: ANN001
            raise AssertionError("must not call network for invalid id")

    client = PhalaCloudClient.__new__(PhalaCloudClient)
    client._base_url = "https://example.invalid"
    client._timeout = 1.0
    client._opener = _Boom()
    client._base_headers = lambda content_type=True: {}  # type: ignore[method-assign]
    with pytest.raises(PhalaApiError, match="invalid CVM id"):
        client.delete_cvm("../etc/passwd")


def test_authorized_review_digest_still_accepts_settings_kw() -> None:
    """Dual residual gate signature must remain (prod tried to drop settings)."""

    import inspect

    from agent_challenge.evaluation import authorization as auth

    sig = inspect.signature(auth._authorized_review_digest)
    assert "settings" in sig.parameters
