"""Disk-aware CPU TDX sizing + stage defaults (offline, no Phala spend).

Covers:
  * stage defaults: review tdx.small/20GB, eval tdx.xlarge/100GB
  * disk validator bounds [20, 500]
  * disk billing in projected cost
  * lifecycle budget includes disk for both stages under the $20 cap
  * provision bodies emit disk_size as a sibling of compose_file
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_challenge.selfdeploy import lifecycle, shapes
from agent_challenge.selfdeploy.eval import EvalDeploymentPlan, HttpEvalPhalaDeployment
from agent_challenge.selfdeploy.review import HttpReviewPhalaDeployment, ReviewDeploymentPlan


def test_stage_instance_defaults_are_split():
    assert shapes.DEFAULT_REVIEW_INSTANCE_TYPE == "tdx.small"
    assert shapes.DEFAULT_EVAL_INSTANCE_TYPE == "tdx.xlarge"
    # Backward-compat alias remains the review default.
    assert shapes.DEFAULT_INSTANCE_TYPE == shapes.DEFAULT_REVIEW_INSTANCE_TYPE
    assert shapes.DEFAULT_REVIEW_DISK_SIZE_GB == 20
    assert shapes.DEFAULT_EVAL_DISK_SIZE_GB == 100


def test_disk_constants_match_decided_billing():
    assert shapes.DISK_USD_PER_GB_HOUR == 0.000139
    assert shapes.MIN_DISK_SIZE_GB == 20
    assert shapes.MAX_DISK_SIZE_GB == 500


@pytest.mark.parametrize("gb", [20, 100, 500])
def test_validate_disk_size_accepts_bounds(gb: int):
    assert shapes.validate_disk_size(gb) == gb


@pytest.mark.parametrize("gb", [0, 19, 501, -1, 20.5, "20", None])
def test_validate_disk_size_refuses_out_of_range(gb: object):
    with pytest.raises(shapes.ShapeError):
        shapes.validate_disk_size(gb)  # type: ignore[arg-type]


def test_projected_cost_includes_disk():
    # tdx.xlarge @ 0.464/h * 6h = 2.784; disk 100GB * 0.000139 * 6 = 0.0834
    cpu_only = shapes.projected_cost_usd("tdx.xlarge", max_runtime_hours=6.0)
    with_disk = shapes.projected_cost_usd(
        "tdx.xlarge",
        max_runtime_hours=6.0,
        disk_size_gb=100,
    )
    assert cpu_only == pytest.approx(0.464 * 6.0)
    assert with_disk == pytest.approx(0.464 * 6.0 + 0.000139 * 100 * 6.0)
    assert with_disk > cpu_only


def test_validate_within_cap_counts_disk():
    # Force over-cap via huge runtime + large disk on xlarge.
    with pytest.raises(shapes.OverCapError):
        shapes.validate_within_cap(
            "tdx.xlarge",
            money_cap_usd=1.0,
            max_runtime_hours=6.0,
            disk_size_gb=500,
        )


def test_default_lifecycle_budget_fits_money_cap():
    cost = lifecycle.validate_lifecycle_budget(
        review_instance_type=shapes.DEFAULT_REVIEW_INSTANCE_TYPE,
        eval_instance_type=shapes.DEFAULT_EVAL_INSTANCE_TYPE,
        review_runtime_hours=shapes.DEFAULT_MAX_RUNTIME_HOURS,
        eval_runtime_hours=shapes.DEFAULT_MAX_RUNTIME_HOURS,
        review_disk_size_gb=shapes.DEFAULT_REVIEW_DISK_SIZE_GB,
        eval_disk_size_gb=shapes.DEFAULT_EVAL_DISK_SIZE_GB,
        money_cap_usd=shapes.DEFAULT_MONEY_CAP_USD,
    )
    assert cost.total_usd <= shapes.DEFAULT_MONEY_CAP_USD
    assert cost.total_usd == pytest.approx(
        lifecycle.projected_lifecycle_cost_usd(
            review_instance_type=shapes.DEFAULT_REVIEW_INSTANCE_TYPE,
            eval_instance_type=shapes.DEFAULT_EVAL_INSTANCE_TYPE,
            review_runtime_hours=shapes.DEFAULT_MAX_RUNTIME_HOURS,
            eval_runtime_hours=shapes.DEFAULT_MAX_RUNTIME_HOURS,
            review_disk_size_gb=shapes.DEFAULT_REVIEW_DISK_SIZE_GB,
            eval_disk_size_gb=shapes.DEFAULT_EVAL_DISK_SIZE_GB,
        )
    )
    # Disk contribution is strictly positive vs CPU-only projection.
    cpu_only = (
        shapes.CPU_TDX_SHAPES[shapes.DEFAULT_REVIEW_INSTANCE_TYPE].usd_per_hour
        * shapes.DEFAULT_MAX_RUNTIME_HOURS
        + shapes.CPU_TDX_SHAPES[shapes.DEFAULT_EVAL_INSTANCE_TYPE].usd_per_hour
        * shapes.DEFAULT_MAX_RUNTIME_HOURS
    )
    assert cost.total_usd > cpu_only


def test_lifecycle_budget_refuses_over_cap_with_disk():
    with pytest.raises(lifecycle.LifecycleBudgetError):
        lifecycle.validate_lifecycle_budget(
            review_instance_type="tdx.small",
            eval_instance_type="tdx.xlarge",
            review_runtime_hours=100.0,
            eval_runtime_hours=100.0,
            review_disk_size_gb=20,
            eval_disk_size_gb=100,
            money_cap_usd=20.0,
        )


def test_lifecycle_budget_refuses_invalid_disk():
    with pytest.raises(lifecycle.LifecycleBudgetError):
        lifecycle.validate_lifecycle_budget(
            review_instance_type="tdx.small",
            eval_instance_type="tdx.xlarge",
            review_runtime_hours=1.0,
            eval_runtime_hours=1.0,
            review_disk_size_gb=10,
            eval_disk_size_gb=100,
        )


def _minimal_eval_plan(*, disk_size_gb: int = 100) -> EvalDeploymentPlan:
    return EvalDeploymentPlan(
        plan={"eval_run_id": "run-1"},
        plan_sha256="a" * 64,
        compose={"manifest_version": 2, "docker_compose_file": "services: {}\n"},
        compose_text="services: {}\n",
        compose_hash="b" * 64,
        app_identity="c" * 40,
        image_ref="ghcr.io/example/eval@sha256:" + ("d" * 64),
        kms_public_key_hex="ab" * 32,
        kms_public_key_sha256="e" * 64,
        measurement={"vm_shape": "tdx.xlarge"},
        eval_run_id="run-1",
        eval_run_token="token-secret",
        instance_type="tdx.xlarge",
        disk_size_gb=disk_size_gb,
    )


def _minimal_review_plan(*, disk_size_gb: int = 20) -> ReviewDeploymentPlan:
    return ReviewDeploymentPlan(
        assignment={"assignment_core": {"assignment_id": "asg-1"}},
        compose={"manifest_version": 2, "docker_compose_file": "services: {}\n"},
        compose_text="services: {}\n",
        compose_hash="b" * 64,
        app_identity="c" * 40,
        image_ref="ghcr.io/example/review@sha256:" + ("d" * 64),
        kms_public_key_hex="ab" * 32,
        kms_public_key_sha256="e" * 64,
        measurement={"vm_shape": "tdx.small"},
        measurement_allowlist_sha256="f" * 64,
        review_session_token="token-secret",
        instance_type="tdx.small",
        disk_size_gb=disk_size_gb,
    )


def test_eval_provision_emits_disk_size_sibling_of_compose():
    captured: list[dict[str, Any]] = []

    class Api:
        def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if path == "/cvms/provision":
                captured.append(dict(payload))
                return {
                    "compose_hash": "b" * 64,
                    "app_id": "c" * 40,
                    "app_env_encrypt_pubkey": "ab" * 32,
                    "os_image_hash": "0" * 64,
                }
            return {"id": "cvm-1", "request_id": "req-1", "created_at_ms": 1}

    plan = _minimal_eval_plan(disk_size_gb=100)
    encrypted = SimpleNamespace(
        eval_run_id=plan.eval_run_id,
        app_identity=plan.app_identity,
        kms_public_key_sha256=plan.kms_public_key_sha256,
        env_keys=("EVAL_RUN_TOKEN",),
        ciphertext="cipher",
    )
    # Avoid OS-identity side checks by stubbing the private verifier.
    dep = HttpEvalPhalaDeployment(Api())  # type: ignore[arg-type]
    dep._verify_provision_os_identity = MagicMock()  # type: ignore[method-assign]
    try:
        dep.deploy(plan, encrypted)  # type: ignore[arg-type]
    except Exception:
        # Create path may still fail closed; provision capture is what we need.
        pass
    assert captured, "provision was never called"
    body = captured[0]
    assert "disk_size" in body
    assert body["disk_size"] == 100
    assert "compose_file" in body
    assert body["compose_file"] is plan.compose
    # Must not mutate compose document.
    assert "disk_size" not in plan.compose


def test_review_provision_emits_disk_size_sibling_of_compose():
    captured: list[dict[str, Any]] = []

    class Api:
        def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            if path == "/cvms/provision":
                captured.append(dict(payload))
                return {
                    "compose_hash": "b" * 64,
                    "app_id": "c" * 40,
                    "app_env_encrypt_pubkey": "ab" * 32,
                    "os_image_hash": "0" * 64,
                }
            return {"id": "cvm-1", "request_id": "req-1", "created_at_ms": 1}

    plan = _minimal_review_plan(disk_size_gb=20)
    encrypted = SimpleNamespace(
        assignment_id="asg-1",
        app_identity=plan.app_identity,
        kms_public_key_sha256=plan.kms_public_key_sha256,
        measurement_allowlist_sha256=plan.measurement_allowlist_sha256,
        env_keys=plan  # placeholder replaced below
    )
    from agent_challenge.selfdeploy import review as review_mod

    encrypted = SimpleNamespace(
        assignment_id="asg-1",
        app_identity=plan.app_identity,
        kms_public_key_sha256=plan.kms_public_key_sha256,
        measurement_allowlist_sha256=plan.measurement_allowlist_sha256,
        env_keys=review_mod.REVIEW_ALLOWED_ENVS,
        ciphertext="cipher",
    )
    dep = HttpReviewPhalaDeployment(Api())  # type: ignore[arg-type]
    dep._verify_provision_response = MagicMock()  # type: ignore[method-assign]
    dep._resolve_created_cvm_id = MagicMock(return_value="cvm-1")  # type: ignore[method-assign]
    try:
        dep.deploy(plan, encrypted)  # type: ignore[arg-type]
    except Exception:
        pass
    assert captured, "provision was never called"
    body = captured[0]
    assert body["disk_size"] == 20
    assert body["compose_file"] is plan.compose
    assert "disk_size" not in plan.compose


def test_cli_stage_defaults_and_disk_flags():
    from agent_challenge.selfdeploy import cli

    parser = cli.build_parser()
    review_ns = parser.parse_args(
        [
            "review",
            "deploy",
            "--base-url",
            "https://example.test",
            "--submission-id",
            "1",
            "--hotkey",
            "5FakeHotkey",
            "--auto-sign",
            "--dry-run",
        ]
    )
    assert review_ns.review_instance_type == "tdx.small"
    assert review_ns.eval_instance_type == "tdx.xlarge"
    assert review_ns.review_disk_size_gb == 20
    assert review_ns.eval_disk_size_gb == 100

    eval_ns = parser.parse_args(
        [
            "eval",
            "deploy",
            "--base-url",
            "https://example.test",
            "--submission-id",
            "1",
            "--hotkey",
            "5FakeHotkey",
            "--dry-run",
            "--eval-disk-size-gb",
            "200",
            "--review-disk-size-gb",
            "40",
        ]
    )
    assert eval_ns.eval_instance_type == "tdx.xlarge"
    assert eval_ns.eval_disk_size_gb == 200
    assert eval_ns.review_disk_size_gb == 40
