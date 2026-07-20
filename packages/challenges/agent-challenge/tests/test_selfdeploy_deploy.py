"""Deploy-path guards + wiring for the miner self-deploy CLI.

Covers VAL-DEPLOY-002 (fetch/prepare → digest-pinned image + valid compose),
VAL-DEPLOY-006 (missing credentials fail clearly with no spend / no key leak),
VAL-DEPLOY-007 (GPU refused before any provisioning), VAL-DEPLOY-008 (smallest
CPU default + over-cap refusal), VAL-DEPLOY-009 (dry-run surfaces the full plan
with zero CVM-creating calls), and VAL-DEPLOY-010 (the operator key-release
endpoint is wired into the run configuration). All offline; no Phala calls.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest
import yaml

from agent_challenge.keyrelease.client import KEY_RELEASE_URL_ENV
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy.plan import (
    PHALA_API_KEY_ENV,
    PrepareError,
    build_deploy_plan,
    prepare_deployment,
)
from agent_challenge.selfdeploy.shapes import (
    DEFAULT_INSTANCE_TYPE,
    SMALLEST_CPU_SHAPES,
    GpuRefusedError,
    OverCapError,
)

DIGEST = "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:" + ("a" * 64)
URL = "https://validator.example/keyrelease"


class SpyDeployer:
    """Records deploy calls so tests can assert zero (or exactly one) Phala calls."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, plan, out_dir):
        self.calls.append((plan, out_dir))
        return {"ok": True}


def _run_cli(argv, *, deployer=None):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv, deployer=deployer)
    return code, out.getvalue(), err.getvalue()


def _docker_compose(prepared) -> dict:
    return yaml.safe_load(prepared.compose["docker_compose_file"])


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-002: fetch/prepare → digest-pinned image + valid generated compose
# --------------------------------------------------------------------------- #
def test_prepare_pins_image_by_digest_not_floating_tag():
    prepared = prepare_deployment(image=DIGEST, key_release_url=URL)
    assert prepared.image == DIGEST
    assert "@sha256:" in prepared.compose["docker_compose_file"]
    service = _docker_compose(prepared)["services"]["orchestrator"]
    assert service["image"] == DIGEST


def test_prepare_refuses_floating_tag():
    with pytest.raises(PrepareError):
        prepare_deployment(
            image="ghcr.io/baseintelligence/agent-challenge-canonical:latest",
            key_release_url=URL,
        )


def test_prepare_requires_key_release_endpoint():
    with pytest.raises(PrepareError):
        prepare_deployment(image=DIGEST, key_release_url="")


def test_prepared_compose_mounts_both_sockets():
    prepared = prepare_deployment(image=DIGEST, key_release_url=URL)
    volumes = _docker_compose(prepared)["services"]["orchestrator"]["volumes"]
    assert any(v.startswith("/var/run/dstack.sock:") for v in volumes)
    assert any(v.startswith("/var/run/docker.sock:") for v in volumes)


def test_prepared_compose_carries_operator_key_release_endpoint():
    prepared = prepare_deployment(image=DIGEST, key_release_url=URL)
    env = prepared.orchestrator_environment()
    # The compose carries the operator-supplied endpoint as a single static env,
    # under exactly the name the in-CVM backend reads.
    kr = [e for e in env if e.startswith(f"{KEY_RELEASE_URL_ENV}")]
    assert kr == [f"{KEY_RELEASE_URL_ENV}={URL}"], kr


def test_prepare_cli_writes_deployable_compose(tmp_path):
    code, out, _ = _run_cli(
        ["prepare", "--image", DIGEST, "--key-release-url", URL, "--out", str(tmp_path)]
    )
    assert code == 0
    payload = json.loads(out)
    compose_path = tmp_path / "app-compose.json"
    assert compose_path.is_file()
    # The digest and both sockets are present in the written bytes.
    written = compose_path.read_text(encoding="utf-8")
    assert "@sha256:" in written
    assert "/var/run/docker.sock" in written and "/var/run/dstack.sock" in written
    assert payload["image"] == DIGEST


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-007: GPU-targeted deploys refused before any provisioning
# --------------------------------------------------------------------------- #
def test_gpu_instance_type_refused_with_zero_deployer_calls():
    spy = SpyDeployer()
    code, _out, err = _run_cli(
        ["deploy", "--image", DIGEST, "--key-release-url", URL, "--instance-type", "h200.small"],
        deployer=spy,
    )
    assert code != 0
    assert spy.calls == []
    assert "cpu-only" in err.lower()


def test_gpu_os_image_refused_with_zero_deployer_calls():
    spy = SpyDeployer()
    code, _out, err = _run_cli(
        [
            "deploy",
            "--image",
            DIGEST,
            "--key-release-url",
            URL,
            "--os-image",
            "dstack-nvidia-0.5.10",
        ],
        deployer=spy,
    )
    assert code != 0
    assert spy.calls == []
    assert "cpu-only" in err.lower()


def test_cpu_tdx_shape_passes_the_same_validation():
    # A CPU Intel TDX shape builds a plan without raising (no GPU/cap refusal).
    plan = build_deploy_plan(image=DIGEST, key_release_url=URL, instance_type="tdx.small")
    assert plan.instance_type == "tdx.small"


def test_gpu_and_overcap_raise_before_prepare_or_phala():
    with pytest.raises(GpuRefusedError):
        build_deploy_plan(image=DIGEST, key_release_url=URL, instance_type="a100.large")


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-008: default smallest CPU shape + over-cap refusal
# --------------------------------------------------------------------------- #
def test_default_shape_is_a_smallest_cpu_shape():
    plan = build_deploy_plan(image=DIGEST, key_release_url=URL)
    assert plan.instance_type == DEFAULT_INSTANCE_TYPE
    assert plan.instance_type in SMALLEST_CPU_SHAPES


def test_over_cap_shape_refused_before_provisioning():
    spy = SpyDeployer()
    # Force the projected cost over the cap via a large runtime budget.
    code, _out, err = _run_cli(
        [
            "deploy",
            "--image",
            DIGEST,
            "--key-release-url",
            URL,
            "--instance-type",
            "tdx.xlarge",
            "--max-runtime-hours",
            "1000",
        ],
        deployer=spy,
    )
    assert code != 0
    assert spy.calls == []
    assert "cap" in err.lower()


def test_over_cap_raises_overcap_error():
    with pytest.raises(OverCapError):
        build_deploy_plan(
            image=DIGEST,
            key_release_url=URL,
            instance_type="tdx.xlarge",
            money_cap_usd=1.0,
            max_runtime_hours=100,
        )


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-006: missing credentials fail clearly, no spend, no key leak
# --------------------------------------------------------------------------- #
def test_missing_credentials_fail_clearly_with_no_deployer_call(monkeypatch):
    monkeypatch.delenv(PHALA_API_KEY_ENV, raising=False)
    spy = SpyDeployer()
    code, _out, err = _run_cli(
        ["deploy", "--image", DIGEST, "--key-release-url", URL],
        deployer=spy,
    )
    assert code != 0
    assert spy.calls == []  # never reached provisioning
    assert PHALA_API_KEY_ENV in err  # names the missing credential


def test_credential_value_is_never_printed(monkeypatch, capsys):
    sentinel = "phak_" + "s" * 32
    monkeypatch.setenv(PHALA_API_KEY_ENV, sentinel)
    spy = SpyDeployer()
    code, out, err = _run_cli(
        ["deploy", "--image", DIGEST, "--key-release-url", URL],
        deployer=spy,
    )
    assert code == 0
    assert len(spy.calls) == 1  # credential present → deploy proceeds
    assert sentinel not in out and sentinel not in err  # key value never echoed


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-009: dry-run surfaces the full plan with zero CVM-creating calls
# --------------------------------------------------------------------------- #
def test_dry_run_surfaces_full_plan_and_makes_no_deployer_call():
    spy = SpyDeployer()
    code, out, _err = _run_cli(
        ["deploy", "--image", DIGEST, "--key-release-url", URL, "--dry-run"],
        deployer=spy,
    )
    assert code == 0
    assert spy.calls == []  # dry-run creates nothing
    plan = json.loads(out)
    assert plan["dry_run"] is True
    # Every plan field a miner needs to review is present.
    assert plan["image"] == DIGEST
    assert plan["instance_type"] == DEFAULT_INSTANCE_TYPE
    assert plan["region"]
    assert plan["key_release_url"] == URL
    assert "@sha256:" in plan["compose"]
    assert plan["compose_hash"]
    assert plan["projected_cost_usd"] <= plan["money_cap_usd"]


def test_dry_run_needs_no_credentials(monkeypatch):
    monkeypatch.delenv(PHALA_API_KEY_ENV, raising=False)
    code, out, _err = _run_cli(["deploy", "--image", DIGEST, "--key-release-url", URL, "--dry-run"])
    assert code == 0
    assert json.loads(out)["dry_run"] is True


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-010: the operator key-release endpoint is wired into the run
# --------------------------------------------------------------------------- #
def test_deploy_plan_wires_operator_endpoint_under_backend_env_name():
    plan = build_deploy_plan(image=DIGEST, key_release_url=URL)
    # The deploy config carries the operator URL under EXACTLY the env name the
    # in-CVM backend reads to reach the key-release endpoint.
    assert plan.encrypted_env[KEY_RELEASE_URL_ENV] == URL


def test_endpoint_is_not_hardcoded_or_defaulted():
    other = "https://other-validator.example/kr"
    plan = build_deploy_plan(image=DIGEST, key_release_url=other)
    assert plan.key_release_url == other
    assert plan.encrypted_env[KEY_RELEASE_URL_ENV] == other
    env = plan.prepared.orchestrator_environment()
    assert f"{KEY_RELEASE_URL_ENV}={other}" in env
