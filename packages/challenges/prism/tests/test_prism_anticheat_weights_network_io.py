from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from conftest import signed_headers, two_script_bundle

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import PrismContainerEvaluator
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code
from prism_challenge.evaluator.scoring import score_prequential_bpb

# Explicit negative-test suite for the no-pretrained-weights / no-network / no-IO-escape anti-cheat
# class (architecture.md sections 4.1, 6, 9). Each malicious vector is asserted REJECTED at the
# static AST sandbox (before any GPU work), NEUTRALIZED at scoring (forced-init step-0 anomaly), or
# blocked by the eval-container runtime posture (network=none, read-only rootfs, no review secret).
# Covers VAL-CHEAT-001..006, 017, 018, 019.

IMPORT_TORCH = "import torch\n"


def _violation(code: str) -> SandboxViolation:
    with pytest.raises(SandboxViolation) as raised:
        inspect_code(code, require_contract=False)
    return raised.value


def _fn(body: str) -> str:
    return IMPORT_TORCH + "\n\ndef use(model, ctx, data):\n" + body + "\n"


# --- VAL-CHEAT-001: smuggled pretrained weights via external load is sandbox-blocked -------------


@pytest.mark.parametrize(
    "body",
    [
        "    return torch.load('/tmp/pretrained.pt')",
        "    return torch.load('/data/fineweb-edu/../weights.pt', weights_only=True)",
        "    return pickle.loads(data)",
    ],
)
def test_anticheat_external_deserialization_load_blocked(body: str) -> None:
    assert _violation(_fn(body)).evidence[0].rule_id == "prism:no-deserialization"


@pytest.mark.parametrize(
    "statement, rule_id",
    [
        ("import pickle", "prism:no-deserialization"),
        ("import safetensors", "prism:no-forbidden-import"),
        ("from safetensors.torch import load_file", "prism:no-forbidden-import"),
        ("import numpy", "prism:no-forbidden-import"),
        ("import joblib", "prism:no-deserialization"),
    ],
)
def test_anticheat_weight_smuggling_imports_blocked(statement: str, rule_id: str) -> None:
    assert _violation(statement + "\n").evidence[0].rule_id == rule_id


def test_anticheat_open_binary_weight_read_blocked() -> None:
    assert _violation(_fn("    return open('/tmp/w.pt', 'rb')")).evidence[0].rule_id == (
        "prism:no-filesystem"
    )


# --- VAL-CHEAT-001 / 006 (hardening): shadowed or variable-built trusted path cannot launder a
#     torch.load/save to an untrusted location past the name-only trusted-root check --------------


@pytest.mark.parametrize(
    "body",
    [
        "    artifacts_dir = '/tmp/pretrained.pt'\n    return torch.load(artifacts_dir)",
        "    checkpoint_dir = '/data/secret/weights.pt'\n    return torch.load(checkpoint_dir)",
        "    resume_checkpoint_dir = '/tmp/w.pt'\n    return torch.load(resume_checkpoint_dir)",
    ],
)
def test_anticheat_shadowed_trusted_name_torch_load_blocked(body: str) -> None:
    # A submission that rebinds a trusted-path name to an external file must NOT be trusted just
    # because the variable NAME matches a harness dir.
    assert _violation(_fn(body)).evidence[0].rule_id == "prism:no-deserialization"


def test_anticheat_attribute_shadow_trusted_path_torch_load_blocked() -> None:
    code = (
        IMPORT_TORCH
        + "\n\ndef use(model, ctx, data):\n"
        + "    ctx.artifacts_dir = '/tmp/pretrained.pt'\n"
        + "    return torch.load(ctx.artifacts_dir)\n"
    )
    assert _violation(code).evidence[0].rule_id == "prism:no-deserialization"


def test_anticheat_incidental_trusted_name_does_not_launder_external_load() -> None:
    # The trusted name appears only inside a sub-expression; the path itself is an external literal.
    body = "    return torch.load('/etc/passwd' + str(len(artifacts_dir)))"
    assert _violation(_fn(body)).evidence[0].rule_id == "prism:no-deserialization"


def test_anticheat_setattr_rebound_trusted_path_torch_load_blocked() -> None:
    # A dynamic setattr rebind of a trusted root (no Store AST node) must drop it from the trusted
    # set so torch.load(ctx.artifacts_dir) cannot be laundered past the static name-only check.
    code = (
        IMPORT_TORCH
        + "\n\ndef use(model, ctx, data):\n"
        + "    setattr(ctx, 'artifacts_dir', '/tmp/pretrained.pt')\n"
        + "    return torch.load(ctx.artifacts_dir)\n"
    )
    assert _violation(code).evidence[0].rule_id == "prism:no-deserialization"


def test_anticheat_setattr_rebound_trusted_path_torch_save_blocked() -> None:
    code = (
        IMPORT_TORCH
        + "\n\ndef use(model, ctx, data):\n"
        + "    setattr(ctx, 'checkpoint_dir', '/etc/cron.d/x')\n"
        + "    torch.save(model.state_dict(), ctx.checkpoint_dir)\n"
        + "    return None\n"
    )
    assert _violation(code).evidence[0].rule_id == "prism:no-filesystem"


def test_anticheat_shadowed_trusted_name_torch_save_blocked() -> None:
    body = (
        "    artifacts_dir = '/etc/cron.d/x'\n"
        "    torch.save(model.state_dict(), artifacts_dir)\n"
        "    return None"
    )
    assert _violation(_fn(body)).evidence[0].rule_id == "prism:no-filesystem"


def test_anticheat_legitimate_trusted_checkpoint_roundtrip_still_allowed() -> None:
    # Regression guard: the harness-provided trusted dirs (a ``checkpoint_dir`` parameter and
    # ``ctx.artifacts_dir`` read) must remain a permitted torch checkpoint round-trip.
    code = (
        IMPORT_TORCH
        + "\n\ndef load_ckpt(model, checkpoint_dir, ctx):\n"
        + "    payload = torch.load(checkpoint_dir / 'model.pt', weights_only=True)\n"
        + "    model.load_state_dict(payload['state_dict'])\n"
        + "    return None\n\n"
        + "def save_ckpt(model, ctx):\n"
        + "    torch.save(model.state_dict(), ctx.artifacts_dir + '/ckpt.pt')\n"
        + "    return None\n"
    )
    report = inspect_code(code, require_contract=False)
    assert "function:load_ckpt" in report.ast_fingerprint
    assert "function:save_ckpt" in report.ast_fingerprint


# --- VAL-CHEAT-005: network / IO imports + dynamic-code builtins blocked before GPU --------------


@pytest.mark.parametrize(
    "statement, rule_id",
    [
        ("import socket", "prism:no-network"),
        ("import requests", "prism:no-network"),
        ("import urllib", "prism:no-network"),
        ("from http import client", "prism:no-network"),
        ("import os", "prism:no-process"),
        ("import sys", "prism:no-process"),
        ("import subprocess", "prism:no-process"),
        ("import ctypes", "prism:no-ffi"),
        ("import importlib", "prism:no-dynamic-import"),
    ],
)
def test_anticheat_network_io_imports_blocked(statement: str, rule_id: str) -> None:
    assert _violation(statement + "\n").evidence[0].rule_id == rule_id


@pytest.mark.parametrize(
    "body",
    [
        "    return eval('1 + 1')",
        "    exec('x = 1')\n    return None",
        "    return compile('1', '<s>', 'eval')",
    ],
)
def test_anticheat_dynamic_code_builtins_blocked(body: str) -> None:
    assert _violation(_fn(body)).evidence[0].rule_id == "prism:no-dynamic-code"


# --- VAL-CHEAT-006: deserialization / attribute-escape smuggling channels blocked ---------------


@pytest.mark.parametrize(
    "body",
    [
        "    return model.__globals__",
        "    return model.__reduce__()",
        "    return type(model).__subclasses__()",
        "    return (1).__class__.__bases__",
        "    return type(model).__mro__",
    ],
)
def test_anticheat_attribute_escape_blocked(body: str) -> None:
    with pytest.raises(SandboxViolation):
        inspect_code(_fn(body), require_contract=False)


def test_anticheat_dynamic_importlib_smuggle_blocked() -> None:
    assert _violation(_fn("    return importlib.import_module('os')")).evidence[0].rule_id == (
        "prism:no-dynamic-import"
    )


@pytest.mark.parametrize(
    "body",
    [
        "    return getattr(torch, 'lo' + 'ad')(data)",
        "    return getattr(__builtins__, 'ev' + 'al')('1')",
    ],
)
def test_anticheat_string_built_indirection_blocked(body: str) -> None:
    assert _violation(_fn(body)).evidence[0].rule_id == "prism:no-dynamic-attr"


# --- VAL-CHEAT-004: outbound network during training fails (network=none) ------------------------


def test_anticheat_outbound_socket_in_source_blocked() -> None:
    # An in-source outbound connection is caught statically; the runtime container has network=none.
    assert _violation("import socket\n").evidence[0].rule_id == "prism:no-network"


def test_anticheat_eval_container_runs_network_none() -> None:
    assert PrismSettings().docker_network == "none"


# --- VAL-CHEAT-017: writes outside artifacts_dir fail --------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "    return open('/etc/cron.d/x', 'w')",
        "    torch.save(model.state_dict(), '/root/evil.pt')\n    return None",
    ],
)
def test_anticheat_write_outside_artifacts_blocked(body: str) -> None:
    assert _violation(_fn(body)).evidence[0].rule_id == "prism:no-filesystem"


@pytest.mark.parametrize("module", ["pathlib", "shutil", "tempfile", "glob"])
def test_anticheat_filesystem_imports_blocked(module: str) -> None:
    assert _violation(f"import {module}\n").evidence[0].rule_id == "prism:no-filesystem"


def test_anticheat_eval_container_rootfs_read_only_except_artifacts() -> None:
    settings = PrismSettings()
    assert settings.base_eval_read_only is True


def test_anticheat_only_artifacts_mount_is_writable(tmp_path: Path) -> None:
    ev = _evaluator(tmp_path)
    mounts = ev._mounts(tmp_path / "ws", tmp_path / "art")
    writable = [mount for mount in mounts if not mount.read_only]
    assert writable, "expected at least one writable mount"
    assert all(mount.target == "/artifacts" for mount in writable)
    assert {mount.target for mount in mounts if mount.read_only}  # workspace stays read-only


def test_anticheat_eval_container_never_mounts_secret_heldout(tmp_path: Path) -> None:
    ev = _evaluator(tmp_path)
    settings = PrismSettings()
    mounts = ev._mounts(tmp_path / "ws", tmp_path / "art")
    targets = {mount.target for mount in mounts}
    # The SECRET held-out val split lives only on the host scorer; the untrusted
    # eval container is handed NO filesystem path to it (every mount target is
    # the workspace or artifacts dir), so miner code cannot read the held-out data.
    assert settings.base_eval_val_data_dir not in targets
    assert not any(mount.target.rstrip("/").endswith("/val") for mount in mounts)
    assert targets <= {"/workspace", "/artifacts"}


# --- VAL-CHEAT-019: the review/host secret never reaches the scored eval container ---------------


def _evaluator(tmp_path: Path) -> PrismContainerEvaluator:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'anticheat.sqlite3'}",
        shared_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
    )
    ctx = PrismContext(vocab_size=32, sequence_length=16, seed=4242)
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


def test_anticheat_eval_env_carries_no_review_or_host_secret(tmp_path: Path) -> None:
    ev = _evaluator(tmp_path)
    env = ev._env("sub-1", "h1", "a1", "base_gpu")
    joined = " ".join(f"{key}={value}" for key, value in env.items()).lower()
    for marker in ("openrouter", "api_key", "authorization", "bearer", "openai", "or_key"):
        assert marker not in joined, f"unexpected secret marker {marker!r} in eval env"
    assert all(not key.lower().endswith("token") or "broker" in key.lower() for key in env)


# --- VAL-CHEAT-002 / 003: embedded weights are inert + step-0 loss anomaly is flagged -----------


def _manifest(*, step0_loss: float, sum_nll_nats: float, step0_anomaly: bool) -> dict:
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": "sub-anticheat",
        "data": {"covered_bytes": 4096, "single_pass": True},
        "metrics": {
            "online_loss": [step0_loss, step0_loss * 0.95, step0_loss * 0.9],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "covered_bytes": 4096,
            "predicted_tokens": 96,
            "step0_loss": step0_loss,
            "consumed_batches": 3,
        },
        "anti_cheat": {
            "step0_anomaly": step0_anomaly,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


def test_anticheat_honest_random_init_scores_without_anomaly() -> None:
    # Embedded weights re-executed under forced random init land in the high random-init band:
    # no step-0 anomaly, a normal (non-zeroed) score.
    score = score_prequential_bpb(
        _manifest(step0_loss=4.8, sum_nll_nats=300.0, step0_anomaly=False)
    )
    assert score.anomaly is False
    assert "step0_anomaly" not in score.flags
    assert score.anti_cheat_multiplier == 1.0
    assert score.final_score > 0.0


def test_anticheat_step0_anomaly_is_flagged_and_zeroed() -> None:
    # An impossibly-low step-0 loss (smuggled pretrained weights surviving) is flagged and the
    # anti-cheat multiplier zeroes the score so it confers no advantage.
    anomalous = score_prequential_bpb(
        _manifest(step0_loss=0.01, sum_nll_nats=2.0, step0_anomaly=True)
    )
    assert anomalous.anomaly is True
    assert "step0_anomaly" in anomalous.flags
    assert anomalous.anti_cheat_multiplier == 0.0
    assert anomalous.final_score == 0.0


def test_anticheat_smuggled_weights_get_no_advantage_over_honest_baseline() -> None:
    honest = score_prequential_bpb(
        _manifest(step0_loss=4.8, sum_nll_nats=300.0, step0_anomaly=False)
    )
    smuggled = score_prequential_bpb(
        _manifest(step0_loss=0.01, sum_nll_nats=2.0, step0_anomaly=True)
    )
    # Even though the smuggled run has a far lower bpb, the anomaly flag removes any advantage.
    assert smuggled.bpb < honest.bpb
    assert smuggled.final_score < honest.final_score
    assert smuggled.final_score == 0.0


# --- VAL-CHEAT-001 / 005 (pipeline): a static hard-block rejects BEFORE any GPU work ------------


def _submit(client, code: str, *, nonce: str) -> str:
    payload = {"code": code, "filename": "bundle.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, hotkey="hk", nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client) -> None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text


def _submission_row(client, submission_id: str) -> dict:
    repository = client.app.state.repository

    async def fetch() -> dict:
        async with repository.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status, error FROM submissions WHERE id=?", (submission_id,)
            )
        return dict(rows[0])

    return anyio.run(fetch)


def _gpu_work_counts(client, submission_id: str) -> tuple[int, int]:
    """GPU leases + GPU-level eval jobs (the ``l1`` static-tracking row is NOT GPU work)."""
    repository = client.app.state.repository

    async def fetch() -> tuple[int, int]:
        async with repository.database.connect() as conn:
            leases = await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM gpu_leases WHERE submission_id=?", (submission_id,)
            )
            jobs = await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM eval_jobs WHERE submission_id=? AND level != 'l1'",
                (submission_id,),
            )
        return int(leases[0]["n"]), int(jobs[0]["n"])

    return anyio.run(fetch)


def test_anticheat_external_weight_load_rejected_before_gpu(client) -> None:
    # torch.load of an external path is not matched by the deterministic LLM safety prefilter
    # (no os.system/open/eval token), so the AST sandbox is the gate that must reject it before
    # any GPU lease/job is created.
    malicious_train = (
        "import torch\n"
        "from architecture import build_model\n\n"
        "def train(ctx):\n"
        "    build_model(ctx)\n"
        "    return torch.load('/tmp/pretrained_weights.pt')\n"
    )
    code = two_script_bundle(train_code=malicious_train)
    submission_id = _submit(client, code, nonce="anticheat-extload-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] == "rejected", row
    assert "torch.load" in str(row["error"]) or "deserialization" in str(row["error"]), row
    leases, jobs = _gpu_work_counts(client, submission_id)
    assert leases == 0, f"expected no gpu lease, found {leases}"
    assert jobs == 0, f"expected no GPU eval job, found {jobs}"


def test_anticheat_shadowed_trusted_load_rejected_before_gpu(client) -> None:
    # The shadowed-trusted-name laundering vector must also be rejected before any GPU work.
    malicious_train = (
        "import torch\n"
        "from architecture import build_model\n\n"
        "def train(ctx):\n"
        "    build_model(ctx)\n"
        "    artifacts_dir = '/tmp/pretrained_weights.pt'\n"
        "    return torch.load(artifacts_dir)\n"
    )
    code = two_script_bundle(train_code=malicious_train)
    submission_id = _submit(client, code, nonce="anticheat-shadowload-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] == "rejected", row
    leases, jobs = _gpu_work_counts(client, submission_id)
    assert leases == 0, f"expected no gpu lease, found {leases}"
    assert jobs == 0, f"expected no GPU eval job, found {jobs}"
