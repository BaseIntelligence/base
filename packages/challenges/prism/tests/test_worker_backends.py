from __future__ import annotations

import json

from base.challenge_sdk.executor import DockerRunResult
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings

# The static param-count phase (architecture.md section 4.1) instantiates build_model under the
# forced seed in the worker before any GPU work, so build_model must construct a real nn.Module.
# The actual training still runs only in the (mocked) remote container.
REMOTE_ONLY_CODE = """
import torch

def build_model(ctx):
    return torch.nn.Linear(8, 8)

def get_recipe(ctx):
    return {'learning_rate': 0.0003, 'batch_size': 2}
"""


def _artifact_dir(spec):
    for mount in spec.mounts:
        if mount.target == "/artifacts":
            return mount.source
    raise AssertionError("container spec has no /artifacts mount")


def _write_v2_manifest(spec) -> None:
    """Write a challenge-authored prism_run_manifest.v2 so the live bpb scoring path finalizes."""
    manifest = {
        "schema_version": "prism_run_manifest.v2",
        "metrics": {
            "covered_bytes": 4096,
            "sum_neg_log_likelihood_nats": 2200.0,
            "online_loss": [3.1, 2.9, 2.4],
            "predicted_tokens": 800,
            "tokens_seen": 800,
        },
    }
    (_artifact_dir(spec) / "prism_run_manifest.v2.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _submit(client: TestClient, code: str, nonce: str = "remote1") -> str:
    payload = {"code": two_script_bundle(arch_code=code), "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body, nonce=nonce), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def test_removed_legacy_remote_provider_backend_is_rejected(tmp_path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'remote.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="remote_provider",
    )
    try:
        create_app(settings)
    except ValueError as exc:
        assert "Unsupported execution backend" in str(exc)
    else:
        raise AssertionError("remote_provider must not be supported")


def test_base_gpu_worker_runs_submission_in_container(tmp_path, monkeypatch):
    captured = {}

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        captured["spec"] = spec
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        _write_v2_manifest(spec)
        return DockerRunResult(
            container_name="prism-eval",
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'base-gpu.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_cpus=3.0,
        base_eval_memory="12g",
        base_eval_gpu_count=2,
        base_eval_gpu_type="l4",
        plagiarism_enabled=False,
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, REMOTE_ONLY_CODE, nonce="base-gpu")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        # The live path scores from the challenge-owned bpb manifest, so the recorded score is a
        # finite positive bpb-derived final_score (not a miner-reported q_arch).
        assert status["q_arch"] is not None
        assert float(status["q_arch"]) > 0.0

    spec = captured["spec"]
    assert spec.image == "ghcr.io/baseintelligence/prism-evaluator:latest"
    assert spec.labels["base.job"] == submission_id
    assert spec.labels["base.task"] == "architecture"
    assert spec.env["PRISM_EXECUTION_BACKEND"] == "base_gpu"
    assert spec.env["PRISM_GPU_COUNT"] == "2"
    assert spec.env["PRISM_GPU_TYPE"] == "l4"
    assert spec.limits.cpus == 3.0
    assert spec.limits.memory == "12g"
    arch_file = next(
        item for item in captured["payload"]["files"] if item["path"].endswith("architecture.py")
    )
    assert arch_file["content"] == REMOTE_ONLY_CODE
    assert captured["payload"]["architecture_entrypoint"].endswith("architecture.py")
    assert captured["payload"]["training_entrypoint"].endswith("training.py")
    assert captured["payload"]["context"]["data_dir"] == "/data/fineweb-edu/train"
    # The outer docker/broker cap is the hard timeout, forced strictly above the graceful
    # wall-clock budget + watchdog grace so the runner can stop gracefully first.
    assert captured["timeout_seconds"] == settings.base_eval_hard_timeout_seconds
    assert captured["payload"]["context"]["budget_seconds"] == settings.base_eval_budget_seconds
    assert (
        captured["payload"]["context"]["watchdog_grace_seconds"]
        == settings.base_eval_watchdog_grace_seconds
    )
    assert (
        captured["payload"]["context"]["artifacts_quota_bytes"]
        == settings.base_eval_artifacts_quota_bytes
    )


def test_base_gpu_rejects_sandbox_violations_before_container(tmp_path, monkeypatch):
    def fail_run(self, spec, timeout_seconds):
        raise AssertionError("container should not run")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fail_run)
    code = """
import os

def build_model(ctx):
    return None

def get_recipe(ctx):
    return {}
"""
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'base-gpu-reject.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        plagiarism_enabled=False,
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, code, nonce="base-gpu-reject")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "rejected"
        assert "forbidden imports: os" in status["error"]


def test_base_gpu_version_advertises_docker_executor(tmp_path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'base-gpu-version.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
    )
    with TestClient(create_app(settings)) as client:
        version = client.get("/version").json()
        assert "challenge.ordinary_proof" in version["capabilities"]
    assert "lium" not in version["capabilities"]


def test_removed_legacy_local_cpu_backend_is_rejected(tmp_path):
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'local.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="local_cpu",
        sequence_length=16,
    )
    try:
        create_app(settings)
    except ValueError as exc:
        assert "Unsupported execution backend" in str(exc)
    else:
        raise AssertionError("local_cpu must not be supported")


CUSTOM_HOOK_CODE = """
import torch
import torch.nn.functional as F
from prism_challenge.evaluator.interface import TrainingRecipe

class HookModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 6)
        self.proj = torch.nn.Linear(6, vocab_size)

    def forward(self, tokens):
        return self.proj(self.emb(tokens))

def build_model(ctx):
    return HookModel(ctx.vocab_size)

def get_recipe(ctx):
    return TrainingRecipe(learning_rate=0.001, batch_size=1)

def configure_optimizer(model, recipe, ctx):
    return torch.optim.SGD(model.parameters(), lr=recipe.learning_rate)

def inference_logits(model, batch, ctx):
    return model(batch.tokens)

def compute_loss(model, batch, ctx):
    logits = inference_logits(model, batch, ctx)
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), batch.targets.reshape(-1) % vocab)

def train_step(model, batch, optimizer, ctx):
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss(model, batch, ctx)
    loss.backward()
    optimizer.step()
    return loss
"""


def test_base_gpu_accepts_custom_training_and_inference_hooks(tmp_path, monkeypatch):
    def fake_run(self, spec, timeout_seconds):
        _write_v2_manifest(spec)
        return DockerRunResult(
            container_name="prism-eval",
            stdout="",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'hooks.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        # Single-process training double; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, CUSTOM_HOOK_CODE, nonce="custom-hooks")
        process = client.post(
            "/internal/v1/worker/process-next",
            headers={"Authorization": "Bearer secret"},
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"
        assert status["q_arch"] is not None
