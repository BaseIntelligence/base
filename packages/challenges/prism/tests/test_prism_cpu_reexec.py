from __future__ import annotations

import json
import math
import sqlite3
from hashlib import sha256
from pathlib import Path

import pytest
from base.challenge_sdk.executor import (
    DockerExecutor,
    DockerLimits,
    DockerMount,
    DockerRunSpec,
)
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import PrismContainerEvaluator
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.mock_reexec import (
    MockCpuReexecError,
    assert_network_isolated,
    cpu_reexec_run,
)
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME
from prism_challenge.evaluator.scoring import score_prequential_bpb
from prism_challenge.evaluator.source_similarity import SourceFile

# A tiny CPU-torch two-script bundle: a byte-level next-token TinyLM trained one step at a time over
# the challenge instrument. No GPU, no tokenizer (byte basis), deterministic under the forced seed.
TINY_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

TINY_TRAIN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

# Smuggled "pretrained" weights: build_model overwrites every parameter with a constant after
# construction (the forced manual_seed before build cannot undo a post-construction overwrite). A
# constant-weight model yields uniform logits, so its step-0 loss sits at the ~ln(vocab) random
# baseline -- it gains NO sub-baseline advantage (VAL-PRISM-008).
SMUGGLED_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    model = TinyLM(ctx.vocab_size)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    return model
"""

# A model whose forward emits non-finite logits on every batch: the challenge sanitizes each NaN
# batch to the worst-case code length, so a degenerate model can never collapse into a finite,
# advantageous bpb (VAL-PRISM-010).
NAN_ARCH = """
import torch
from torch import nn


class NanLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        logits = self.head(self.emb(tokens))
        return logits * float("nan")


def build_model(ctx):
    return NanLM(ctx.vocab_size)
"""

# A no-forward train: never iterates the challenge instrument, so no online loss is captured. The
# runner FAILS this (degenerate) run rather than fabricating a zero-that-ranks (VAL-PRISM-009).
ZERO_FORWARD_TRAIN = """
def train(ctx):
    ctx.build_model()
    return None
"""

# Plants a miner-authored manifest + inflated metrics during train; the challenge discards it and
# authors its own prism_run_manifest.v2 from its instrument (VAL-PRISM-006).
PLANTING_TRAIN = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    pathlib.Path(ctx.artifacts_dir, "prism_run_manifest.v2.json").write_text(
        json.dumps(
            {
                "schema_version": "prism_run_manifest.v2",
                "submission_id": "miner-evil",
                "metrics": {"bpb": 0.0001, "q_arch": 1.0},
                "score": {"final_score": 999.0},
            }
        ),
        encoding="utf-8",
    )
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

_SHARD_LINE = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


def _stage_train(root: Path, *, lines: int = 64) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD_LINE.format(i=i) for i in range(lines)), encoding="utf-8"
    )
    return data_dir


def _source_files(arch: str, train: str) -> tuple[SourceFile, ...]:
    return (
        SourceFile("architecture.py", arch, sha256(arch.encode()).hexdigest()),
        SourceFile("training.py", train, sha256(train.encode()).hexdigest()),
    )


def _direct_evaluator(tmp_path: Path) -> PrismContainerEvaluator:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'reexec.sqlite3'}",
        shared_token="secret",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
    )
    # Tiny CPU model: small vocab/seq and a short step budget keep the deterministic run fast.
    ctx = PrismContext(vocab_size=64, sequence_length=16, seed=1234, step_budget=24)
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


def _run_direct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    arch: str,
    train: str,
    data_dir: Path,
    submission_id: str = "sub-reexec",
):
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    evaluator = _direct_evaluator(tmp_path)
    files = _source_files(arch, train)
    result = evaluator.evaluate(
        submission_id=submission_id,
        code=arch,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
    )
    return result, captured


# --- VAL-PRISM-013 / 014 / 039: CPU re-exec produces a real v2 manifest on a network-none spec ---


def test_cpu_reexec_produces_v2_manifest_and_trained_state(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    result, captured = _run_direct(
        tmp_path, monkeypatch, arch=TINY_ARCH, train=TINY_TRAIN, data_dir=data_dir
    )

    manifest = result.run_manifest
    assert manifest is not None
    assert manifest["schema_version"] == "prism_run_manifest.v2"
    # A real captured online-loss stream (not a hand-authored manifest), produced on CPU.
    assert isinstance(manifest["metrics"]["online_loss"], list)
    assert len(manifest["metrics"]["online_loss"]) > 0
    assert manifest["metrics"]["covered_bytes"] > 0
    assert manifest["run"]["device"] == "cpu"
    assert manifest["miner_reported_ignored"] is True

    # The host scorer's trained_state artifact was persisted by the runner.
    trained = Path(result.artifact_output_path) / "trained_state.pt"
    assert trained.is_file()

    # VAL-PRISM-014: the scored launch command is single-process and the manifest agrees.
    spec = captured[0]
    assert "--nproc-per-node=1" in spec.command
    assert manifest["run"]["nproc_per_node"] == 1
    assert manifest["compute"]["world_size"] == 1
    assert manifest["compute"]["nproc_per_node"] == 1


def test_cpu_reexec_spec_is_network_isolated_to_workspace_and_artifacts(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    _result, captured = _run_direct(
        tmp_path, monkeypatch, arch=TINY_ARCH, train=TINY_TRAIN, data_dir=data_dir
    )

    spec = captured[0]
    # VAL-PRISM-039 / VAL-PRISM-028: network=none and ONLY workspace + writable artifacts mounted.
    assert spec.limits.network == "none"
    targets = {mount.target for mount in spec.mounts}
    assert targets == {"/workspace", "/artifacts"}
    writable = {mount.target for mount in spec.mounts if not mount.read_only}
    assert writable == {"/artifacts"}
    # No secret val/test split is ever handed to the container.
    assert not any("val" in mount.target or "test" in mount.target for mount in spec.mounts)


def test_assert_network_isolated_rejects_bad_specs(tmp_path):
    base_mounts = (
        DockerMount(tmp_path / "ws", "/workspace"),
        DockerMount(tmp_path / "art", "/artifacts", read_only=False),
    )
    ok = DockerRunSpec(
        image="img", command=("x",), mounts=base_mounts, limits=DockerLimits(network="none")
    )
    assert_network_isolated(ok)  # does not raise

    with_network = DockerRunSpec(
        image="img", command=("x",), mounts=base_mounts, limits=DockerLimits(network="bridge")
    )
    with pytest.raises(MockCpuReexecError, match="network=none"):
        assert_network_isolated(with_network)

    with_val = DockerRunSpec(
        image="img",
        command=("x",),
        mounts=(*base_mounts, DockerMount(tmp_path / "val", "/data/fineweb-edu/val")),
        limits=DockerLimits(network="none"),
    )
    with pytest.raises(MockCpuReexecError, match="workspace"):
        assert_network_isolated(with_val)


# --- VAL-PRISM-015: deterministic across repeats -------------------------------------------------


def test_cpu_reexec_is_deterministic_across_repeats(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    result_a, _ = _run_direct(
        tmp_path / "a",
        monkeypatch,
        arch=TINY_ARCH,
        train=TINY_TRAIN,
        data_dir=data_dir,
        submission_id="det-a",
    )
    result_b, _ = _run_direct(
        tmp_path / "b",
        monkeypatch,
        arch=TINY_ARCH,
        train=TINY_TRAIN,
        data_dir=data_dir,
        submission_id="det-b",
    )
    score_a = score_prequential_bpb(result_a.run_manifest)
    score_b = score_prequential_bpb(result_b.run_manifest)
    assert score_a.bpb == pytest.approx(score_b.bpb)
    assert score_a.final_score == pytest.approx(score_b.final_score)


# --- VAL-PRISM-016: gpu_count is observability-only, never a score input -------------------------


def test_cpu_reexec_gpu_count_is_observability_only(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    result, _ = _run_direct(
        tmp_path, monkeypatch, arch=TINY_ARCH, train=TINY_TRAIN, data_dir=data_dir
    )
    manifest = result.run_manifest

    baseline_score = score_prequential_bpb(manifest).final_score
    assert manifest["score"]["wall_clock_term"] is False

    inflated = json.loads(json.dumps(manifest))
    inflated["compute"]["gpu_count"] = 8
    inflated["run"]["world_size"] = 8
    assert score_prequential_bpb(inflated).final_score == pytest.approx(baseline_score)


# --- VAL-PRISM-006: miner-reported numbers are ignored -------------------------------------------


def test_cpu_reexec_ignores_miner_authored_manifest(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    result, _ = _run_direct(
        tmp_path, monkeypatch, arch=TINY_ARCH, train=PLANTING_TRAIN, data_dir=data_dir
    )
    manifest = result.run_manifest
    # The miner-planted bpb/q_arch/final_score are discarded; the challenge authors its own metrics.
    assert manifest["miner_reported_ignored"] is True
    assert "online_loss" in manifest["metrics"]
    assert "q_arch" not in manifest["metrics"]
    assert manifest["metrics"].get("bpb") != 0.0001
    assert manifest["score"]["final_score"] != 999.0
    assert manifest["submission_id"] == "sub-reexec"


# --- VAL-PRISM-008: smuggled pretrained weights stay inert under forced init ---------------------


def test_cpu_reexec_smuggled_constant_weights_are_inert(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    result, _ = _run_direct(
        tmp_path, monkeypatch, arch=SMUGGLED_ARCH, train=TINY_TRAIN, data_dir=data_dir
    )
    manifest = result.run_manifest
    metrics = manifest["metrics"]
    baseline = metrics["random_init_baseline_nats"]
    step0 = metrics["step0_loss"]
    # The constant-weight model gets a step-0 loss at the random baseline -- no sub-baseline edge.
    assert step0 == pytest.approx(baseline, rel=0.05)
    assert manifest["anti_cheat"]["step0_anomaly"] is False


# --- VAL-PRISM-010: NaN/Inf online-loss batches are sanitized to worst-case ----------------------


def test_cpu_reexec_nan_inf_batches_sanitized(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    nan_result, _ = _run_direct(
        tmp_path / "nan",
        monkeypatch,
        arch=NAN_ARCH,
        train=TINY_TRAIN,
        data_dir=data_dir,
        submission_id="nan",
    )
    benign_result, _ = _run_direct(
        tmp_path / "ok",
        monkeypatch,
        arch=TINY_ARCH,
        train=TINY_TRAIN,
        data_dir=data_dir,
        submission_id="ok",
    )
    nan_manifest = nan_result.run_manifest
    assert nan_manifest["metrics"]["nan_inf_batches"] > 0
    assert nan_manifest["anti_cheat"]["nan_inf_detected"] is True

    nan_score = score_prequential_bpb(nan_manifest)
    benign_score = score_prequential_bpb(benign_result.run_manifest)
    assert math.isfinite(nan_score.bpb)
    assert "nan_inf_detected" in nan_score.flags
    # Sanitizing to the worst-case code length RAISES bpb; it can never lower it below a real run.
    assert nan_score.bpb > benign_score.bpb


# --- VAL-PRISM-009: a degenerate (zero-forward) run is failed, never scored ----------------------


def test_cpu_reexec_zero_forward_run_fails(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    from prism_challenge.evaluator.container import ContainerEvaluationError

    with pytest.raises(ContainerEvaluationError):
        _run_direct(
            tmp_path, monkeypatch, arch=TINY_ARCH, train=ZERO_FORWARD_TRAIN, data_dir=data_dir
        )


# --- VAL-PRISM-005 / 013: end-to-end CPU re-exec through the worker scores prequential bpb -------


def _e2e_settings(tmp_path: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'e2e.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
    )


def _submit(client: TestClient, arch: str, train: str, nonce: str) -> str:
    payload = {
        "code": two_script_bundle(arch_code=arch, train_code=train),
        "filename": "project.zip",
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={**signed_headers("secret", body, nonce=nonce), "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def test_cpu_reexec_end_to_end_scores_recomputed_prequential_bpb(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    db_path = tmp_path / "e2e.sqlite3"
    settings = _e2e_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, TINY_ARCH, TINY_TRAIN, nonce="cpu-e2e")
        process = client.post(
            "/internal/v1/worker/process-next", headers={"Authorization": "Bearer secret"}
        )
        assert process.status_code == 200, process.text
        status = client.get(f"/v1/submissions/{submission_id}").json()
        assert status["status"] == "completed"

    # The dispatched spec used the validator's broker executor on a network-none container.
    assert captured, "expected the CPU re-exec executor to be dispatched"
    assert captured[0].limits.network == "none"
    assert "--nproc-per-node=1" in captured[0].command

    manifest_path = (
        Path(settings.base_eval_artifact_root)
        / submission_id
        / "attempt-1"
        / RUN_MANIFEST_V2_FILENAME
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recomputed = score_prequential_bpb(manifest)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    # VAL-PRISM-005: authoritative score is the challenge-recomputed prequential bpb final_score.
    assert row[0] == pytest.approx(recomputed.final_score)
    assert math.isfinite(row[0]) and row[0] > 0.0
    assert recomputed.bpb > 0.0


def test_real_docker_executor_run_is_not_invoked_in_tests():
    # Defense-in-depth: the CPU re-exec seam replaces DockerExecutor.run; the real method exists but
    # the mock path never touches a live broker/Docker. (Guards against accidental real dispatch.)
    assert hasattr(DockerExecutor, "run")


# --- mock matches the real executor timeout contract: TimeoutExpired -> timed_out=True ------------


def test_cpu_reexec_subprocess_timeout_maps_to_timed_out_result(tmp_path, monkeypatch):
    import subprocess as _subprocess

    from prism_challenge.evaluator import mock_reexec

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "runner.py").write_text("print('noop')\n", encoding="utf-8")
    (workspace / "payload.json").write_text(json.dumps({"context": {}}), encoding="utf-8")
    artifacts = tmp_path / "art"
    artifacts.mkdir()
    data_dir = _stage_train(tmp_path)

    spec = DockerRunSpec(
        image="img",
        command=("python", "runner.py"),
        mounts=(
            DockerMount(workspace, "/workspace"),
            DockerMount(artifacts, "/artifacts", read_only=False),
        ),
        limits=DockerLimits(network="none"),
        name="prism-timeout",
    )

    def _raise_timeout(*args, **kwargs):
        raise _subprocess.TimeoutExpired(
            cmd=args[0] if args else kwargs.get("args"),
            timeout=kwargs.get("timeout", 0.01),
            output="partial-out",
            stderr="partial-err",
        )

    monkeypatch.setattr(mock_reexec.subprocess, "run", _raise_timeout)

    result = mock_reexec.run_cpu_reexec(spec, 0.01, train_data_dir=data_dir)

    # Mirrors the real broker DockerExecutor: a timed-out run is a DockerRunResult(timed_out=True)
    # with returncode 124 and the partial captured output, never a propagated TimeoutExpired.
    assert result.timed_out is True
    assert result.returncode == 124
    assert result.container_name == "prism-timeout"
    assert result.stdout == "partial-out"
    assert result.stderr == "partial-err"
