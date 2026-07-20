from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import (
    _CONTAINER_EVAL_SCRIPT,
    PrismContainerEvaluator,
    _runner_launch_command,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.modes import execution_mode_from_value
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME
from prism_challenge.evaluator.source_similarity import SourceFile

ARCH_PROBE = """
import torch
from torch import nn

IMPORT_PROBE = torch.randn(4).tolist()


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

# Consume the CHALLENGE instrument so the run is not a zero-forward run; the loop body is
# identical to the benign twin so the forced-init invariant can be checked.
TRAIN_TAMPER = """
import json
import pathlib

import torch
import torch.nn.functional as F
from architecture import IMPORT_PROBE


def train(ctx):
    torch.manual_seed(987654321)  # miner re-seed attempt -> forced init must win
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
    flat = [x for p in model.parameters() for x in p.detach().flatten().tolist()]
    params = [round(float(x), 6) for x in flat]
    shards = sorted(pathlib.Path(ctx.data_dir).glob("*.jsonl"))
    data_text = shards[0].read_text(encoding="utf-8") if shards else ""
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps(
            {
                "import_probe": IMPORT_PROBE,
                "params": params,
                "data_text": data_text,
                "seed": ctx.seed,
            }
        ),
        encoding="utf-8",
    )
"""

TRAIN_BENIGN = """
import json
import pathlib

import torch
import torch.nn.functional as F
from architecture import IMPORT_PROBE


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
    flat = [x for p in model.parameters() for x in p.detach().flatten().tolist()]
    params = [round(float(x), 6) for x in flat]
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"import_probe": IMPORT_PROBE, "params": params}),
        encoding="utf-8",
    )
"""

TRAIN_FAKE_MANIFEST = """
import json
import pathlib

import torch
import torch.nn.functional as F
from architecture import build_model  # noqa: F401


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
    pathlib.Path(ctx.artifacts_dir, "prism_run_manifest.v2.json").write_text(
        json.dumps(
            {
                "schema_version": "prism_run_manifest.v2",
                "submission_id": "miner-evil",
                "metrics": {"bpb": 0.0001, "q_arch": 1.0},
            }
        ),
        encoding="utf-8",
    )
    print("PRISM_METRICS_JSON=" + json.dumps({"q_arch": 1.0, "bpb": 0.0001}))
"""

LOCKED_SHARD = '{"text": "the locked fineweb-edu train split bytes"}\n'


def _run_runner(
    tmp_path: Path,
    *,
    run_name: str,
    train_code: str,
    arch_code: str = ARCH_PROBE,
    seed: int = 1337,
    submission_id: str = "sub-abc123",
    data_files: dict[str, str] | None = None,
    plant_manifest: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path / run_name
    project = root / "project"
    project.mkdir(parents=True)
    (project / "architecture.py").write_text(arch_code, encoding="utf-8")
    (project / "training.py").write_text(train_code, encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    files = {"train-00000.jsonl": LOCKED_SHARD} if data_files is None else data_files
    for name, content in files.items():
        (data_dir / name).write_text(content, encoding="utf-8")

    artifacts = root / "artifacts"
    artifacts.mkdir()
    if plant_manifest is not None:
        (artifacts / RUN_MANIFEST_V2_FILENAME).write_text(plant_manifest, encoding="utf-8")

    payload = {
        "submission_id": submission_id,
        "architecture_entrypoint": "architecture.py",
        "training_entrypoint": "training.py",
        "build_model_symbol": "build_model",
        "train_symbol": "train",
        "execution_mode": "gpu_proxy_eval",
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "context": {
            "vocab_size": 32,
            "sequence_length": 16,
            "max_layers": 2,
            "max_parameters": 1000,
            "seed": seed,
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts),
            "reference_tokenizer_dir": str(root / "tok"),
            "token_budget": 64,
            "step_budget": 1,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "distributed_backend": None,
        },
    }
    runner = root / "runner.py"
    runner.write_text(_CONTAINER_EVAL_SCRIPT, encoding="utf-8")
    payload_path = root / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    env = dict(os.environ)
    env["PRISM_PROJECT_ROOT"] = str(project)
    env["PRISM_DATA_DIR"] = str(data_dir)
    env["PRISM_ARTIFACT_OUTPUT_PATH"] = str(artifacts)
    proc = subprocess.run(
        [sys.executable, str(runner), str(payload_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return proc, artifacts


def _read_manifest(artifacts: Path) -> dict:
    return json.loads((artifacts / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))


def test_harness_runner_forces_seed_before_miner_import(tmp_path):
    proc_a, artifacts_a = _run_runner(tmp_path, run_name="a", train_code=TRAIN_TAMPER)
    proc_b, artifacts_b = _run_runner(tmp_path, run_name="b", train_code=TRAIN_TAMPER)
    assert proc_a.returncode == 0, proc_a.stderr
    assert proc_b.returncode == 0, proc_b.stderr

    lines = proc_a.stdout.splitlines()
    forced_idx = next(i for i, line in enumerate(lines) if "forced seed=" in line)
    import_idx = next(i for i, line in enumerate(lines) if "imported architecture" in line)
    assert "before miner import" in lines[forced_idx]
    assert forced_idx < import_idx

    probe_a = json.loads((artifacts_a / "miner_probe.json").read_text(encoding="utf-8"))
    probe_b = json.loads((artifacts_b / "miner_probe.json").read_text(encoding="utf-8"))
    assert probe_a["import_probe"] == probe_b["import_probe"]
    assert probe_a["seed"] == 1337


def test_harness_runner_miner_cannot_override_forced_init(tmp_path):
    tamper, tamper_artifacts = _run_runner(tmp_path, run_name="tamper", train_code=TRAIN_TAMPER)
    benign, benign_artifacts = _run_runner(tmp_path, run_name="benign", train_code=TRAIN_BENIGN)
    assert tamper.returncode == 0, tamper.stderr
    assert benign.returncode == 0, benign.stderr
    tamper_probe = json.loads((tamper_artifacts / "miner_probe.json").read_text(encoding="utf-8"))
    benign_probe = json.loads((benign_artifacts / "miner_probe.json").read_text(encoding="utf-8"))
    assert tamper_probe["params"] == benign_probe["params"]


def test_harness_runner_trains_on_locked_data_not_random(tmp_path):
    proc, artifacts = _run_runner(tmp_path, run_name="locked", train_code=TRAIN_TAMPER)
    assert proc.returncode == 0, proc.stderr
    probe = json.loads((artifacts / "miner_probe.json").read_text(encoding="utf-8"))
    assert probe["data_text"] == LOCKED_SHARD
    manifest = _read_manifest(artifacts)
    assert manifest["data"]["source"] == "locked-fineweb-edu-train"
    assert manifest["data"]["random_token_fallback"] is False
    assert manifest["data"]["available_bytes"] == len(LOCKED_SHARD.encode("utf-8"))
    assert "randint" not in proc.stdout.lower()


def test_harness_runner_missing_locked_data_fails_fast(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="empty", train_code=TRAIN_BENIGN, data_files={}
    )
    assert proc.returncode != 0
    assert "no random-token fallback" in proc.stderr
    assert not (artifacts / RUN_MANIFEST_V2_FILENAME).exists()


def test_harness_runner_ignores_miner_written_manifest(tmp_path):
    proc, artifacts = _run_runner(tmp_path, run_name="evilmanifest", train_code=TRAIN_FAKE_MANIFEST)
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["schema_version"] == "prism_run_manifest.v2"
    assert manifest["submission_id"] == "sub-abc123"
    # The challenge authors the metrics from its own instrumentation; the miner-fabricated
    # values (bpb / q_arch) are discarded.
    assert "online_loss" in manifest["metrics"]
    assert "q_arch" not in manifest["metrics"]
    assert "bpb" not in manifest["metrics"]
    assert manifest["miner_reported_ignored"] is True


def test_harness_runner_fresh_artifacts_discards_stale_manifest(tmp_path):
    stale = json.dumps(
        {
            "schema_version": "prism_run_manifest.v2",
            "submission_id": "stale-old",
            "run_id": "prism-reexec-stale-old",
            "metrics": {"bpb": 9.99},
        }
    )
    proc, artifacts = _run_runner(
        tmp_path, run_name="stale", train_code=TRAIN_BENIGN, plant_manifest=stale
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["run_id"] == "prism-reexec-sub-abc123"
    assert manifest["submission_id"] == "sub-abc123"
    # The stale planted manifest (bpb 9.99) is discarded; metrics come from this run.
    assert "online_loss" in manifest["metrics"]
    assert manifest["metrics"].get("bpb") != 9.99


def test_container_runner_launch_command_is_loopback_single_proc():
    assert _runner_launch_command(1) == (
        "torchrun",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=1",
        "/workspace/runner.py",
        "/workspace/payload.json",
    )


def _evaluator(tmp_path: Path) -> PrismContainerEvaluator:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'harness.sqlite3'}",
        shared_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
    )
    ctx = PrismContext(vocab_size=32, sequence_length=16, seed=4242)
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


def test_container_payload_uses_locked_data_dir_and_loopback(tmp_path):
    ev = _evaluator(tmp_path)
    files = (SourceFile("project/architecture.py", ARCH_PROBE, "h1"),)
    payload = ev._payload(
        submission_id="sub-1",
        code_hash="h1",
        arch_hash="a1",
        files=files,
        architecture_entrypoint="project/architecture.py",
        training_entrypoint="project/training.py",
        build_model_symbol="build_model",
        train_symbol="train",
        gpu_allocation=ev._gpu_allocation(None),
        execution_mode=execution_mode_from_value(None),
    )
    assert payload["context"]["data_dir"] == "/data/fineweb-edu/train"
    assert payload["context"]["artifacts_dir"] == "/artifacts"
    assert payload["master_addr"] == "127.0.0.1"
    assert payload["artifact_output"]["manifest_path"] == f"/artifacts/{RUN_MANIFEST_V2_FILENAME}"


def test_container_env_sets_loopback_and_locked_data(tmp_path):
    ev = _evaluator(tmp_path)
    env = ev._env("sub-1", "h1", "a1", "base_gpu")
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == "29500"
    assert env["PRISM_DATA_DIR"] == "/data/fineweb-edu/train"
    assert env["PRISM_RUN_MANIFEST_PATH"] == f"/artifacts/{RUN_MANIFEST_V2_FILENAME}"


def test_container_fresh_artifact_output_discards_stale(tmp_path):
    ev = _evaluator(tmp_path)
    first = ev._fresh_artifact_output("sub-1", 1)
    (first / RUN_MANIFEST_V2_FILENAME).write_text('{"stale": true}', encoding="utf-8")
    again = ev._fresh_artifact_output("sub-1", 1)
    assert again == first
    assert list(again.glob("prism_run_manifest*.json")) == []
