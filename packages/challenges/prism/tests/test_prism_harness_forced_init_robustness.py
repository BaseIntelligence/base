from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import (
    _CONTAINER_EVAL_SCRIPT,
    PrismContainerEvaluator,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.modes import execution_mode_from_value
from prism_challenge.evaluator.source_similarity import SourceFile

# Hardening for the two determinism-robustness gaps the M3 scrutiny flagged in the forced-init
# runner (architecture.md section 6.3):
#   (1) strict torch.use_deterministic_algorithms(True) needs CUBLAS_WORKSPACE_CONFIG=:4096:8 on
#       CUDA, so the runner env must carry it before training (determinism cannot silently regress
#       on a torch/base-image bump);
#   (2) VAL-HARNESS-004 residual gap: a miner that re-seeds then builds the architecture via a
#       DIRECT import (bypassing ctx.build_model) must still get the harness-forced random init.

ARCH_LM = """
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

# Reads the runtime env (the runner executes training.py directly; the AST sandbox is a separate
# static-only gate) so the test can prove the runner forced CUBLAS_WORKSPACE_CONFIG before torch.
TRAIN_PROBE_CUBLAS = """
import json
import os
import pathlib


def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        break
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"cublas": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}), encoding="utf-8"
    )
"""

# DIRECT-import path (module-level ``from architecture import build_model``) that bypasses
# ctx.build_model and re-seeds INSIDE train before building -- the forced init must still win.
TRAIN_DIRECT_FROM_IMPORT_RESEED = """
import json
import pathlib

import torch
from architecture import build_model


def train(ctx):
    torch.manual_seed(424242)
    model = build_model(ctx)
    flat = [round(float(x), 6) for p in model.parameters() for x in p.detach().flatten().tolist()]
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        break
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"params": flat}), encoding="utf-8"
    )
"""

# Same direct path but no re-seed: the benign twin builds under the forced init.
TRAIN_DIRECT_FROM_IMPORT_BENIGN = """
import json
import pathlib

from architecture import build_model


def train(ctx):
    model = build_model(ctx)
    flat = [round(float(x), 6) for p in model.parameters() for x in p.detach().flatten().tolist()]
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        break
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"params": flat}), encoding="utf-8"
    )
"""

# DIRECT attribute path (``import architecture`` then ``architecture.build_model``) + re-seed.
TRAIN_DIRECT_ATTR_RESEED = """
import json
import pathlib

import architecture
import torch


def train(ctx):
    torch.manual_seed(424242)
    model = architecture.build_model(ctx)
    flat = [round(float(x), 6) for p in model.parameters() for x in p.detach().flatten().tolist()]
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        break
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"params": flat}), encoding="utf-8"
    )
"""

LOCKED_TEXT_LINE = (
    '{"id": "doc-%d", "text": "the locked fineweb-edu train split sample sentence %d"}\n'
)


def _locked_shard(lines: int) -> str:
    return "".join(LOCKED_TEXT_LINE % (i, i) for i in range(lines))


def _run_runner(
    tmp_path: Path,
    *,
    run_name: str,
    train_code: str,
    arch_code: str = ARCH_LM,
    seed: int = 1337,
    submission_id: str = "sub-robust",
    clear_cublas: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path / run_name
    project = root / "project"
    project.mkdir(parents=True)
    (project / "architecture.py").write_text(arch_code, encoding="utf-8")
    (project / "training.py").write_text(train_code, encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "train-00000.jsonl").write_text(_locked_shard(40), encoding="utf-8")

    artifacts = root / "artifacts"
    artifacts.mkdir()

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
            "vocab_size": 64,
            "sequence_length": 16,
            "max_layers": 2,
            "max_parameters": 5_000_000,
            "seed": seed,
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts),
            "reference_tokenizer_dir": str(root / "tok"),
            "token_budget": 256,
            "step_budget": 8,
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
    if clear_cublas:
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    proc = subprocess.run(
        [sys.executable, str(runner), str(payload_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc, artifacts


def _probe(artifacts: Path) -> dict:
    return json.loads((artifacts / "miner_probe.json").read_text(encoding="utf-8"))


def _evaluator(tmp_path: Path) -> PrismContainerEvaluator:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'robust.sqlite3'}",
        shared_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
    )
    ctx = PrismContext(vocab_size=32, sequence_length=16, seed=4242)
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


# --- Gap (1): CUBLAS_WORKSPACE_CONFIG=:4096:8 forced for strict CUDA determinism ----------------


def test_harness_container_env_sets_cublas_workspace_config(tmp_path: Path) -> None:
    env = _evaluator(tmp_path)._env("sub-1", "h1", "a1", "base_gpu")
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_harness_runner_forces_cublas_workspace_config_before_torch() -> None:
    script = _CONTAINER_EVAL_SCRIPT
    assert "CUBLAS_WORKSPACE_CONFIG" in script
    assert ":4096:8" in script
    cublas_idx = script.index("CUBLAS_WORKSPACE_CONFIG")
    torch_import_idx = script.index("\nimport torch")
    assert cublas_idx < torch_import_idx, "CUBLAS_WORKSPACE_CONFIG must be set before import torch"


def test_harness_runner_sets_cublas_default_at_runtime(tmp_path: Path) -> None:
    proc, artifacts = _run_runner(
        tmp_path, run_name="cublas", train_code=TRAIN_PROBE_CUBLAS, clear_cublas=True
    )
    assert proc.returncode == 0, proc.stderr
    assert _probe(artifacts)["cublas"] == ":4096:8"


# --- Gap (2): VAL-HARNESS-004 direct-import reseed -> forced init still authoritative -----------


def test_harness_direct_from_import_reseed_cannot_override_forced_init(tmp_path: Path) -> None:
    tamper, tamper_art = _run_runner(
        tmp_path, run_name="direct-tamper", train_code=TRAIN_DIRECT_FROM_IMPORT_RESEED
    )
    benign, benign_art = _run_runner(
        tmp_path, run_name="direct-benign", train_code=TRAIN_DIRECT_FROM_IMPORT_BENIGN
    )
    assert tamper.returncode == 0, tamper.stderr
    assert benign.returncode == 0, benign.stderr
    assert _probe(tamper_art)["params"] == _probe(benign_art)["params"]


def test_harness_direct_attr_reseed_cannot_override_forced_init(tmp_path: Path) -> None:
    tamper, tamper_art = _run_runner(
        tmp_path, run_name="attr-tamper", train_code=TRAIN_DIRECT_ATTR_RESEED
    )
    benign, benign_art = _run_runner(
        tmp_path, run_name="attr-benign", train_code=TRAIN_DIRECT_FROM_IMPORT_BENIGN
    )
    assert tamper.returncode == 0, tamper.stderr
    assert benign.returncode == 0, benign.stderr
    assert _probe(tamper_art)["params"] == _probe(benign_art)["params"]


def test_harness_evaluator_payload_files_roundtrip(tmp_path: Path) -> None:
    # Regression guard: the evaluator still assembles a payload with the two-script roles.
    ev = _evaluator(tmp_path)
    files = (SourceFile("project/architecture.py", ARCH_LM, "h1"),)
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
    assert payload["build_model_symbol"] == "build_model"
    assert payload["train_symbol"] == "train"


if __name__ == "__main__":  # pragma: no cover - convenience for ad-hoc runs
    pytest.main([__file__, "-q"])
