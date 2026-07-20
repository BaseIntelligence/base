from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from prism_challenge.evaluator import reference_tokenizers as rt
from prism_challenge.evaluator.container import _CONTAINER_EVAL_SCRIPT
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME


def _network_available() -> bool:
    try:
        with socket.create_connection(("huggingface.co", 443), timeout=8):
            return True
    except OSError:
        return False


# A tiny token-in/logits-out LM the challenge instrument can score (forward -> [B, T, V]).
ARCH_LM = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 16)
        self.head = nn.Linear(16, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

# A miner that consumes the CHALLENGE instrument (predict-then-train single pass).
TRAIN_LEARN = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    miner_losses = []
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
        miner_losses.append(float(loss))
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"miner_losses": miner_losses}), encoding="utf-8"
    )
"""

# A no-op / dead loop: returns immediately, never iterates the instrument (no forward).
TRAIN_ZERO_BATCH = """
def train(ctx):
    ctx.build_model()
    return
"""

# An architecture whose forward yields non-finite logits -> NaN/Inf online loss.
ARCH_NAN = """
import torch
from torch import nn


class NanLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens)) * float("inf")


def build_model(ctx):
    return NanLM(ctx.vocab_size)
"""

# A "smuggled-weights" architecture: a hard-coded bias makes the model predict byte 97 ('a')
# with near-certainty regardless of input, so on an all-'a' corpus the step-0 loss is ~0.
ARCH_SMUGGLED = """
import torch
from torch import nn


class SmuggledLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 4)
        self.head = nn.Linear(4, vocab)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.bias[97] = 50.0

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return SmuggledLM(ctx.vocab_size)
"""

# A miner that probes whether the reference-tokenizer module is importable from inside the
# runner's shadowed prism_challenge.evaluator package (the m3-forced-init regression).
TRAIN_REFTOK_PROBE = """
import importlib
import json
import pathlib


def train(ctx):
    model = ctx.build_model()
    probe = {}
    try:
        mod = importlib.import_module("prism_challenge.evaluator.reference_tokenizers")
        probe["module_imported"] = True
        probe["has_loader"] = hasattr(mod, "load_reference_tokenizer")
    except Exception as exc:  # noqa: BLE001
        probe["module_imported"] = False
        probe["error_type"] = type(exc).__name__
    try:
        ctx.reference_tokenizer("gpt2")
        probe["reference_call_error"] = None
    except Exception as exc:  # noqa: BLE001
        probe["reference_call_error"] = type(exc).__name__
    # Consume at least one instrumented batch so this is not a zero-forward run.
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        break
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps(probe), encoding="utf-8"
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
    arch_code: str,
    train_code: str,
    vocab_size: int = 128,
    sequence_length: int = 16,
    seed: int = 1337,
    submission_id: str = "sub-online",
    data_files: dict[str, str] | None = None,
    token_budget: int | None = None,
    step_budget: int | None = None,
    reference_tokenizer_dir: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path / run_name
    project = root / "project"
    project.mkdir(parents=True)
    (project / "architecture.py").write_text(arch_code, encoding="utf-8")
    (project / "training.py").write_text(train_code, encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    files = {"train-00000.jsonl": _locked_shard(40)} if data_files is None else data_files
    for name, content in files.items():
        (data_dir / name).write_text(content, encoding="utf-8")

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
            "vocab_size": vocab_size,
            "sequence_length": sequence_length,
            "max_layers": 2,
            "max_parameters": 5_000_000,
            "seed": seed,
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts),
            "reference_tokenizer_dir": reference_tokenizer_dir or str(root / "tok"),
            "token_budget": token_budget,
            "step_budget": step_budget,
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
        timeout=300,
    )
    return proc, artifacts


def _read_manifest(artifacts: Path) -> dict:
    return json.loads((artifacts / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))


def test_harness_online_loss_capture_predict_then_train(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="capture", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    online = metrics["online_loss"]
    # The challenge captures one online loss per consumed batch (predict-before-update).
    assert isinstance(online, list)
    assert len(online) >= 2
    assert len(online) == metrics["consumed_batches"]
    assert metrics["consumed_batches"] == manifest["data"]["consumed_batches"]
    assert all(math.isfinite(v) for v in online)
    # The stream is challenge-captured, not the miner's reported numbers.
    miner = json.loads((artifacts / "miner_probe.json").read_text(encoding="utf-8"))
    assert len(miner["miner_losses"]) == len(online)
    assert manifest["anti_cheat"]["no_learning"] is False
    assert manifest["miner_reported_ignored"] is True


def test_harness_step0_baseline_within_random_init_band(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="step0", arch_code=ARCH_LM, train_code=TRAIN_LEARN, vocab_size=128
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    baseline = metrics["random_init_baseline_nats"]
    assert baseline == math.log(128)
    step0 = metrics["step0_loss"]
    assert step0 is not None
    assert metrics["online_loss"][0] == step0
    # Random init => step-0 loss sits in the from-scratch band (~ln(vocab)); not anomalous.
    assert 0.5 * baseline <= step0 <= 2.5 * baseline
    assert manifest["anti_cheat"]["step0_anomaly"] is False


def test_harness_step0_anomaly_flags_smuggled_low_loss(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="smuggled",
        arch_code=ARCH_SMUGGLED,
        train_code=TRAIN_LEARN,
        vocab_size=128,
        data_files={"train-00000.jsonl": '{"id": "a", "text": "%s"}\n' % ("a" * 400)},
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    baseline = metrics["random_init_baseline_nats"]
    # The hard-coded model predicts the next byte with near-certainty => impossibly low step-0.
    assert metrics["step0_loss"] < 0.5 * baseline
    assert manifest["anti_cheat"]["step0_anomaly"] is True


def test_harness_nan_inf_online_loss_sanitized_to_worst_case(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="nan", arch_code=ARCH_NAN, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    online = metrics["online_loss"]
    assert online, "expected captured (sanitized) online losses"
    # NaN/Inf never collapses into a finite low/good value: every entry is finite and high.
    assert all(math.isfinite(v) for v in online)
    baseline = metrics["random_init_baseline_nats"]
    assert all(v >= baseline for v in online)
    assert manifest["anti_cheat"]["nan_inf_detected"] is True
    assert manifest["anti_cheat"]["nan_inf_batches"] == metrics["nan_inf_batches"]
    assert metrics["nan_inf_batches"] >= 1


def test_harness_zero_batch_no_forward_flagged_failed(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="zero", arch_code=ARCH_LM, train_code=TRAIN_ZERO_BATCH
    )
    # A zero-batch / no-forward run is flag-FAILED (non-zero exit), no fabricated score.
    assert proc.returncode != 0
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    assert metrics["online_loss"] == []
    assert metrics["covered_bytes"] == 0
    assert metrics["step0_loss"] is None
    assert manifest["data"]["covered_bytes"] == 0
    assert manifest["anti_cheat"]["no_learning"] is True
    assert manifest["anti_cheat"]["zero_forward"] is True


def test_harness_single_pass_no_repeats_covered_bytes_increasing(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="singlepass",
        arch_code=ARCH_LM,
        train_code=TRAIN_LEARN,
        data_files={
            "train-00000.jsonl": _locked_shard(20),
            "train-00001.jsonl": _locked_shard(20),
        },
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    data = manifest["data"]
    assert data["single_pass"] is True
    assert data["random_token_fallback"] is False
    # No shard/offset revisited within the run.
    assert data["distinct_offsets"] == data["consumed_offsets"]
    cumulative = data["covered_bytes_cumulative"]
    assert len(cumulative) == manifest["metrics"]["consumed_batches"]
    assert all(b < a for b, a in zip(cumulative, cumulative[1:], strict=False))
    assert data["covered_bytes"] == manifest["metrics"]["covered_bytes"]
    assert data["covered_bytes"] > 0


def test_harness_reference_tokenizer_module_importable_from_runner(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="reftok", arch_code=ARCH_LM, train_code=TRAIN_REFTOK_PROBE
    )
    assert proc.returncode == 0, proc.stderr
    probe = json.loads((artifacts / "miner_probe.json").read_text(encoding="utf-8"))
    # The runner module-shadowing must NOT break importing reference_tokenizers.
    assert probe["module_imported"] is True
    assert probe["has_loader"] is True
    # The reference_tokenizer() call must not fail with an import error from the shadow.
    assert probe["reference_call_error"] not in ("ModuleNotFoundError", "ImportError")


# A miner that resolves a pre-staged gpt2 reference tokenizer at runtime and round-trips text,
# proving VAL-CONTRACT-022 / VAL-DATA-013 on the REAL re-execution path (not just standalone).
TRAIN_REFTOK_GPT2 = """
import json
import pathlib


def train(ctx):
    model = ctx.build_model()
    enc = ctx.reference_tokenizer("gpt2")
    roundtrip = enc.decode(enc.encode("prism online loss"))
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"n_vocab": int(enc.n_vocab), "roundtrip": roundtrip}), encoding="utf-8"
    )
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        break
"""


@pytest.mark.skipif(
    not _network_available(), reason="staging the gpt2 tiktoken cache needs network (prep step)"
)
def test_harness_reference_tokenizer_gpt2_loads_on_runner_path(tmp_path):
    staged = tmp_path / "reference-tokenizers"
    rt.stage_gpt2(staged)
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="reftok-gpt2",
        arch_code=ARCH_LM,
        train_code=TRAIN_REFTOK_GPT2,
        reference_tokenizer_dir=str(staged),
    )
    assert proc.returncode == 0, proc.stderr
    probe = json.loads((artifacts / "miner_probe.json").read_text(encoding="utf-8"))
    assert probe["n_vocab"] == rt.GPT2_VOCAB_SIZE
    assert probe["roundtrip"] == "prism online loss"
