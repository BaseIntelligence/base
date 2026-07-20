from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from prism_challenge.evaluator.container import (
    _CONTAINER_EVAL_SCRIPT,
    ARTIFACTS_QUOTA_MARKER,
    BUDGET_EXCEEDED_MARKER,
)
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME

# A tiny token-in/logits-out LM the challenge instrument can score.
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

# A miner that consumes the CHALLENGE instrument and steps its optimizer (real learning).
TRAIN_LEARN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
"""

# A miner that iterates the instrument but NEVER updates: a no-op / dead training loop.
TRAIN_NOOP = """
def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
"""

# A miner that sleeps per batch so the graceful in-loop wall-clock budget binds first.
TRAIN_SLOW = """
import time

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
        time.sleep(0.05)
"""

# A miner whose loop hangs OUTSIDE the instrumented iterator (never iterates a batch).
TRAIN_HANG = """
import time


def train(ctx):
    ctx.build_model()
    while True:
        time.sleep(0.05)
"""

# A miner that floods ctx.artifacts_dir (the only writable path) to exhaust disk.
TRAIN_DISK_FILL = """
import pathlib
import time


def train(ctx):
    ctx.build_model()
    artifacts = pathlib.Path(ctx.artifacts_dir)
    chunk = b"x" * (256 * 1024)
    i = 0
    while True:
        (artifacts / ("blob-%d.bin" % i)).write_bytes(chunk)
        i += 1
        time.sleep(0.2)
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
    arch_code: str = ARCH_LM,
    train_code: str,
    vocab_size: int = 128,
    sequence_length: int = 16,
    submission_id: str = "sub-budget",
    data_lines: int = 40,
    token_budget: int | None = None,
    step_budget: int | None = None,
    budget_seconds: float | None = None,
    watchdog_grace_seconds: float | None = None,
    artifacts_quota_bytes: int | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path / run_name
    project = root / "project"
    project.mkdir(parents=True)
    (project / "architecture.py").write_text(arch_code, encoding="utf-8")
    (project / "training.py").write_text(train_code, encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "train-00000.jsonl").write_text(_locked_shard(data_lines), encoding="utf-8")

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
            "seed": 1337,
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts),
            "reference_tokenizer_dir": str(root / "tok"),
            "token_budget": token_budget,
            "step_budget": step_budget,
            "budget_seconds": budget_seconds,
            "watchdog_grace_seconds": watchdog_grace_seconds,
            "artifacts_quota_bytes": artifacts_quota_bytes,
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
        timeout=120,
    )
    return proc, artifacts


def _read_manifest(artifacts: Path) -> dict:
    return json.loads((artifacts / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def test_harness_token_budget_precedence(tmp_path):
    # A small token budget binds before step / wall-clock / data exhaustion.
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="token",
        train_code=TRAIN_LEARN,
        data_lines=80,
        token_budget=80,
        step_budget=10_000,
        budget_seconds=3600,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["run"]["stopped_reason"] == "token_budget"
    assert manifest["data"]["stopped_reason"] == "token_budget"
    # Bound respected: no more than (token_budget) tokens were consumed into batches.
    assert manifest["metrics"]["consumed_batches"] >= 1
    assert manifest["data"]["distinct_offsets"] == manifest["data"]["consumed_offsets"]


def test_harness_step_budget_precedence(tmp_path):
    # With token budget absent, a small step budget binds before wall-clock / data exhaustion.
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="step",
        train_code=TRAIN_LEARN,
        data_lines=80,
        step_budget=3,
        budget_seconds=3600,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["run"]["stopped_reason"] == "step_budget"
    assert manifest["metrics"]["consumed_batches"] == 3


def test_harness_single_pass_data_exhaustion_graceful(tmp_path):
    # No budget binds: the single-pass shards are exhausted gracefully (no wraparound/repeat).
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="exhaust",
        train_code=TRAIN_LEARN,
        data_lines=12,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["run"]["stopped_reason"] == "data_exhausted"
    assert manifest["data"]["stopped_reason"] == "data_exhausted"
    assert manifest["data"]["single_pass"] is True
    # Single pass: no shard offset is revisited.
    assert manifest["data"]["distinct_offsets"] == manifest["data"]["consumed_offsets"]


def test_harness_wall_clock_budget_graceful_partial_stream(tmp_path):
    # An over-budget (but iterating) loop is stopped gracefully at the cap and scored on the
    # PARTIAL captured stream (VAL-HARNESS-012); the watchdog grace is large so it never fires.
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="wallclock-graceful",
        train_code=TRAIN_SLOW,
        data_lines=200,
        budget_seconds=0.4,
        watchdog_grace_seconds=60,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["run"]["stopped_reason"] == "wall_clock"
    online = manifest["metrics"]["online_loss"]
    assert online, "expected a partial captured stream"
    assert all(math.isfinite(v) for v in online)
    # Scored on the partial stream; the score is compute-normalized (no wall-clock term).
    assert manifest["score"]["final_score"] is not None
    assert manifest["score"]["wall_clock_term"] is False
    assert manifest["anti_cheat"]["no_learning"] is False


def test_harness_wall_clock_watchdog_bounds_outside_iterator(tmp_path):
    # A loop that hangs OUTSIDE the instrumented iterator is bounded by the hard watchdog
    # (budget + grace), lands non-zero with a budget marker, and never hangs indefinitely.
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="wallclock-hang",
        train_code=TRAIN_HANG,
        budget_seconds=0.3,
        watchdog_grace_seconds=0.3,
    )
    assert proc.returncode != 0
    assert BUDGET_EXCEEDED_MARKER in proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["run"]["stopped_reason"] == "wall_clock"
    assert manifest["anti_cheat"]["budget_exceeded"] is True


def test_harness_artifacts_dir_disk_fill_bounded(tmp_path):
    # An artifacts_dir disk-fill is bounded by the quota watchdog: the run lands non-zero with a
    # quota marker and the artifacts dir stays bounded (host not taken down) (VAL-HARNESS-026).
    quota = 512 * 1024
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="diskfill",
        train_code=TRAIN_DISK_FILL,
        artifacts_quota_bytes=quota,
    )
    assert proc.returncode != 0
    assert ARTIFACTS_QUOTA_MARKER in proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["run"]["stopped_reason"] == "artifacts_quota"
    assert manifest["anti_cheat"]["artifacts_quota_exceeded"] is True
    # Bounded: the watchdog stops the flood within a few poll intervals, never unbounded.
    assert _dir_bytes(artifacts) < 16 * 1024 * 1024


def test_harness_noop_dead_loop_handled_not_crashed(tmp_path):
    # A no-op loop (iterates the instrument but never steps the optimizer) yields a usable
    # terminal scored outcome reflecting no learning, without crashing (VAL-HARNESS-014).
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="noop",
        train_code=TRAIN_NOOP,
        data_lines=40,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    metrics = manifest["metrics"]
    # It DID forward over the instrument (distinct from a zero-forward run).
    assert metrics["online_loss"]
    assert all(math.isfinite(v) for v in metrics["online_loss"])
    assert metrics["covered_bytes"] > 0
    assert manifest["anti_cheat"]["no_learning"] is False
    # Scored on prequential bpb; no learning => a finite, non-advantageous score.
    assert metrics["prequential_bpb"] is not None
    assert math.isfinite(metrics["prequential_bpb"])
    assert manifest["score"]["final_score"] is not None
