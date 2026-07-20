"""Resume-from-public-checkpoint + validator-reported metrics.

Covers VAL-PRISM-023 (a reassigned run resumes from the last public HF checkpoint), VAL-PRISM-024
(a first attempt with no checkpoint starts fresh, no download), VAL-PRISM-025 (resume preserves
forced-random-init + step-0 anti-cheat), and VAL-PRISM-026 (the validator container reports only the
online-loss stream + trained_state, never the secret-val-derived held-out delta). HuggingFace is the
in-memory mock publisher (no real network).
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from base.challenge_sdk.executor import DockerRunSpec

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.checkpoint_publisher import (
    CheckpointUpload,
    MockCheckpointPublisher,
    PublishedCheckpoint,
    revision_for,
)
from prism_challenge.evaluator.checkpoints import checkpoint_workspace, persist_checkpoint
from prism_challenge.evaluator.container import PrismContainerEvaluator
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.evaluator.scoring import score_prequential_bpb
from prism_challenge.evaluator.source_similarity import SourceFile

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

# A resume-aware miner: it records the resume context the host handed it (proving the resume payload
# reached the in-container runner), listing the staged public-checkpoint files when present, then
# trains normally on the challenge instrument.
RESUME_AWARE_TRAIN = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    resume_dir = ctx.resume_checkpoint_dir
    resume_files = (
        sorted(p.name for p in pathlib.Path(resume_dir).glob("*"))
        if resume_dir and pathlib.Path(resume_dir).is_dir()
        else []
    )
    pathlib.Path(ctx.artifacts_dir, "resume_context.json").write_text(
        json.dumps(
            {
                "is_resume": ctx.is_resume,
                "attempt": ctx.attempt,
                "resume_checkpoint_dir": resume_dir,
                "resume_files": resume_files,
            }
        )
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

# A "smuggled pretrained weights" model delivered on resume: its forward biases its prediction
# toward the NEXT input token (a lookahead/memorization cheat), so step-0 loss sits well below the
# random baseline (0.5*ln(vocab)). The challenge step-0 anomaly must still catch it on a resumed run
# and zero the score (VAL-PRISM-025). The bias is modest so the bpb stays finite and positive.
LOOKAHEAD_ARCH = """
import torch
from torch import nn


class CheatLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        rows, cols = tokens.shape
        logits = torch.zeros(rows, cols, self.vocab)
        if cols >= 2:
            cols_idx = torch.arange(cols - 1)
            for r in range(rows):
                logits[r, cols_idx, tokens[r, 1:]] = 3.0
        return logits


def build_model(ctx):
    return CheatLM(ctx.vocab_size)
"""

CHEAT_TRAIN = """
def train(ctx):
    model = ctx.build_model()
    for _ in ctx.iter_train_batches(model, batch_size=1):
        pass
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


def _evaluator(tmp_path: Path, *, checkpoint_publisher=None) -> PrismContainerEvaluator:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'resume.sqlite3'}",
        shared_token="secret",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
    )
    ctx = PrismContext(vocab_size=64, sequence_length=16, seed=1234, step_budget=24)
    return PrismContainerEvaluator(
        settings=settings, ctx=ctx, checkpoint_publisher=checkpoint_publisher
    )


class _SpyPublisher:
    """Wraps a MockCheckpointPublisher to record download calls (no real network)."""

    def __init__(self, inner: MockCheckpointPublisher) -> None:
        self.inner = inner
        self.download_calls: list[str] = []

    def publish(self, upload: CheckpointUpload) -> PublishedCheckpoint:
        return self.inner.publish(upload)

    def download(self, checkpoint_ref: str, dest_dir: Path) -> Path:
        self.download_calls.append(checkpoint_ref)
        return self.inner.download(checkpoint_ref, dest_dir)


def _publish_checkpoint(
    inner: MockCheckpointPublisher,
    tmp_path: Path,
    *,
    submission_id: str = "sub-res",
    attempt: int = 1,
) -> PublishedCheckpoint:
    workspace = checkpoint_workspace(
        tmp_path / "ckpt-src", submission_id=submission_id, attempt=attempt
    )
    current = persist_checkpoint(
        workspace,
        state_files={"model.pt": b"warm-start-weights"},
        code_hash="codehash",
        arch_hash="archhash",
        recipe_fingerprint="recipe",
        created_at="2026-06-27T00:00:00Z",
    )
    names = tuple(sorted(p.name for p in current.glob("*")))
    upload = CheckpointUpload(
        submission_id=submission_id,
        attempt=attempt,
        checkpoint_dir=current,
        files=names,
        revision=revision_for(submission_id, attempt, names),
    )
    return inner.publish(upload)


def _read_resume_context(result) -> dict:
    return json.loads(
        (Path(result.artifact_output_path) / "resume_context.json").read_text(encoding="utf-8")
    )


# --- VAL-PRISM-023: a reassigned run resumes from the last public HF checkpoint -------------------


def test_reassigned_run_downloads_and_stages_published_checkpoint(tmp_path, monkeypatch):
    inner = MockCheckpointPublisher()
    published = _publish_checkpoint(inner, tmp_path)
    spy = _SpyPublisher(inner)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    evaluator = _evaluator(tmp_path, checkpoint_publisher=spy)
    files = _source_files(TINY_ARCH, RESUME_AWARE_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-res",
        code=TINY_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
        attempt=2,
        resume_checkpoint_ref=published.checkpoint_ref,
    )

    # The mocked publisher download was called with the prior public checkpoint ref...
    assert spy.download_calls == [published.checkpoint_ref]
    # ...and the run resumes from it (staged + signalled to the in-container runner), not scratch.
    context = _read_resume_context(result)
    assert context["is_resume"] is True
    assert context["attempt"] == 2
    assert context["resume_checkpoint_dir"]
    assert "model.pt" in context["resume_files"]


# --- VAL-PRISM-024: a from-scratch run with no checkpoint starts fresh ----------------------------


def test_first_attempt_without_checkpoint_starts_fresh(tmp_path, monkeypatch):
    spy = _SpyPublisher(MockCheckpointPublisher())
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    evaluator = _evaluator(tmp_path, checkpoint_publisher=spy)
    files = _source_files(TINY_ARCH, RESUME_AWARE_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-fresh",
        code=TINY_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
        attempt=1,
        resume_checkpoint_ref=None,
    )

    # No publisher download is attempted and no resume dir is staged: a clean from-scratch run.
    assert spy.download_calls == []
    context = _read_resume_context(result)
    assert context["is_resume"] is False
    assert context["attempt"] == 1
    assert context["resume_checkpoint_dir"] is None
    assert context["resume_files"] == []


def test_fresh_run_clears_planted_trained_state(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    captured: list[DockerRunSpec] = []
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir, captured_specs=captured),
    )
    evaluator = _evaluator(tmp_path)
    # Plant a hostile pickle at the trained_state path BEFORE the run.
    artifact_dir = tmp_path / "artifacts" / "sub-plant" / "attempt-1"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "trained_state.pt").write_bytes(b"planted-not-a-real-state")
    files = _source_files(TINY_ARCH, TINY_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-plant",
        code=TINY_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
        attempt=1,
    )
    # The challenge re-authored trained_state from its own forced-init run (not the planted one).
    trained = Path(result.artifact_output_path) / "trained_state.pt"
    assert trained.is_file()
    assert trained.read_bytes() != b"planted-not-a-real-state"
    assert result.run_manifest["metrics"]["step0_loss"] is not None


# --- VAL-PRISM-025: resume preserves forced-random-init + step-0 anti-cheat -----------------------


def test_resume_preserves_anti_cheat_shape(tmp_path, monkeypatch):
    inner = MockCheckpointPublisher()
    published = _publish_checkpoint(inner, tmp_path)
    spy = _SpyPublisher(inner)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    evaluator = _evaluator(tmp_path, checkpoint_publisher=spy)
    files = _source_files(TINY_ARCH, RESUME_AWARE_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-res",
        code=TINY_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
        attempt=2,
        resume_checkpoint_ref=published.checkpoint_ref,
    )
    manifest = result.run_manifest
    assert manifest is not None
    # Scored identically in shape: the challenge manifest + anti-cheat block are still authored.
    assert "anti_cheat" in manifest
    assert manifest["anti_cheat"]["step0_anomaly"] is False
    assert manifest["metrics"]["step0_loss"] is not None
    assert score_prequential_bpb(manifest).final_score > 0.0
    # The resume payload genuinely reached the in-container runner.
    context = _read_resume_context(result)
    assert context["is_resume"] is True
    assert "model.pt" in context["resume_files"]


def test_resume_cannot_evade_step0_anomaly(tmp_path, monkeypatch):
    inner = MockCheckpointPublisher()
    published = _publish_checkpoint(inner, tmp_path)
    spy = _SpyPublisher(inner)
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    evaluator = _evaluator(tmp_path, checkpoint_publisher=spy)
    files = _source_files(LOOKAHEAD_ARCH, CHEAT_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-res",
        code=LOOKAHEAD_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
        attempt=2,
        resume_checkpoint_ref=published.checkpoint_ref,
    )
    manifest = result.run_manifest
    assert manifest is not None
    # A sub-baseline step-0 (smuggled-weights signature) is caught even on a resumed run.
    assert manifest["anti_cheat"]["step0_anomaly"] is True
    assert manifest["score"]["anti_cheat_multiplier"] == 0.0
    assert score_prequential_bpb(manifest).final_score == 0.0


# --- VAL-PRISM-026: validator reports only the online-loss stream + trained_state -----------------


def test_validator_container_manifest_reports_only_online_loss_and_trained_state(
    tmp_path, monkeypatch
):
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    # No val split is configured (the default path does not exist), so the host held-out is skipped:
    # the validator's container manifest carries online_loss + trained_state but no held-out delta.
    evaluator = _evaluator(tmp_path)
    files = _source_files(TINY_ARCH, TINY_TRAIN)
    result = evaluator.evaluate(
        submission_id="sub-report",
        code=TINY_ARCH,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend="base_gpu",
        files=files,
    )
    manifest = result.run_manifest
    assert manifest is not None
    assert len(manifest["metrics"]["online_loss"]) > 0
    assert manifest["artifacts"]["trained_state"] == "trained_state.pt"
    # The secret-val-derived held-out delta is NOT computed/reported by the validator container.
    for key in ("heldout_delta", "held_out_delta"):
        assert key not in manifest["metrics"]
        assert key not in manifest["score"]
    # The trained_state artifact the master will later read for held-out is persisted by the run.
    assert (Path(result.artifact_output_path) / "trained_state.pt").is_file()
