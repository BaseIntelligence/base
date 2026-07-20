from __future__ import annotations

import math
from pathlib import Path

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import container as container_mod
from prism_challenge.evaluator.container import PrismContainerEvaluator
from prism_challenge.evaluator.heldout import (
    DEFAULT_HELDOUT_VAL_BYTE_BUDGET,
    compute_heldout_metrics,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.source_similarity import SourceFile

# A tiny token-in/logits-out LM (forward -> [B, T, V]); host-instantiable on CPU.
HELDOUT_ARCH = """
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

HELDOUT_TRAIN = """
def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
"""

VAL_LINE = '{"id": "val-%d", "text": "the locked fineweb edu held out sample sentence number %d"}\n'

SEED = 1337
VOCAB = 128
SEQ = 16


def _host_ctx() -> PrismContext:
    return PrismContext(vocab_size=VOCAB, sequence_length=SEQ, seed=SEED, max_parameters=5_000_000)


def _stage_val(root: Path, *, lines: int) -> Path:
    val_dir = root / "val-data"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "val-00000.jsonl").write_text(
        "".join(VAL_LINE % (i, i) for i in range(lines)), encoding="utf-8"
    )
    return val_dir


def _full_val_bytes(val_dir: Path) -> int:
    import json

    total = 0
    for line in (val_dir / "val-00000.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += len(json.loads(line)["text"].encode("utf-8"))
    return total


def _max_doc_bytes(val_dir: Path) -> int:
    import json

    largest = 0
    for line in (val_dir / "val-00000.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            largest = max(largest, len(json.loads(line)["text"].encode("utf-8")))
    return largest


def _build_tiny_lm():
    import torch
    from torch import nn

    class TinyLM(nn.Module):
        def __init__(self, vocab: int) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, 8)
            self.head = nn.Linear(8, vocab)

        def forward(self, tokens):  # type: ignore[no-untyped-def]
            return self.head(self.emb(tokens))

    torch.manual_seed(SEED)
    return TinyLM(VOCAB)


def _save_state(state: dict, path: Path) -> Path:
    import torch

    torch.save(state, path)
    return path


def _noop_trained_state() -> dict:
    """A trained_state IDENTICAL to the forced-seed random-init twin (a true no-op)."""
    model = _build_tiny_lm()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _learner_trained_state(val_dir: Path) -> dict:
    """A lightly-trained TinyLM so the trained model beats the random-init twin on val."""
    import json

    import torch
    import torch.nn.functional as functional

    model = _build_tiny_lm()
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    text = (val_dir / "val-00000.jsonl").read_text(encoding="utf-8")
    byte_ids = [b % VOCAB for line in text.splitlines() for b in json.loads(line)["text"].encode()]
    for _epoch in range(3):
        for start in range(0, len(byte_ids) - SEQ - 1, SEQ):
            chunk = byte_ids[start : start + SEQ + 1]
            tokens = torch.tensor(chunk[:-1]).view(1, -1)
            targets = torch.tensor(chunk[1:]).view(-1)
            opt.zero_grad()
            logits = model(tokens)
            loss = functional.cross_entropy(logits.reshape(-1, VOCAB), targets)
            loss.backward()
            opt.step()
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def test_harness_heldout_val_byte_budget_bounds_subsample(tmp_path):
    # The held-out eval is capped to a FIXED, deterministic val byte budget: only a stable prefix
    # of the (large) secret val split is scored, so the host-side delta completes within budget.
    val_dir = _stage_val(tmp_path, lines=600)
    full_bytes = _full_val_bytes(val_dir)
    budget = 2048
    assert budget < full_bytes  # the split is genuinely larger than the budget

    state_path = _save_state(_learner_trained_state(val_dir), tmp_path / "trained_state.pt")
    result = compute_heldout_metrics(
        files={"architecture.py": HELDOUT_ARCH, "training.py": HELDOUT_TRAIN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=state_path,
        val_data_dir=val_dir,
        train_bpb=6.0,
        val_byte_budget=budget,
    )
    assert result is not None
    # Only a bounded prefix was scored (budget + at most one boundary-crossing doc), strictly less
    # than the full split. This is what makes the host-side held-out complete LIVE within budget.
    assert result.val_covered_bytes > 0
    assert result.val_covered_bytes <= budget + _max_doc_bytes(val_dir)
    assert result.val_covered_bytes < full_bytes


def test_harness_heldout_delta_deterministic_under_budget(tmp_path):
    # Determinism: same submission + same seed + same budget => identical delta within tolerance.
    val_dir = _stage_val(tmp_path, lines=300)
    state_path = _save_state(_learner_trained_state(val_dir), tmp_path / "trained_state.pt")
    kwargs = dict(
        files={"architecture.py": HELDOUT_ARCH, "training.py": HELDOUT_TRAIN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=state_path,
        val_data_dir=val_dir,
        train_bpb=6.0,
        val_byte_budget=4096,
    )
    first = compute_heldout_metrics(**kwargs)
    second = compute_heldout_metrics(**kwargs)
    assert first is not None and second is not None
    assert first.heldout_delta == pytest.approx(second.heldout_delta, abs=1e-9)
    assert first.val_covered_bytes == second.val_covered_bytes
    assert first.val_bpb_trained == pytest.approx(second.val_bpb_trained, abs=1e-9)


def test_harness_heldout_learner_positive_delta_under_budget(tmp_path):
    # Directional correctness preserved under the bounded budget: a genuine learner beats its own
    # random-init twin on the held-out prefix (delta > 0, tokenizer-agnostic byte denominator).
    val_dir = _stage_val(tmp_path, lines=300)
    state_path = _save_state(_learner_trained_state(val_dir), tmp_path / "trained_state.pt")
    result = compute_heldout_metrics(
        files={"architecture.py": HELDOUT_ARCH, "training.py": HELDOUT_TRAIN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=state_path,
        val_data_dir=val_dir,
        train_bpb=6.0,
        val_byte_budget=4096,
    )
    assert result is not None
    assert result.heldout_delta > 0.0
    assert result.val_bpb_trained < result.val_bpb_random_init
    assert math.isfinite(result.val_bpb_trained) and result.val_bpb_trained > 0.0


def test_harness_heldout_noop_delta_zero_under_budget(tmp_path):
    # A no-op (trained weights == the forced-seed random-init twin) yields ~0 delta: the twin and
    # the trained model are scored over the SAME deterministic val subsample, so they coincide.
    val_dir = _stage_val(tmp_path, lines=300)
    state_path = _save_state(_noop_trained_state(), tmp_path / "trained_state.pt")
    result = compute_heldout_metrics(
        files={"architecture.py": HELDOUT_ARCH, "training.py": HELDOUT_TRAIN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=state_path,
        val_data_dir=val_dir,
        train_bpb=6.0,
        val_byte_budget=4096,
    )
    assert result is not None
    assert result.heldout_delta == pytest.approx(0.0, abs=1e-9)
    assert result.val_bpb_trained == pytest.approx(result.val_bpb_random_init, abs=1e-9)


def test_harness_heldout_metrics_include_val_covered_bytes(tmp_path):
    # The bounded val subsample size is recorded in the manifest metrics (evidence the held-out
    # delta was computed within a fixed compute budget).
    val_dir = _stage_val(tmp_path, lines=300)
    state_path = _save_state(_learner_trained_state(val_dir), tmp_path / "trained_state.pt")
    result = compute_heldout_metrics(
        files={"architecture.py": HELDOUT_ARCH, "training.py": HELDOUT_TRAIN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=state_path,
        val_data_dir=val_dir,
        train_bpb=6.0,
        val_byte_budget=4096,
    )
    assert result is not None
    metrics = result.as_metrics()
    assert metrics["val_covered_bytes"] == result.val_covered_bytes
    assert isinstance(metrics["val_covered_bytes"], int)
    assert metrics["val_covered_bytes"] > 0


def test_container_augment_passes_configured_heldout_budget_and_timeout(tmp_path, monkeypatch):
    # The container scorer plumbs the CONFIGURABLE compute budget + raised timeout into the
    # held-out computation (so a deploy can tune them for the live scorer).
    val_dir = _stage_val(tmp_path, lines=40)
    artifact_root = tmp_path / "artifact-root"
    settings = PrismSettings(
        base_eval_val_data_dir=str(val_dir),
        base_eval_artifact_root=artifact_root,
        base_eval_heldout_val_byte_budget=12345,
        base_eval_heldout_timeout_seconds=321.0,
    )
    ctx = _host_ctx()
    evaluator = PrismContainerEvaluator(settings=settings, ctx=ctx)
    artifact_output = artifact_root / "sub" / "attempt-1"
    artifact_output.mkdir(parents=True)
    _save_state(_noop_trained_state(), artifact_output / "trained_state.pt")

    captured: dict = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(container_mod, "compute_heldout_metrics", _spy)

    manifest = {
        "metrics": {"prequential_bpb": 6.0, "train_bpb_basis": "bytes"},
        "score": {"prequential_bpb": 6.0},
        "anti_cheat": {},
        "artifacts": {"trained_state": "trained_state.pt"},
    }
    files = (
        SourceFile("architecture.py", HELDOUT_ARCH, "h1"),
        SourceFile("training.py", HELDOUT_TRAIN, "h2"),
    )
    evaluator._augment_with_heldout(
        manifest,
        files=files,
        architecture_entrypoint="architecture.py",
        build_model_symbol="build_model",
        artifact_output=artifact_output,
    )
    assert captured["val_byte_budget"] == 12345
    assert captured["timeout_seconds"] == pytest.approx(321.0)


def test_harness_heldout_default_byte_budget_is_bounded():
    # A sane, bounded default budget exists so the host-side held-out completes within budget even
    # when no explicit budget is configured.
    assert isinstance(DEFAULT_HELDOUT_VAL_BYTE_BUDGET, int)
    assert 0 < DEFAULT_HELDOUT_VAL_BYTE_BUDGET <= 4 * 1024 * 1024
