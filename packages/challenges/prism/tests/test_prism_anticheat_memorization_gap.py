"""Anti-memorization / held-out-secrecy anti-cheat negatives (VAL-CHEAT-007/008/009).

These drive the REAL forced-init re-execution runner to produce trained weights, then run the
challenge held-out scorer (``evaluator/heldout.py``) on a SECRET val split the miner never sees.
They prove (a) the held-out split is not exposed via ``PrismContext``; (b) a hardcoded/constant
emitter earns no held-out improvement; (c) a genuine memorizer (overfits the exposed train split)
is flagged by the train-vs-held-out gap, while a benign learner is not; and (d) the gap uses the
CONVERGED (final-checkpoint) train bpb as its train reference, which is strictly more sensitive
than the curve-averaged prequential AUC (closing the false-negative hole).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest
from test_prism_harness_online_loss import (
    ARCH_LM,
    TRAIN_LEARN,
    _read_manifest,
    _run_runner,
)

from prism_challenge.evaluator.heldout import compute_heldout_metrics
from prism_challenge.evaluator.interface import PrismContext

# A hardcoded "lookup"/constant emitter: forward IGNORES the input tokens AND the (trainable)
# weights, always emitting the same logits. Training cannot change its output, so it can earn NO
# held-out improvement over the forced-random-init twin (VAL-CHEAT-008).
ARCH_HARDCODE = """
import torch
from torch import nn


class HardcodeLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)
        self.register_buffer("hardcoded", torch.zeros(vocab))
        with torch.no_grad():
            self.hardcoded[97] = 30.0

    def forward(self, tokens):
        b, t = tokens.shape
        base = self.hardcoded.view(1, 1, -1).expand(b, t, self.vocab)
        # A zeroed grad path keeps the miner's backward() working while contributing nothing:
        # the output (and thus the held-out bpb) is fixed regardless of training or input.
        grad_path = self.head(self.emb(tokens)) * 0.0
        return base + grad_path


def build_model(ctx):
    return HardcodeLM(ctx.vocab_size)
"""

# A NARROW train distribution (only bytes a/b/space). A model that overfits ("memorizes") it
# predicts a DISJOINT held-out val (w/x/y/z/space) badly => large train-vs-held-out gap.
_MEM_TRAIN_TEXT = "aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb"
_MEM_VAL_TEXT = "wxyz wxyz wxyz wxyz wxyz wxyz wxyz wxyz"


def _benign_text(i: int) -> str:
    return f"the locked fineweb edu sample sentence number {i} about science and learning"


def _jsonl(prefix: str, text_fn: Callable[[int], str], n: int) -> str:
    return "".join(json.dumps({"id": f"{prefix}-{i}", "text": text_fn(i)}) + "\n" for i in range(n))


def _write_val(root: Path, text_fn: Callable[[int], str], n: int = 60) -> Path:
    val_dir = root / "val-data"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "val-00000.jsonl").write_text(_jsonl("val", text_fn, n), encoding="utf-8")
    return val_dir


def _host_ctx() -> PrismContext:
    # Mirrors the runner payload context for the _run_runner default fixture (vocab 128, seq 16).
    return PrismContext(vocab_size=128, sequence_length=16, seed=1337, max_parameters=5_000_000)


def test_anticheat_heldout_split_not_exposed_to_miner_context() -> None:
    # VAL-CHEAT-007 (static surface): the PrismContext threaded into the miner scripts exposes ONLY
    # the read-only train split (data_dir); it carries no val/test handle/path the miner could read.
    ctx = _host_ctx()
    field_names = set(vars(ctx))
    assert "data_dir" in field_names
    for forbidden in ("val_dir", "val_data_dir", "test_dir", "test_data_dir", "heldout_dir"):
        assert forbidden not in field_names
        assert not hasattr(ctx, forbidden)
    # The held-out scorer reads val from a SEPARATE host path argument, never from ctx.data_dir.
    assert "val_data_dir" in compute_heldout_metrics.__code__.co_varnames
    assert "train_data_dir" in compute_heldout_metrics.__code__.co_varnames


def test_anticheat_hardcoded_outputs_earn_no_heldout_improvement(tmp_path) -> None:
    # VAL-CHEAT-008: a constant/lookup emitter (output independent of input AND weights) cannot beat
    # the forced-random-init twin on the SECRET val => held-out delta ~ 0 (no advantage), unlike a
    # genuine learner which improves over the twin.
    val_dir = _write_val(tmp_path / "hard", _benign_text)

    hard_proc, hard_art = _run_runner(
        tmp_path, run_name="hardcode", arch_code=ARCH_HARDCODE, train_code=TRAIN_LEARN
    )
    assert hard_proc.returncode == 0, hard_proc.stderr
    hard = compute_heldout_metrics(
        files={"architecture.py": ARCH_HARDCODE, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=hard_art / "trained_state.pt",
        val_data_dir=val_dir,
        train_bpb=1.0,
    )
    assert hard is not None
    # Hardcoding yields no held-out improvement over the random-init twin.
    assert hard.heldout_delta == pytest.approx(0.0, abs=1e-6)

    learn_proc, learn_art = _run_runner(
        tmp_path, run_name="genuine", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert learn_proc.returncode == 0, learn_proc.stderr
    learn = compute_heldout_metrics(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=learn_art / "trained_state.pt",
        val_data_dir=val_dir,
        train_bpb=1.0,
    )
    assert learn is not None
    # A genuine learner DOES improve on the unseen val (delta clearly above the hardcoded baseline).
    assert learn.heldout_delta > hard.heldout_delta


def test_anticheat_memorizer_flagged_via_converged_gap(tmp_path) -> None:
    # VAL-CHEAT-009 (positive): a model trained on a NARROW exposed train split, evaluated on a
    # disjoint SECRET val, shows a large train-vs-held-out gap measured against the CONVERGED
    # (final-checkpoint) train bpb => flagged as memorization.
    train_shard = _jsonl("doc", lambda _i: _MEM_TRAIN_TEXT, 60)
    val_dir = _write_val(tmp_path / "memflag", lambda _i: _MEM_VAL_TEXT)

    proc, artifacts = _run_runner(
        tmp_path,
        run_name="memorizer",
        arch_code=ARCH_LM,
        train_code=TRAIN_LEARN,
        data_files={"train-00000.jsonl": train_shard},
    )
    assert proc.returncode == 0, proc.stderr
    train_dir = artifacts.parent / "data"

    result = compute_heldout_metrics(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=val_dir,
        train_data_dir=train_dir,
        train_bpb=_read_manifest(artifacts)["score"]["prequential_bpb"],
    )
    assert result is not None
    assert result.gap_basis == "converged"
    assert result.train_bpb_converged is not None
    # Converged-on-train fits much better than held-out val => excessive gap => flagged.
    assert result.train_bpb_converged < result.val_bpb_trained
    assert result.train_heldout_gap == pytest.approx(
        result.val_bpb_trained - result.train_bpb_converged
    )
    assert result.train_heldout_gap > 1.0
    assert result.memorization_flag is True


def test_anticheat_benign_learner_not_flagged_via_converged_gap(tmp_path) -> None:
    # VAL-CHEAT-009 (negative / VAL-SCORE-009): a generalizing learner on a same-distribution
    # train/val pair has converged train bpb ~ val bpb (small gap) => NOT flagged.
    train_shard = _jsonl("doc", _benign_text, 60)
    val_dir = _write_val(tmp_path / "benign", _benign_text)

    proc, artifacts = _run_runner(
        tmp_path,
        run_name="benign",
        arch_code=ARCH_LM,
        train_code=TRAIN_LEARN,
        data_files={"train-00000.jsonl": train_shard},
    )
    assert proc.returncode == 0, proc.stderr
    train_dir = artifacts.parent / "data"

    result = compute_heldout_metrics(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=val_dir,
        train_data_dir=train_dir,
        train_bpb=_read_manifest(artifacts)["score"]["prequential_bpb"],
    )
    assert result is not None
    assert result.gap_basis == "converged"
    assert result.train_heldout_gap is not None
    assert result.train_heldout_gap < 1.0
    assert result.memorization_flag is False


def test_anticheat_converged_gap_more_sensitive_than_prequential(tmp_path) -> None:
    # The hardening: the prequential (curve-averaged) AUC train bpb is inflated by early high-loss
    # steps, SHRINKING the gap and risking a missed memorizer. The converged (final-checkpoint)
    # train reference is strictly lower => a strictly larger gap => catches memorizers the
    # prequential reference can miss.
    train_shard = _jsonl("doc", lambda _i: _MEM_TRAIN_TEXT, 60)
    val_dir = _write_val(tmp_path / "sensit", lambda _i: _MEM_VAL_TEXT)

    proc, artifacts = _run_runner(
        tmp_path,
        run_name="sensitivity",
        arch_code=ARCH_LM,
        train_code=TRAIN_LEARN,
        data_files={"train-00000.jsonl": train_shard},
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    prequential_train_bpb = manifest["score"]["prequential_bpb"]
    train_dir = artifacts.parent / "data"

    common = dict(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=val_dir,
        train_bpb=prequential_train_bpb,
    )
    converged = compute_heldout_metrics(train_data_dir=train_dir, **common)
    prequential = compute_heldout_metrics(**common)
    assert converged is not None and prequential is not None

    # The converged final-checkpoint train bpb is strictly below the inflated prequential AUC.
    assert converged.train_bpb_converged is not None
    assert converged.train_bpb_converged < prequential_train_bpb
    # ... so the converged gap is strictly larger (more sensitive) than the prequential-reference
    # gap, and it flags the memorizer.
    assert prequential.gap_basis == "prequential"
    assert prequential.train_heldout_gap is not None
    assert converged.train_heldout_gap is not None
    assert converged.train_heldout_gap > prequential.train_heldout_gap
    assert converged.memorization_flag is True
    assert math.isfinite(converged.train_heldout_gap)
