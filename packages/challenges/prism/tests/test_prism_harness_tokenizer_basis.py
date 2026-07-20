from __future__ import annotations

from pathlib import Path

from test_prism_harness_online_loss import (
    ARCH_LM,
    TRAIN_LEARN,
    _read_manifest,
    _run_runner,
)

from prism_challenge.evaluator.heldout import compute_heldout_metrics
from prism_challenge.evaluator.interface import PrismContext

# A miner that trains with its OWN in-code tokenizer (NOT the challenge byte-level default), so the
# challenge's prequential TRAIN bpb is measured on a native-tokenizer basis (VAL-CONTRACT-021).
TRAIN_TOKENIZER = """
import torch
import torch.nn.functional as F


class CharTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    tok = CharTokenizer()
    for batch in ctx.iter_train_batches(model, batch_size=1, tokenizer=tok):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
"""

VAL_LINE = '{"id": "val-%d", "text": "the locked fineweb-edu train split sample sentence %d"}\n'


def _write_val_split(root: Path, lines: int = 40) -> Path:
    val_dir = root / "val-data"
    val_dir.mkdir(parents=True)
    (val_dir / "val-00000.jsonl").write_text(
        "".join(VAL_LINE % (i, i) for i in range(lines)), encoding="utf-8"
    )
    return val_dir


def _host_ctx() -> PrismContext:
    return PrismContext(vocab_size=128, sequence_length=16, seed=1337, max_parameters=5_000_000)


def test_harness_runner_records_tokenizer_basis_when_tokenizer_used(tmp_path):
    # The runner records that the prequential TRAIN bpb was measured on a (native) tokenizer basis.
    proc, artifacts = _run_runner(
        tmp_path, run_name="tokbasis", arch_code=ARCH_LM, train_code=TRAIN_TOKENIZER
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["metrics"]["train_bpb_basis"] == "tokenizer"


def test_harness_runner_records_byte_basis_without_tokenizer(tmp_path):
    # With no tokenizer the challenge instrument feeds raw UTF-8 bytes: the TRAIN basis is "bytes".
    proc, artifacts = _run_runner(
        tmp_path, run_name="bytebasis", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["metrics"]["train_bpb_basis"] == "bytes"


def test_harness_tokenizer_learner_not_memorization_flagged(tmp_path):
    # DEFECT 2 / VAL-SCORE-009 + VAL-SCORE-004: a benign tokenizer-using learner is NOT
    # memorization-flagged even with a tiny train bpb, because the host measures val on the byte
    # basis and the cross-basis gap is not applied (basis mismatch).
    proc, artifacts = _run_runner(
        tmp_path, run_name="tokmem", arch_code=ARCH_LM, train_code=TRAIN_TOKENIZER
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    val_dir = _write_val_split(tmp_path / "tokmem")

    result = compute_heldout_metrics(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_TOKENIZER},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=val_dir,
        train_bpb=0.0,  # pretend perfect train "memorization" to provoke a large gap
        train_bpb_basis=manifest["metrics"]["train_bpb_basis"],
    )
    assert result is not None
    assert result.memorization_flag is False
    assert result.train_heldout_gap is None
