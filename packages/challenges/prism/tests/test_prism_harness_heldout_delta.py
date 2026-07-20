from __future__ import annotations

import math
from pathlib import Path

import pytest
from test_prism_harness_online_loss import (
    ARCH_LM,
    TRAIN_LEARN,
    _read_manifest,
    _run_runner,
)

from prism_challenge.evaluator.heldout import compute_heldout_metrics, val_split_present
from prism_challenge.evaluator.interface import PrismContext

# A no-op loop: iterates the instrumented batches (predict-then-train) but never updates the model,
# so the online loss stays flat at the random-init baseline.
TRAIN_NOOP = """
def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
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
    # Mirrors the runner payload context for the _run_runner default fixture (vocab 128, seq 16).
    return PrismContext(vocab_size=128, sequence_length=16, seed=1337, max_parameters=5_000_000)


def test_harness_runner_persists_trained_state_artifact(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="trained-state", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    # The runner persists the trained weights so the HOST scorer can run held-out on the secret val.
    state_file = artifacts / "trained_state.pt"
    assert state_file.is_file()
    manifest = _read_manifest(artifacts)
    assert manifest["artifacts"]["trained_state"] == "trained_state.pt"


def test_harness_heldout_delta_computed_on_secret_val(tmp_path):
    # VAL-HARNESS-015 / VAL-SCORE-007: the challenge computes the held-out delta on a SECRET val
    # split that is SEPARATE from the miner's train data_dir (never exposed to the miner script).
    proc, artifacts = _run_runner(
        tmp_path, run_name="heldout", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    train_bpb = manifest["score"]["prequential_bpb"]

    val_dir = _write_val_split(tmp_path / "heldout")
    assert val_split_present(val_dir)

    result = compute_heldout_metrics(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=val_dir,
        train_bpb=train_bpb,
    )
    assert result is not None
    # Both component bpb values are finite/positive and the delta = random_init - trained.
    assert result.val_bpb_trained > 0.0 and math.isfinite(result.val_bpb_trained)
    assert result.val_bpb_random_init > 0.0 and math.isfinite(result.val_bpb_random_init)
    assert result.heldout_delta == pytest.approx(
        result.val_bpb_random_init - result.val_bpb_trained
    )
    # A genuinely-learning model improves over the random-init twin on held-out val: delta > 0.
    assert result.heldout_delta > 0.0
    assert result.val_bpb_trained < result.val_bpb_random_init
    # A non-memorizing learner on same-distribution val is not flagged.
    assert result.memorization_flag is False


def test_harness_heldout_skipped_without_val_split(tmp_path):
    # No secret val split -> held-out gracefully skipped (the run still scores on prequential bpb).
    proc, artifacts = _run_runner(
        tmp_path, run_name="noval", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    missing = tmp_path / "noval" / "does-not-exist"
    assert not val_split_present(missing)
    result = compute_heldout_metrics(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=missing,
        train_bpb=1.0,
    )
    assert result is None


def test_harness_learner_scores_better_than_noop_near_baseline(tmp_path):
    # VAL-SCORE-005 / VAL-SCORE-006: a genuine learner has strictly LOWER prequential bpb than a
    # no-op loop, and the no-op sits at ~ the random-init baseline bpb (flat online-loss curve).
    learn_proc, learn_artifacts = _run_runner(
        tmp_path, run_name="learner", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    noop_proc, noop_artifacts = _run_runner(
        tmp_path, run_name="noop", arch_code=ARCH_LM, train_code=TRAIN_NOOP
    )
    assert learn_proc.returncode == 0, learn_proc.stderr
    assert noop_proc.returncode == 0, noop_proc.stderr
    learn = _read_manifest(learn_artifacts)["metrics"]
    noop = _read_manifest(noop_artifacts)["metrics"]

    learner_bpb = learn["prequential_bpb"]
    noop_bpb = noop["prequential_bpb"]
    assert learner_bpb < noop_bpb

    # The no-op never updates the model, so its online loss is flat at ~ the random-init baseline.
    baseline_bits = noop["random_init_baseline_nats"] / math.log(2.0)
    assert 0.5 * baseline_bits <= noop_bpb <= 2.5 * baseline_bits


def test_harness_heldout_memorization_gap_recorded(tmp_path):
    # VAL-HARNESS-016: the train-vs-held-out gap is recorded; with a train_bpb far below the val
    # bpb the gap is large and the run is flagged.
    proc, artifacts = _run_runner(
        tmp_path, run_name="memgap", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    val_dir = _write_val_split(tmp_path / "memgap")
    # Pretend the online train bpb was ~0 (perfect memorization of train) so the gap is excessive.
    result = compute_heldout_metrics(
        files={"architecture.py": ARCH_LM, "training.py": TRAIN_LEARN},
        entrypoint="architecture.py",
        ctx=_host_ctx(),
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=val_dir,
        train_bpb=0.0,
    )
    assert result is not None
    assert result.train_heldout_gap is not None
    assert result.train_heldout_gap == pytest.approx(result.val_bpb_trained)
    assert result.memorization_flag is True
