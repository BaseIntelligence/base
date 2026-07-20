from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import (
    _CONTAINER_EVAL_SCRIPT,
    PrismContainerEvaluator,
)
from prism_challenge.evaluator.heldout import compute_heldout_metrics
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME
from prism_challenge.evaluator.source_similarity import SourceFile

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

VAL_LINE = '{"id": "val-%d", "text": "the locked fineweb edu held out sample sentence %d"}\n'
LOCKED_LINE = '{"id": "doc-%d", "text": "the locked fineweb-edu train split sample sentence %d"}\n'


def _stage_val(root: Path, lines: int = 40) -> Path:
    val_dir = root / "val-data"
    val_dir.mkdir(parents=True, exist_ok=True)
    (val_dir / "val-00000.jsonl").write_text(
        "".join(VAL_LINE % (i, i) for i in range(lines)), encoding="utf-8"
    )
    return val_dir


def _touch_sentinel(path: str) -> int:
    """Module-level so a pickle ``__reduce__`` can reference it by import path (the RCE payload)."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("pwned")
    return 0


class _Exploit:
    """A malicious object whose ``__reduce__`` runs code on unpickle (weights_only must refuse)."""

    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (_touch_sentinel, (self.sentinel,))


def _benign_trained_state(vocab: int) -> dict:
    import torch
    from torch import nn

    class TinyLM(nn.Module):
        def __init__(self, vocab: int) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, 8)
            self.head = nn.Linear(8, vocab)

        def forward(self, tokens):  # type: ignore[no-untyped-def]
            return self.head(self.emb(tokens))

    torch.manual_seed(1337)
    model = TinyLM(vocab)
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _v2_manifest(*, vocab: int, record_trained_state: bool) -> dict:
    covered_bytes = 1200
    bpb = 6.0
    sum_nll_nats = bpb * covered_bytes * math.log(2.0)
    bits = sum_nll_nats / math.log(2.0)
    manifest: dict = {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": "rce-test",
        "run_id": "prism-reexec-rce-test",
        "mode": "gpu_proxy_eval",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": [4.5, 4.0, 3.5],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "sum_neg_log2_likelihood_bits": bits,
            "cumulative_codelength_bits": bits,
            "covered_bytes": covered_bytes,
            "total_bytes_covered": covered_bytes,
            "predicted_tokens": 1100,
            "tokens_seen": 1100,
            "prequential_bpb": bpb,
            "bits_per_byte": bpb,
            "step0_loss": 4.5,
            "consumed_batches": 3,
            "random_init_baseline_nats": math.log(vocab),
            "nan_inf_batches": 0,
            "train_bpb_basis": "bytes",
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "score": {
            "schema": "prism_score.v2",
            "primary_metric": "heldout_delta",
            "secondary_metric": "prequential_bpb",
            "emission_ranking": "heldout_primary_bpb_secondary",
            "prequential_bpb": bpb,
            "bits_per_byte": bpb,
            "final_score": 1.0 / (1.0 + bpb),
            "lower_is_better": True,
            "tie_breaker": "prequential_bpb",
        },
        "artifacts": {},
        "miner_reported_ignored": True,
    }
    if record_trained_state:
        manifest["artifacts"]["trained_state"] = "trained_state.pt"
    return manifest


def _evaluator(val_dir: Path, artifact_root: Path) -> PrismContainerEvaluator:
    settings = PrismSettings(
        base_eval_val_data_dir=str(val_dir),
        base_eval_artifact_root=artifact_root,
    )
    ctx = PrismContext(vocab_size=128, sequence_length=16, seed=1337, max_parameters=5_000_000)
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


def test_anticheat_heldout_refuses_hostile_pickle_and_fails_safe(tmp_path):
    # DEFECT 1a: a hostile pickle planted at trained_state.pt must NOT execute on the host
    # (weights_only refusal) and the held-out step must fail safe (return None, run still scores).
    val_dir = _stage_val(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    sentinel = tmp_path / "PWNED"
    import torch

    torch.save(_Exploit(str(sentinel)), artifacts / "trained_state.pt")

    ctx = PrismContext(vocab_size=128, sequence_length=16, seed=1337, max_parameters=5_000_000)
    result = compute_heldout_metrics(
        files={"architecture.py": HELDOUT_ARCH, "training.py": HELDOUT_TRAIN},
        entrypoint="architecture.py",
        ctx=ctx,
        trained_state_path=artifacts / "trained_state.pt",
        val_data_dir=val_dir,
        train_bpb=6.0,
    )
    assert result is None
    assert not sentinel.exists(), "hostile __reduce__ executed: weights_only refusal missing"


def test_anticheat_heldout_skips_trained_state_not_recorded_in_manifest(tmp_path):
    # DEFECT 1b: the host scorer must only read the trained_state artifact the CHALLENGE-AUTHORED
    # manifest recorded for THIS run; a bare is_file() on the miner-writable artifacts_dir is an
    # RCE sink. With a (benign) file present but NOT recorded, the held-out is skipped entirely.
    val_dir = _stage_val(tmp_path)
    artifact_root = tmp_path / "artifact-root"
    evaluator = _evaluator(val_dir, artifact_root)
    artifact_output = artifact_root / "rce-test" / "attempt-1"
    artifact_output.mkdir(parents=True)

    import torch

    torch.save(_benign_trained_state(128), artifact_output / "trained_state.pt")

    manifest = _v2_manifest(vocab=128, record_trained_state=False)
    files = (
        SourceFile("architecture.py", HELDOUT_ARCH, "h1"),
        SourceFile("training.py", HELDOUT_TRAIN, "h2"),
    )
    augmented = evaluator._augment_with_heldout(
        manifest,
        files=files,
        architecture_entrypoint="architecture.py",
        build_model_symbol="build_model",
        artifact_output=artifact_output,
    )
    assert "heldout_delta" not in augmented["metrics"]
    assert "val_bpb_trained" not in augmented["metrics"]


def test_anticheat_heldout_reads_trained_state_when_manifest_records_it(tmp_path):
    # Positive control: when the manifest DOES record the artifact, the held-out is computed.
    val_dir = _stage_val(tmp_path)
    artifact_root = tmp_path / "artifact-root"
    evaluator = _evaluator(val_dir, artifact_root)
    artifact_output = artifact_root / "rce-test" / "attempt-1"
    artifact_output.mkdir(parents=True)

    import torch

    torch.save(_benign_trained_state(128), artifact_output / "trained_state.pt")

    manifest = _v2_manifest(vocab=128, record_trained_state=True)
    files = (
        SourceFile("architecture.py", HELDOUT_ARCH, "h1"),
        SourceFile("training.py", HELDOUT_TRAIN, "h2"),
    )
    augmented = evaluator._augment_with_heldout(
        manifest,
        files=files,
        architecture_entrypoint="architecture.py",
        build_model_symbol="build_model",
        artifact_output=artifact_output,
    )
    assert "heldout_delta" in augmented["metrics"]
    assert math.isfinite(augmented["metrics"]["val_bpb_trained"])


def _drive_runner_with_planted_state(
    tmp_path: Path, *, run_name: str, train_code: str, planted: bytes
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the challenge runner after PRE-PLANTING a hostile trained_state.pt in artifacts_dir."""
    root = tmp_path / run_name
    project = root / "project"
    project.mkdir(parents=True)
    (project / "architecture.py").write_text(HELDOUT_ARCH, encoding="utf-8")
    (project / "training.py").write_text(train_code, encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "train-00000.jsonl").write_text(
        "".join(LOCKED_LINE % (i, i) for i in range(40)), encoding="utf-8"
    )

    artifacts = root / "artifacts"
    artifacts.mkdir()
    # The miner-writable artifacts_dir already holds a hostile file before the runner writes.
    (artifacts / "trained_state.pt").write_bytes(planted)

    payload = {
        "submission_id": "sub-rce",
        "architecture_entrypoint": "architecture.py",
        "training_entrypoint": "training.py",
        "build_model_symbol": "build_model",
        "train_symbol": "train",
        "execution_mode": "gpu_proxy_eval",
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "context": {
            "vocab_size": 128,
            "sequence_length": 16,
            "max_layers": 2,
            "max_parameters": 5_000_000,
            "seed": 1337,
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts),
            "reference_tokenizer_dir": str(root / "tok"),
            "token_budget": None,
            "step_budget": None,
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


def test_anticheat_runner_removes_planted_trained_state_on_zero_forward(tmp_path):
    # DEFECT 1c: a miner that makes state_dict() unsavable (here: a zero-forward run that skips the
    # best-effort save) must NOT be able to leave a pre-planted hostile trained_state.pt behind. The
    # runner unconditionally OWNS the path: the planted file is gone and the manifest records None.
    zero_batch = "def train(ctx):\n    ctx.build_model()\n    return\n"
    proc, artifacts = _drive_runner_with_planted_state(
        tmp_path, run_name="planted-zero", train_code=zero_batch, planted=b"HOSTILE-PICKLE-BYTES"
    )
    # A zero-forward run is flag-failed (non-zero exit) but still authors the manifest.
    assert proc.returncode != 0
    assert not (artifacts / "trained_state.pt").exists()
    manifest = json.loads((artifacts / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))
    assert manifest["artifacts"]["trained_state"] is None


def test_anticheat_runner_overwrites_planted_trained_state_on_real_train(tmp_path):
    # A genuine training run overwrites the planted file with challenge-owned weights the host can
    # safely load with weights_only=True (no leftover hostile pickle).
    proc, artifacts = _drive_runner_with_planted_state(
        tmp_path,
        run_name="planted-learn",
        train_code=HELDOUT_TRAIN,
        planted=b"HOSTILE-PICKLE-BYTES",
    )
    assert proc.returncode == 0, proc.stderr
    state_file = artifacts / "trained_state.pt"
    assert state_file.is_file()
    manifest = json.loads((artifacts / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))
    assert manifest["artifacts"]["trained_state"] == "trained_state.pt"
    import torch

    loaded = torch.load(state_file, map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict) and loaded
