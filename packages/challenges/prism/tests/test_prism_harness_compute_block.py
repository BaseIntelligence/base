from __future__ import annotations

from test_prism_harness_online_loss import (
    ARCH_LM,
    TRAIN_LEARN,
    _read_manifest,
    _run_runner,
)

from prism_challenge.evaluator.schemas import COMPUTE_BLOCK_SCHEMA, ComputeBlock
from prism_challenge.evaluator.scoring import score_prequential_bpb


def test_harness_manifest_carries_typed_compute_block(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="compute", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    compute = manifest["compute"]
    # The block parses as the typed compute schema.
    parsed = ComputeBlock.model_validate(compute)
    assert compute["schema"] == COMPUTE_BLOCK_SCHEMA
    # Scored nproc=1 path: gpu_count == 1 (the GPUs leased for this run).
    assert parsed.gpu_count == 1
    assert compute["gpu_count"] == 1
    assert compute["device"]


def test_harness_compute_block_consistent_with_run_block(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="compute-consistent", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    compute = manifest["compute"]
    run = manifest["run"]
    # Internally consistent with run.world_size / run.nproc_per_node for the same eval.
    assert compute["gpu_count"] == run["world_size"]
    assert compute["world_size"] == run["world_size"]
    assert compute["nproc_per_node"] == run["nproc_per_node"]
    assert compute["device"] == run["device"]


def test_harness_compute_gpu_count_not_input_to_final_score(tmp_path):
    proc, artifacts = _run_runner(
        tmp_path, run_name="compute-score", arch_code=ARCH_LM, train_code=TRAIN_LEARN
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    baseline = score_prequential_bpb(manifest)
    # Mutating the recorded gpu_count cannot change the challenge-computed final_score.
    manifest["compute"]["gpu_count"] = 8
    manifest["compute"]["world_size"] = 8
    manifest["compute"]["nproc_per_node"] = 8
    mutated = score_prequential_bpb(manifest)
    assert mutated.final_score == baseline.final_score
    assert mutated.bpb == baseline.bpb
