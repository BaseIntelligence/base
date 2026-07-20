from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from prism_challenge.evaluator.schemas import COMPUTE_BLOCK_SCHEMA, ComputeBlock
from prism_challenge.evaluator.scoring import build_compute_block, score_prequential_bpb


def _manifest(
    *,
    sum_nll_nats: float = 80.0,
    covered_bytes: int = 256,
    compute: dict | None = None,
) -> dict:
    manifest: dict = {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": "sub-compute",
        "run": {
            "seed": 1337,
            "forced_init": True,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": "cpu",
            "nproc_per_node": 1,
        },
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": [2.5, 2.0, 1.5, 1.0],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "covered_bytes": covered_bytes,
            "predicted_tokens": 64,
            "step0_loss": 2.0,
            "consumed_batches": 4,
            "random_init_baseline_nats": math.log(128),
            "nan_inf_batches": 0,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }
    if compute is not None:
        manifest["compute"] = compute
    return manifest


def test_build_compute_block_is_typed_and_consistent() -> None:
    block = build_compute_block(
        gpu_count=1, world_size=1, nproc_per_node=1, device="cuda:0", max_gpu_count=8
    )
    assert block["schema"] == COMPUTE_BLOCK_SCHEMA
    assert block["gpu_count"] == 1
    assert block["world_size"] == 1
    assert block["nproc_per_node"] == 1
    assert block["device"] == "cuda:0"
    assert block["max_gpu_count"] == 8
    # The scored nproc=1 path: gpu_count is internally consistent with world_size / nproc_per_node.
    assert block["gpu_count"] == block["world_size"] == block["nproc_per_node"] == 1
    # The block round-trips through the typed schema.
    parsed = ComputeBlock.model_validate(block)
    assert parsed.gpu_count == 1


def test_build_compute_block_omits_unset_max_gpu_count() -> None:
    block = build_compute_block(gpu_count=1, world_size=1, nproc_per_node=1, device="cpu")
    assert "max_gpu_count" not in block


def test_compute_block_rejects_invalid_launch_shape() -> None:
    with pytest.raises(ValidationError):
        ComputeBlock(gpu_count=1, world_size=0, nproc_per_node=1, device="cpu")
    with pytest.raises(ValidationError):
        ComputeBlock(gpu_count=-1, world_size=1, nproc_per_node=1, device="cpu")
    with pytest.raises(ValidationError):
        ComputeBlock(gpu_count=1, world_size=1, nproc_per_node=1, device="")


def test_gpu_count_is_not_an_input_to_final_score() -> None:
    # final_score derives ONLY from compute-normalized learning metrics (prequential bpb + held-out
    # delta tie-break). gpu_count is observability-only: varying it must NOT move final_score.
    baseline = score_prequential_bpb(_manifest())
    for gpu_count in (1, 2, 4, 8):
        compute = build_compute_block(
            gpu_count=gpu_count,
            world_size=gpu_count,
            nproc_per_node=gpu_count,
            device="cuda:0",
            max_gpu_count=8,
        )
        scored = score_prequential_bpb(_manifest(compute=compute))
        assert scored.final_score == baseline.final_score
        assert scored.bpb == baseline.bpb


def test_no_gpu_scaling_reward_term_in_score_outputs() -> None:
    # No GPU-count reward / scaling bonus appears anywhere in the score surface.
    score = score_prequential_bpb(
        _manifest(
            compute=build_compute_block(
                gpu_count=8, world_size=8, nproc_per_node=8, device="cuda:0"
            )
        )
    )
    payload_text = " ".join(str(k) for k in score.metrics_payload()).lower()
    block_text = " ".join(str(k) for k in score.manifest_score_block()).lower()
    assert "gpu" not in payload_text
    assert "gpu" not in block_text
    assert "scaling" not in payload_text
    assert "scaling" not in block_text


def test_final_score_deterministic_regardless_of_compute_block() -> None:
    run1 = score_prequential_bpb(_manifest())
    run2 = score_prequential_bpb(
        _manifest(
            compute=build_compute_block(
                gpu_count=4, world_size=4, nproc_per_node=4, device="cuda:0"
            )
        )
    )
    assert run1.final_score == run2.final_score
