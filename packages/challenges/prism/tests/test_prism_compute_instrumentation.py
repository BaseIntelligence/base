"""Runner compute instrumentation + host reconciliation (architecture-lab API contract, Group B).

The in-container runner measures peak VRAM / peak host RSS / wall-clock around ``miner_train(ctx)``
and the host derives the 6ND FLOPs estimate during reconciliation. These are observability-only and
MUST never feed the score; the host re-author must PRESERVE the runner-measured telemetry.
"""

from __future__ import annotations

import json
import math

# Reuse the CPU re-exec harness helpers to validate the real runner instrumentation end-to-end.
from test_prism_cpu_reexec import TINY_ARCH, TINY_TRAIN, _run_direct, _stage_train

from prism_challenge.evaluator.container import _ensure_compute_block
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME, ComputeBlock
from prism_challenge.evaluator.scoring import build_compute_block, score_prequential_bpb


def _manifest(*, model_params: int | None, tokens: int, compute: dict | None = None) -> dict:
    covered_bytes = 256
    sum_nll_nats = 80.0
    bits = sum_nll_nats / math.log(2.0)
    metrics: dict = {
        "online_loss": [2.5, 2.0, 1.5, 1.0],
        "sum_neg_log_likelihood_nats": sum_nll_nats,
        "sum_neg_log2_likelihood_bits": bits,
        "covered_bytes": covered_bytes,
        "predicted_tokens": tokens,
        "tokens_seen": tokens,
        "step0_loss": 2.5,
        "consumed_batches": 4,
        "random_init_baseline_nats": math.log(128),
        "nan_inf_batches": 0,
    }
    if model_params is not None:
        metrics["model_params"] = model_params
    manifest: dict = {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": "sub-instr",
        "run": {
            "seed": 1337,
            "forced_init": True,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": "cuda:0",
            "nproc_per_node": 1,
        },
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": metrics,
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


def test_build_compute_block_round_trips_new_telemetry() -> None:
    block = build_compute_block(
        gpu_count=1,
        world_size=1,
        nproc_per_node=1,
        device="cuda:0",
        peak_vram_bytes=12884901888,
        peak_rss_bytes=5368709120,
        wall_clock_seconds=412.5,
        estimated_flops=7.31e15,
    )
    parsed = ComputeBlock.model_validate(block)
    assert parsed.peak_vram_bytes == 12884901888
    assert parsed.peak_rss_bytes == 5368709120
    assert parsed.wall_clock_seconds == 412.5
    assert parsed.estimated_flops == 7.31e15


def test_build_compute_block_omits_unset_telemetry() -> None:
    block = build_compute_block(gpu_count=1, world_size=1, nproc_per_node=1, device="cpu")
    for key in ("peak_vram_bytes", "peak_rss_bytes", "wall_clock_seconds", "estimated_flops"):
        assert key not in block


def test_ensure_compute_block_preserves_telemetry_and_computes_flops(tmp_path) -> None:
    runner_compute = {
        "schema": "prism_compute.v1",
        "gpu_count": 1,
        "world_size": 1,
        "nproc_per_node": 1,
        "device": "cuda:0",
        "peak_vram_bytes": 12884901888,
        "peak_rss_bytes": 5368709120,
        "wall_clock_seconds": 412.5,
    }
    manifest = _manifest(model_params=1000, tokens=2000, compute=runner_compute)
    (tmp_path / RUN_MANIFEST_V2_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    before = score_prequential_bpb(manifest).final_score

    _ensure_compute_block(manifest, {"actual_gpu_count": 1, "max_gpu_count": 8}, tmp_path)

    compute = manifest["compute"]
    # The runner-measured telemetry survives the host re-author (instead of being dropped).
    assert compute["peak_vram_bytes"] == 12884901888
    assert compute["peak_rss_bytes"] == 5368709120
    assert compute["wall_clock_seconds"] == 412.5
    # estimated_flops = 6 * model_params * tokens_consumed (host-computed).
    assert compute["estimated_flops"] == 6.0 * 1000 * 2000
    assert compute["model_params"] == 1000
    # Telemetry is observability-only: the score is unchanged.
    assert score_prequential_bpb(manifest).final_score == before
    # Persisted to disk where manifest-inspect reads it.
    on_disk = json.loads((tmp_path / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))
    assert on_disk["compute"]["wall_clock_seconds"] == 412.5


def test_ensure_compute_block_omits_flops_without_model_params(tmp_path) -> None:
    manifest = _manifest(model_params=None, tokens=2000)
    _ensure_compute_block(manifest, {"actual_gpu_count": 1}, tmp_path)
    assert "estimated_flops" not in manifest["compute"]


def test_cpu_reexec_runner_records_wall_clock_rss_and_flops(tmp_path, monkeypatch) -> None:
    data_dir = _stage_train(tmp_path)
    result, _ = _run_direct(
        tmp_path, monkeypatch, arch=TINY_ARCH, train=TINY_TRAIN, data_dir=data_dir
    )
    compute = result.run_manifest["compute"]
    # The runner measures wall-clock + peak host RSS (stdlib rusage); a CPU run has no CUDA device
    # so peak VRAM is simply omitted.
    assert isinstance(compute["wall_clock_seconds"], int | float)
    assert compute["wall_clock_seconds"] >= 0.0
    assert isinstance(compute["peak_rss_bytes"], int)
    assert compute["peak_rss_bytes"] > 0
    assert "peak_vram_bytes" not in compute
    # The host derived the 6ND FLOPs estimate from the runner's model_params + tokens.
    assert isinstance(compute["estimated_flops"], int | float)
    assert compute["estimated_flops"] > 0
    # None of this feeds the score.
    assert result.run_manifest["score"]["wall_clock_term"] is False
