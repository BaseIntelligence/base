from __future__ import annotations

import pytest
import torch

from prism_challenge.evaluator.gloo_functional import (
    GlooFunctionalResult,
    NcclAdvisoryResult,
    run_gloo_functional,
    run_nccl_advisory,
)

# Multi-GPU 1-GPU validation strategy, Gate B (architecture.md section 8, data-multigpu.md):
# a gloo multi-rank functional test that drives a miner-style DDP training loop under a forced
# seed at WORLD_SIZE 2 and 4 on CPU. It proves DDP grad-sync correctness (loss decreases AND
# parameters stay byte-identical across ranks), disjoint per-rank data sharding, world-consistent
# all-reduced metrics, clean collective teardown (barrier + destroy_process_group, no hang/orphan),
# and a rank-0-only checkpoint/manifest writer. An advisory NCCL nproc=2 single-GPU launch is
# exercised non-gating (VAL-GPU-009..015).

# These multi-rank gloo collectives can deadlock if a collective hangs. They are quarantined behind
# the `distributed_gloo` marker so CI excludes them from the publish-gating `test` job (they run in
# a separate, non-gating job); a per-test timeout bounds any hang so it fails fast instead of
# running to GitHub's ~6h job ceiling and silently stalling image publication.
pytestmark = [
    pytest.mark.distributed_gloo,
    pytest.mark.timeout(300),
]


@pytest.fixture(scope="module")
def gloo_ws2() -> GlooFunctionalResult:
    return run_gloo_functional(world_size=2)


@pytest.fixture(scope="module")
def gloo_ws4() -> GlooFunctionalResult:
    return run_gloo_functional(world_size=4)


# --- VAL-GPU-009 / VAL-GPU-010: loss decreases AND params byte-identical across ranks ---


def test_distributed_gloo_world_size_2_loss_decreases_and_params_identical(
    gloo_ws2: GlooFunctionalResult,
) -> None:
    result = gloo_ws2
    assert result.world_size == 2
    assert len(result.ranks) == 2
    assert result.world_loss_decreased
    assert result.ranks[0].reduced_loss_last < result.ranks[0].reduced_loss_first
    # Byte-identical parameters across ranks (DDP grad-sync correctness).
    assert result.params_synced
    assert len({rank.param_hash for rank in result.ranks}) == 1


def test_distributed_gloo_world_size_4_loss_decreases_and_params_identical(
    gloo_ws4: GlooFunctionalResult,
) -> None:
    result = gloo_ws4
    assert result.world_size == 4
    assert len(result.ranks) == 4
    assert result.world_loss_decreased
    assert result.ranks[0].reduced_loss_last < result.ranks[0].reduced_loss_first
    assert result.params_synced
    assert len({rank.param_hash for rank in result.ranks}) == 1


# --- VAL-GPU-011: per-rank data sharding is disjoint (no duplicated batches) ---


def test_distributed_gloo_data_sharding_disjoint(
    gloo_ws2: GlooFunctionalResult, gloo_ws4: GlooFunctionalResult
) -> None:
    for result in (gloo_ws2, gloo_ws4):
        assert result.sharding_disjoint
        seen: set[int] = set()
        for rank in result.ranks:
            idx = set(rank.consumed_indices)
            assert idx, "each rank must consume at least one sample"
            assert not (seen & idx), "ranks consumed overlapping/duplicated batches"
            seen |= idx
        assert seen == set(range(result.num_samples))


# --- VAL-GPU-013: all-reduced metrics are world-consistent ---


def test_distributed_gloo_all_reduced_metrics_world_consistent(
    gloo_ws2: GlooFunctionalResult, gloo_ws4: GlooFunctionalResult
) -> None:
    for result in (gloo_ws2, gloo_ws4):
        assert result.metrics_world_consistent
        reduced_tokens = {rank.reduced_tokens_seen for rank in result.ranks}
        assert len(reduced_tokens) == 1, "reduced tokens differ across ranks"
        assert next(iter(reduced_tokens)) == sum(rank.local_tokens_seen for rank in result.ranks)
        reduced_losses = {round(rank.reduced_loss_last, 6) for rank in result.ranks}
        assert len(reduced_losses) == 1, "all-reduced loss differs across ranks"


# --- VAL-GPU-012: clean collective teardown (barrier + destroy_process_group, no hang/orphan) ---


def test_distributed_gloo_clean_teardown_no_orphans(
    gloo_ws2: GlooFunctionalResult, gloo_ws4: GlooFunctionalResult
) -> None:
    for result in (gloo_ws2, gloo_ws4):
        assert result.clean_teardown
        assert all(code == 0 for code in result.exit_codes)
        assert all(rank.clean_exit for rank in result.ranks)
        assert len(result.exit_codes) == result.world_size


# --- VAL-GPU-015: only rank 0 writes the checkpoint/manifest (no duplicate/corrupt writes) ---


def test_distributed_gloo_only_rank0_writes_artifacts(
    gloo_ws2: GlooFunctionalResult, gloo_ws4: GlooFunctionalResult
) -> None:
    for result in (gloo_ws2, gloo_ws4):
        assert result.rank0_is_sole_writer
        assert result.ranks[0].wrote_artifacts
        assert all(not rank.wrote_artifacts for rank in result.ranks[1:])
        assert sorted(result.artifacts_files) == [
            "checkpoint.pt",
            "prism_run_manifest.v2.json",
        ]
        assert result.manifest_valid


# --- VAL-GPU-014: advisory NCCL nproc=2 single-GPU launch (advisory, not a gate) ---


def test_distributed_gloo_nccl_advisory_is_nonblocking() -> None:
    # The advisory NCCL launch must NEVER raise or gate: it returns a structured result whether
    # or not a CUDA device is present (single-device NCCL is a non-standard config and may hang).
    result = run_nccl_advisory(nproc=2)
    assert isinstance(result, NcclAdvisoryResult)
    assert result.p2p_disabled is True
    assert result.cuda_available == torch.cuda.is_available()
    if not torch.cuda.is_available():
        assert result.attempted is False
        assert result.succeeded is False
        assert "skip" in result.detail.lower() or "no cuda" in result.detail.lower()
