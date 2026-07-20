"""Resource-aware concurrency sizing for the in-CVM DooD orchestrator (M2).

Black-box tests for :mod:`agent_challenge.evaluation.own_runner.concurrency`,
encoding the M2 validation-contract assertions:

* VAL-ORCH-001  concurrency is auto-computed from the CVM shape, never hardcoded
* VAL-ORCH-002  a tiny 1-vCPU CVM yields concurrency exactly 1
* VAL-ORCH-003  CPU-bound sizing == floor(nproc / per-task cpus) when memory is ample
* VAL-ORCH-004  memory-bound sizing == floor(usable_mem / per-task memory) when tighter
* VAL-ORCH-005  final concurrency == min(cpu_bound, mem_bound), clamped to >= 1
* VAL-ORCH-006  computed concurrency never exceeds nproc
* VAL-ORCH-007  per-task task.toml cpus/memory are read and change the bound (default fallback)
* VAL-ORCH-008  an explicit config concurrency cap further bounds the auto-computed value

All values are injected (nproc / MemTotal / per-task limits) so the sizing
decision is exercised as a pure function with no live enclave.
"""

from __future__ import annotations

import math

import pytest

from agent_challenge.evaluation.own_runner.concurrency import (
    DEFAULT_TASK_CPUS,
    DEFAULT_TASK_MEMORY_MB,
    TaskResourceCost,
    aggregate_task_cost,
    auto_concurrency,
    compute_max_concurrency,
    read_mem_total_kib,
    read_nproc,
    task_resource_cost,
)
from agent_challenge.evaluation.own_runner.taskdefs import ResourceLimits

KIB_PER_GIB = 1024 * 1024
KIB_PER_MIB = 1024


def _gib_kib(gib: float) -> int:
    """MemTotal (kiB, as /proc/meminfo reports it) for ``gib`` GiB."""
    return int(gib * KIB_PER_GIB)


# The observed tdx.small usable MemTotal (~1.85 GiB) from the live spike.
TDX_SMALL_MEM_KIB = _gib_kib(1.85)
# A memory size large enough to never bind the CPU sweeps below.
AMPLE_MEM_KIB = _gib_kib(512)


# --------------------------------------------------------------------------- #
# VAL-ORCH-001: auto-computed from the shape, never a fixed constant
# --------------------------------------------------------------------------- #
def test_concurrency_changes_with_cvm_shape() -> None:
    small = compute_max_concurrency(
        nproc=1,
        mem_total_kib=_gib_kib(2),
        task_cpus=1.0,
        task_memory_mb=512,
    )
    large = compute_max_concurrency(
        nproc=8,
        mem_total_kib=_gib_kib(16),
        task_cpus=1.0,
        task_memory_mb=512,
    )
    # Same per-task limits, two different shapes -> two different values.
    assert small != large
    assert small == 1
    assert large == 8


def test_concurrency_is_a_pure_function_of_inputs() -> None:
    kwargs = dict(nproc=4, mem_total_kib=_gib_kib(8), task_cpus=1.0, task_memory_mb=1024)
    first = compute_max_concurrency(**kwargs)
    second = compute_max_concurrency(**kwargs)
    assert first == second


# --------------------------------------------------------------------------- #
# VAL-ORCH-002: tiny 1-vCPU CVM yields concurrency exactly 1
# --------------------------------------------------------------------------- #
def test_tiny_one_vcpu_cvm_yields_exactly_one() -> None:
    value = compute_max_concurrency(
        nproc=1,
        mem_total_kib=TDX_SMALL_MEM_KIB,
    )
    assert value == 1


def test_tiny_cvm_never_zero_even_when_memory_is_tight() -> None:
    # Even with the default per-task memory exceeding a tiny MemTotal, clamp to 1.
    value = compute_max_concurrency(nproc=1, mem_total_kib=_gib_kib(0.5))
    assert value == 1


# --------------------------------------------------------------------------- #
# VAL-ORCH-003: CPU-bound == floor(nproc / cpus) when memory is ample
# --------------------------------------------------------------------------- #
def test_cpu_bound_matches_floor_nproc_over_cpus() -> None:
    for nproc, cpus in [(8, 2), (8, 3), (16, 4), (7, 2), (12, 5)]:
        value = compute_max_concurrency(
            nproc=nproc,
            mem_total_kib=AMPLE_MEM_KIB,
            task_cpus=cpus,
            task_memory_mb=64,
        )
        assert value == max(1, math.floor(nproc / cpus)), (nproc, cpus)


# --------------------------------------------------------------------------- #
# VAL-ORCH-004: memory-bound == floor(usable_mem / memory) when tighter
# --------------------------------------------------------------------------- #
def test_memory_bound_governs_when_it_is_the_tighter_bound() -> None:
    # 6 GiB usable, 2048 MiB per task -> memory allows exactly 3, CPU allows 64.
    value = compute_max_concurrency(
        nproc=64,
        mem_total_kib=_gib_kib(6),
        task_cpus=1.0,
        task_memory_mb=2048,
    )
    assert value == 3


def test_memory_bound_is_not_ignored_for_various_sizes() -> None:
    for gib, mem_mb, expected in [(8, 2048, 4), (10, 2048, 5), (3, 1024, 3)]:
        value = compute_max_concurrency(
            nproc=128,
            mem_total_kib=_gib_kib(gib),
            task_cpus=1.0,
            task_memory_mb=mem_mb,
        )
        assert value == expected, (gib, mem_mb)


# --------------------------------------------------------------------------- #
# VAL-ORCH-005: final == min(cpu_bound, mem_bound), clamped to >= 1
# --------------------------------------------------------------------------- #
def test_final_is_min_of_cpu_and_memory_bounds() -> None:
    # CPU allows 8, memory allows 3 -> min is 3.
    value = compute_max_concurrency(
        nproc=8,
        mem_total_kib=_gib_kib(6),
        task_cpus=1.0,
        task_memory_mb=2048,
    )
    assert value == 3


def test_oversized_task_clamps_to_one() -> None:
    # Per-task memory exceeds usable MemTotal -> serialized (1), never 0.
    mem_clamp = compute_max_concurrency(
        nproc=8,
        mem_total_kib=_gib_kib(2),
        task_cpus=1.0,
        task_memory_mb=4096,
    )
    assert mem_clamp == 1
    # Per-task cpus exceeds nproc -> serialized (1), never 0.
    cpu_clamp = compute_max_concurrency(
        nproc=2,
        mem_total_kib=AMPLE_MEM_KIB,
        task_cpus=4.0,
        task_memory_mb=64,
    )
    assert cpu_clamp == 1


# --------------------------------------------------------------------------- #
# VAL-ORCH-006: never exceeds nproc
# --------------------------------------------------------------------------- #
def test_concurrency_never_exceeds_nproc_across_sweep() -> None:
    for nproc in [1, 2, 4, 8, 16, 64]:
        for cpus in [0.5, 1.0, 2.0, 3.0]:
            for gib in [2, 16, 512]:
                value = compute_max_concurrency(
                    nproc=nproc,
                    mem_total_kib=_gib_kib(gib),
                    task_cpus=cpus,
                    task_memory_mb=256,
                )
                assert 1 <= value <= nproc, (nproc, cpus, gib, value)


# --------------------------------------------------------------------------- #
# VAL-ORCH-007: per-task cpus/memory are read and change the bound
# --------------------------------------------------------------------------- #
def test_increasing_cpus_monotonically_lowers_concurrency() -> None:
    prev = None
    for cpus in [1.0, 2.0, 4.0, 8.0]:
        value = compute_max_concurrency(
            nproc=16,
            mem_total_kib=AMPLE_MEM_KIB,
            task_cpus=cpus,
            task_memory_mb=256,
        )
        if prev is not None:
            assert value <= prev
        prev = value
    assert prev == 2  # nproc=16 / cpus=8


def test_increasing_memory_monotonically_lowers_concurrency() -> None:
    prev = None
    for mem_mb in [2048, 4096, 8192, 16384]:
        value = compute_max_concurrency(
            nproc=16,
            mem_total_kib=_gib_kib(32),
            task_cpus=1.0,
            task_memory_mb=mem_mb,
        )
        if prev is not None:
            assert value <= prev
        prev = value


def test_no_limit_task_uses_documented_default_not_zero_cost() -> None:
    # A task with no explicit limits must be charged the documented default cost,
    # NOT treated as zero-cost (which would inflate concurrency to nproc).
    no_limit = compute_max_concurrency(
        nproc=64,
        mem_total_kib=_gib_kib(8),
    )
    explicit_default = compute_max_concurrency(
        nproc=64,
        mem_total_kib=_gib_kib(8),
        task_cpus=DEFAULT_TASK_CPUS,
        task_memory_mb=DEFAULT_TASK_MEMORY_MB,
    )
    assert no_limit == explicit_default
    # 8 GiB / 2048 MiB default => 4, far below nproc (proves not zero-cost).
    assert no_limit == 4
    assert no_limit < 64


def test_task_resource_cost_applies_defaults_for_missing_fields() -> None:
    empty = task_resource_cost(ResourceLimits())
    assert empty == TaskResourceCost(cpus=DEFAULT_TASK_CPUS, memory_mb=DEFAULT_TASK_MEMORY_MB)

    partial = task_resource_cost(ResourceLimits(cpus=2))
    assert partial.cpus == 2.0
    assert partial.memory_mb == DEFAULT_TASK_MEMORY_MB

    full = task_resource_cost(ResourceLimits(cpus=3, memory_mb=4096))
    assert full == TaskResourceCost(cpus=3.0, memory_mb=4096)


def test_aggregate_task_cost_takes_worst_case_across_tasks() -> None:
    cost = aggregate_task_cost(
        [
            ResourceLimits(cpus=1, memory_mb=1024),
            ResourceLimits(cpus=4, memory_mb=2048),
            ResourceLimits(),  # defaults
        ]
    )
    # Worst-case cpus (max of 1, 4, default 1) and memory (max of 1024, 2048, default 2048).
    assert cost.cpus == 4.0
    assert cost.memory_mb == DEFAULT_TASK_MEMORY_MB


def test_aggregate_task_cost_empty_falls_back_to_defaults() -> None:
    cost = aggregate_task_cost([])
    assert cost == TaskResourceCost(cpus=DEFAULT_TASK_CPUS, memory_mb=DEFAULT_TASK_MEMORY_MB)


# --------------------------------------------------------------------------- #
# VAL-ORCH-008: an explicit config cap further bounds the auto-computed value
# --------------------------------------------------------------------------- #
def test_config_cap_lowers_when_smaller() -> None:
    value = compute_max_concurrency(
        nproc=8,
        mem_total_kib=AMPLE_MEM_KIB,
        task_cpus=1.0,
        task_memory_mb=64,
        config_cap=3,
    )
    assert value == 3


def test_config_cap_is_noop_when_larger() -> None:
    value = compute_max_concurrency(
        nproc=8,
        mem_total_kib=AMPLE_MEM_KIB,
        task_cpus=1.0,
        task_memory_mb=64,
        config_cap=100,
    )
    assert value == 8


def test_no_config_cap_uses_pure_auto_value() -> None:
    value = compute_max_concurrency(
        nproc=8,
        mem_total_kib=AMPLE_MEM_KIB,
        task_cpus=1.0,
        task_memory_mb=64,
        config_cap=None,
    )
    assert value == 8


def test_config_cap_never_overrides_smaller_auto_upward() -> None:
    # auto is 1 (tiny CVM); a large cap must NOT raise it.
    value = compute_max_concurrency(
        nproc=1,
        mem_total_kib=TDX_SMALL_MEM_KIB,
        config_cap=64,
    )
    assert value == 1


# --------------------------------------------------------------------------- #
# System readers + integration convenience
# --------------------------------------------------------------------------- #
def test_read_mem_total_kib_parses_proc_meminfo(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:        1939865 kB\nMemFree:          123456 kB\nMemAvailable:     654321 kB\n",
        encoding="utf-8",
    )
    assert read_mem_total_kib(meminfo) == 1939865


def test_read_nproc_returns_at_least_one() -> None:
    assert read_nproc() >= 1


def test_read_mem_total_kib_raises_when_absent(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemFree: 123456 kB\n", encoding="utf-8")
    with pytest.raises(ValueError, match="MemTotal"):
        read_mem_total_kib(meminfo)


def test_auto_concurrency_uses_injected_shape_and_task_resources() -> None:
    value = auto_concurrency(
        resources=[ResourceLimits(cpus=2, memory_mb=1024)],
        nproc=8,
        mem_total_kib=_gib_kib(16),
    )
    # CPU allows floor(8/2)=4; memory allows 16 GiB / 1 GiB = 16 -> min is 4.
    assert value == 4


def test_auto_concurrency_honors_config_cap() -> None:
    value = auto_concurrency(
        resources=[ResourceLimits(cpus=1, memory_mb=512)],
        nproc=8,
        mem_total_kib=_gib_kib(16),
        config_cap=2,
    )
    assert value == 2


def test_auto_concurrency_tiny_shape_is_one() -> None:
    value = auto_concurrency(
        resources=[ResourceLimits()],
        nproc=1,
        mem_total_kib=TDX_SMALL_MEM_KIB,
    )
    assert value == 1
