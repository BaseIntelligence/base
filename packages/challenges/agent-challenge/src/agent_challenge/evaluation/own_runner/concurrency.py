"""Resource-aware concurrency sizing for the in-CVM DooD orchestrator (M2).

The orchestrator runs INSIDE whatever Phala TDX CPU CVM the miner chose to
deploy, so the number of Terminal-Bench task containers it may run in parallel
cannot be a hardcoded worker count: it must self-size to the CVM shape. This
module derives that maximum concurrency purely from

* ``nproc`` -- the usable CPU count of the CVM, and
* ``MemTotal`` -- the usable RAM reported by ``/proc/meminfo``, and
* the per-task ``[environment]`` ``cpus`` / ``memory_mb`` declared in each
  task's ``task.toml`` (with documented defaults for tasks that omit them),

as ``min(cpu_bound, mem_bound)`` where ``cpu_bound = floor(nproc / cpus)`` and
``mem_bound = floor(usable_mem / memory)``, clamped to ``[1, nproc]`` and
optionally further bounded by an explicit config cap. A tiny 1-vCPU CVM
therefore yields concurrency ``1`` and a larger CVM a bounded value that never
exceeds ``nproc``.

The functions here are pure (system introspection is factored into
:func:`read_nproc` / :func:`read_mem_total_kib`, both injectable), so the sizing
decision is fully testable offline without a live enclave.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

#: Per-task CPU cost charged when a task's ``task.toml`` omits ``[environment].cpus``.
#: A no-limit task is charged one core (never zero-cost, which would let it
#: inflate the CPU bound up to ``nproc``).
DEFAULT_TASK_CPUS = 1.0
#: Per-task memory cost (MiB) charged when a task's ``task.toml`` omits
#: ``[environment].memory_mb``. A no-limit task is charged this much RAM (never
#: zero-cost, which would let it inflate the memory bound without limit).
DEFAULT_TASK_MEMORY_MB = 2048

#: Where MemTotal is read from inside the CVM.
MEMINFO_PATH = "/proc/meminfo"

#: ``/proc/meminfo`` reports sizes in kiB; docker ``--memory`` (and task.toml
#: ``memory_mb``) are MiB. Both are binary, so 1 MiB == 1024 kiB.
_KIB_PER_MIB = 1024


@dataclass(frozen=True)
class TaskResourceCost:
    """The per-task resource cost that governs one concurrency slot."""

    cpus: float
    memory_mb: int


def read_nproc() -> int:
    """Return the usable CPU count for this process (cgroup/affinity aware).

    Prefers :func:`os.sched_getaffinity` (honors the container/cgroup CPU set the
    CVM actually grants) and falls back to :func:`os.cpu_count`. Always at least
    ``1``.
    """

    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux fallback
        count = os.cpu_count() or 1
    return max(1, count)


def read_mem_total_kib(meminfo_path: str | Path = MEMINFO_PATH) -> int:
    """Return usable ``MemTotal`` (kiB) parsed from ``/proc/meminfo``.

    ``MemTotal`` is already net of the memory the kernel/hardware reserves, so it
    is the ``usable`` figure the sizing math consumes.
    """

    text = Path(meminfo_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1])
    raise ValueError(f"MemTotal not found in {meminfo_path}")


def task_resource_cost(resources: object) -> TaskResourceCost:
    """Resolve one task's resource cost, applying documented defaults.

    ``resources`` is duck-typed (any object exposing ``cpus`` / ``memory_mb``,
    e.g. a :class:`~agent_challenge.evaluation.own_runner.taskdefs.ResourceLimits`)
    so this module stays import-light for the lean canonical image. A missing or
    non-positive field falls back to the documented default (never zero-cost).
    """

    cpus = getattr(resources, "cpus", None)
    memory_mb = getattr(resources, "memory_mb", None)
    cpus_val = float(cpus) if (cpus is not None and float(cpus) > 0) else DEFAULT_TASK_CPUS
    mem_val = (
        int(memory_mb) if (memory_mb is not None and int(memory_mb) > 0) else DEFAULT_TASK_MEMORY_MB
    )
    return TaskResourceCost(cpus=cpus_val, memory_mb=mem_val)


def aggregate_task_cost(resources: Iterable[object]) -> TaskResourceCost:
    """Worst-case per-task cost across ``resources`` (so peak never oversubscribes).

    Takes the maximum ``cpus`` and maximum ``memory_mb`` requested by any task in
    the job, so the computed concurrency is safe even for the heaviest task. An
    empty job falls back to the documented defaults.
    """

    costs = [task_resource_cost(r) for r in resources]
    if not costs:
        return TaskResourceCost(cpus=DEFAULT_TASK_CPUS, memory_mb=DEFAULT_TASK_MEMORY_MB)
    return TaskResourceCost(
        cpus=max(c.cpus for c in costs),
        memory_mb=max(c.memory_mb for c in costs),
    )


def compute_max_concurrency(
    *,
    nproc: int,
    mem_total_kib: int,
    task_cpus: float = DEFAULT_TASK_CPUS,
    task_memory_mb: int = DEFAULT_TASK_MEMORY_MB,
    config_cap: int | None = None,
) -> int:
    """Compute the max concurrent task containers for a CVM shape + per-task cost.

    ``= min(floor(nproc / task_cpus), floor(usable_mem / task_memory_mb), nproc)``
    clamped to at least ``1``, then further bounded by ``config_cap`` when given.
    The value is a pure function of its inputs (no hardcoded worker count) and
    never exceeds ``nproc``.
    """

    nproc = max(1, int(nproc))
    mem_total_kib = max(0, int(mem_total_kib))

    cpus = float(task_cpus) if (task_cpus and float(task_cpus) > 0) else DEFAULT_TASK_CPUS
    mem_mb = (
        int(task_memory_mb)
        if (task_memory_mb and int(task_memory_mb) > 0)
        else DEFAULT_TASK_MEMORY_MB
    )

    cpu_bound = math.floor(nproc / cpus)
    mem_bound = mem_total_kib // (mem_mb * _KIB_PER_MIB)

    # Never assign more concurrent tasks than we have cores (caps sub-1-cpu tasks
    # too), and never drop below serialized execution.
    auto = max(1, min(cpu_bound, mem_bound, nproc))

    if config_cap is not None:
        return max(1, min(auto, int(config_cap)))
    return auto


def auto_concurrency(
    *,
    resources: Iterable[object] = (),
    config_cap: int | None = None,
    nproc: int | None = None,
    mem_total_kib: int | None = None,
) -> int:
    """Auto-size concurrency from the CVM shape + the job's per-task resources.

    Reads ``nproc`` / ``MemTotal`` from the running CVM unless overridden (tests
    inject both). ``resources`` is the per-task resource set for the job (worst
    case is used). ``config_cap`` optionally lowers the result.
    """

    cost = aggregate_task_cost(resources)
    resolved_nproc = read_nproc() if nproc is None else nproc
    resolved_mem = read_mem_total_kib() if mem_total_kib is None else mem_total_kib
    return compute_max_concurrency(
        nproc=resolved_nproc,
        mem_total_kib=resolved_mem,
        task_cpus=cost.cpus,
        task_memory_mb=cost.memory_mb,
        config_cap=config_cap,
    )


__all__ = [
    "DEFAULT_TASK_CPUS",
    "DEFAULT_TASK_MEMORY_MB",
    "MEMINFO_PATH",
    "TaskResourceCost",
    "aggregate_task_cost",
    "auto_concurrency",
    "compute_max_concurrency",
    "read_mem_total_kib",
    "read_nproc",
    "task_resource_cost",
]
