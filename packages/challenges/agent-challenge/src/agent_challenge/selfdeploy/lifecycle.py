"""Shared self-deploy lifecycle safety guards."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_challenge.selfdeploy.shapes import (
    CPU_TDX_SHAPES,
    DEFAULT_EVAL_DISK_SIZE_GB,
    DEFAULT_MONEY_CAP_USD,
    DEFAULT_REVIEW_DISK_SIZE_GB,
    ShapeError,
    projected_disk_cost_usd,
    validate_cpu_only,
    validate_disk_size,
)


class LifecycleBudgetError(ShapeError):
    """The combined review + Eval lifecycle exceeds the money cap."""


@dataclass(frozen=True)
class LifecycleCost:
    review_usd: float
    eval_usd: float
    total_usd: float
    money_cap_usd: float


def projected_lifecycle_cost_usd(
    *,
    review_instance_type: str,
    eval_instance_type: str,
    review_runtime_hours: float,
    eval_runtime_hours: float,
    review_disk_size_gb: int = DEFAULT_REVIEW_DISK_SIZE_GB,
    eval_disk_size_gb: int = DEFAULT_EVAL_DISK_SIZE_GB,
) -> float:
    """Compute both CVM projections together (compute + disk), never budget each alone."""

    for instance_type in (review_instance_type, eval_instance_type):
        try:
            validate_cpu_only(instance_type=instance_type)
        except ShapeError as exc:
            raise LifecycleBudgetError(str(exc)) from exc
    try:
        review_disk = validate_disk_size(review_disk_size_gb)
        eval_disk = validate_disk_size(eval_disk_size_gb)
    except ShapeError as exc:
        raise LifecycleBudgetError(str(exc)) from exc
    if (
        not math.isfinite(review_runtime_hours)
        or not math.isfinite(eval_runtime_hours)
        or review_runtime_hours < 0
        or eval_runtime_hours < 0
    ):
        raise LifecycleBudgetError("runtime hours must be non-negative")
    return (
        CPU_TDX_SHAPES[review_instance_type].usd_per_hour * review_runtime_hours
        + projected_disk_cost_usd(review_disk, max_runtime_hours=review_runtime_hours)
        + CPU_TDX_SHAPES[eval_instance_type].usd_per_hour * eval_runtime_hours
        + projected_disk_cost_usd(eval_disk, max_runtime_hours=eval_runtime_hours)
    )


def validate_lifecycle_budget(
    *,
    review_instance_type: str,
    eval_instance_type: str,
    review_runtime_hours: float,
    eval_runtime_hours: float,
    money_cap_usd: float = 20.0,
    review_disk_size_gb: int = DEFAULT_REVIEW_DISK_SIZE_GB,
    eval_disk_size_gb: int = DEFAULT_EVAL_DISK_SIZE_GB,
) -> LifecycleCost:
    """Refuse a combined lifecycle that could exceed the shared cap."""

    if not math.isfinite(money_cap_usd) or money_cap_usd < 0:
        raise LifecycleBudgetError("money cap must be a finite non-negative number")
    if money_cap_usd > DEFAULT_MONEY_CAP_USD:
        raise LifecycleBudgetError(
            f"money cap cannot exceed the mission ${DEFAULT_MONEY_CAP_USD:.2f} cap"
        )
    review_cost = CPU_TDX_SHAPES.get(review_instance_type)
    eval_cost = CPU_TDX_SHAPES.get(eval_instance_type)
    total = projected_lifecycle_cost_usd(
        review_instance_type=review_instance_type,
        eval_instance_type=eval_instance_type,
        review_runtime_hours=review_runtime_hours,
        eval_runtime_hours=eval_runtime_hours,
        review_disk_size_gb=review_disk_size_gb,
        eval_disk_size_gb=eval_disk_size_gb,
    )
    if total > money_cap_usd:
        raise LifecycleBudgetError(
            f"projected review+eval cost ${total:.2f} exceeds the ${money_cap_usd:.2f} cap"
        )
    assert review_cost is not None and eval_cost is not None
    review_disk = validate_disk_size(review_disk_size_gb)
    eval_disk = validate_disk_size(eval_disk_size_gb)
    return LifecycleCost(
        review_usd=(
            review_cost.usd_per_hour * review_runtime_hours
            + projected_disk_cost_usd(review_disk, max_runtime_hours=review_runtime_hours)
        ),
        eval_usd=(
            eval_cost.usd_per_hour * eval_runtime_hours
            + projected_disk_cost_usd(eval_disk, max_runtime_hours=eval_runtime_hours)
        ),
        total_usd=total,
        money_cap_usd=money_cap_usd,
    )


__all__ = [
    "LifecycleBudgetError",
    "LifecycleCost",
    "projected_lifecycle_cost_usd",
    "validate_lifecycle_budget",
]
