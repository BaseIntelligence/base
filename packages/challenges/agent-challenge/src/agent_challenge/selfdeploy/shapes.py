"""CPU Intel TDX shape catalog + money/GPU deploy guards (AGENTS.md boundaries).

The mission is **CPU Intel TDX only** (no GPU, not available to this account) with
a hard **$20** spend cap and a preference for the smallest CPU shape that works
(``tdx.small``/``tdx.medium``). This module is the single source of truth for the
CPU shape catalog and the pure, side-effect-free guard functions the deploy path
runs BEFORE any provisioning:

* :func:`validate_cpu_only` refuses a GPU instance type (e.g. ``h200.small``) or a
  GPU OS image (e.g. ``dstack-nvidia-*``) and any unknown shape (VAL-DEPLOY-007);
* :func:`select_default_instance_type` picks the smallest CPU shape when the miner
  gives none (VAL-DEPLOY-008);
* :func:`validate_within_cap` refuses a shape whose projected cost would breach the
  money cap (VAL-DEPLOY-008).

Hourly rates are the account's observed CPU TDX prices (library/phala.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class ShapeError(ValueError):
    """A requested VM shape/OS image is not deployable under the mission boundaries."""


class GpuRefusedError(ShapeError):
    """A GPU instance type or GPU OS image was requested (mission is CPU-only)."""


class OverCapError(ShapeError):
    """The requested shape's projected cost would breach the money cap."""


@dataclass(frozen=True)
class CpuShape:
    """A CPU Intel TDX shape: vCPU/RAM and the account's observed hourly USD rate."""

    name: str
    vcpus: int
    memory_gib: int
    usd_per_hour: float


#: CPU Intel TDX shapes (library/phala.md), smallest first.
CPU_TDX_SHAPES: dict[str, CpuShape] = {
    "tdx.small": CpuShape("tdx.small", 1, 2, 0.058),
    "tdx.medium": CpuShape("tdx.medium", 2, 4, 0.116),
    "tdx.large": CpuShape("tdx.large", 4, 8, 0.232),
    "tdx.xlarge": CpuShape("tdx.xlarge", 8, 16, 0.464),
}

#: The smallest CPU shapes the mission prefers (AGENTS.md).
SMALLEST_CPU_SHAPES: tuple[str, ...] = ("tdx.small", "tdx.medium")

#: Default instance type when the miner does not pick one: the smallest CPU shape.
DEFAULT_INSTANCE_TYPE = "tdx.small"

#: Default CPU dstack OS image. Live teepods (prod5/prod9) currently ship up to
#: dstack-0.5.9 (product default was 0.5.10 but that image is not mounted on
#: available nodes → provision ports a different os_image_hash and dual-flag
#: allowlists fail closed). Prefer real non-dev 0.5.9. GPU images are refused.
DEFAULT_OS_IMAGE = "dstack-0.5.9"

#: Hard mission spend cap (USD).
DEFAULT_MONEY_CAP_USD = 20.0

#: Conservative projected max runtime (hours) used for the cost-cap guard. A
#: deploy's projected cost is ``usd_per_hour * max_runtime_hours``; a shape whose
#: projected cost exceeds the money cap is refused before provisioning.
DEFAULT_MAX_RUNTIME_HOURS = 6.0

#: GPU instance-type prefixes/markers that are always refused (CPU-only mission).
_GPU_INSTANCE_RE = re.compile(r"^(?:h100|h200|a100|a10|l40|l4|t4|rtx)", re.IGNORECASE)

#: GPU OS image markers (e.g. ``dstack-nvidia-*``) that are always refused.
_GPU_IMAGE_RE = re.compile(r"nvidia|gpu|cuda", re.IGNORECASE)


def is_gpu_instance_type(instance_type: str) -> bool:
    """Whether ``instance_type`` names a GPU shape (refused on a CPU-only mission)."""

    token = (instance_type or "").strip().lower()
    if not token:
        return False
    if "gpu" in token or "nvidia" in token:
        return True
    return bool(_GPU_INSTANCE_RE.match(token))


def is_gpu_os_image(os_image: str) -> bool:
    """Whether ``os_image`` is a GPU/NVIDIA dstack image (refused CPU-only)."""

    return bool(_GPU_IMAGE_RE.search(os_image or ""))


def validate_cpu_only(*, instance_type: str, os_image: str = DEFAULT_OS_IMAGE) -> CpuShape:
    """Refuse GPU/unknown targets; return the resolved CPU shape (VAL-DEPLOY-007).

    Raises :class:`GpuRefusedError` for a GPU instance type or GPU OS image, and
    :class:`ShapeError` for an unknown (non-catalog) CPU shape. Pure and
    side-effect free: it never touches Phala, so a refusal makes zero provision
    calls.
    """

    if is_gpu_os_image(os_image):
        raise GpuRefusedError(
            f"CPU-only mission: refusing GPU OS image {os_image!r}; use a CPU dstack image "
            f"(default {DEFAULT_OS_IMAGE!r}). No GPU deploys are permitted."
        )
    if is_gpu_instance_type(instance_type):
        raise GpuRefusedError(
            f"CPU-only mission: refusing GPU instance type {instance_type!r}; use a CPU Intel "
            f"TDX shape (one of {sorted(CPU_TDX_SHAPES)}). No GPU deploys are permitted."
        )
    shape = CPU_TDX_SHAPES.get((instance_type or "").strip())
    if shape is None:
        raise ShapeError(
            f"unknown CPU Intel TDX shape {instance_type!r}; expected one of "
            f"{sorted(CPU_TDX_SHAPES)}"
        )
    return shape


def select_default_instance_type() -> str:
    """The smallest CPU shape used when the miner supplies none (VAL-DEPLOY-008)."""

    return DEFAULT_INSTANCE_TYPE


def projected_cost_usd(
    instance_type: str,
    *,
    max_runtime_hours: float = DEFAULT_MAX_RUNTIME_HOURS,
) -> float:
    """Projected deploy cost = ``usd_per_hour * max_runtime_hours`` for a CPU shape."""

    shape = CPU_TDX_SHAPES.get((instance_type or "").strip())
    if shape is None:
        raise ShapeError(f"unknown CPU Intel TDX shape {instance_type!r}")
    if max_runtime_hours < 0:
        raise ShapeError("max_runtime_hours must be non-negative")
    return shape.usd_per_hour * float(max_runtime_hours)


def validate_within_cap(
    instance_type: str,
    *,
    money_cap_usd: float = DEFAULT_MONEY_CAP_USD,
    max_runtime_hours: float = DEFAULT_MAX_RUNTIME_HOURS,
) -> float:
    """Refuse a shape whose projected cost breaches the money cap (VAL-DEPLOY-008).

    Returns the projected cost when within cap; raises :class:`OverCapError`
    otherwise. Pure/side-effect-free (no Phala call), so an over-cap request is
    refused before any provisioning.
    """

    cost = projected_cost_usd(instance_type, max_runtime_hours=max_runtime_hours)
    if cost > money_cap_usd:
        raise OverCapError(
            f"projected cost ${cost:.2f} ({instance_type} @ "
            f"${CPU_TDX_SHAPES[instance_type].usd_per_hour}/h x {max_runtime_hours}h) "
            f"exceeds the ${money_cap_usd:.2f} money cap; choose a smaller shape or lower "
            "the runtime budget"
        )
    return cost


__all__ = [
    "CPU_TDX_SHAPES",
    "DEFAULT_INSTANCE_TYPE",
    "DEFAULT_MAX_RUNTIME_HOURS",
    "DEFAULT_MONEY_CAP_USD",
    "DEFAULT_OS_IMAGE",
    "SMALLEST_CPU_SHAPES",
    "CpuShape",
    "GpuRefusedError",
    "OverCapError",
    "ShapeError",
    "is_gpu_instance_type",
    "is_gpu_os_image",
    "projected_cost_usd",
    "select_default_instance_type",
    "validate_cpu_only",
    "validate_within_cap",
]
