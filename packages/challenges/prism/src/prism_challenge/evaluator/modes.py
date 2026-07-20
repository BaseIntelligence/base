from __future__ import annotations

from .schemas import ExecutionMode

GPU_PROXY_TOKEN_TARGET = 10_000_000_000
FULL_SCALE_PHASE_1_TOKEN_TARGET = 10_000_000_000
FULL_SCALE_PHASE_2_PARAMETER_TARGET = 1_000_000_000
FULL_SCALE_PHASE_2_TOKEN_TARGET = 100_000_000_000


def execution_mode_from_value(value: str | ExecutionMode | None) -> ExecutionMode:
    if value is None:
        return ExecutionMode.GPU_PROXY_EVAL
    return value if isinstance(value, ExecutionMode) else ExecutionMode(str(value))
