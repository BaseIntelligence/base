from __future__ import annotations

from prism_challenge.queue import SUPPORTED_EXECUTION_BACKENDS


def test_lium_backends_are_not_supported() -> None:
    assert "remote_provider" not in SUPPORTED_EXECUTION_BACKENDS
    assert "local_cpu" not in SUPPORTED_EXECUTION_BACKENDS
    assert "base_gpu" in SUPPORTED_EXECUTION_BACKENDS
