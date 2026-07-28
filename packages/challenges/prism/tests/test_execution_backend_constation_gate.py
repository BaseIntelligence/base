"""Execution-backend gate: bare ``lium`` is allowed (T14 unattested / master-owned).

Container backends stay always-on. ``constation_bundle`` remains an API-compat
kwarg but does **not** gate Lium selection. ``constation_ok`` score elevation is
a separate ingestion path; constation modules are not deleted.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.constation import ConstationBundle
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.queue import (
    GATED_EXECUTION_BACKENDS,
    LIUM_EXECUTION_BACKEND,
    SUPPORTED_EXECUTION_BACKENDS,
    PrismWorker,
    is_execution_backend_supported,
    require_execution_backend,
)


def _bundle() -> ConstationBundle:
    return ConstationBundle(
        commit_sha="a" * 40,
        tree_sha="b" * 40,
        variant="cuda",
        digest="sha256:" + ("1" * 64),
        work_unit_id="wu-1",
        miner_hotkey="hk",
        pod_id="pod-1",
        nonce="n-1",
        signed_attestation={"sig": "x"},
        expected_sealed_manifest_hashes={"h.py": "c" * 64},
        reported_sealed_manifest_hashes={"h.py": "c" * 64},
        lium_declared_digest="sha256:" + ("1" * 64),
        constation_gap_budget_seconds=30.0,
        constation_observed_max_gap_seconds=1.0,
    )


def test_base_gpu_always_supported_without_bundle() -> None:
    assert "base_gpu" in SUPPORTED_EXECUTION_BACKENDS
    assert is_execution_backend_supported("base_gpu") is True
    assert is_execution_backend_supported("base_gpu", constation_bundle=None) is True
    require_execution_backend("base_gpu")  # no raise


def test_lium_without_bundle_accepted_unattested() -> None:
    """T14: bare ``lium`` is a supported compute backend (no constation required)."""
    assert LIUM_EXECUTION_BACKEND in SUPPORTED_EXECUTION_BACKENDS
    assert LIUM_EXECUTION_BACKEND not in GATED_EXECUTION_BACKENDS
    assert is_execution_backend_supported(LIUM_EXECUTION_BACKEND) is True
    assert (
        is_execution_backend_supported(LIUM_EXECUTION_BACKEND, constation_bundle=None)
        is True
    )
    require_execution_backend(LIUM_EXECUTION_BACKEND)  # no raise
    require_execution_backend("lium", constation_bundle=None)  # no raise


def test_lium_with_full_bundle_still_accepted() -> None:
    """Bundle remains optional API-compat; still accepted when supplied."""
    bundle = _bundle()
    assert is_execution_backend_supported("lium", constation_bundle=bundle) is True
    require_execution_backend("lium", constation_bundle=bundle)  # no raise


def test_prism_worker_constructs_with_bare_lium() -> None:
    """PrismWorker(execution_backend=lium) constructs without constation_bundle."""
    repo = SimpleNamespace(claim_next=AsyncMock(return_value=None))
    worker = PrismWorker(
        repository=repo,  # type: ignore[arg-type]
        ctx=PrismContext(),
        execution_backend="lium",
        settings=PrismSettings(docker_enabled=False, plagiarism_enabled=False),
    )
    assert worker.execution_backend == "lium"
    assert worker._constation_bundle is None  # noqa: SLF001


def test_remote_provider_and_local_cpu_still_rejected() -> None:
    assert "remote_provider" not in SUPPORTED_EXECUTION_BACKENDS
    assert "local_cpu" not in SUPPORTED_EXECUTION_BACKENDS
    with pytest.raises(ValueError, match="Unsupported execution backend"):
        require_execution_backend("remote_provider")
    with pytest.raises(ValueError, match="Unsupported execution backend"):
        require_execution_backend("local_cpu", constation_bundle=_bundle())
