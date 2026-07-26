"""Lium execution backend is gated on a full constation bundle (todo 19).

Renamed from the misleading test_lium_client.py (which tested no client).
"""

from __future__ import annotations

import pytest

from prism_challenge.constation import ConstationBundle
from prism_challenge.queue import (
    LIUM_EXECUTION_BACKEND,
    SUPPORTED_EXECUTION_BACKENDS,
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


def test_lium_without_bundle_rejected() -> None:
    assert is_execution_backend_supported(LIUM_EXECUTION_BACKEND) is False
    assert is_execution_backend_supported(LIUM_EXECUTION_BACKEND, constation_bundle=None) is False
    with pytest.raises(ValueError, match="constation bundle required for lium"):
        require_execution_backend(LIUM_EXECUTION_BACKEND)
    with pytest.raises(ValueError, match="constation bundle required for lium"):
        require_execution_backend("lium", constation_bundle=None)


def test_lium_with_full_bundle_accepted() -> None:
    bundle = _bundle()
    assert is_execution_backend_supported("lium", constation_bundle=bundle) is True
    require_execution_backend("lium", constation_bundle=bundle)  # no raise


def test_remote_provider_and_local_cpu_still_rejected() -> None:
    assert "remote_provider" not in SUPPORTED_EXECUTION_BACKENDS
    assert "local_cpu" not in SUPPORTED_EXECUTION_BACKENDS
    with pytest.raises(ValueError, match="Unsupported execution backend"):
        require_execution_backend("remote_provider")
    with pytest.raises(ValueError, match="Unsupported execution backend"):
        require_execution_backend("local_cpu", constation_bundle=_bundle())
