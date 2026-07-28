"""Unit tests for Lium orphan pod reconciler (prefix-owned money-leak guard)."""

from __future__ import annotations

from typing import Any

import pytest

from base.compute.lium_orphan import OrphanTermination, reconcile_orphan_pods


class _FakeLiumClient:
    """In-memory stand-in for list_pods / terminate / verify_terminated."""

    def __init__(self, pods: list[dict[str, Any]]) -> None:
        self._pods = [dict(p) for p in pods]
        self.terminated_ids: list[str] = []

    async def list_pods(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self._pods]

    async def terminate(self, instance_id: str) -> None:
        self.terminated_ids.append(str(instance_id))
        remaining: list[dict[str, Any]] = []
        for pod in self._pods:
            pod_id = str(pod.get("id") or pod.get("pod_id") or "")
            if pod_id != str(instance_id):
                remaining.append(pod)
        self._pods = remaining

    async def verify_terminated(self, instance_id: str) -> bool:
        for pod in self._pods:
            pod_id = str(pod.get("id") or pod.get("pod_id") or "")
            if pod_id == str(instance_id):
                return False
        return True


@pytest.mark.asyncio
async def test_orphan_terminator_only_prefix() -> None:
    """Prefix orphan is terminated; foreign pods are left alone."""
    client = _FakeLiumClient(
        [
            {"id": "orphan-1", "pod_name": "prism-train-job-aaa"},
            {"id": "other-1", "name": "user-notebook"},
        ]
    )

    results = await reconcile_orphan_pods(
        client,
        active_lease_pod_ids=set(),
    )

    assert client.terminated_ids == ["orphan-1"]
    assert len(results) == 1
    assert results[0] == OrphanTermination(
        pod_id="orphan-1",
        pod_name="prism-train-job-aaa",
        verified=True,
        skipped_reason=None,
    )


@pytest.mark.asyncio
async def test_orphan_skips_leased() -> None:
    """Leased prefix pods (by id or name) are not terminated."""
    client = _FakeLiumClient(
        [
            {"id": "leased-id", "pod_name": "prism-train-active-1"},
            {"id": "leased-name-id", "name": "prism-train-active-2"},
            {"id": "true-orphan", "pod_name": "prism-train-dead"},
        ]
    )

    results = await reconcile_orphan_pods(
        client,
        active_lease_pod_ids={"leased-id"},
        active_lease_pod_names={"prism-train-active-2"},
    )

    assert client.terminated_ids == ["true-orphan"]
    kept = [r.pod_id for r in results if r.skipped_reason is None]
    assert kept == ["true-orphan"]


@pytest.mark.asyncio
async def test_orphan_skips_non_prefix() -> None:
    """Non-prefix pods never reach terminate."""
    client = _FakeLiumClient(
        [
            {"id": "a", "pod_name": "miner-pod"},
            # Does not start with prism-train-
            {"id": "b", "name": "prism-trainX-nope"},
            {"id": "c", "pod_name": "other-prism-train-suffix"},
        ]
    )

    results = await reconcile_orphan_pods(
        client,
        active_lease_pod_ids=set(),
    )

    assert client.terminated_ids == []
    assert results == []


@pytest.mark.asyncio
async def test_empty_pods_ok() -> None:
    """Empty account yields empty results and no terminate calls."""
    client = _FakeLiumClient([])

    results = await reconcile_orphan_pods(
        client,
        active_lease_pod_ids=frozenset({"anything"}),
    )

    assert results == []
    assert client.terminated_ids == []


@pytest.mark.asyncio
async def test_orphan_skips_unidentified_fail_closed() -> None:
    """Missing id/name is fail-closed: skip terminate rather than guess."""
    client = _FakeLiumClient(
        [
            {"pod_name": "prism-train-no-id"},
            {"id": "mystery-1"},
            {"id": "ok-orphan", "name": "prism-train-ok"},
        ]
    )

    results = await reconcile_orphan_pods(
        client,
        active_lease_pod_ids=set(),
    )

    assert client.terminated_ids == ["ok-orphan"]
    skipped = [r for r in results if r.skipped_reason is not None]
    assert len(skipped) == 2
    assert all(r.verified is False for r in skipped)
    terminated = [r for r in results if r.skipped_reason is None]
    assert len(terminated) == 1
    assert terminated[0].pod_id == "ok-orphan"
    assert terminated[0].verified is True
