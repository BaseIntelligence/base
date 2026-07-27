"""TDD tests for pure Lium pod hotkey + running-status helpers.

Fail-closed: never invent miner_hotkey; only top-level status == RUNNING.
"""

from __future__ import annotations

from typing import Any

from base.compute.constation_pod import (
    assert_pod_bound,
    extract_miner_hotkey,
    pod_is_running,
)
from base.compute.constation_types import ConstationFailCode, ConstationVerdict

_HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"
_OTHER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"


def _pod(
    *,
    miner_hotkey: object = _HOTKEY,
    status: object = "RUNNING",
    executor: object | None = ...,  # type: ignore[assignment]
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Minimal Lium PodDetailResponse-shaped mapping (mirrors sidecar tests)."""
    if executor is ...:
        executor_val: object = {
            "executor_ip_address": "1.2.3.4",
            "miner_hotkey": miner_hotkey,
            "validator_hotkey": "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
        }
    else:
        executor_val = executor
    raw: dict[str, Any] = {
        "id": "pod-1",
        "status": status,
        "executor": executor_val,
    }
    if extra:
        raw.update(extra)
    return raw


def test_extract_miner_hotkey_when_nested_executor_present() -> None:
    # Given executor.miner_hotkey is a non-blank string
    # When extract_miner_hotkey runs
    result = extract_miner_hotkey(_pod(miner_hotkey=f"  {_HOTKEY}  "))
    # Then stripped hotkey is returned
    assert result == _HOTKEY


def test_extract_miner_hotkey_when_executor_missing() -> None:
    # Given no executor mapping
    # When extract_miner_hotkey runs
    assert extract_miner_hotkey(_pod(executor=None)) is None
    assert extract_miner_hotkey({"id": "pod-1", "status": "RUNNING"}) is None
    assert extract_miner_hotkey(_pod(executor="not-a-map")) is None


def test_extract_miner_hotkey_when_hotkey_absent_or_blank() -> None:
    # Given executor without usable miner_hotkey
    # When extract_miner_hotkey runs
    # Then fail-closed None (never invent)
    assert extract_miner_hotkey(_pod(miner_hotkey=None)) is None
    assert extract_miner_hotkey(_pod(miner_hotkey="")) is None
    assert extract_miner_hotkey(_pod(miner_hotkey="   ")) is None
    assert extract_miner_hotkey(_pod(miner_hotkey=12345)) is None
    # Top-level miner_hotkey must not be used as a fallback
    assert (
        extract_miner_hotkey(
            _pod(
                executor={"executor_ip_address": "1.2.3.4"},
                extra={"miner_hotkey": _HOTKEY},
            )
        )
        is None
    )


def test_pod_is_running_when_status_running() -> None:
    # Given top-level status RUNNING (any case / padding)
    assert pod_is_running(_pod(status="RUNNING")) is True
    assert pod_is_running(_pod(status="running")) is True
    assert pod_is_running(_pod(status=" Running ")) is True


def test_pod_is_running_when_stopped_or_absent() -> None:
    # Given non-RUNNING statuses or missing field
    assert pod_is_running(_pod(status="STOPPED")) is False
    assert pod_is_running(_pod(status="PENDING")) is False
    assert pod_is_running(_pod(status="FAILED")) is False
    assert pod_is_running(_pod(status="CREATION_FAILED")) is False
    assert pod_is_running(_pod(status="BROKEN")) is False
    assert pod_is_running(_pod(status="")) is False
    assert pod_is_running(_pod(status=None)) is False
    assert (
        pod_is_running({"id": "pod-1", "executor": {"miner_hotkey": _HOTKEY}}) is False
    )
    assert pod_is_running(_pod(status=1)) is False


def test_assert_pod_bound_ok_when_hotkey_matches_and_running() -> None:
    # Given matching hotkey and RUNNING
    # When assert_pod_bound runs
    result = assert_pod_bound(pod_raw=_pod(), expected_hotkey=_HOTKEY)
    # Then typed ok verdict
    assert isinstance(result, ConstationVerdict)
    assert result.ok is True
    assert result.reason is ConstationFailCode.OK
    assert bool(result) is True


def test_assert_pod_bound_fails_hotkey_mismatch() -> None:
    # Given running pod bound to a different miner
    result = assert_pod_bound(
        pod_raw=_pod(miner_hotkey=_OTHER), expected_hotkey=_HOTKEY
    )
    assert result.ok is False
    assert result.reason is ConstationFailCode.POD_HOTKEY_MISMATCH
    assert bool(result) is False


def test_assert_pod_bound_fails_when_hotkey_absent() -> None:
    # Given missing nested hotkey — treat as mismatch (fail-closed)
    result = assert_pod_bound(
        pod_raw=_pod(executor={"executor_ip_address": "1.2.3.4"}),
        expected_hotkey=_HOTKEY,
    )
    assert result.ok is False
    assert result.reason is ConstationFailCode.POD_HOTKEY_MISMATCH


def test_assert_pod_bound_fails_when_not_running() -> None:
    # Given matching hotkey but STOPPED
    result = assert_pod_bound(
        pod_raw=_pod(status="STOPPED"),
        expected_hotkey=_HOTKEY,
    )
    assert result.ok is False
    assert result.reason is ConstationFailCode.POD_NOT_RUNNING


def test_assert_pod_bound_mismatch_precedes_not_running() -> None:
    # Given wrong hotkey AND not running — hotkey check wins first
    result = assert_pod_bound(
        pod_raw=_pod(miner_hotkey=_OTHER, status="STOPPED"),
        expected_hotkey=_HOTKEY,
    )
    assert result.ok is False
    assert result.reason is ConstationFailCode.POD_HOTKEY_MISMATCH


def test_assert_pod_bound_strips_expected_hotkey() -> None:
    # Given expected_hotkey with surrounding whitespace
    result = assert_pod_bound(pod_raw=_pod(), expected_hotkey=f"  {_HOTKEY}  ")
    assert result.ok is True
    assert result.reason is ConstationFailCode.OK
