"""Pure helpers: miner hotkey + running status from a raw Lium pod payload.

Fail-closed. Reads only documented Lium ``PodDetailResponse`` fields:

* ``executor.miner_hotkey`` — never invents or falls back to top-level keys
* top-level ``status`` — running iff case-insensitive equality with ``RUNNING``

No network I/O. Callers pass ``LiumPodRead.raw`` (or any equivalent mapping).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from base.compute.constation_types import ConstationFailCode, ConstationVerdict

_RUNNING_STATUS = "RUNNING"


def extract_miner_hotkey(pod_raw: Mapping[str, Any]) -> str | None:
    """Return stripped ``executor.miner_hotkey``, or None if absent/unusable."""
    executor = pod_raw.get("executor")
    if not isinstance(executor, Mapping):
        return None
    raw = executor.get("miner_hotkey")
    if not isinstance(raw, str):
        return None
    hotkey = raw.strip()
    if not hotkey:
        return None
    return hotkey


def pod_is_running(pod_raw: Mapping[str, Any]) -> bool:
    """True only when top-level ``status`` is case-insensitively ``RUNNING``."""
    status = pod_raw.get("status")
    if not isinstance(status, str):
        return False
    return status.strip().upper() == _RUNNING_STATUS


def assert_pod_bound(
    *,
    pod_raw: Mapping[str, Any],
    expected_hotkey: str,
) -> ConstationVerdict:
    """Fail-closed bind check: hotkey match then running status.

    Precedence:
    1. missing/blank/mismatched ``executor.miner_hotkey`` → POD_HOTKEY_MISMATCH
    2. status not RUNNING → POD_NOT_RUNNING
    3. otherwise ok
    """
    want = expected_hotkey.strip()
    got = extract_miner_hotkey(pod_raw)
    if got is None or got != want:
        return ConstationVerdict(
            ok=False, reason=ConstationFailCode.POD_HOTKEY_MISMATCH
        )
    if not pod_is_running(pod_raw):
        return ConstationVerdict(ok=False, reason=ConstationFailCode.POD_NOT_RUNNING)
    return ConstationVerdict(ok=True, reason=ConstationFailCode.OK)


__all__ = [
    "assert_pod_bound",
    "extract_miner_hotkey",
    "pod_is_running",
]
