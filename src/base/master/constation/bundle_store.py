"""In-memory + optional durable store for sealed ConstationBundle wire dicts."""

from __future__ import annotations

import copy
from typing import Any


class ConstationBundleStore:
    """Keyed by work_unit_id; last-write wins. Forwarder attaches stored bundles."""

    def __init__(self) -> None:
        self._by_wu: dict[str, dict[str, Any]] = {}

    def put(self, work_unit_id: str, bundle: dict[str, Any]) -> None:
        key = work_unit_id.strip()
        if not key:
            raise ValueError("work_unit_id must be non-empty")
        if not isinstance(bundle, dict):
            raise TypeError("bundle must be a dict")
        self._by_wu[key] = copy.deepcopy(bundle)

    def get(self, work_unit_id: str) -> dict[str, Any] | None:
        key = work_unit_id.strip()
        blob = self._by_wu.get(key)
        return copy.deepcopy(blob) if blob is not None else None

    def pop(self, work_unit_id: str) -> dict[str, Any] | None:
        key = work_unit_id.strip()
        blob = self._by_wu.pop(key, None)
        return copy.deepcopy(blob) if blob is not None else None


__all__ = ["ConstationBundleStore"]
