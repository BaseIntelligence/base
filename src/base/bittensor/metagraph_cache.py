from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetagraphCache:
    netuid: int
    ttl_seconds: int = 300
    subtensor: Any | None = None
    _hotkey_to_uid: dict[str, int] = field(default_factory=dict)
    _validator_permits: dict[str, bool] = field(default_factory=dict)
    _stakes: dict[str, float] = field(default_factory=dict)
    _updated_at: float = 0.0

    @property
    def hotkey_to_uid(self) -> dict[str, int]:
        return dict(self._hotkey_to_uid)

    def expired(self) -> bool:
        return time.time() - self._updated_at > self.ttl_seconds

    def update_from_hotkeys(self, hotkeys: list[str]) -> dict[str, int]:
        return self.update_from_metagraph(hotkeys)

    def update_from_metagraph(
        self,
        hotkeys: list[str],
        *,
        validator_permits: list[bool] | None = None,
        stakes: list[float] | None = None,
    ) -> dict[str, int]:
        """Replace the cached metagraph snapshot keyed by hotkey.

        ``validator_permits`` and ``stakes`` are positional per-uid sequences
        aligned with ``hotkeys``; missing entries default to ``False``/``0.0``.
        """

        self._hotkey_to_uid = {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
        permits = list(validator_permits or [])
        stake_values = list(stakes or [])
        self._validator_permits = {
            hotkey: bool(permits[uid])
            for uid, hotkey in enumerate(hotkeys)
            if uid < len(permits)
        }
        self._stakes = {
            hotkey: float(stake_values[uid])
            for uid, hotkey in enumerate(hotkeys)
            if uid < len(stake_values)
        }
        self._updated_at = time.time()
        return self.hotkey_to_uid

    def refresh(self) -> dict[str, int]:
        if self.subtensor is None:
            raise RuntimeError("Subtensor is required to refresh metagraph")
        metagraph = self.subtensor.metagraph(self.netuid)
        hotkeys = list(getattr(metagraph, "hotkeys", []))
        permits = [bool(value) for value in getattr(metagraph, "validator_permit", [])]
        stakes = [float(value) for value in getattr(metagraph, "S", [])]
        return self.update_from_metagraph(
            hotkeys, validator_permits=permits, stakes=stakes
        )

    def get(self, *, force: bool = False) -> dict[str, int]:
        if force or self.expired():
            return self.refresh()
        return self.hotkey_to_uid

    def validator_permit(self, hotkey: str) -> bool:
        """Return whether ``hotkey`` holds a validator permit in the snapshot."""

        return self._validator_permits.get(hotkey, False)

    def stake(self, hotkey: str) -> float:
        """Return the cached stake for ``hotkey`` (``0.0`` when unknown)."""

        return self._stakes.get(hotkey, 0.0)

    def is_validator(self, hotkey: str) -> bool:
        """True when ``hotkey`` is on the metagraph AND holds a validator permit."""

        return hotkey in self._hotkey_to_uid and self.validator_permit(hotkey)
