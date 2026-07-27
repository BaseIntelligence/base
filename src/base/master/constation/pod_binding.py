"""Miner Lium API key + instance_id binding (domain; no HTTP).

Registration is fail-closed:

1. Probe the API key (same path as ``LiumKeyCustody``)
2. ``get_pod_raw(instance_id)`` via the probed client
3. :func:`~base.compute.constation_pod.assert_pod_bound` (running + hotkey match)
4. Only then store encrypted key + ``instance_id`` keyed by miner hotkey

Never logs ``api_key``. Routes live in a later task.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from base.compute.constation_custody import LiumKeyCustody
from base.compute.constation_pod import assert_pod_bound
from base.compute.constation_types import ConstationFailCode, ConstationVerdict
from base.compute.lium import (
    LiumAuthError,
    LiumError,
    LiumNotFoundError,
    LiumRateLimitError,
)

logger = logging.getLogger(__name__)


@dataclass
class MinerPodBinding:
    """In-memory miner hotkey → (encrypted Lium key, instance_id) binding.

    Composes :class:`LiumKeyCustody` for Fernet key storage. Instance ids are
    plain strings (not secret). ``repr`` never includes api keys.
    """

    custody: LiumKeyCustody
    _instance_by_hotkey: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __repr__(self) -> str:
        return f"MinerPodBinding(bound={len(self._instance_by_hotkey)})"

    def has_binding(self, miner_hotkey: str) -> bool:
        hotkey = miner_hotkey.strip()
        return hotkey in self._instance_by_hotkey and self.custody.has_key(hotkey)

    def get_instance_id(self, miner_hotkey: str) -> str | None:
        return self._instance_by_hotkey.get(miner_hotkey.strip())

    async def register(
        self,
        *,
        miner_hotkey: str,
        api_key: str,
        instance_id: str,
    ) -> ConstationVerdict:
        """Probe key, bind pod, then store encrypted key + instance_id.

        Fail closed on bad key / mismatch / not running / pod fetch errors.
        Never logs ``api_key``.
        """
        hotkey = _require_nonblank("miner_hotkey", miner_hotkey)
        key = _require_nonblank("api_key", api_key)
        pod_id = _require_nonblank("instance_id", instance_id)

        client = self.custody.client_factory(key)
        try:
            await self.custody.probe_fn(client)
        except LiumAuthError:
            logger.warning("lium key probe rejected (401) for miner_hotkey=%s", hotkey)
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.LIUM_AUTH_REVOKED,
                detail="probe_401",
            )
        except LiumError as exc:
            logger.warning(
                "lium key probe failed for miner_hotkey=%s status=%s",
                hotkey,
                getattr(exc, "status_code", None),
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.PROBE_FAILED,
                detail=type(exc).__name__,
            )

        try:
            pod = await client.get_pod_raw(pod_id)
        except LiumAuthError:
            logger.warning(
                "lium get_pod rejected (401) for miner_hotkey=%s instance_id=%s",
                hotkey,
                pod_id,
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.LIUM_AUTH_REVOKED,
                detail="get_pod_401",
            )
        except LiumNotFoundError:
            logger.warning(
                "lium get_pod not found for miner_hotkey=%s instance_id=%s",
                hotkey,
                pod_id,
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.POD_HOTKEY_MISMATCH,
                detail="pod_not_found",
            )
        except LiumRateLimitError:
            logger.warning(
                "lium get_pod rate limited for miner_hotkey=%s instance_id=%s",
                hotkey,
                pod_id,
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.LIUM_RATE_LIMITED,
                detail="get_pod_429",
            )
        except LiumError as exc:
            logger.warning(
                "lium get_pod failed for miner_hotkey=%s instance_id=%s status=%s",
                hotkey,
                pod_id,
                getattr(exc, "status_code", None),
            )
            return ConstationVerdict(
                ok=False,
                reason=ConstationFailCode.PROBE_FAILED,
                detail=type(exc).__name__,
            )

        bound = assert_pod_bound(pod_raw=pod.raw, expected_hotkey=hotkey)
        if not bound.ok:
            logger.warning(
                "pod bind failed for miner_hotkey=%s instance_id=%s reason=%s",
                hotkey,
                pod_id,
                bound.reason,
            )
            return bound

        self.custody.store_probed_key(miner_hotkey=hotkey, api_key=key)
        self._instance_by_hotkey[hotkey] = pod_id
        logger.info(
            "miner pod bound for miner_hotkey=%s instance_id=%s",
            hotkey,
            pod_id,
        )
        return ConstationVerdict(ok=True, reason=ConstationFailCode.OK)


def _require_nonblank(field_name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return normalized


__all__ = ["MinerPodBinding"]
