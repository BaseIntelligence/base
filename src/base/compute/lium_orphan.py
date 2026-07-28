"""Terminate master-owned Lium pods that no longer hold an active lease.

Ownership marker is the pod name prefix (default ``prism-train-``). Active
lease sets are supplied by the caller so this module stays independent of
capacity/lease bookkeeping.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_DEFAULT_POD_NAME_PREFIX = "prism-train-"


@runtime_checkable
class LiumOrphanClient(Protocol):
    """Minimal Lium surface required by :func:`reconcile_orphan_pods`."""

    async def list_pods(self) -> list[dict[str, Any]]:
        """Return raw pod dicts from the account."""
        ...

    async def terminate(self, instance_id: str) -> None:
        """Request pod deletion (idempotent on 404)."""
        ...

    async def verify_terminated(self, instance_id: str) -> bool:
        """Return True when ``instance_id`` is absent from list_pods."""
        ...


@dataclass(frozen=True, slots=True)
class OrphanTermination:
    """Outcome for one pod considered during orphan reconciliation."""

    pod_id: str
    pod_name: str
    verified: bool
    skipped_reason: str | None = None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_pod_id(pod: Mapping[str, Any]) -> str | None:
    for key in ("id", "pod_id", "uuid"):
        found = _optional_text(pod.get(key))
        if found is not None:
            return found
    return None


def _extract_pod_name(pod: Mapping[str, Any]) -> str | None:
    for key in ("pod_name", "name"):
        found = _optional_text(pod.get(key))
        if found is not None:
            return found
    return None


def _is_actively_leased(
    *,
    pod_id: str,
    pod_name: str,
    active_lease_pod_ids: Set[str],
    active_lease_pod_names: Set[str] | None,
) -> bool:
    if pod_id in active_lease_pod_ids:
        return True
    if active_lease_pod_names is not None and pod_name in active_lease_pod_names:
        return True
    return False


async def reconcile_orphan_pods(
    client: LiumOrphanClient,
    *,
    pod_name_prefix: str = _DEFAULT_POD_NAME_PREFIX,
    active_lease_pod_ids: Set[str],
    active_lease_pod_names: Set[str] | None = None,
) -> list[OrphanTermination]:
    """Terminate prefix-owned pods that are not covered by an active lease.

    Only pods whose name starts with ``pod_name_prefix`` are candidates.
    Pods missing a usable id or name are skipped fail-closed (never terminated).
    Non-prefix pods are ignored entirely.
    """
    pods: Sequence[Mapping[str, Any]] = await client.list_pods()
    results: list[OrphanTermination] = []

    for raw in pods:
        if not isinstance(raw, Mapping):
            logger.warning("lium orphan reconcile skipped non-mapping pod entry")
            continue

        pod_id = _extract_pod_id(raw)
        pod_name = _extract_pod_name(raw)

        if pod_name is None:
            results.append(
                OrphanTermination(
                    pod_id=pod_id or "",
                    pod_name="",
                    verified=False,
                    skipped_reason="missing_pod_name",
                )
            )
            logger.warning(
                "lium orphan reconcile skipped pod with missing name (id=%r)",
                pod_id,
            )
            continue

        if not pod_name.startswith(pod_name_prefix):
            continue

        if pod_id is None:
            results.append(
                OrphanTermination(
                    pod_id="",
                    pod_name=pod_name,
                    verified=False,
                    skipped_reason="missing_pod_id",
                )
            )
            logger.warning(
                "lium orphan reconcile skipped prefix pod with missing id (name=%r)",
                pod_name,
            )
            continue

        if _is_actively_leased(
            pod_id=pod_id,
            pod_name=pod_name,
            active_lease_pod_ids=active_lease_pod_ids,
            active_lease_pod_names=active_lease_pod_names,
        ):
            continue

        await client.terminate(pod_id)
        verified = await client.verify_terminated(pod_id)
        results.append(
            OrphanTermination(
                pod_id=pod_id,
                pod_name=pod_name,
                verified=verified,
                skipped_reason=None,
            )
        )
        if not verified:
            logger.warning(
                "lium orphan terminate issued but pod still listed (id=%s name=%s)",
                pod_id,
                pod_name,
            )

    return results
