"""Load custody master key and build production constation runtime (fail-closed)."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from base.compute.constation_custody import LiumKeyCustody
from base.compute.constation_poller import PollerConfig
from base.master.constation.orchestrator import ProductionConstationOrchestrator
from base.master.constation.pod_binding import MinerPodBinding
from base.security.admin_auth import read_secret

logger = logging.getLogger(__name__)


class _HasConstation(Protocol):
    @property
    def constation(self) -> _ConstationSurface: ...


class _ConstationSurface(Protocol):
    enabled: bool
    custody_master_key: str | None
    custody_master_key_file: Path | None
    gap_budget_seconds: float
    min_interval_seconds: float
    max_interval_seconds: float
    max_polls: int
    sidecar_internal_port: int
    poll_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ConstationRuntime:
    """Optional production constation services attached at master boot."""

    enabled: bool
    pod_binding: MinerPodBinding | None
    orchestrator: ProductionConstationOrchestrator | None


def load_custody_master_key(settings: _HasConstation) -> bytes | None:
    """Read Fernet master key from settings (inline or file). Empty → None."""
    cs = settings.constation
    file_path = cs.custody_master_key_file
    raw = read_secret(
        cs.custody_master_key,
        str(file_path) if file_path is not None else None,
    )
    text = raw.strip() if raw else ""
    if not text:
        return None
    return text.encode("utf-8")


def poller_config_from_settings(cs: _ConstationSurface) -> PollerConfig:
    """Map ConstationSettings poll fields onto PollerConfig."""
    max_polls = int(cs.max_polls)
    return PollerConfig(
        gap_budget_seconds=float(cs.gap_budget_seconds),
        min_interval_seconds=float(cs.min_interval_seconds),
        max_interval_seconds=float(cs.max_interval_seconds),
        max_polls=max_polls,
        max_cost_units=float(max_polls),
    )


def build_constation_runtime(
    settings: _HasConstation,
    *,
    nonce_service: Any,
    bundle_store: Any,
) -> ConstationRuntime:
    """Construct custody + binding + orchestrator when enabled and key present.

    Fail-closed: enabled without a usable master key logs an error and returns
    ``pod_binding=None`` / ``orchestrator=None`` so master boot continues and
    register_miner_key stays 503. Never logs key material.
    """
    cs = settings.constation
    if not cs.enabled:
        return ConstationRuntime(enabled=False, pod_binding=None, orchestrator=None)

    master_key = load_custody_master_key(settings)
    if master_key is None:
        logger.error(
            "constation.enabled is True but custody master key is missing "
            "(set custody_master_key or custody_master_key_file); "
            "constation custody/orchestrator disabled"
        )
        return ConstationRuntime(enabled=True, pod_binding=None, orchestrator=None)

    try:
        custody = LiumKeyCustody(master_key=master_key)
    except (ValueError, TypeError) as exc:
        logger.error(
            "constation.enabled is True but custody master key is invalid "
            "(%s); constation custody/orchestrator disabled",
            type(exc).__name__,
        )
        return ConstationRuntime(enabled=True, pod_binding=None, orchestrator=None)

    pod_binding = MinerPodBinding(custody=custody)
    orchestrator = ProductionConstationOrchestrator(
        pod_binding=pod_binding,
        nonce_service=nonce_service,
        bundle_store=bundle_store,
        poller_config=poller_config_from_settings(cs),
        now_fn=time.monotonic,
        sleep_fn=asyncio.sleep,
        rng_fn=random.random,
        sidecar_internal_port=int(cs.sidecar_internal_port),
        sidecar_timeout_seconds=float(cs.poll_timeout_seconds),
    )
    return ConstationRuntime(
        enabled=True, pod_binding=pod_binding, orchestrator=orchestrator
    )


def make_constation_pre_forward_hook(
    orchestrator: ProductionConstationOrchestrator | None,
    *,
    duration_seconds: float,
):
    """Return async hook for WorkerReconciliationService (or None).

    Invokes orchestrator only when work-unit metadata carries full identity
    (required_digest, commit/tree/variant, sealed_manifest_hashes). Incomplete
    identity is skipped with a debug log — services remain wired for register.
    Hook errors are logged and do not block result forward.
    """
    if orchestrator is None:
        return None

    async def _hook(
        *,
        work_unit_id: str,
        miner_hotkey: str,
        metadata: Mapping[str, Any],
    ) -> None:
        from base.master.constation.orchestrator import (
            ConstationOrchestrationRequest,
        )

        md = dict(metadata or {})
        required_digest = md.get("required_digest") or md.get("digest")
        commit_sha = md.get("commit_sha")
        tree_sha = md.get("tree_sha")
        variant = md.get("variant")
        sealed = md.get("sealed_manifest_hashes")
        if not (
            isinstance(required_digest, str)
            and required_digest
            and isinstance(commit_sha, str)
            and commit_sha
            and isinstance(tree_sha, str)
            and tree_sha
            and variant is not None
            and isinstance(sealed, Mapping)
            and sealed
        ):
            logger.debug(
                "constation pre-forward hook skipped: incomplete identity "
                "on work_unit_id=%s miner_hotkey=%s",
                work_unit_id,
                miner_hotkey,
            )
            return
        try:
            await orchestrator.run(
                ConstationOrchestrationRequest(
                    work_unit_id=work_unit_id,
                    miner_hotkey=miner_hotkey,
                    required_digest=required_digest,
                    commit_sha=commit_sha,
                    tree_sha=tree_sha,
                    variant=variant,
                    sealed_manifest_hashes=dict(sealed),
                    duration_seconds=float(
                        md.get("duration_seconds", duration_seconds)
                    ),
                    pod_id=md.get("pod_id")
                    if isinstance(md.get("pod_id"), str)
                    else None,
                    instance_id=(
                        md.get("instance_id")
                        if isinstance(md.get("instance_id"), str)
                        else None
                    ),
                )
            )
        except Exception:
            logger.exception(
                "constation orchestrator failed for work_unit_id=%s "
                "(forward continues)",
                work_unit_id,
            )

    return _hook


__all__ = [
    "ConstationRuntime",
    "build_constation_runtime",
    "load_custody_master_key",
    "make_constation_pre_forward_hook",
    "poller_config_from_settings",
]
