"""Build training-locked Lium clients/schedulers from LiumTrainingSettings.

Fail-closed: ``lium_training.enabled=True`` without a usable API key raises.
Disabled plane returns ``None`` from the client factory (no network client).
Never logs key material.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from base.compute.lium import LiumClient
from base.compute.lium_capacity import LiumCapacityClient, LiumCapacityScheduler
from base.compute.worker_deployment import WORKER_IMAGE
from base.security.admin_auth import read_secret

logger = logging.getLogger(__name__)
# Default training pod image (no digest pin yet — T14). Prefer prism-evaluator
# over the scheduler placeholder ``ghcr.io/base/prism-train:latest``.
DEFAULT_LIUM_TRAINING_IMAGE = WORKER_IMAGE


class _LiumTrainingSurface(Protocol):
    enabled: bool
    api_key: str | None
    api_key_file: Path | None
    concurrency_cap: int
    pod_name_prefix: str
    max_price_per_hour: float
    max_lifetime_hours: float
    ssh_public_key_file: Path | None


class _HasLiumTraining(Protocol):
    @property
    def lium_training(self) -> _LiumTrainingSurface: ...


def resolve_lium_training_api_key(lt: _LiumTrainingSurface) -> str | None:
    """Resolve API key from inline ``api_key`` or ``api_key_file`` (strip).

    Empty / whitespace-only → ``None``. Never logs the key.
    """
    file_path = lt.api_key_file
    raw = read_secret(
        lt.api_key,
        str(file_path) if file_path is not None else None,
    )
    text = raw.strip() if raw else ""
    if not text:
        return None
    return text


def build_lium_training_client(settings: _HasLiumTraining) -> LiumClient | None:
    """Construct a training-locked :class:`LiumClient` when the plane is on.

    * ``enabled=False`` → ``None``
    * ``enabled=True`` without key → :class:`ValueError` (fail-closed)
    * ``enabled=True`` with key → :meth:`LiumClient.for_prism_training`
    """
    lt = settings.lium_training
    if not lt.enabled:
        return None

    api_key = resolve_lium_training_api_key(lt)
    if api_key is None:
        raise ValueError(
            "lium_training.enabled is True but API key is missing "
            "(set lium_training.api_key or lium_training.api_key_file)"
        )
    return LiumClient.for_prism_training(api_key)


def _load_ssh_public_keys(lt: _LiumTrainingSurface) -> tuple[str, ...] | None:
    path = lt.ssh_public_key_file
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(
            "lium_training.ssh_public_key_file is set but is not a readable file"
        )
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("lium_training.ssh_public_key_file is empty")
    return (text,)


def build_lium_capacity_scheduler(
    settings: _HasLiumTraining,
    *,
    image: str | None = None,
    client_factory: Callable[[], LiumCapacityClient] | None = None,
    store: object | None = None,
    spend_gate: Callable[[], bool] | None = None,
) -> LiumCapacityScheduler:
    """Build :class:`LiumCapacityScheduler` from settings + training-locked client.

    Requires ``lium_training.enabled=True`` and a resolvable API key (fail-closed).
    Image defaults to :data:`DEFAULT_LIUM_TRAINING_IMAGE` (prism-evaluator, no digest).
    """
    lt = settings.lium_training
    if not lt.enabled:
        raise ValueError(
            "lium_training.enabled is False; refuse to build LiumCapacityScheduler"
        )

    factory = client_factory
    if factory is None:
        # Capture key once so factory does not re-read settings on every call
        # in a way that could race; still fail-closed if key missing.
        api_key = resolve_lium_training_api_key(lt)
        if api_key is None:
            raise ValueError(
                "lium_training.enabled is True but API key is missing "
                "(set lium_training.api_key or lium_training.api_key_file)"
            )

        def factory() -> LiumCapacityClient:
            return LiumClient.for_prism_training(api_key)

    ssh_keys = _load_ssh_public_keys(lt)
    kwargs: dict[str, object] = {
        "concurrency_cap": int(lt.concurrency_cap),
        "pod_name_prefix": str(lt.pod_name_prefix),
        "max_price_per_hour": float(lt.max_price_per_hour),
        "max_lifetime_hours": float(lt.max_lifetime_hours),
        "image": image if image is not None else DEFAULT_LIUM_TRAINING_IMAGE,
    }
    if store is not None:
        kwargs["store"] = store
    if spend_gate is not None:
        kwargs["spend_gate"] = spend_gate
    if ssh_keys is not None:
        kwargs["ssh_public_keys"] = ssh_keys

    return LiumCapacityScheduler(factory, **kwargs)  # type: ignore[arg-type]


def try_build_lium_capacity_scheduler(
    settings: _HasLiumTraining,
    *,
    image: str | None = None,
    client_factory: Callable[[], LiumCapacityClient] | None = None,
    store: object | None = None,
    spend_gate: Callable[[], bool] | None = None,
) -> LiumCapacityScheduler | None:
    """Build scheduler when ``lium_training.enabled``; else ``None``.

    Fail-closed on missing key when enabled: logs and returns ``None`` so the
    master still boots (worker-plane / validator path unchanged). Callers that
    need hard fail should use :func:`build_lium_capacity_scheduler` directly.
    """
    if not settings.lium_training.enabled:
        return None
    try:
        return build_lium_capacity_scheduler(
            settings,
            image=image,
            client_factory=client_factory,
            store=store,
            spend_gate=spend_gate,
        )
    except Exception:
        logger.exception(
            "lium_training.enabled but LiumCapacityScheduler build failed; "
            "continuing without master-owned Lium admission"
        )
        return None


async def run_lium_capacity_tick(
    scheduler: LiumCapacityScheduler | None,
) -> None:
    """Safe one-shot tick for an optional scheduler (no-op if ``None``).

    Intended for a dedicated background loop if orchestration is not the
    tick owner. Failures are logged; capacity never raises to the caller.
    """
    if scheduler is None:
        return
    try:
        await scheduler.tick()
    except Exception:
        logger.exception("lium capacity tick failed")
