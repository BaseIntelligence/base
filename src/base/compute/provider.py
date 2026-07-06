"""Provider-agnostic compute contract for miner-funded GPU instances.

The worker plane provisions GPU instances on external providers (Lium, Targon)
that a miner pays for. This module defines the shared contract every provider
client implements plus the cost guardrails that are part of that contract:

* Every :class:`InstanceSpec` MUST carry a bounded ``max_lifetime_hours`` and a
  ``max_price_per_hour``; :meth:`ProviderClient.provision` refuses a spec that is
  unbounded (raising :class:`CostGuardrailError`) BEFORE any network call.
* Every ``provision`` code path, including exceptions, MUST attempt to
  ``terminate`` + ``verify_terminated`` the instance it created so a failed
  provisioning never leaks a billable pod (architecture.md sec 3.1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Base error for provider client operations."""


class CostGuardrailError(ProviderError):
    """A provision request violated a cost guardrail.

    Raised for an unbounded/non-positive lifetime, a missing price bound, or an
    offer whose price exceeds the spec's ``max_price_per_hour``. It is a typed
    error so callers can distinguish a guardrail refusal (never billed) from a
    generic transport/API failure.
    """


@dataclass(frozen=True)
class InstanceSpec:
    """A provider-agnostic request to provision one compute instance.

    ``max_lifetime_hours`` and ``max_price_per_hour`` are mandatory cost
    guardrails: they are typed optional only so a caller can construct an invalid
    spec that :meth:`ProviderClient.provision` will reject up front, but a valid
    provisioning requires both to be positive.
    """

    name: str
    template_ref: str | None = None
    image: str | None = None
    image_digest: str | None = None
    dockerfile_content: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    ports: tuple[int, ...] = (22,)
    ssh_public_keys: tuple[str, ...] = ()
    ssh_key_name: str | None = None
    max_lifetime_hours: float | None = None
    max_price_per_hour: float | None = None
    gpu_count: int = 1


@dataclass(frozen=True)
class Offer:
    """A rentable GPU offer exposed by a provider.

    ``price_per_hour`` is the per-GPU hourly price the cost guardrails filter on
    (Lium prices per GPU; the mission budget is expressed per GPU/hour).
    """

    id: str
    gpu_type: str
    gpu_count: int
    price_per_hour: float
    provider: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Instance:
    """A provisioned compute instance (a Lium pod / Targon workload)."""

    id: str
    status: str
    provider: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProviderClient(Protocol):
    """The contract every compute provider client implements."""

    async def list_offers(
        self, *, max_price_per_hour: float | None = None
    ) -> list[Offer]:
        """Return rentable offers, optionally filtered to ``<= price`` per hour."""
        ...

    async def provision(self, spec: InstanceSpec) -> Instance:
        """Provision an instance for ``spec`` under the cost guardrails."""
        ...

    async def status(self, instance_id: str) -> Instance:
        """Return the current state of a provisioned instance."""
        ...

    def stream_logs(self, instance_id: str) -> AsyncIterator[str]:
        """Yield log lines for a provisioned instance."""
        ...

    async def terminate(self, instance_id: str) -> None:
        """Terminate an instance; idempotent (a missing instance is success)."""
        ...

    async def verify_terminated(self, instance_id: str) -> bool:
        """Return ``True`` once the instance is absent from the provider."""
        ...
