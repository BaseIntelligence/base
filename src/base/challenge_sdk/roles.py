"""Canonical, inspectable role and capability contracts.

The registry in this module is deliberately data-only.  It is safe to import
from a challenge, validator, or worker without importing the master runtime.
Runtime identity is installed by the server process with :func:`activate_role`;
request headers and payload claims are never consulted.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from types import MappingProxyType
from typing import Any, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")


CAPABILITY_REGISTRY_VERSION = "1"


class Role(StrEnum):
    MASTER = "master"
    VALIDATOR = "validator"
    CHALLENGE = "challenge"
    WORKER = "worker"


class Capability(StrEnum):
    MASTER_COORDINATION = "master.coordination"
    MASTER_REGISTRY = "master.registry"
    MASTER_PERSISTENCE = "master.persistence"
    MASTER_RAW_WEIGHT_INGRESS = "master.raw_weight_ingress"
    MASTER_AGGREGATION = "master.aggregation"
    MASTER_VECTOR_READ = "master.vector_read"
    MASTER_WATCHER = "master.watcher"
    VALIDATOR_REGISTRATION = "validator.registration"
    VALIDATOR_HEARTBEAT = "validator.heartbeat"
    VALIDATOR_ASSIGNMENT_PULL = "validator.assignment_pull"
    VALIDATOR_ASSIGNMENT_PROGRESS = "validator.assignment_progress"
    VALIDATOR_ASSIGNMENT_RESULT = "validator.assignment_result"
    VALIDATOR_VECTOR_READ = "validator.vector_read"
    VALIDATOR_OWN_SET_WEIGHTS = "validator.own_set_weights"
    CHALLENGE_SCORING = "challenge.scoring"
    CHALLENGE_ORDINARY_PROOF = "challenge.ordinary_proof"
    CHALLENGE_TEE_VERIFICATION = "challenge.tee_verification"
    CHALLENGE_STATE = "challenge.state"
    CHALLENGE_RAW_WEIGHT_PUSH = "challenge.raw_weight_push"
    WORKER_ASSIGNMENT_EXECUTION = "worker.assignment_execution"
    WORKER_RESULT_REPORTING = "worker.result_reporting"


@dataclass(frozen=True)
class CapabilitySpec:
    """One registry entry and its callable operation identity."""

    token: str
    role: Role
    operation: str
    route: str | None
    cli_operation: str | None
    credential_domain: str
    side_effect_class: str


@dataclass(frozen=True)
class RoleContext:
    """Authenticated server identity available to decorated operations."""

    role: Role
    capabilities: tuple[str, ...]


class RoleContractError(PermissionError):
    """Raised before an operation when the server role is not authorized."""


_CAPABILITY_SPECS: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        Capability.MASTER_COORDINATION,
        Role.MASTER,
        "coordination",
        "/v1/validators/*",
        "master coordination",
        "validator-signature",
        "persistence",
    ),
    CapabilitySpec(
        Capability.MASTER_REGISTRY,
        Role.MASTER,
        "registry",
        "/v1/registry",
        "master registry",
        "admin",
        "persistence",
    ),
    CapabilitySpec(
        Capability.MASTER_PERSISTENCE,
        Role.MASTER,
        "persistence",
        None,
        None,
        "server",
        "persistence",
    ),
    CapabilitySpec(
        Capability.MASTER_RAW_WEIGHT_INGRESS,
        Role.MASTER,
        "raw weight ingress",
        None,
        "master weights ingest",
        "challenge-signature",
        "persistence",
    ),
    CapabilitySpec(
        Capability.MASTER_AGGREGATION,
        Role.MASTER,
        "aggregation",
        None,
        "master weights aggregate",
        "server",
        "computation",
    ),
    CapabilitySpec(
        Capability.MASTER_VECTOR_READ,
        Role.MASTER,
        "vector read",
        "/v1/weights/latest",
        "master weights status",
        "public",
        "read",
    ),
    CapabilitySpec(
        Capability.MASTER_WATCHER,
        Role.MASTER,
        "watcher",
        None,
        "master watcher",
        "server",
        "docker",
    ),
    CapabilitySpec(
        Capability.VALIDATOR_REGISTRATION,
        Role.VALIDATOR,
        "registration",
        "/v1/validators/register",
        "validator register",
        "validator-signature",
        "persistence",
    ),
    CapabilitySpec(
        Capability.VALIDATOR_HEARTBEAT,
        Role.VALIDATOR,
        "heartbeat",
        "/v1/validators/heartbeat",
        "validator heartbeat",
        "validator-signature",
        "persistence",
    ),
    CapabilitySpec(
        Capability.VALIDATOR_ASSIGNMENT_PULL,
        Role.VALIDATOR,
        "assignment pull",
        "/v1/assignments/pull",
        "validator pull",
        "validator-signature",
        "read",
    ),
    CapabilitySpec(
        Capability.VALIDATOR_ASSIGNMENT_PROGRESS,
        Role.VALIDATOR,
        "assignment progress",
        "/v1/assignments/{id}/progress",
        "validator progress",
        "validator-signature",
        "persistence",
    ),
    CapabilitySpec(
        Capability.VALIDATOR_ASSIGNMENT_RESULT,
        Role.VALIDATOR,
        "assignment result",
        "/v1/assignments/{id}/result",
        "validator result",
        "validator-signature",
        "persistence",
    ),
    CapabilitySpec(
        Capability.VALIDATOR_VECTOR_READ,
        Role.VALIDATOR,
        "vector read",
        "/v1/weights/latest",
        "validator weights fetch",
        "validator-signature",
        "read",
    ),
    CapabilitySpec(
        Capability.VALIDATOR_OWN_SET_WEIGHTS,
        Role.VALIDATOR,
        "own set weights",
        None,
        "validator set-weights",
        "validator-wallet",
        "chain",
    ),
    CapabilitySpec(
        Capability.CHALLENGE_SCORING,
        Role.CHALLENGE,
        "scoring",
        "/v1/submissions",
        None,
        "challenge-auth",
        "persistence",
    ),
    CapabilitySpec(
        Capability.CHALLENGE_ORDINARY_PROOF,
        Role.CHALLENGE,
        "ordinary proof",
        "/internal/v1/work_units/result",
        None,
        "challenge-auth",
        "verification",
    ),
    CapabilitySpec(
        Capability.CHALLENGE_TEE_VERIFICATION,
        Role.CHALLENGE,
        "TEE verification",
        "/internal/v1/work_units/result",
        None,
        "challenge-auth",
        "verification",
    ),
    CapabilitySpec(
        Capability.CHALLENGE_STATE,
        Role.CHALLENGE,
        "state",
        "/health",
        None,
        "server",
        "persistence",
    ),
    CapabilitySpec(
        Capability.CHALLENGE_RAW_WEIGHT_PUSH,
        Role.CHALLENGE,
        "raw weight push",
        None,
        None,
        "challenge-signature",
        "network",
    ),
    CapabilitySpec(
        Capability.WORKER_ASSIGNMENT_EXECUTION,
        Role.WORKER,
        "assignment execution",
        "/v1/workers/assignments/pull",
        "worker execute",
        "worker-signature",
        "execution",
    ),
    CapabilitySpec(
        Capability.WORKER_RESULT_REPORTING,
        Role.WORKER,
        "result reporting",
        "/v1/workers/assignments/{id}/result",
        "worker result",
        "worker-signature",
        "network",
    ),
)


@dataclass(frozen=True)
class CapabilityRegistry:
    """Immutable registry shared by all runtime roles."""

    version: str
    capabilities: Mapping[str, CapabilitySpec]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capabilities", MappingProxyType(dict(self.capabilities))
        )

    def get(self, capability: Capability | str) -> CapabilitySpec:
        key = capability.value if isinstance(capability, Capability) else capability
        try:
            return self.capabilities[key]
        except KeyError as exc:
            raise RoleContractError(f"unknown capability: {capability}") from exc

    def for_role(self, role: Role | str) -> tuple[CapabilitySpec, ...]:
        normalized = Role(role)
        return tuple(
            spec for spec in self.capabilities.values() if spec.role == normalized
        )


ROLE_REGISTRY = CapabilityRegistry(
    CAPABILITY_REGISTRY_VERSION,
    {spec.token: spec for spec in _CAPABILITY_SPECS},
)


def capabilities_for_role(
    role: Role | str,
    *,
    tee_verification: bool = True,
) -> tuple[str, ...]:
    """Return the immutable server capability projection for ``role``.

    Base challenge services do not claim TEE verification.  Prism opts into
    that capability only from its server-side application factory, never from
    a request header or body field.
    """

    normalized = Role(role)
    return tuple(
        spec.token
        for spec in ROLE_REGISTRY.for_role(normalized)
        if tee_verification or spec.token != Capability.CHALLENGE_TEE_VERIFICATION.value
    )


_current_role: contextvars.ContextVar[RoleContext | None] = contextvars.ContextVar(
    "base_challenge_sdk_role", default=None
)


@contextlib.contextmanager
def activate_role(
    role: Role | str,
    *,
    capabilities: tuple[str, ...] | None = None,
) -> Iterator[RoleContext]:
    """Install authenticated server identity for a request or process scope."""

    normalized = Role(role)
    granted = (
        capabilities if capabilities is not None else capabilities_for_role(normalized)
    )
    expected = set(capabilities_for_role(normalized))
    if not set(granted).issubset(expected):
        raise RoleContractError("role context contains an unauthorized capability")
    context = RoleContext(normalized, tuple(granted))
    token = _current_role.set(context)
    try:
        yield context
    finally:
        _current_role.reset(token)


def current_role() -> RoleContext:
    context = _current_role.get()
    if context is None:
        raise RoleContractError("authenticated server role is not established")
    return context


def role_contract(
    *,
    role: Role | str,
    capability: Capability | str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate an operation with inspectable metadata and a fail-closed guard."""

    normalized_role = Role(role)
    normalized_capability = ROLE_REGISTRY.get(capability)
    if normalized_capability.role != normalized_role:
        raise ValueError("capability is not owned by the declared role")
    metadata = MappingProxyType(
        {
            "role": normalized_role,
            "capability": normalized_capability.token,
            "registry_version": CAPABILITY_REGISTRY_VERSION,
        }
    )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        guarded: Any
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def guarded_async(*args: P.args, **kwargs: P.kwargs) -> R:
                context = current_role()
                if context.role != normalized_role:
                    raise RoleContractError(
                        f"operation requires role {normalized_role.value!r}"
                    )
                if normalized_capability.token not in context.capabilities:
                    raise RoleContractError(
                        f"operation requires capability {normalized_capability.token!r}"
                    )
                return await func(*args, **kwargs)  # type: ignore[misc]

            guarded = guarded_async
        else:

            @wraps(func)
            def guarded_sync(*args: P.args, **kwargs: P.kwargs) -> R:
                context = current_role()
                if context.role != normalized_role:
                    raise RoleContractError(
                        f"operation requires role {normalized_role.value!r}"
                    )
                if normalized_capability.token not in context.capabilities:
                    raise RoleContractError(
                        f"operation requires capability {normalized_capability.token!r}"
                    )
                return func(*args, **kwargs)

            guarded = guarded_sync

        guarded.__base_role_contract__ = metadata  # type: ignore[attr-defined]
        return cast(Callable[P, R], guarded)

    return decorator


def _role_decorator(
    role: Role,
    func: Callable[..., Any] | None = None,
    *,
    capability: Capability | str | None = None,
) -> Any:
    selected = capability or ROLE_REGISTRY.for_role(role)[0].token
    decorator = role_contract(role=role, capability=selected)
    if func is not None:
        return decorator(func)
    return decorator


def master_only(
    func: Callable[..., Any] | None = None,
    *,
    capability: Capability | str | None = None,
) -> Any:
    return _role_decorator(Role.MASTER, func, capability=capability)


def validator_only(
    func: Callable[..., Any] | None = None,
    *,
    capability: Capability | str | None = None,
) -> Any:
    return _role_decorator(Role.VALIDATOR, func, capability=capability)


def challenge_only(
    func: Callable[..., Any] | None = None,
    *,
    capability: Capability | str | None = None,
) -> Any:
    return _role_decorator(Role.CHALLENGE, func, capability=capability)


def worker_only(
    func: Callable[..., Any] | None = None,
    *,
    capability: Capability | str | None = None,
) -> Any:
    return _role_decorator(Role.WORKER, func, capability=capability)


def public_route(
    *,
    tags: list[str] | None = None,
    auth_required: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Mark a route as part of a challenge's published HTTP surface."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        func.__base_public_route__ = True  # type: ignore[attr-defined]
        func.__base_public_tags__ = tuple(tags or ())  # type: ignore[attr-defined]
        func.__base_public_auth_required__ = auth_required  # type: ignore[attr-defined]
        return func

    return decorator


def is_public_route(func: Callable[..., object]) -> bool:
    return bool(getattr(func, "__base_public_route__", False))


__all__ = [
    "CAPABILITY_REGISTRY_VERSION",
    "Capability",
    "CapabilityRegistry",
    "CapabilitySpec",
    "Role",
    "RoleContext",
    "ROLE_REGISTRY",
    "RoleContractError",
    "activate_role",
    "capabilities_for_role",
    "challenge_only",
    "current_role",
    "is_public_route",
    "master_only",
    "public_route",
    "role_contract",
    "validator_only",
    "worker_only",
]
