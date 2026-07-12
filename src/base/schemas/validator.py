"""API schemas for the validator coordination plane."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ValidatorView(BaseModel):
    """Public view of a registered validator row."""

    hotkey: str
    uid: int | None = None
    status: str
    capabilities: list[str] = Field(default_factory=list)
    subscriptions: list[str] = Field(default_factory=list)
    version: str | None = None
    registered_at: datetime
    last_heartbeat_at: datetime | None = None
    last_heartbeat_sequence: int = 0
    last_seen_meta: dict[str, Any] = Field(default_factory=dict)


class ValidatorListResponse(BaseModel):
    """Response for the token-gated admin validator read view."""

    validators: list[ValidatorView] = Field(default_factory=list)


class ValidatorRegisterRequest(BaseModel):
    """Body for ``POST /v1/validators/register``."""

    model_config = ConfigDict(extra="forbid")

    capabilities: list[str] = Field(default_factory=lambda: ["cpu"])
    version: str | None = None
    last_seen_meta: dict[str, Any] | None = None


class ValidatorRegisterResponse(BaseModel):
    """Response for a successful validator registration."""

    validator: ValidatorView
    heartbeat_interval_seconds: int
    idempotent: bool = False


class ValidatorHeartbeatRequest(BaseModel):
    """Body for ``POST /v1/validators/heartbeat``."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(default=0, ge=0)
    last_seen_meta: dict[str, Any] | None = None
    capabilities: list[str] | None = None
    version: str | None = None

    @field_validator("sequence")
    @classmethod
    def reject_boolean_sequence(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("boolean is not a sequence integer")
        return value


class ValidatorHeartbeatResponse(BaseModel):
    """Response for a successful validator heartbeat."""

    status: str
    now: datetime
    sequence: int = 0
    idempotent: bool = False


class ValidatorSubscriptionRequest(BaseModel):
    """Body for ``POST /v1/validators/subscriptions``.

    ``slugs`` is the set of challenge slugs the validator opts in to. An empty
    list clears the subscription (the validator validates ALL challenges).
    """

    model_config = ConfigDict(extra="forbid")

    slugs: list[str] = Field(default_factory=list)


class ValidatorSubscriptionResponse(BaseModel):
    """Response for a successful subscription update."""

    validator: ValidatorView
    subscriptions: list[str] = Field(default_factory=list)


class PublicIdentityView(BaseModel):
    """Render-safe resolved identity (display name + logo URL).

    Used for both a validator's resolved identity and the top-level subnet
    identity. A self-declared identity is UNTRUSTED; consumers MUST sanitize it
    on render and never execute the logo URL.
    """

    display_name: str | None = None
    logo_url: str | None = None


class PublicValidatorView(BaseModel):
    """Safe, anonymous-facing view of a validator for the open directory API.

    Exposes ONLY fields safe for public consumption. It deliberately omits raw
    ``last_seen_meta``, tokens, and any other secret; it is NOT the privileged
    admin :class:`ValidatorView`.
    """

    hotkey: str
    uid: int | None = None
    status: str
    online: bool
    capabilities: list[str] = Field(default_factory=list)
    subscriptions: list[str] = Field(default_factory=list)
    last_heartbeat_at: datetime | None = None
    identity: PublicIdentityView | None = None


class PublicValidatorsResponse(BaseModel):
    """Response for the open ``GET /v1/validators/public`` directory API."""

    validators: list[PublicValidatorView] = Field(default_factory=list)
    subnet: PublicIdentityView | None = None
