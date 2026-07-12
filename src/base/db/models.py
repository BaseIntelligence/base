"""Normalized challenge models for the base master database."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from base.db.base import Base

DEFAULT_VALIDATOR_VERSION = "unknown"
"""Fallback ``validators.version`` when a registration omits one.

The column is non-null with a matching server default so a direct/raw insert and
the coordination route converge on the same value.
"""


class ChallengeStatus(StrEnum):
    """Lifecycle states supported by the master challenge registry."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DISABLED = "disabled"
    DRAFT = "draft"


class ValidatorStatus(StrEnum):
    """Liveness states for a registered validator in the coordination plane."""

    ONLINE = "online"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class ValidatorHealthEventType(StrEnum):
    """Audit events recorded for a validator's lifecycle transitions."""

    REGISTERED = "registered"
    ONLINE = "online"
    OFFLINE = "offline"
    CRASH_DETECTED = "crash_detected"


class WorkAssignmentStatus(StrEnum):
    """Lifecycle states for a work unit coordinated to a validator.

    ``disputed`` is a terminal worker-plane outcome (architecture.md sec 3.3):
    a gpu unit whose replica manifest hashes diverged is disputed and NEVER
    forwarded to the challenge (before or after audit); it is only ever set by
    worker-plane reconciliation, so the validator plane never observes it.
    """

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DISPUTED = "disputed"


class WorkerStatus(StrEnum):
    """Lifecycle states for a miner-funded GPU worker (architecture.md sec 3.3).

    ``pending`` -> ``active`` -> ``stale`` -> ``retired``. ``active`` requires a
    verified miner binding AND a heartbeat within the freshness window
    (``compute.worker_heartbeat_ttl_seconds``); ``retired`` is terminal (a
    retired worker is never assignable and a heartbeat never resurrects it).
    """

    PENDING = "pending"
    ACTIVE = "active"
    STALE = "stale"
    RETIRED = "retired"


class TimestampMixin:
    """Created/updated timestamp columns shared by mutable tables."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class Challenge(Base, TimestampMixin):
    """A registered challenge managed by the base master."""

    __tablename__ = "challenges"
    __table_args__ = (
        Index("ix_challenges_status", "status"),
        Index("ix_challenges_slug_status", "slug", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slug: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[ChallengeStatus] = mapped_column(
        Enum(
            ChallengeStatus,
            name="challenge_status",
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        nullable=False,
        default=ChallengeStatus.DRAFT,
        server_default=ChallengeStatus.DRAFT.value,
    )
    emission_percent: Mapped[Decimal] = mapped_column(
        Numeric(8, 4),
        nullable=False,
        default=Decimal("0"),
        server_default="0",
    )
    version: Mapped[str] = mapped_column(Text, nullable=False)
    api_version: Mapped[str] = mapped_column(
        Text, nullable=False, default="1.0", server_default="1.0"
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    image: Mapped[ChallengeImage | None] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
        single_parent=True,
        uselist=False,
    )
    auth: Mapped[ChallengeAuth | None] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
        single_parent=True,
        uselist=False,
    )
    resources: Mapped[list[ChallengeResource]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
    )
    volumes: Mapped[list[ChallengeVolume]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
    )
    secrets: Mapped[list[ChallengeSecret]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
    )
    env: Mapped[list[ChallengeEnv]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
    )
    capabilities: Mapped[list[ChallengeCapability]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
    )
    routes: Mapped[list[ChallengeRoute]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
    )
    health_events: Mapped[list[ChallengeHealthEvent]] = relationship(
        back_populates="challenge",
        cascade="all, delete-orphan",
        order_by="ChallengeHealthEvent.checked_at.desc()",
    )


class ChallengeImage(Base):
    """Container image coordinates for a challenge."""

    __tablename__ = "challenge_images"
    __table_args__ = (
        UniqueConstraint("challenge_id", name="uq_challenge_images_challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    registry_name: Mapped[str] = mapped_column("registry", Text, nullable=False)
    repository: Mapped[str] = mapped_column(Text, nullable=False)
    tag: Mapped[str] = mapped_column(Text, nullable=False)
    digest: Mapped[str | None] = mapped_column(Text)
    pull_policy: Mapped[str] = mapped_column(
        Text, nullable=False, default="if_not_present", server_default="if_not_present"
    )

    challenge: Mapped[Challenge] = relationship(back_populates="image")


class ChallengeAuth(Base):
    """Hashed authentication material for challenge internal endpoints."""

    __tablename__ = "challenge_auth"
    __table_args__ = (
        UniqueConstraint("challenge_id", name="uq_challenge_auth_challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    token_hint: Mapped[str | None] = mapped_column(Text)
    broker_token_hash: Mapped[str | None] = mapped_column(Text)
    broker_token_hint: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    challenge: Mapped[Challenge] = relationship(back_populates="auth")


class ChallengeResource(Base):
    """A named runtime resource value requested by a challenge."""

    __tablename__ = "challenge_resources"
    __table_args__ = (
        UniqueConstraint(
            "challenge_id", "key", name="uq_challenge_resources_challenge_key"
        ),
        Index("ix_challenge_resources_challenge_id", "challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)

    challenge: Mapped[Challenge] = relationship(back_populates="resources")


class ChallengeVolume(Base):
    """A Docker volume mount requested by a challenge."""

    __tablename__ = "challenge_volumes"
    __table_args__ = (
        UniqueConstraint(
            "challenge_id", "name", name="uq_challenge_volumes_challenge_name"
        ),
        Index("ix_challenge_volumes_challenge_id", "challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    mount_path: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)

    challenge: Mapped[Challenge] = relationship(back_populates="volumes")


class ChallengeSecret(Base):
    """A file secret mounted into a challenge container."""

    __tablename__ = "challenge_secrets"
    __table_args__ = (
        UniqueConstraint(
            "challenge_id", "name", name="uq_challenge_secrets_challenge_name"
        ),
        Index("ix_challenge_secrets_challenge_id", "challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    mount_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)

    challenge: Mapped[Challenge] = relationship(back_populates="secrets")


class ChallengeEnv(Base):
    """An environment variable definition for a challenge container."""

    __tablename__ = "challenge_env"
    __table_args__ = (
        UniqueConstraint("challenge_id", "key", name="uq_challenge_env_challenge_key"),
        Index("ix_challenge_env_challenge_id", "challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    key: Mapped[str] = mapped_column(Text, nullable=False)
    value_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    is_secret: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )

    challenge: Mapped[Challenge] = relationship(back_populates="env")


class ChallengeCapability(Base):
    """A named capability advertised by a challenge."""

    __tablename__ = "challenge_capabilities"
    __table_args__ = (
        UniqueConstraint(
            "challenge_id", "name", name="uq_challenge_capabilities_challenge_name"
        ),
        Index("ix_challenge_capabilities_challenge_id", "challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str | None] = mapped_column(Text)

    challenge: Mapped[Challenge] = relationship(back_populates="capabilities")


class ChallengeRoute(Base):
    """A public route prefix exposed by a challenge through the proxy."""

    __tablename__ = "challenge_routes"
    __table_args__ = (
        UniqueConstraint(
            "challenge_id", "public_prefix", name="uq_challenge_routes_challenge_prefix"
        ),
        Index("ix_challenge_routes_challenge_id", "challenge_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    public_prefix: Mapped[str] = mapped_column(Text, nullable=False)
    proxy_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="1"
    )

    challenge: Mapped[Challenge] = relationship(back_populates="routes")


class ChallengeHealthEvent(Base):
    """Historical health/version observations for a challenge."""

    __tablename__ = "challenge_health_events"
    __table_args__ = (
        Index(
            "ix_challenge_health_events_challenge_checked", "challenge_id", "checked_at"
        ),
        Index("ix_challenge_health_events_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("challenges.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str | None] = mapped_column(Text)
    message: Mapped[str | None] = mapped_column(Text)
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    challenge: Mapped[Challenge] = relationship(back_populates="health_events")


class Validator(Base, TimestampMixin):
    """A validator registered with the master coordination plane."""

    __tablename__ = "validators"
    __table_args__ = (
        Index("ix_validators_status", "status"),
        Index("ix_validators_last_heartbeat_at", "last_heartbeat_at"),
        Index("ix_validators_registered_at", "registered_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    hotkey: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    uid: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[ValidatorStatus] = mapped_column(
        Enum(
            ValidatorStatus,
            name="validator_status",
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        nullable=False,
        default=ValidatorStatus.UNKNOWN,
        server_default=ValidatorStatus.UNKNOWN.value,
    )
    capabilities: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
    )
    version: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default=DEFAULT_VALIDATOR_VERSION,
        server_default=DEFAULT_VALIDATOR_VERSION,
    )
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_heartbeat_sequence: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0, server_default="0"
    )
    last_heartbeat_payload_digest: Mapped[str | None] = mapped_column(Text)
    last_seen_meta: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    subscriptions: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
    )


class ValidatorHealthEvent(Base):
    """Append-only audit log of validator liveness transitions."""

    __tablename__ = "validator_health_events"
    __table_args__ = (
        Index(
            "ix_validator_health_events_hotkey_created",
            "validator_hotkey",
            "created_at",
            "seq",
        ),
        Index("ix_validator_health_events_event", "event"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    event: Mapped[ValidatorHealthEventType] = mapped_column(
        Enum(
            ValidatorHealthEventType,
            name="validator_health_event_type",
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        nullable=False,
    )
    message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    seq: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        server_default="0",
    )


class WorkAssignment(Base, TimestampMixin):
    """A unit of evaluation work coordinated to an online validator.

    The master fans submissions out into work units (agent-challenge: one per
    selected task; prism: exactly one per submission) and assigns pending units
    to eligible online validators. The master only coordinates; validators
    execute the work on their own brokers.
    """

    __tablename__ = "work_assignments"
    __table_args__ = (
        UniqueConstraint(
            "challenge_slug",
            "work_unit_id",
            name="uq_work_assignments_challenge_work_unit",
        ),
        Index("ix_work_assignments_challenge_slug", "challenge_slug"),
        Index("ix_work_assignments_status", "status"),
        Index(
            "ix_work_assignments_assigned_validator_hotkey",
            "assigned_validator_hotkey",
        ),
        Index(
            "ix_work_assignments_status_validator",
            "status",
            "assigned_validator_hotkey",
        ),
        Index("ix_work_assignments_status_deadline", "status", "deadline_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_slug: Mapped[str] = mapped_column(Text, nullable=False)
    work_unit_id: Mapped[str] = mapped_column(Text, nullable=False)
    submission_ref: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    required_capability: Mapped[str] = mapped_column(
        Text, nullable=False, default="cpu", server_default="cpu"
    )
    assigned_validator_hotkey: Mapped[str | None] = mapped_column(Text)
    status: Mapped[WorkAssignmentStatus] = mapped_column(
        Enum(
            WorkAssignmentStatus,
            name="work_assignment_status",
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        nullable=False,
        default=WorkAssignmentStatus.PENDING,
        server_default=WorkAssignmentStatus.PENDING.value,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3, server_default="3"
    )
    revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    payload_digest: Mapped[str | None] = mapped_column(Text)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_progress_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    checkpoint_ref: Mapped[str | None] = mapped_column(Text)
    result_ref: Mapped[str | None] = mapped_column(Text)


class WorkResult(Base):
    """A validator-reported result for a coordinated work unit.

    The master persists each reported result so weight computation can consume
    validator-reported evaluation outcomes. The owning ``work_assignments`` row
    points back at the persisted result via ``result_ref``. Stores no secret
    material; ``payload`` carries the challenge-specific result descriptor.
    """

    __tablename__ = "work_results"
    __table_args__ = (
        UniqueConstraint("assignment_id", name="uq_work_results_assignment_id"),
        Index("ix_work_results_assignment_id", "assignment_id"),
        Index("ix_work_results_challenge_slug", "challenge_slug"),
        Index("ix_work_results_validator_hotkey", "validator_hotkey"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    assignment_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    challenge_slug: Mapped[str] = mapped_column(Text, nullable=False)
    work_unit_id: Mapped[str] = mapped_column(Text, nullable=False)
    submission_ref: Mapped[str] = mapped_column(Text, nullable=False)
    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    result_digest: Mapped[str | None] = mapped_column(Text)
    checkpoint_ref: Mapped[str | None] = mapped_column(Text)
    proof: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ValidatorRequestNonce(Base):
    """Replay protection for signed validator coordination requests."""

    __tablename__ = "validator_request_nonces"
    __table_args__ = (
        UniqueConstraint(
            "hotkey", "nonce", name="uq_validator_request_nonces_hotkey_nonce"
        ),
        Index("ix_validator_request_nonces_created_at", "created_at"),
        Index("ix_validator_request_nonces_hotkey", "hotkey"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    nonce: Mapped[str] = mapped_column(Text, nullable=False)
    body_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class MinerRequestNonce(Base):
    """Replay protection for signed miner uploads accepted by the proxy."""

    __tablename__ = "miner_request_nonces"
    __table_args__ = (
        UniqueConstraint(
            "netuid",
            "challenge_slug",
            "hotkey",
            "nonce",
            name="uq_miner_request_nonces_scope",
        ),
        Index("ix_miner_request_nonces_created_at", "created_at"),
        Index("ix_miner_request_nonces_hotkey", "hotkey"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    netuid: Mapped[int] = mapped_column(Integer, nullable=False)
    challenge_slug: Mapped[str] = mapped_column(Text, nullable=False)
    hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    nonce: Mapped[str] = mapped_column(Text, nullable=False)
    body_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class WorkerRegistration(Base, TimestampMixin):
    """A miner-funded GPU worker enrolled with the master (architecture.md 3.3).

    A worker is bound to exactly one miner hotkey via a miner-signed sr25519
    binding (``worker-binding:{worker_pubkey}:{miner_hotkey}:{nonce}``). The
    ``worker_pubkey`` is unique, so a pubkey has a single stable owner: a
    re-registration under a DIFFERENT miner hotkey is rejected (no silent
    rebind). ``worker_id`` is the public identifier used by the heartbeat route
    and returned to the agent.
    """

    __tablename__ = "worker_registrations"
    __table_args__ = (
        Index("ix_worker_registrations_status", "status"),
        Index("ix_worker_registrations_miner_hotkey", "miner_hotkey"),
        Index("ix_worker_registrations_last_heartbeat_at", "last_heartbeat_at"),
        Index(
            "ix_worker_registrations_status_miner_hotkey",
            "status",
            "miner_hotkey",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    worker_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    worker_pubkey: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    miner_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    binding_signature: Mapped[str] = mapped_column(Text, nullable=False)
    binding_nonce: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    provider_instance_ref: Mapped[str | None] = mapped_column(Text)
    capabilities: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
    )
    status: Mapped[WorkerStatus] = mapped_column(
        Enum(
            WorkerStatus,
            name="worker_status",
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        nullable=False,
        default=WorkerStatus.PENDING,
        server_default=WorkerStatus.PENDING.value,
    )
    last_seen_meta: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorkerFault(Base):
    """A fault attributed to a worker replica during reconciliation/audit.

    Written when a worker's ``ExecutionProof.manifest_sha256`` diverges from the
    authoritative validator replay (architecture.md sec 3.3). Recording a fault
    does NOT change the worker's ``worker_registrations.status``; faults are
    surfaced read-only in the fleet view.
    """

    __tablename__ = "worker_faults"
    __table_args__ = (
        Index("ix_worker_faults_worker_id", "worker_id"),
        Index("ix_worker_faults_work_unit_id", "work_unit_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    worker_id: Mapped[str] = mapped_column(Text, nullable=False)
    work_unit_id: Mapped[str] = mapped_column(Text, nullable=False)
    challenge_slug: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class WorkerRequestNonce(Base):
    """Replay protection for worker-plane nonces.

    Serves two purposes keyed on ``(hotkey, nonce)``: the miner binding nonce
    (``hotkey`` = the miner hotkey) consumed at ``register`` and the request
    envelope nonce (``hotkey`` = the worker pubkey) consumed on signed
    heartbeat/fleet-read requests. Miner hotkeys and worker pubkeys never
    collide, so one table safely dedups both.
    """

    __tablename__ = "worker_request_nonces"
    __table_args__ = (
        UniqueConstraint(
            "hotkey", "nonce", name="uq_worker_request_nonces_hotkey_nonce"
        ),
        Index("ix_worker_request_nonces_created_at", "created_at"),
        Index("ix_worker_request_nonces_hotkey", "hotkey"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    nonce: Mapped[str] = mapped_column(Text, nullable=False)
    body_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class WorkerAssignment(Base, TimestampMixin):
    """A gpu work-unit replica coordinated to a miner-funded worker.

    Distinct from :class:`WorkAssignment` (the validator plane): a single gpu
    work unit is replicated to multiple workers bound to DISTINCT owners
    (architecture.md sec 3.3 R=2), so this table keys one row PER (work unit,
    worker) rather than one row per unit. ``worker_pubkey`` is the authenticated
    identity the worker pull/post routes gate on; ``miner_hotkey`` is the owner
    used for anti-collusion accounting. The reported result and its
    ``ExecutionProof`` (and the extracted ``manifest_sha256`` used for
    reconciliation) are stored on the row.
    """

    __tablename__ = "worker_assignments"
    __table_args__ = (
        UniqueConstraint(
            "work_unit_id",
            "worker_id",
            name="uq_worker_assignments_work_unit_worker",
        ),
        Index("ix_worker_assignments_work_unit_id", "work_unit_id"),
        Index("ix_worker_assignments_status", "status"),
        Index("ix_worker_assignments_worker_pubkey", "worker_pubkey"),
        Index(
            "ix_worker_assignments_status_worker_pubkey",
            "status",
            "worker_pubkey",
        ),
        Index("ix_worker_assignments_status_deadline", "status", "deadline_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_slug: Mapped[str] = mapped_column(Text, nullable=False)
    work_unit_id: Mapped[str] = mapped_column(Text, nullable=False)
    submission_ref: Mapped[str] = mapped_column(Text, nullable=False)
    worker_id: Mapped[str] = mapped_column(Text, nullable=False)
    worker_pubkey: Mapped[str] = mapped_column(Text, nullable=False)
    miner_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    required_capability: Mapped[str] = mapped_column(
        Text, nullable=False, default="gpu", server_default="gpu"
    )
    status: Mapped[WorkAssignmentStatus] = mapped_column(
        Enum(
            WorkAssignmentStatus,
            name="work_assignment_status",
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        nullable=False,
        default=WorkAssignmentStatus.PENDING,
        server_default=WorkAssignmentStatus.PENDING.value,
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3, server_default="3"
    )
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_progress_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    checkpoint_ref: Mapped[str | None] = mapped_column(Text)
    result_success: Mapped[bool | None] = mapped_column(Boolean)
    result_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    manifest_sha256: Mapped[str | None] = mapped_column(Text)


class AggregationEpochStatus(StrEnum):
    """Durable state for master aggregation epochs."""

    OPEN = "open"
    SEALED = "sealed"
    WITHHELD = "withheld"


class RawWeightSnapshot(Base):
    """Immutable authenticated challenge raw-weight snapshot (push ingress).

    One row per accepted ``(challenge_slug, epoch, revision)``. Higher accepted
    revisions supersede selection while the matching aggregation epoch remains
    open; sealing freezes the selected source. Exact concurrent delivery is
    idempotent and conflicts preserve the original canonical bytes/digest.
    """

    __tablename__ = "raw_weight_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "challenge_slug",
            "epoch",
            "revision",
            name="uq_raw_weight_snapshots_challenge_epoch_revision",
        ),
        UniqueConstraint(
            "challenge_slug",
            "nonce",
            name="uq_raw_weight_snapshots_challenge_nonce",
        ),
        Index("ix_raw_weight_snapshots_challenge_epoch", "challenge_slug", "epoch"),
        Index(
            "ix_raw_weight_snapshots_selected",
            "challenge_slug",
            "epoch",
            "is_selected_source",
        ),
        Index("ix_raw_weight_snapshots_payload_digest", "payload_digest"),
        Index("ix_raw_weight_snapshots_received_at", "received_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_slug: Mapped[str] = mapped_column(Text, nullable=False)
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    protocol_version: Mapped[str] = mapped_column(Text, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    nonce: Mapped[str] = mapped_column(Text, nullable=False)
    payload_digest: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_payload: Mapped[str] = mapped_column(Text, nullable=False)
    weights: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    is_selected_source: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class RawWeightNonce(Base):
    """Authentication and push-operation nonce ledger for raw weight ingress.

    Distinct from validator/miner/worker nonces. Exact body-digest retry is
    idempotent; nonce reuse with different bytes is a conflict.
    """

    __tablename__ = "raw_weight_nonces"
    __table_args__ = (
        UniqueConstraint(
            "challenge_slug",
            "nonce",
            name="uq_raw_weight_nonces_challenge_nonce",
        ),
        Index("ix_raw_weight_nonces_created_at", "created_at"),
        Index("ix_raw_weight_nonces_challenge", "challenge_slug"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    challenge_slug: Mapped[str] = mapped_column(Text, nullable=False)
    nonce: Mapped[str] = mapped_column(Text, nullable=False)
    body_hash: Mapped[str] = mapped_column(Text, nullable=False)
    payload_digest: Mapped[str] = mapped_column(Text, nullable=False)
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("raw_weight_snapshots.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class AggregationEpoch(Base):
    """Durable aggregation-epoch lifecycle (open, sealed, or withheld).

    Opening records the expected active challenge set and emission policy for
    the barrier. Sealing produces one immutable final vector when every expected
    source is available; missing or invalid active sources transition the epoch
    to ``withheld`` without inventing/carrying-forward contributions.
    """

    __tablename__ = "aggregation_epochs"
    __table_args__ = (
        UniqueConstraint("epoch", name="uq_aggregation_epochs_epoch"),
        Index("ix_aggregation_epochs_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[AggregationEpochStatus] = mapped_column(
        Enum(
            AggregationEpochStatus,
            name="aggregation_epoch_status",
            values_callable=lambda obj: [e.value for e in obj],
            native_enum=False,
        ),
        nullable=False,
        default=AggregationEpochStatus.OPEN,
        server_default=AggregationEpochStatus.OPEN.value,
    )
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expected_challenges: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    emission_policy_version: Mapped[str | None] = mapped_column(Text)
    emission_shares: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    source_outcome_policy_version: Mapped[str | None] = mapped_column(Text)
    burn_policy_version: Mapped[str | None] = mapped_column(Text)
    mapping_policy_version: Mapped[str | None] = mapped_column(Text)
    outcome_reason: Mapped[str | None] = mapped_column(Text)
    vector_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("final_weight_vectors.id", ondelete="SET NULL"),
        nullable=True,
    )
    source_outcomes: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class FinalWeightVector(Base):
    """Immutable master-owned final weight vector with full provenance.

    One sealed vector per successful epoch. Validators fetch these bytes only;
    re-aggregation on read is forbidden.
    """

    __tablename__ = "final_weight_vectors"
    __table_args__ = (
        UniqueConstraint("vector_digest", name="uq_final_weight_vectors_digest"),
        UniqueConstraint("epoch", name="uq_final_weight_vectors_epoch"),
        Index("ix_final_weight_vectors_computed_at", "computed_at"),
        Index("ix_final_weight_vectors_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    protocol_version: Mapped[str] = mapped_column(Text, nullable=False)
    netuid: Mapped[int] = mapped_column(Integer, nullable=False)
    chain_endpoint: Mapped[str] = mapped_column(Text, nullable=False, default="")
    vector_digest: Mapped[str] = mapped_column(Text, nullable=False)
    uids: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    weights: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    hotkey_weights: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    chain_domain_bytes: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_payload: Mapped[str] = mapped_column(Text, nullable=False)
    source_snapshot_ids: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    source_snapshot_digests: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    source_outcomes: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    emission_policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    emission_shares: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    burn_policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    mapping_policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    metagraph_block: Mapped[int | None] = mapped_column(BigInteger)
    metagraph_hash: Mapped[str | None] = mapped_column(Text)
    metagraph_identity: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    hotkey_to_uid: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    metagraph_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ValidatorSubmissionObservation(Base):
    """Non-authoritative validator-reported chain-submission observations.

    Keyed by validator identity + immutable vector identity. Exact retries are
    idempotent; conflicts are auditable. Does not mutate final_weight_vectors
    and never implies the master performed set_weights.
    """

    __tablename__ = "validator_submission_observations"
    __table_args__ = (
        UniqueConstraint(
            "validator_hotkey",
            "vector_id",
            "vector_digest",
            "outcome",
            "attempt",
            name="uq_validator_submission_observation_identity",
        ),
        Index("ix_validator_submission_observations_vector", "vector_id"),
        Index("ix_validator_submission_observations_hotkey", "validator_hotkey"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    validator_hotkey: Mapped[str] = mapped_column(Text, nullable=False)
    vector_id: Mapped[str] = mapped_column(Text, nullable=False)
    vector_digest: Mapped[str] = mapped_column(Text, nullable=False)
    netuid: Mapped[int] = mapped_column(Integer, nullable=False)
    chain_endpoint: Mapped[str] = mapped_column(Text, nullable=False, default="")
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    error_code: Mapped[str | None] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ChallengeWatcherState(Base, TimestampMixin):
    """Durable Compose challenge-watcher intent and rollout state.

    One row per challenge slug. Survives master restart so backoff, preferred
    rollback digest, and in-flight phase can be reconstructed without mistaking
    the desired registry image for the currently-running digest.
    """

    __tablename__ = "challenge_watcher_state"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_challenge_watcher_state_slug"),
        Index("ix_challenge_watcher_state_phase", "phase"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    desired_digest: Mapped[str | None] = mapped_column(Text)
    current_digest: Mapped[str | None] = mapped_column(Text)
    rollback_digest: Mapped[str | None] = mapped_column(Text)
    desired_image: Mapped[str | None] = mapped_column(Text)
    rollback_image: Mapped[str | None] = mapped_column(Text)
    phase: Mapped[str] = mapped_column(Text, nullable=False, default="idle")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_eligible_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    last_result: Mapped[str | None] = mapped_column(Text)
    last_health_ok: Mapped[bool | None] = mapped_column(Boolean)
    last_version_ok: Mapped[bool | None] = mapped_column(Boolean)
    alerted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    project_name: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict, server_default="{}"
    )
