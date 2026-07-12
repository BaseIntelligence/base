"""API schemas for the master worker coordination plane (architecture.md 3.3)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class WorkerFaultView(BaseModel):
    """A fault attributed to a worker, surfaced read-only in the fleet view."""

    work_unit_id: str
    challenge_slug: str | None = None
    detail: str | None = None
    created_at: datetime


class WorkerView(BaseModel):
    """Fleet view of a worker: status, owner, provider, last-seen, faults."""

    worker_id: str
    worker_pubkey: str
    miner_hotkey: str
    provider: str
    provider_instance_ref: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    status: str
    last_heartbeat_at: datetime | None = None
    created_at: datetime
    faults: list[WorkerFaultView] = Field(default_factory=list)


class WorkerRegisterRequest(BaseModel):
    """Body for ``POST /v1/workers/register``.

    ``binding_signature`` is the miner's sr25519 signature over
    ``worker-binding:{worker_pubkey}:{miner_hotkey}:{nonce}``.
    """

    worker_pubkey: str
    miner_hotkey: str
    binding_signature: str
    nonce: str
    provider: str
    provider_instance_ref: str | None = None
    capabilities: list[str] = Field(default_factory=lambda: ["gpu"])
    last_seen_meta: dict[str, Any] | None = None


class WorkerRegisterResponse(BaseModel):
    """Response for a successful worker registration."""

    worker: WorkerView
    heartbeat_ttl_seconds: int


class WorkerHeartbeatRequest(BaseModel):
    """Body for ``POST /v1/workers/{worker_id}/heartbeat``."""

    last_seen_meta: dict[str, Any] | None = None


class WorkerHeartbeatResponse(BaseModel):
    """Response for a successful worker heartbeat."""

    status: str
    now: datetime


class WorkerListResponse(BaseModel):
    """Response for ``GET /v1/workers`` and ``GET /v1/workers/active``."""

    workers: list[WorkerView] = Field(default_factory=list)


class WorkerReplicaView(BaseModel):
    """One worker replica of a gpu unit, surfaced in the unit-status view."""

    worker_id: str
    miner_hotkey: str
    status: str
    manifest_sha256: str | None = None
    has_proof: bool = False


class WorkerAuditUnitView(BaseModel):
    """The validator AUDIT unit linked to a disputed gpu unit.

    ``outcome`` is the audit's terminal state: ``pending`` (not yet resolved),
    ``mismatch-resolved`` (resolved with a fault attributed to the divergent
    worker(s)), or ``passed`` (resolved with no fault).
    """

    work_unit_id: str
    executor_kind: str
    outcome: str


class WorkerUnitStatusView(BaseModel):
    """Per primary gpu unit status for operator dispute discovery (VAL-CROSS-011).

    Exposes the unit id + status (INCLUDING ``disputed``), its replicas
    (worker/owner/manifest/proof-presence), and, when disputed, the linked
    validator audit unit with its terminal outcome -- so the full dispute ->
    audit -> invalidation -> fault chain is reconstructable via APIs alone.
    """

    work_unit_id: str
    challenge_slug: str
    submission_ref: str
    status: str
    replicas: list[WorkerReplicaView] = Field(default_factory=list)
    audit: WorkerAuditUnitView | None = None


class WorkerUnitStatusListResponse(BaseModel):
    """Response for ``GET /v1/workers/units``."""

    units: list[WorkerUnitStatusView] = Field(default_factory=list)


class ProviderInfo(BaseModel):
    """Provider/pod identity carried by an ExecutionProof (architecture 3.4)."""

    name: str
    executor_id: str | None = None
    pod_id: str | None = None
    miner_hotkey: str | None = None


class WorkerSignature(BaseModel):
    """The worker's sr25519 signature binding a manifest hash to a work unit."""

    worker_pubkey: str
    sig: str


#: Tier value marking an ExecutionProof as a Phala Intel TDX attestation
#: (architecture.md sec 6). Distinct from the integer worker-plane tiers so the
#: Phala envelope is self-describing without disturbing tier 0/1/2.
PHALA_TDX_TIER = "phala-tdx"


class PhalaMeasurement(BaseModel):
    """TDX measurement registers for a canonical Phala eval image (arch sec 6/7).

    The static, allowlist-pinnable ``canonical_measurement`` is the subset
    ``{mrtd, rtmr0, rtmr1, rtmr2, compose_hash, os_image_hash}``. ``rtmr3`` is the
    runtime event-log register (it carries the live compose-hash event); it is
    kept on the record for completeness but excluded from the pinned canonical
    measurement bound into ``report_data``.
    """

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    compose_hash: str
    os_image_hash: str

    def canonical(self) -> dict[str, str]:
        """The static, allowlist-pinnable subset (excludes runtime ``rtmr3``)."""

        return {
            "mrtd": self.mrtd,
            "rtmr0": self.rtmr0,
            "rtmr1": self.rtmr1,
            "rtmr2": self.rtmr2,
            "compose_hash": self.compose_hash,
            "os_image_hash": self.os_image_hash,
        }


class PhalaAttestation(BaseModel):
    """Phala Intel TDX attestation payload carried by a Phala-tier ExecutionProof.

    Populates the ``attestation`` block of an ExecutionProof whose ``tier`` is
    :data:`PHALA_TDX_TIER` (architecture.md sec 6). The architecture's
    ``tdx_quote_b64`` / ``report_data_hex`` spellings are accepted as input
    aliases; serialization always uses the canonical field names.
    """

    model_config = ConfigDict(populate_by_name=True)

    tdx_quote: str = Field(validation_alias=AliasChoices("tdx_quote", "tdx_quote_b64"))
    event_log: list[dict[str, Any]] = Field(default_factory=list)
    report_data: str = Field(
        validation_alias=AliasChoices("report_data", "report_data_hex")
    )
    measurement: PhalaMeasurement
    vm_config: dict[str, Any] = Field(default_factory=dict)


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REGISTER_PATTERN = r"^[0-9a-f]{96}$"
_REPORT_DATA_PATTERN = r"^[0-9a-f]{128}$"
_EVEN_HEX_PATTERN = r"^(?:[0-9a-f]{2})*$"
_NONEMPTY_EVEN_HEX_PATTERN = r"^(?:[0-9a-f]{2})+$"
_VISIBLE_ID_PATTERN = r"^[!-~]{1,128}$"

# Eval result wire limits are deliberately fixed at the schema boundary.  The
# direct result endpoint uses the same defaults, but the BASE conformance model
# must remain safe when it is used independently of that endpoint.
EVAL_MAX_QUOTE_BYTES = 64 * 1024
EVAL_MAX_EVENT_LOG_ENTRIES = 4096
EVAL_MAX_EVENT_LOG_BYTES = 2 * 1024 * 1024
EVAL_MAX_VM_CONFIG_BYTES = 64 * 1024
EVAL_MAX_STRING_BYTES = 16 * 1024
EVAL_MAX_PAYLOAD_BYTES = EVAL_MAX_STRING_BYTES
EVAL_MAX_INTEGER = (1 << 63) - 1


class EvalPhalaMeasurement(BaseModel):
    """Exact canonical measurement wire schema for Eval Phala attestations."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mrtd: str = Field(pattern=_REGISTER_PATTERN)
    rtmr0: str = Field(pattern=_REGISTER_PATTERN)
    rtmr1: str = Field(pattern=_REGISTER_PATTERN)
    rtmr2: str = Field(pattern=_REGISTER_PATTERN)
    rtmr3: str = Field(pattern=_REGISTER_PATTERN)
    compose_hash: str = Field(pattern=_SHA256_PATTERN)
    os_image_hash: str = Field(pattern=_SHA256_PATTERN)

    def canonical(self) -> dict[str, str]:
        """The static, allowlist-pinnable subset, excluding runtime ``rtmr3``."""

        return {
            "mrtd": self.mrtd,
            "rtmr0": self.rtmr0,
            "rtmr1": self.rtmr1,
            "rtmr2": self.rtmr2,
            "compose_hash": self.compose_hash,
            "os_image_hash": self.os_image_hash,
        }


class EvalPhalaEventLogEntry(BaseModel):
    """One schema-closed event-log entry on the canonical Eval wire."""

    model_config = ConfigDict(extra="forbid", strict=True)

    imr: int = Field(ge=0, le=EVAL_MAX_INTEGER)
    event_type: int = Field(ge=0, le=EVAL_MAX_INTEGER)
    digest: str = Field(pattern=_REGISTER_PATTERN)
    event: str = Field(pattern=_VISIBLE_ID_PATTERN, max_length=EVAL_MAX_STRING_BYTES)
    event_payload: str = Field(
        pattern=_EVEN_HEX_PATTERN,
        max_length=EVAL_MAX_PAYLOAD_BYTES,
    )


class EvalPhalaVmConfig(BaseModel):
    """Evidence-only VM configuration carried in a canonical Eval attestation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    vcpu: int = Field(ge=1, le=EVAL_MAX_INTEGER)
    memory_mb: int = Field(ge=1, le=EVAL_MAX_INTEGER)
    os_image_hash: str | None = Field(pattern=_SHA256_PATTERN)


class EvalPhalaAttestation(BaseModel):
    """Strict Eval Phala attestation boundary, with no legacy field aliases."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tdx_quote: str = Field(
        pattern=_NONEMPTY_EVEN_HEX_PATTERN,
        max_length=2 * EVAL_MAX_QUOTE_BYTES,
    )
    event_log: list[EvalPhalaEventLogEntry]
    report_data: str = Field(pattern=_REPORT_DATA_PATTERN)
    measurement: EvalPhalaMeasurement
    vm_config: EvalPhalaVmConfig

    @model_validator(mode="before")
    @classmethod
    def validate_transport_bounds(cls, value: Any) -> Any:
        """Reject large nested transports before quote verification/allocation."""

        if not isinstance(value, Mapping):
            return value
        event_log = value.get("event_log")
        if isinstance(event_log, list):
            if len(event_log) > EVAL_MAX_EVENT_LOG_ENTRIES:
                raise ValueError("event_log exceeds its entry bound")
            encoded_bytes = 2 + max(0, len(event_log) - 1)
            for event in event_log:
                if not isinstance(event, Mapping):
                    continue
                if set(event) != {
                    "imr",
                    "event_type",
                    "digest",
                    "event",
                    "event_payload",
                }:
                    raise ValueError("event_log entry has invalid fields")
                for field, limit in (
                    ("digest", len("a" * 96)),
                    ("event", EVAL_MAX_STRING_BYTES),
                    ("event_payload", EVAL_MAX_PAYLOAD_BYTES),
                ):
                    field_value = event.get(field)
                    if isinstance(field_value, str) and len(field_value) > limit:
                        raise ValueError(f"event_log.{field} exceeds its string bound")
                try:
                    encoded_bytes += len(
                        json.dumps(
                            event,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode("utf-8")
                    )
                except (TypeError, ValueError, UnicodeEncodeError) as exc:
                    raise ValueError("event_log is not encodable") from exc
                if encoded_bytes > EVAL_MAX_EVENT_LOG_BYTES:
                    raise ValueError("event_log exceeds its byte bound")
        vm_config = value.get("vm_config")
        if isinstance(vm_config, Mapping):
            if set(vm_config) != {"vcpu", "memory_mb", "os_image_hash"}:
                raise ValueError("vm_config has invalid fields")
            os_image_hash = vm_config.get("os_image_hash")
            if isinstance(os_image_hash, str) and len(os_image_hash) > 64:
                raise ValueError("vm_config.os_image_hash exceeds its string bound")
            try:
                encoded = json.dumps(
                    vm_config,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeEncodeError) as exc:
                raise ValueError("vm_config is not encodable") from exc
            if len(encoded) > EVAL_MAX_VM_CONFIG_BYTES:
                raise ValueError("vm_config exceeds its byte bound")
        return value


class EvalWorkerSignature(BaseModel):
    """The sole in-CVM worker-signature placeholder accepted on the Eval wire."""

    model_config = ConfigDict(extra="forbid", strict=True)

    worker_pubkey: Literal[""]
    sig: Literal[""]


class EvalExecutionProof(BaseModel):
    """Schema-closed canonical Eval ``ExecutionProof`` wire envelope.

    This is intentionally separate from the permissive legacy
    :class:`ExecutionProof` model. The direct Eval result endpoint validates this
    model before replacing its exact empty signature placeholder with a
    validator-owned signature.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    tier: Literal["phala-tdx"]
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    image_digest: str = Field(
        pattern=r"^[^@\s]+@sha256:[0-9a-f]{64}$",
        max_length=EVAL_MAX_STRING_BYTES,
    )
    provider: Literal[None]
    worker_signature: EvalWorkerSignature
    attestation: EvalPhalaAttestation

    def to_execution_proof(self) -> ExecutionProof:
        """Convert validated canonical wire data to the legacy transport model."""

        return ExecutionProof.model_validate(self.model_dump(mode="json"))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """JSON object hook that rejects duplicate member names before Pydantic."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_eval_execution_proof_json(
    data: str | bytes | bytearray,
) -> EvalExecutionProof:
    """Parse a canonical Eval proof while rejecting duplicate JSON object keys."""

    decoded = json.loads(data, object_pairs_hook=_reject_duplicate_json_keys)
    return EvalExecutionProof.model_validate(decoded)


class ExecutionProof(BaseModel):
    """Proof envelope attached to every worker result (architecture 3.4).

    Tier 0 (mandatory, all backends) carries the deterministic
    ``manifest_sha256`` plus the worker's sr25519 ``worker_signature`` over the
    pinned message (``sha256(f"{manifest_sha256}:{unit_id}")``). Tier 1 adds
    ``image_digest`` + a populated ``provider`` block; tier 2 adds a non-null
    ``attestation``. The base worker plane emits tier 0; prism fills the higher
    tiers. The Phala TDX tier (``tier == PHALA_TDX_TIER``) carries a
    :class:`PhalaAttestation` payload in ``attestation`` (architecture.md sec 6);
    hence ``tier`` accepts the string Phala value in addition to the integer
    worker-plane tiers.
    """

    version: int = 1
    tier: int | str = 0
    manifest_sha256: str
    image_digest: str | None = None
    provider: ProviderInfo | None = None
    worker_signature: WorkerSignature
    attestation: dict[str, Any] | None = None
