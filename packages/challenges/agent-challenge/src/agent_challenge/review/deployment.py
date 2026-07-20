"""Validator-owned review deployment identity and acknowledgement validation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

from agent_challenge.review.canonical import canonical_sha256
from agent_challenge.sdk.config import ChallengeSettings

from .compose import (
    DEFAULT_REVIEW_APP_IDENTITY,
    ReviewComposeError,
    generate_review_app_compose,
    review_app_compose_hash,
)
from .schemas import (
    AssignmentSchemaError,
    ReviewInputConfig,
    _require_id,
    _require_sha256,
    _require_time_ms,
    validate_review_assignment,
)

CANONICAL_MEASUREMENT_FIELDS = (
    "mrtd",
    "rtmr0",
    "rtmr1",
    "rtmr2",
    "compose_hash",
    "os_image_hash",
)
REVIEW_DEPLOYED_ACK_SCHEMA_VERSION = 1
DEPLOYED_ACK_FIELDS = frozenset(
    {
        "schema_version",
        "assignment_id",
        "cvm_id",
        "phala_create_receipt",
        "compose_identity",
    }
)
PHALA_CREATE_RECEIPT_FIELDS = frozenset(
    {
        "request_id",
        "app_id",
        "cvm_id",
        "receipt_sha256",
        "created_at_ms",
    }
)
COMPOSE_IDENTITY_FIELDS = frozenset(
    {
        "image_ref",
        "compose_hash",
        "app_kms_public_key_sha256",
    }
)
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_DIGEST_IMAGE_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")


class ReviewDeploymentError(ValueError):
    """A review deployment identity is absent, malformed, or untrusted."""


def _canonical_measurement(measurement: Mapping[str, Any], compose_hash: str) -> dict[str, str]:
    expected = {
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "os_image_hash",
        "key_provider",
        "vm_shape",
    }
    if set(measurement) != expected:
        raise ReviewDeploymentError("review measurement must be schema-closed")
    values = {name: str(measurement[name]) for name in expected}
    return {
        "mrtd": values["mrtd"],
        "rtmr0": values["rtmr0"],
        "rtmr1": values["rtmr1"],
        "rtmr2": values["rtmr2"],
        "compose_hash": compose_hash,
        "os_image_hash": values["os_image_hash"],
    }


def _normalize_allowlist_entries(
    value: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != set(CANONICAL_MEASUREMENT_FIELDS):
            raise ReviewDeploymentError(f"{field} entries must use the canonical six-field shape")
        values = {name: str(entry[name]) for name in CANONICAL_MEASUREMENT_FIELDS}
        widths = (96, 96, 96, 96, 64, 64)
        ordered = tuple(values[name] for name in CANONICAL_MEASUREMENT_FIELDS)
        if any(
            len(item) != width or not _HEX_RE.fullmatch(item)
            for item, width in zip(ordered, widths, strict=True)
        ):
            raise ReviewDeploymentError(f"{field} entries must use exact lowercase-hex widths")
        if ordered in seen:
            raise ReviewDeploymentError(f"{field} entries must be unique")
        seen.add(ordered)
        entries.append(values)
    if not entries:
        raise ReviewDeploymentError(f"{field} must be non-empty")
    return entries


def _allowlist_digest(entries: Sequence[Mapping[str, str]]) -> str:
    return canonical_sha256({"entries": list(entries)})


def review_input_config_from_settings(settings: ChallengeSettings) -> ReviewInputConfig:
    """Build review assignment identity from validator config, never miner input.

    Review cannot start unless the configured image, deterministic compose,
    signed X25519 key, review measurement, and dedicated allowlist agree.  The
    eval application identity (image/compose/KMS/allowlist) must share no exact
    identity field or measurement entry with review.
    """

    image_ref = settings.review_app_image_ref
    app_identity = settings.review_app_identity
    public_key = settings.review_app_kms_public_key_hex
    measurement = settings.review_app_measurement
    if not isinstance(measurement, Mapping):
        raise ReviewDeploymentError("validator review measurement is unavailable")
    # Phala deterministic assigned app_id is a 40-hex string. Compose ``name``
    # still uses the product moniker so measured compose_hash stays stable.
    # Never invent moniker->hex melt; pin equality is handled on provision verify.
    compose_name = app_identity
    if isinstance(app_identity, str) and re.fullmatch(r"[0-9a-f]{40}", app_identity.lower() or ""):
        compose_name = DEFAULT_REVIEW_APP_IDENTITY
        app_identity = app_identity.lower()
    try:
        compose = generate_review_app_compose(review_image=image_ref, app_identity=compose_name)
    except ReviewComposeError as exc:
        raise ReviewDeploymentError(str(exc)) from exc
    computed_compose_hash = review_app_compose_hash(compose)
    if settings.review_app_compose_hash != computed_compose_hash:
        raise ReviewDeploymentError(
            "validator review compose hash does not match canonical compose"
        )
    if app_identity == DEFAULT_REVIEW_APP_IDENTITY and not image_ref:
        raise ReviewDeploymentError("validator review image is unavailable")

    review_entries = _normalize_allowlist_entries(
        settings.review_app_measurement_allowlist,
        field="review_app_measurement_allowlist",
    )
    eval_entries = _normalize_allowlist_entries(
        settings.eval_app_measurement_allowlist,
        field="eval_app_measurement_allowlist",
    )
    review_digest = _allowlist_digest(review_entries)

    config = ReviewInputConfig(
        image_ref=image_ref,
        compose_hash=computed_compose_hash,
        app_identity=app_identity,
        kms_public_key_hex=public_key,
        measurement=dict(measurement),
        measurement_allowlist=tuple(review_entries),
        measurement_allowlist_sha256=review_digest,
    )
    # Assignment schema is the single source for exact scalar validation,
    # including the literal ``x25519`` algorithm, public-key digest, and the
    # bound allowlist identity.
    try:
        provisional = {
            "image_ref": config.image_ref,
            "compose_hash": config.compose_hash,
            "app_identity": config.app_identity,
            "kms_key_algorithm": config.kms_key_algorithm,
            "kms_public_key_hex": config.kms_public_key_hex,
            "kms_public_key_sha256": sha256(bytes.fromhex(config.kms_public_key_hex)).hexdigest(),
            "measurement": config.resolved_measurement(),
            "measurement_allowlist": config.resolved_measurement_allowlist(),
            "measurement_allowlist_sha256": config.resolved_measurement_allowlist_sha256(),
        }
        from .schemas import _validate_review_app

        _validate_review_app(provisional, AssignmentSchemaError)
    except (AssignmentSchemaError, ValueError) as exc:
        raise ReviewDeploymentError("validator review app identity is malformed") from exc

    review_measurement = _canonical_measurement(
        config.resolved_measurement(),
        computed_compose_hash,
    )
    review_tuple = tuple(review_measurement[name] for name in CANONICAL_MEASUREMENT_FIELDS)
    review_entry_tuples = {
        tuple(entry[name] for name in CANONICAL_MEASUREMENT_FIELDS) for entry in review_entries
    }
    eval_entry_tuples = {
        tuple(entry[name] for name in CANONICAL_MEASUREMENT_FIELDS) for entry in eval_entries
    }
    if review_tuple not in review_entry_tuples:
        raise ReviewDeploymentError("review measurement is not in validator-owned review allowlist")
    if review_entry_tuples & eval_entry_tuples:
        raise ReviewDeploymentError("review and eval measurement allowlists must be disjoint")

    eval_identity = settings.eval_app_identity
    eval_image = settings.eval_app_image_ref
    eval_compose_hash = settings.eval_app_compose_hash
    eval_kms = settings.eval_app_kms_public_key_hex
    if eval_identity and eval_identity == app_identity:
        raise ReviewDeploymentError("review and eval app identities must be disjoint")
    if eval_image and eval_image == image_ref:
        raise ReviewDeploymentError("review and eval image refs must be disjoint")
    if eval_compose_hash and eval_compose_hash == computed_compose_hash:
        raise ReviewDeploymentError("review and eval compose hashes must be disjoint")
    if eval_kms and eval_kms == public_key:
        raise ReviewDeploymentError("review and eval KMS public keys must be disjoint")
    return config


def validate_review_deployed_acknowledgement(
    assignment: Mapping[str, Any],
    acknowledgement: Mapping[str, Any],
) -> None:
    """Accept only nested Review deployed acknowledgement v1 bound to assignment."""

    try:
        validate_review_assignment(assignment)
    except AssignmentSchemaError as exc:
        raise ReviewDeploymentError("stored review assignment is invalid") from exc
    if not isinstance(acknowledgement, Mapping) or set(acknowledgement) != DEPLOYED_ACK_FIELDS:
        raise ReviewDeploymentError("review deployed acknowledgement must be schema-closed")
    if acknowledgement["schema_version"] != REVIEW_DEPLOYED_ACK_SCHEMA_VERSION:
        raise ReviewDeploymentError("unsupported review deployed acknowledgement schema version")

    core = assignment["assignment_core"]
    review_app = core["review_app"]
    try:
        _require_id(acknowledgement["assignment_id"], "assignment_id", ReviewDeploymentError)
        _require_id(acknowledgement["cvm_id"], "cvm_id", ReviewDeploymentError)
    except ValueError as exc:
        raise ReviewDeploymentError(str(exc)) from exc
    if acknowledgement["assignment_id"] != core["assignment_id"]:
        raise ReviewDeploymentError(
            "review deployed acknowledgement assignment_id mismatches assignment"
        )

    receipt = acknowledgement["phala_create_receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != PHALA_CREATE_RECEIPT_FIELDS:
        raise ReviewDeploymentError(
            "review deployed acknowledgement phala_create_receipt must be schema-closed"
        )
    try:
        _require_id(receipt["request_id"], "request_id", ReviewDeploymentError)
        _require_id(receipt["app_id"], "app_id", ReviewDeploymentError)
        _require_id(receipt["cvm_id"], "receipt cvm_id", ReviewDeploymentError)
        _require_sha256(receipt["receipt_sha256"], "receipt_sha256", ReviewDeploymentError)
        _require_time_ms(receipt["created_at_ms"], "created_at_ms", ReviewDeploymentError)
    except ValueError as exc:
        raise ReviewDeploymentError(
            f"review deployed acknowledgement receipt is invalid: {exc}"
        ) from exc
    if receipt["cvm_id"] != acknowledgement["cvm_id"]:
        raise ReviewDeploymentError(
            "review deployed acknowledgement receipt cvm_id mismatches top-level cvm_id"
        )
    if receipt["app_id"] != review_app["app_identity"]:
        raise ReviewDeploymentError(
            "review deployed acknowledgement receipt app_id mismatches assignment"
        )

    compose_identity = acknowledgement["compose_identity"]
    if (
        not isinstance(compose_identity, Mapping)
        or set(compose_identity) != COMPOSE_IDENTITY_FIELDS
    ):
        raise ReviewDeploymentError(
            "review deployed acknowledgement compose_identity must be schema-closed"
        )
    expected_identity = {
        "image_ref": review_app["image_ref"],
        "compose_hash": review_app["compose_hash"],
        "app_kms_public_key_sha256": review_app["kms_public_key_sha256"],
    }
    for field, expected in expected_identity.items():
        if compose_identity.get(field) != expected:
            raise ReviewDeploymentError(
                f"review deployed acknowledgement compose_identity {field} mismatches assignment"
            )
    if not isinstance(compose_identity["image_ref"], str) or not _DIGEST_IMAGE_RE.fullmatch(
        compose_identity["image_ref"]
    ):
        raise ReviewDeploymentError(
            "review deployed acknowledgement image_ref must be digest-pinned"
        )
    try:
        _require_sha256(compose_identity["compose_hash"], "compose_hash", ReviewDeploymentError)
        _require_sha256(
            compose_identity["app_kms_public_key_sha256"],
            "app_kms_public_key_sha256",
            ReviewDeploymentError,
        )
    except ValueError as exc:
        raise ReviewDeploymentError(str(exc)) from exc


def build_review_deployed_acknowledgement(
    *,
    assignment: Mapping[str, Any],
    cvm_id: str,
    request_id: str,
    receipt_sha256: str,
    created_at_ms: int,
) -> dict[str, Any]:
    """Emit the exact nested Review deployed acknowledgement v1 object."""

    try:
        validate_review_assignment(assignment)
    except AssignmentSchemaError as exc:
        raise ReviewDeploymentError("stored review assignment is invalid") from exc
    review_app = assignment["assignment_core"]["review_app"]
    acknowledgement = {
        "schema_version": REVIEW_DEPLOYED_ACK_SCHEMA_VERSION,
        "assignment_id": assignment["assignment_core"]["assignment_id"],
        "cvm_id": cvm_id,
        "phala_create_receipt": {
            "request_id": request_id,
            "app_id": review_app["app_identity"],
            "cvm_id": cvm_id,
            "receipt_sha256": receipt_sha256,
            "created_at_ms": created_at_ms,
        },
        "compose_identity": {
            "image_ref": review_app["image_ref"],
            "compose_hash": review_app["compose_hash"],
            "app_kms_public_key_sha256": review_app["kms_public_key_sha256"],
        },
    }
    validate_review_deployed_acknowledgement(assignment, acknowledgement)
    return acknowledgement


__all__ = [
    "CANONICAL_MEASUREMENT_FIELDS",
    "COMPOSE_IDENTITY_FIELDS",
    "DEPLOYED_ACK_FIELDS",
    "PHALA_CREATE_RECEIPT_FIELDS",
    "REVIEW_DEPLOYED_ACK_SCHEMA_VERSION",
    "ReviewDeploymentError",
    "build_review_deployed_acknowledgement",
    "review_input_config_from_settings",
    "validate_review_deployed_acknowledgement",
]
