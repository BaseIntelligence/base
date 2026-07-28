"""Discover Phala app identity from provision responses (handle, not trust pin).

``app_id`` is a deployment handle derived from (deployer_account, nonce). It must
be discovered from the provision response rather than asserted against a static
assignment pin. Real trust anchors remain ``compose_hash`` and the OS /
measurement identity (account-independent for the same compose/OS/shape).

Pure helpers only — no network I/O, no env reads. Review and eval self-deploy
paths call these; this module does not own those call sites.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from dstack_sdk import verify_env_encrypt_public_key

from agent_challenge.selfdeploy.measurements import (
    ProvisionOsIdentityError,
    verify_provision_os_identity,
)

#: Production Phala CREATE-style app_id (20-byte address as lowercase hex).
_APP_ID_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
#: X25519 public key as 32 raw bytes → 64 hex chars (case-insensitive on wire).
_PUBKEY_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ProvisionIdentityError(ValueError):
    """Provision response identity or trust-anchor check failed (fail-closed)."""


@dataclass(frozen=True, slots=True)
class DiscoveredPhalaAppIdentity:
    """Phala app handle + env-encrypt material discovered from provision."""

    app_id: str
    app_env_encrypt_pubkey: str
    kms_public_key_sha256: str
    signature: bytes | None = None
    timestamp: int | None = None


def env_keys_from_allowed(
    allowed: Sequence[str],
    selected: Collection[str] | None = None,
) -> list[str]:
    """Return env *names* for a provision request (no ciphertext).

    Order follows ``allowed``. When ``selected`` is given, only names present in
    both are returned (still in ``allowed`` order). Unknown selected names raise.
    """

    if selected is None:
        return list(allowed)
    allowed_set = set(allowed)
    unknown = sorted(name for name in selected if name not in allowed_set)
    if unknown:
        raise ProvisionIdentityError(
            f"env key selection contains names outside the allowed set ({len(unknown)} unknown)"
        )
    selected_set = set(selected)
    return [name for name in allowed if name in selected_set]


def parse_discovered_identity(provision: Mapping[str, Any]) -> DiscoveredPhalaAppIdentity:
    """Parse app_id + env-encrypt pubkey from a provision response (fail-closed).

    Never embeds the full pubkey or any secret in exception messages.
    """

    app_id = provision.get("app_id")
    if not isinstance(app_id, str) or not app_id:
        raise ProvisionIdentityError("provision app_id is missing or not a non-empty string")
    if _APP_ID_HEX40_RE.fullmatch(app_id) is None:
        raise ProvisionIdentityError(
            "provision app_id is not a lowercase 40-hex Phala CREATE-style address"
        )

    pubkey = provision.get("app_env_encrypt_pubkey")
    if not isinstance(pubkey, str) or not pubkey:
        raise ProvisionIdentityError(
            "provision app_env_encrypt_pubkey is missing or not a non-empty string"
        )
    if _PUBKEY_HEX64_RE.fullmatch(pubkey) is None:
        raise ProvisionIdentityError(
            "provision app_env_encrypt_pubkey is not a 64-hex X25519 public key"
        )
    pubkey_norm = pubkey.lower()
    try:
        pubkey_bytes = bytes.fromhex(pubkey_norm)
    except ValueError as exc:
        raise ProvisionIdentityError("provision app_env_encrypt_pubkey is not valid hex") from exc
    if len(pubkey_bytes) != 32:
        raise ProvisionIdentityError(
            "provision app_env_encrypt_pubkey is not a 32-byte X25519 public key"
        )

    signature = _optional_signature(provision)
    timestamp = _optional_timestamp(provision)

    return DiscoveredPhalaAppIdentity(
        app_id=app_id,
        app_env_encrypt_pubkey=pubkey_norm,
        kms_public_key_sha256=sha256(pubkey_bytes).hexdigest(),
        signature=signature,
        timestamp=timestamp,
    )


def assert_provision_trust_anchors(
    *,
    plan_compose_hash: str,
    plan_measurement: Mapping[str, str],
    provision: Mapping[str, Any],
) -> None:
    """Hard-fail when compose_hash or OS/measurement identity mismatches the plan.

    These are the real trust anchors (account-independent). ``app_id`` is not
    checked here — discover it via :func:`parse_discovered_identity`.
    """

    if provision.get("compose_hash") != plan_compose_hash:
        raise ProvisionIdentityError("provision compose_hash mismatches plan compose hash")
    try:
        verify_provision_os_identity(
            measurement=plan_measurement,
            provision_os=provision.get("os_image_hash"),
            mismatch_message=("provision os_image_hash mismatches plan measurement"),
        )
    except ProvisionOsIdentityError as exc:
        raise ProvisionIdentityError(str(exc)) from exc


def optional_verify_env_encrypt_pubkey(
    identity: DiscoveredPhalaAppIdentity,
    *,
    max_age_seconds: int = 300,
) -> None:
    """KMS-signature hardening hook; skip when signature material is absent.

    Present-and-invalid signatures always fail closed. Never compares against an
    assignment pin.
    """

    if identity.signature is None and identity.timestamp is None:
        return
    if identity.signature is None or identity.timestamp is None:
        raise ProvisionIdentityError("provision env-encrypt signature material is incomplete")
    pubkey_bytes = bytes.fromhex(identity.app_env_encrypt_pubkey)
    signer = verify_env_encrypt_public_key(
        pubkey_bytes,
        identity.signature,
        identity.app_id,
        identity.timestamp,
        max_age_seconds=max_age_seconds,
    )
    if signer is None:
        raise ProvisionIdentityError(
            "provision env-encrypt public key signature verification failed"
        )


def _optional_signature(provision: Mapping[str, Any]) -> bytes | None:
    raw = provision.get("signature")
    if raw is None:
        return None
    if isinstance(raw, bytes | bytearray):
        return bytes(raw)
    if isinstance(raw, str) and raw:
        try:
            return bytes.fromhex(raw.removeprefix("0x"))
        except ValueError as exc:
            raise ProvisionIdentityError("provision signature is not valid hex") from exc
    raise ProvisionIdentityError("provision signature has an unsupported type")


def _optional_timestamp(provision: Mapping[str, Any]) -> int | None:
    raw = provision.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ProvisionIdentityError("provision timestamp must be an int when present")
    return raw
