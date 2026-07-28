"""Provision identity discovery: app_id is a handle, not a trust pin.

Trust anchors remain compose_hash + OS/measurement. These tests lock the shared
helpers that review/eval self-deploy will call later — discovery must accept a
live Phala app_id that differs from any assignment placeholder.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any
from unittest.mock import patch

import pytest

from agent_challenge.selfdeploy.provision_identity import (
    DiscoveredPhalaAppIdentity,
    ProvisionIdentityError,
    assert_provision_trust_anchors,
    env_keys_from_allowed,
    optional_verify_env_encrypt_pubkey,
    parse_discovered_identity,
)

# Assignment-style placeholder (operator-minted pin that must NOT gate discovery).
_ASSIGNMENT_PLACEHOLDER_APP_ID = "f024ea23" + ("ab" * 16)
# Live Phala CREATE-style app_id for a different deployer account / nonce 0.
_LIVE_DISCOVERED_APP_ID = "1850aa11" + ("cd" * 16)
_VALID_PUBKEY_HEX = "11" * 32
_VALID_COMPOSE_HASH = "aa" * 32
_VALID_OS_HASH = "bb" * 32


def _valid_provision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "app_id": _LIVE_DISCOVERED_APP_ID,
        "app_env_encrypt_pubkey": _VALID_PUBKEY_HEX,
        "compose_hash": _VALID_COMPOSE_HASH,
        "os_image_hash": _VALID_OS_HASH,
    }
    base.update(overrides)
    return base


def test_parse_accepts_app_id_different_from_assignment_placeholder() -> None:
    """Given: provision app_id ≠ any assignment pin. When: parse. Then: discovered."""

    assert _LIVE_DISCOVERED_APP_ID != _ASSIGNMENT_PLACEHOLDER_APP_ID
    provision = _valid_provision(app_id=_LIVE_DISCOVERED_APP_ID)
    identity = parse_discovered_identity(provision)
    assert isinstance(identity, DiscoveredPhalaAppIdentity)
    assert identity.app_id == _LIVE_DISCOVERED_APP_ID
    assert identity.app_env_encrypt_pubkey == _VALID_PUBKEY_HEX
    expected_sha = sha256(bytes.fromhex(_VALID_PUBKEY_HEX)).hexdigest()
    assert identity.kms_public_key_sha256 == expected_sha
    assert identity.signature is None
    assert identity.timestamp is None


def test_assert_trust_anchors_raises_on_compose_hash_mismatch() -> None:
    """Given: provision compose_hash ≠ plan. When: assert anchors. Then: raises."""

    provision = _valid_provision(compose_hash="cc" * 32)
    with pytest.raises(ProvisionIdentityError, match="compose"):
        assert_provision_trust_anchors(
            plan_compose_hash=_VALID_COMPOSE_HASH,
            plan_measurement={"os_image_hash": _VALID_OS_HASH},
            provision=provision,
        )


def test_parse_raises_on_missing_app_id() -> None:
    """Given: no app_id. When: parse. Then: domain error (no secret leakage)."""

    provision = _valid_provision()
    del provision["app_id"]
    with pytest.raises(ProvisionIdentityError, match="app_id"):
        parse_discovered_identity(provision)


def test_parse_raises_on_malformed_app_id() -> None:
    """Given: non-40-hex app_id. When: parse. Then: raises."""

    with pytest.raises(ProvisionIdentityError, match="app_id"):
        parse_discovered_identity(_valid_provision(app_id="not-a-phala-app-id"))
    with pytest.raises(ProvisionIdentityError, match="app_id"):
        parse_discovered_identity(_valid_provision(app_id="F024EA23" + ("AB" * 16)))
    with pytest.raises(ProvisionIdentityError, match="app_id"):
        parse_discovered_identity(_valid_provision(app_id=""))


def test_parse_raises_on_malformed_pubkey() -> None:
    """Given: bad app_env_encrypt_pubkey. When: parse. Then: raises without full key."""

    bad = "zz" * 32
    with pytest.raises(ProvisionIdentityError, match="pubkey|public.key|encrypt") as exc_info:
        parse_discovered_identity(_valid_provision(app_env_encrypt_pubkey=bad))
    assert bad not in str(exc_info.value)
    with pytest.raises(ProvisionIdentityError):
        parse_discovered_identity(_valid_provision(app_env_encrypt_pubkey="11" * 16))
    with pytest.raises(ProvisionIdentityError):
        parse_discovered_identity(_valid_provision(app_env_encrypt_pubkey=12345))


def test_env_keys_from_allowed_returns_names_only_deterministic() -> None:
    """Given: allowed names (+ optional selected). When: env_keys. Then: ordered names."""

    allowed = ("REVIEW_SESSION_TOKEN", "REVIEW_API_BASE_URL", "OPENROUTER_API_KEY")
    assert env_keys_from_allowed(allowed) == list(allowed)
    assert env_keys_from_allowed(allowed, selected=None) == list(allowed)
    selected = {"OPENROUTER_API_KEY", "REVIEW_SESSION_TOKEN"}
    assert env_keys_from_allowed(allowed, selected=selected) == [
        "REVIEW_SESSION_TOKEN",
        "OPENROUTER_API_KEY",
    ]
    # No ciphertext / values involved — pure name list.
    assert all(isinstance(name, str) for name in env_keys_from_allowed(allowed))


def test_optional_verify_raises_when_signature_present_and_invalid() -> None:
    """Given: signature+timestamp present, verifier returns None. When: verify. Then: raises."""

    identity = DiscoveredPhalaAppIdentity(
        app_id=_LIVE_DISCOVERED_APP_ID,
        app_env_encrypt_pubkey=_VALID_PUBKEY_HEX,
        kms_public_key_sha256=sha256(bytes.fromhex(_VALID_PUBKEY_HEX)).hexdigest(),
        signature=b"\x00" * 65,
        timestamp=1_700_000_000,
    )
    with patch(
        "agent_challenge.selfdeploy.provision_identity.verify_env_encrypt_public_key",
        return_value=None,
    ):
        with pytest.raises(ProvisionIdentityError, match="signature|encrypt"):
            optional_verify_env_encrypt_pubkey(identity)


def test_optional_verify_passes_when_signature_absent() -> None:
    """Given: no signature fields. When: optional verify. Then: silent pass."""

    identity = DiscoveredPhalaAppIdentity(
        app_id=_LIVE_DISCOVERED_APP_ID,
        app_env_encrypt_pubkey=_VALID_PUBKEY_HEX,
        kms_public_key_sha256=sha256(bytes.fromhex(_VALID_PUBKEY_HEX)).hexdigest(),
        signature=None,
        timestamp=None,
    )
    optional_verify_env_encrypt_pubkey(identity)


def test_assert_trust_anchors_passes_when_compose_and_os_match() -> None:
    """Happy path: compose_hash + os_image_hash match plan measurement."""

    provision = _valid_provision()
    assert_provision_trust_anchors(
        plan_compose_hash=_VALID_COMPOSE_HASH,
        plan_measurement={"os_image_hash": _VALID_OS_HASH},
        provision=provision,
    )
