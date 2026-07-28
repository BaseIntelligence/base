"""Review self-deploy: discover Phala app_id from provision (handle, not pin).

Trust anchors remain compose_hash + OS/measurement. Assignment 40-hex
app_identity is advisory only; moniker app_identity still binds compose name.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from agent_challenge.review.canonical import canonical_sha256
from agent_challenge.review.compose import (
    DEFAULT_REVIEW_APP_IDENTITY,
    generate_review_app_compose,
    review_app_compose_hash,
)
from agent_challenge.review.deployment import (
    ReviewDeploymentError as ReviewAckError,
)
from agent_challenge.review.deployment import (
    build_review_deployed_acknowledgement,
    validate_review_deployed_acknowledgement,
)
from agent_challenge.review.schemas import ReviewInputConfig, build_review_assignment
from agent_challenge.selfdeploy.review import (
    ReviewDeploymentError,
    ReviewPhalaDeployment,
    build_review_deployment_plan,
    encrypt_review_secrets,
)

# Operator-minted assignment pin (must NOT gate deploy when Phala returns another).
_ASSIGNMENT_PIN_APP_ID = "f024ea23" + ("ab" * 16)
# Live Phala CREATE-style app_id for a different deployer account at nonce 0.
_LIVE_DISCOVERED_APP_ID = "1850aa11" + ("cd" * 16)
# Pre-change moniker compose_hash baseline (image a*64 + moniker name).
_MONIKER_COMPOSE_HASH_BASELINE = "4bcdae7cc06e733e03790391dd0563b86994a4d7f37b823e0c98fc1d32503a97"
_DEFAULT_MONIKER_COMPOSE_HASH_BASELINE = (
    "20c80f5e1c7a5ef7ff10951de4bd065ae7a1e6e87650e8a13c9ebc882373d4a5"
)

REVIEW_IMAGE = "docker.io/example/agent-challenge-review@sha256:" + ("a" * 64)
MEASUREMENT = {
    "mrtd": "01" * 48,
    "rtmr0": "02" * 48,
    "rtmr1": "03" * 48,
    "rtmr2": "04" * 48,
    "os_image_hash": "05" * 32,
    "key_provider": "phala",
    "vm_shape": "tdx.small",
}
TOKEN = "review-session-token-discovery"


def _pubkey_hex() -> tuple[X25519PrivateKey, str]:
    private_key = X25519PrivateKey.generate()
    public_key_hex = (
        private_key.public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        .hex()
    )
    return private_key, public_key_hex


def _assignment(
    *,
    public_key_hex: str,
    app_identity: str,
    compose_name_for_hash: str,
) -> tuple[dict[str, Any], str]:
    compose = generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity=compose_name_for_hash,
    )
    compose_hash = review_app_compose_hash(compose)
    allowlisted = {
        "mrtd": MEASUREMENT["mrtd"],
        "rtmr0": MEASUREMENT["rtmr0"],
        "rtmr1": MEASUREMENT["rtmr1"],
        "rtmr2": MEASUREMENT["rtmr2"],
        "compose_hash": compose_hash,
        "os_image_hash": MEASUREMENT["os_image_hash"],
    }
    config = ReviewInputConfig(
        image_ref=REVIEW_IMAGE,
        compose_hash=compose_hash,
        app_identity=app_identity,
        kms_public_key_hex=public_key_hex,
        measurement=MEASUREMENT,
        measurement_allowlist=(allowlisted,),
        measurement_allowlist_sha256=canonical_sha256({"entries": [allowlisted]}),
    )
    assignment, _bytes, _digest = build_review_assignment(
        session_id="rs-discovery",
        assignment_id="ra-discovery",
        attempt=1,
        submission_id="17",
        artifact={
            "agent_hash": "10" * 32,
            "zip_sha256": "20" * 32,
            "zip_size_bytes": 1,
            "manifest_sha256": "30" * 32,
            "manifest_entries_sha256": "40" * 32,
            "fetch_path": "/review/v1/assignments/ra-discovery/artifact",
        },
        rules_snapshot_sha256_value="50" * 32,
        rules_revision_id="rules-v1",
        review_nonce="rn-discovery",
        issued_at_ms=1,
        expires_at_ms=2,
        session_token_sha256=hashlib.sha256(TOKEN.encode()).hexdigest(),
        config=config,
    )
    return assignment, TOKEN


def _secrets(token: str) -> dict[str, str]:
    return {
        "OPENROUTER_API_KEY": "or-discovery-sentinel",
        "REVIEW_API_BASE_URL": "https://chain.joinbase.ai/challenges/agent-challenge",
        "REVIEW_SESSION_TOKEN": token,
    }


def test_deploy_proceeds_when_provision_app_id_differs_from_assignment_pin() -> None:
    """S1: provision app_id ≠ assignment pin → deploy uses discovered handle."""

    assert _LIVE_DISCOVERED_APP_ID != _ASSIGNMENT_PIN_APP_ID
    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name_for_hash=DEFAULT_REVIEW_APP_IDENTITY,
    )
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    assert plan.app_identity == _ASSIGNMENT_PIN_APP_ID
    assert plan.compose_name == DEFAULT_REVIEW_APP_IDENTITY
    encrypted = encrypt_review_secrets(plan, _secrets(token))
    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": _LIVE_DISCOVERED_APP_ID,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": public_key_hex,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
        create_response={
            "id": "cvm-discovered-1",
            "request_id": "req-discovered-1",
            "created_at_ms": 1000,
        },
    )
    acknowledgement = deployment.deploy(plan, encrypted)
    assert deployment.create_requests, "create must run after successful discovery"
    assert deployment.create_requests[0]["app_id"] == _LIVE_DISCOVERED_APP_ID
    assert acknowledgement["phala_create_receipt"]["app_id"] == _LIVE_DISCOVERED_APP_ID
    assert acknowledgement["cvm_id"] == "cvm-discovered-1"
    # Provision must send env names only (no ciphertext values).
    prov = deployment.provision_requests[0]
    assert "encrypted_env" not in prov
    assert set(prov["env_keys"]) == {
        "OPENROUTER_API_KEY",
        "REVIEW_API_BASE_URL",
        "REVIEW_SESSION_TOKEN",
    }


def test_deploy_hard_fails_on_compose_hash_mismatch() -> None:
    """S2: compose_hash mismatch remains a hard fail (trust anchor)."""

    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name_for_hash=DEFAULT_REVIEW_APP_IDENTITY,
    )
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    encrypted = encrypt_review_secrets(plan, _secrets(token))
    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": _LIVE_DISCOVERED_APP_ID,
            "compose_hash": "ff" * 32,
            "app_env_encrypt_pubkey": public_key_hex,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
        create_response={"id": "cvm-x", "request_id": "r", "created_at_ms": 1},
    )
    with pytest.raises(ReviewDeploymentError, match="compose"):
        deployment.deploy(plan, encrypted)
    assert deployment.create_requests == []


def test_deploy_hard_fails_on_os_measurement_mismatch() -> None:
    """S3: OS/measurement mismatch remains a hard fail (trust anchor)."""

    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name_for_hash=DEFAULT_REVIEW_APP_IDENTITY,
    )
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    encrypted = encrypt_review_secrets(plan, _secrets(token))
    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": _LIVE_DISCOVERED_APP_ID,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": public_key_hex,
            "os_image_hash": "de" + "9" * 62,
        },
        create_response={"id": "cvm-x", "request_id": "r", "created_at_ms": 1},
    )
    with pytest.raises(ReviewDeploymentError, match="os_image_hash|measurement"):
        deployment.deploy(plan, encrypted)
    assert deployment.create_requests == []


def test_ack_accepts_discovered_app_id_not_assignment_pin() -> None:
    """S4: ack validation accepts receipt app_id = discovered handle ≠ pin."""

    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name_for_hash=DEFAULT_REVIEW_APP_IDENTITY,
    )
    assert assignment["assignment_core"]["review_app"]["app_identity"] == _ASSIGNMENT_PIN_APP_ID
    acknowledgement = build_review_deployed_acknowledgement(
        assignment=assignment,
        cvm_id="cvm-ack-1",
        request_id="req-ack-1",
        receipt_sha256="6" * 64,
        created_at_ms=1_000,
        app_id=_LIVE_DISCOVERED_APP_ID,
    )
    assert acknowledgement["phala_create_receipt"]["app_id"] == _LIVE_DISCOVERED_APP_ID
    validate_review_deployed_acknowledgement(assignment, acknowledgement)

    # Pin equality must not be required: a receipt still carrying the old pin
    # is fine too when it is a valid id, but a garbage id still fails closed.
    bad = dict(acknowledgement)
    bad_receipt = dict(acknowledgement["phala_create_receipt"])
    bad_receipt["app_id"] = ""
    bad["phala_create_receipt"] = bad_receipt
    with pytest.raises(ReviewAckError):
        validate_review_deployed_acknowledgement(assignment, bad)


def test_moniker_path_compose_hash_byte_identical() -> None:
    """S5: non-hex moniker app_identity still feeds compose name; hash unchanged."""

    custom_moniker = "custom-moniker-review"
    compose = generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity=custom_moniker,
    )
    assert compose["name"] == custom_moniker
    assert review_app_compose_hash(compose) == _MONIKER_COMPOSE_HASH_BASELINE

    default_compose = generate_review_app_compose(
        review_image=REVIEW_IMAGE,
        app_identity=DEFAULT_REVIEW_APP_IDENTITY,
    )
    assert review_app_compose_hash(default_compose) == _DEFAULT_MONIKER_COMPOSE_HASH_BASELINE

    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=custom_moniker,
        compose_name_for_hash=custom_moniker,
    )
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    assert plan.compose_name == custom_moniker
    assert plan.app_identity == custom_moniker
    assert plan.compose_hash == _MONIKER_COMPOSE_HASH_BASELINE
    assert plan.phala_app_nonce is None


def test_discovery_provision_request_omits_nonce_and_app_id() -> None:
    """S-shape: production 40-hex path must not send nonce or app_id (live 422)."""

    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name_for_hash=DEFAULT_REVIEW_APP_IDENTITY,
    )
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    assert _APP_ID_IS_HEX40(plan.app_identity)
    encrypted = encrypt_review_secrets(plan, _secrets(token))
    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": _LIVE_DISCOVERED_APP_ID,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": public_key_hex,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
        create_response={
            "id": "cvm-shape-1",
            "request_id": "req-shape-1",
            "created_at_ms": 1000,
        },
    )
    deployment.deploy(plan, encrypted)
    prov = deployment.provision_requests[0]
    assert "nonce" not in prov, f"discovery must omit nonce; got keys={sorted(prov)}"
    assert "app_id" not in prov, f"discovery must omit app_id; got keys={sorted(prov)}"
    # Create still binds the discovered handle from the provision response.
    assert deployment.create_requests[0]["app_id"] == _LIVE_DISCOVERED_APP_ID


def test_moniker_provision_sends_app_id_without_nonce() -> None:
    """S-legacy: moniker path may send app_id alone (legal Phala combo)."""

    custom_moniker = "custom-moniker-review"
    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=custom_moniker,
        compose_name_for_hash=custom_moniker,
    )
    plan = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    encrypted = encrypt_review_secrets(plan, _secrets(token))
    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": _LIVE_DISCOVERED_APP_ID,
            "compose_hash": plan.compose_hash,
            "app_env_encrypt_pubkey": public_key_hex,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
        create_response={
            "id": "cvm-moniker-1",
            "request_id": "req-moniker-1",
            "created_at_ms": 1000,
        },
    )
    deployment.deploy(plan, encrypted)
    prov = deployment.provision_requests[0]
    assert "nonce" not in prov
    assert prov.get("app_id") == custom_moniker
    assert deployment.create_requests[0]["app_id"] == _LIVE_DISCOVERED_APP_ID


def test_public_api_cannot_emit_nonce_without_app_id() -> None:
    """S-guard: even a hand-built plan with a nonce must not emit illegal shape."""

    from agent_challenge.selfdeploy.review import ReviewDeploymentPlan

    _private, public_key_hex = _pubkey_hex()
    assignment, token = _assignment(
        public_key_hex=public_key_hex,
        app_identity=_ASSIGNMENT_PIN_APP_ID,
        compose_name_for_hash=DEFAULT_REVIEW_APP_IDENTITY,
    )
    base = build_review_deployment_plan({"assignment": assignment, "review_session_token": token})
    # Reconstruct with an explicit nonce to prove deploy refuses the illegal combo.
    poisoned = ReviewDeploymentPlan(
        assignment=base.assignment,
        compose=base.compose,
        compose_text=base.compose_text,
        compose_hash=base.compose_hash,
        app_identity=base.app_identity,
        image_ref=base.image_ref,
        kms_public_key_hex=base.kms_public_key_hex,
        kms_public_key_sha256=base.kms_public_key_sha256,
        measurement=base.measurement,
        measurement_allowlist_sha256=base.measurement_allowlist_sha256,
        review_session_token=token,
        instance_type=base.instance_type,
        region=base.region,
        os_image=base.os_image,
        compose_name=base.compose_name,
        phala_app_nonce=0,
    )
    encrypted = encrypt_review_secrets(poisoned, _secrets(token))
    deployment = ReviewPhalaDeployment(
        provision_response={
            "app_id": _LIVE_DISCOVERED_APP_ID,
            "compose_hash": poisoned.compose_hash,
            "app_env_encrypt_pubkey": public_key_hex,
            "os_image_hash": MEASUREMENT["os_image_hash"],
        },
        create_response={
            "id": "cvm-guard-1",
            "request_id": "req-guard-1",
            "created_at_ms": 1,
        },
    )
    deployment.deploy(poisoned, encrypted)
    prov = deployment.provision_requests[0]
    has_nonce = "nonce" in prov
    has_app_id = "app_id" in prov
    assert not (has_nonce and not has_app_id), (
        f"illegal Phala shape nonce-without-app_id: keys={sorted(prov)}"
    )


def _APP_ID_IS_HEX40(value: str) -> bool:
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower())
