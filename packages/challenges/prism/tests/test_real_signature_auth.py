from __future__ import annotations

import hmac
import json
import time
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.auth import canonical_submission_message, verify_hotkey_signature
from prism_challenge.config import PrismSettings

ALICE_SS58 = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
ALICE_NONCE = "test-nonce-13"
ALICE_TIMESTAMP = "1750000000"
ALICE_BODY = b'{"code":"print(1)","filename":"model.py"}'
ALICE_SIG_HEX = (
    "bc88793cb7dbcd7148c5592da54cee8d3b40956c36a6f367281d4341bedfce14"
    "4b43bf85a664fa108faf9d1ec97da027f144e67e47aea3a955de4e0d2069dc89"
)
FORGED_SIG_HEX = (
    "1e6c877697c7adc1a2fe951a462883bfd76658f01a875748dade03c83d45be68"
    "890a819de4f7500e2eac539019c6832f3031a7823cecb367b09e9842f2701885"
)


def _alice_message() -> bytes:
    return canonical_submission_message(
        hotkey=ALICE_SS58, nonce=ALICE_NONCE, timestamp=ALICE_TIMESTAMP, body=ALICE_BODY
    )


def test_real_sr25519_signature_accepted_by_production_verifier() -> None:
    assert verify_hotkey_signature(ALICE_SS58, _alice_message(), ALICE_SIG_HEX) is True


def test_forged_signature_rejected_by_production_verifier() -> None:
    assert verify_hotkey_signature(ALICE_SS58, _alice_message(), FORGED_SIG_HEX) is False


def test_wrong_message_rejected_by_production_verifier() -> None:
    tampered = canonical_submission_message(
        hotkey=ALICE_SS58, nonce="tampered-nonce", timestamp=ALICE_TIMESTAMP, body=ALICE_BODY
    )
    assert verify_hotkey_signature(ALICE_SS58, tampered, ALICE_SIG_HEX) is False


def test_unsigned_garbage_rejected_by_production_verifier() -> None:
    assert verify_hotkey_signature(ALICE_SS58, _alice_message(), "not-a-real-signature") is False


def test_dev_hmac_is_not_a_valid_sr25519_signature() -> None:
    message = _alice_message()
    dev_hmac = hmac.new(b"secret", message, sha256).hexdigest()
    assert verify_hotkey_signature(ALICE_SS58, message, dev_hmac) is False


@pytest.fixture
def secure_client(tmp_path: Path) -> TestClient:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=False,
        fineweb_sample_count=4,
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def _submission_body(code: str = "print(1)") -> bytes:
    payload = {"code": code, "filename": "model.py"}
    return json.dumps(payload, separators=(",", ":")).encode()


def test_live_signed_submission_authenticates(secure_client: TestClient) -> None:
    pytest.importorskip("bittensor")
    from prism_challenge.keypair import keypair_from_uri

    keypair = keypair_from_uri("//Alice")
    body = _submission_body()
    nonce = "live-accept-1"
    timestamp = str(int(time.time()))
    message = canonical_submission_message(
        hotkey=keypair.ss58_address, nonce=nonce, timestamp=timestamp, body=body
    )
    signature = keypair.sign(message).hex()
    response = secure_client.post(
        "/v1/submissions",
        content=body,
        headers={
            "X-Hotkey": keypair.ss58_address,
            "X-Signature": signature,
            "X-Nonce": nonce,
            "X-Timestamp": timestamp,
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text


def test_live_forged_submission_rejected(secure_client: TestClient) -> None:
    pytest.importorskip("bittensor")
    from prism_challenge.keypair import keypair_from_uri

    keypair = keypair_from_uri("//Alice")
    attacker = keypair_from_uri("//Bob")
    body = _submission_body()
    nonce = "live-forge-1"
    timestamp = str(int(time.time()))
    message = canonical_submission_message(
        hotkey=keypair.ss58_address, nonce=nonce, timestamp=timestamp, body=body
    )
    forged = attacker.sign(message).hex()
    response = secure_client.post(
        "/v1/submissions",
        content=body,
        headers={
            "X-Hotkey": keypair.ss58_address,
            "X-Signature": forged,
            "X-Nonce": nonce,
            "X-Timestamp": timestamp,
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 401, response.text


def test_dev_hmac_submission_rejected_when_insecure_disabled(secure_client: TestClient) -> None:
    pytest.importorskip("bittensor")
    from prism_challenge.keypair import keypair_from_uri

    keypair = keypair_from_uri("//Alice")
    body = _submission_body()
    nonce = "live-hmac-1"
    timestamp = str(int(time.time()))
    message = canonical_submission_message(
        hotkey=keypair.ss58_address, nonce=nonce, timestamp=timestamp, body=body
    )
    dev_hmac = hmac.new(b"secret", message, sha256).hexdigest()
    response = secure_client.post(
        "/v1/submissions",
        content=body,
        headers={
            "X-Hotkey": keypair.ss58_address,
            "X-Signature": dev_hmac,
            "X-Nonce": nonce,
            "X-Timestamp": timestamp,
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 401, response.text
