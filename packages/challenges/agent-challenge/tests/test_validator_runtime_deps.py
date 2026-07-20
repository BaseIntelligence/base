"""Validator-runtime dependency guards for VAL-CODE-VRT-001.

These prove the agent-challenge signing/verification path runs on
``bittensor.Keypair`` and that the legacy ``substrate-interface``/``scalecodec``
stack is no longer a runtime dependency, so ``base-validator-runtime:latest`` can
build bittensor-only and the validator can auto-update onto it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from agent_challenge.auth.security import (
    SignatureVerifierUnavailable,
    _verify_signature,
    canonical_request_string,
    verify_substrate_signature,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"


def _canonical() -> str:
    return canonical_request_string(
        method="POST",
        path="/submissions",
        query_string="",
        timestamp="2026-01-01T00:00:00+00:00",
        nonce="nonce-1",
        raw_body=b'{"name":"x","artifact_zip_base64":"QUJD"}',
    )


def _client_signature(keypair, message: str) -> str:
    """Mirror ``scripts/submit_agent.py::_sign`` (hex, ``0x``-prefixed)."""
    signature = keypair.sign(message)
    if isinstance(signature, bytes | bytearray):
        return "0x" + bytes(signature).hex()
    text = str(signature)
    return text if text.startswith("0x") else "0x" + text


def test_pyproject_drops_legacy_substrate_stack() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    dependencies = data["project"]["dependencies"]
    joined = "\n".join(dependencies).lower()

    assert "substrate-interface" not in joined
    assert "substrateinterface" not in joined
    assert "scalecodec" not in joined
    assert any(dep.lower().startswith("bittensor") for dep in dependencies), dependencies


def test_dispatch_module_imports_without_substrate_interface() -> None:
    """The build smoke: dispatch imports succeed and never pull substrate-interface."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import agent_challenge.validator_dispatch; "
                "assert 'substrateinterface' not in sys.modules, "
                "'validator_dispatch pulled the legacy substrate-interface stack'; "
                "print('dispatch-import-ok')"
            ),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dispatch-import-ok" in result.stdout


def test_verify_substrate_signature_round_trips_with_bittensor() -> None:
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    canonical = _canonical()
    signature = _client_signature(keypair, canonical)

    assert verify_substrate_signature(keypair.ss58_address, canonical, signature) is True


def test_verify_substrate_signature_rejects_tampered_message_and_wrong_hotkey() -> None:
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_uri("//Alice")
    other = bt.Keypair.create_from_uri("//Bob")
    canonical = _canonical()
    signature = _client_signature(keypair, canonical)

    assert verify_substrate_signature(keypair.ss58_address, canonical + "x", signature) is False
    assert verify_substrate_signature(other.ss58_address, canonical, signature) is False
    assert verify_substrate_signature(keypair.ss58_address, canonical, "0xdeadbeef") is False


def test_verify_substrate_signature_unavailable_is_soft_failure(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bittensor", None)

    with pytest.raises(SignatureVerifierUnavailable):
        verify_substrate_signature("hotkey", "message", "signature")

    # The auth path swallows unavailability into a rejection (never a crash).
    assert _verify_signature(verify_substrate_signature, "hotkey", "message", "signature") is False
