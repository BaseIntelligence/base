"""Bittensor Keypair resolution across v9-v11 (VAL-CIGREEN-002).

``bittensor.Keypair`` was removed in v11. Product code must use the shared
resolver in ``prism_challenge.keypair`` rather than the removed attribute.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from prism_challenge.keypair import (
    keypair_from_ss58,
    keypair_from_uri,
    reset_keypair_type_cache,
    resolve_keypair_type,
)


@pytest.fixture(autouse=True)
def _clear_keypair_cache() -> None:
    reset_keypair_type_cache()
    yield
    reset_keypair_type_cache()


def test_resolve_keypair_type_usable_for_sign_and_verify() -> None:
    pytest.importorskip("bittensor")
    cls = resolve_keypair_type()
    assert callable(getattr(cls, "create_from_uri", None))
    kp = keypair_from_uri("//Alice")
    msg = b"prism-keypair-compat"
    sig = kp.sign(msg)
    pub = keypair_from_ss58(kp.ss58_address)
    assert pub.verify(msg, sig) is True


def test_resolve_prefers_sp_core_when_top_level_removed(monkeypatch) -> None:
    """Emulate v11: accessing bittensor.Keypair raises AttributeError."""
    pytest.importorskip("bittensor")

    class Removed:
        def __getattr__(self, name: str) -> object:
            if name == "Keypair":
                raise AttributeError(
                    "bittensor.Keypair was removed in v11 — keypairs come from "
                    "bittensor.wallet.Wallet; the low-level type is bittensor.sp_core.Keypair."
                )
            raise AttributeError(name)

    real = resolve_keypair_type()
    import sys

    # Only sp_core/wallet-style modules supply Keypair; top-level bt refuses.
    sys.modules["bittensor.sp_core"] = SimpleNamespace(Keypair=real)  # type: ignore[assignment]
    sys.modules["bittensor.wallet"] = SimpleNamespace(Keypair=real)  # type: ignore[assignment]
    sys.modules["bittensor"] = Removed()  # type: ignore[assignment]
    sys.modules.pop("bittensor_wallet", None)

    reset_keypair_type_cache()
    # Prefer bittensor_wallet first in resolver; force that import to fail so sp_core wins.
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "bittensor_wallet" or name.startswith("bittensor_wallet."):
            raise ImportError("no bittensor_wallet in v11 emulation")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    cls = resolve_keypair_type()
    assert cls is real
    kp = keypair_from_uri("//Bob")
    assert isinstance(kp.ss58_address, str) and len(kp.ss58_address) > 10


def test_auth_verify_uses_compat_not_bt_keypair() -> None:
    """Production verifier path must not raise on v11-style installs."""
    pytest.importorskip("bittensor")
    from prism_challenge.auth import verify_hotkey_signature

    kp = keypair_from_uri("//Alice")
    msg = b"auth-compat"
    sig = kp.sign(msg)
    sig_hex = sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    assert verify_hotkey_signature(kp.ss58_address, msg, sig_hex) is True
    assert verify_hotkey_signature(kp.ss58_address, msg, "00" * 64) is False


def test_worker_signer_from_key_uri() -> None:
    pytest.importorskip("bittensor")
    from prism_challenge.proof import worker_signer_from_key

    signer = worker_signer_from_key("//WorkerAlice")
    assert signer.worker_pubkey.startswith("5")
    sig = signer.sign(b"proof-compat")
    assert isinstance(sig, str) and sig.startswith("0x")
