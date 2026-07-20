"""Bittensor sr25519 Keypair resolution compatible with v9–v11.

``bittensor.Keypair`` was removed in bittensor v11. The low-level type lives on
``bittensor.sp_core.Keypair`` (re-exported as ``bittensor.wallet.Keypair``); older
installs still expose ``bittensor.Keypair`` and/or the standalone
``bittensor_wallet.Keypair`` package. Prism never pin-imports the removed attribute.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

_KeypairType: type[Any] | None = None


def resolve_keypair_type() -> type[Any]:
    """Return the first available sr25519 Keypair class (cached)."""
    global _KeypairType
    if _KeypairType is not None:
        return _KeypairType

    candidates: tuple[tuple[str, str], ...] = (
        ("bittensor_wallet", "Keypair"),
        ("bittensor.sp_core", "Keypair"),
        ("bittensor.wallet", "Keypair"),
        ("bittensor", "Keypair"),
    )
    errors: list[str] = []
    for module_name, attr in candidates:
        try:
            module = __import__(module_name, fromlist=[attr])
            keypair_cls = getattr(module, attr, None)
            if keypair_cls is None:
                errors.append(f"{module_name}.{attr} missing")
                continue
            # v11 raises AttributeError on bittensor.Keypair via __getattr__ only when
            # accessed; getattr on a real class succeeds. Reject the removed stub just in case.
            if not callable(getattr(keypair_cls, "create_from_uri", None)):
                errors.append(f"{module_name}.{attr} has no create_from_uri")
                continue
            _KeypairType = cast(type[Any], keypair_cls)
            return _KeypairType
        except Exception as exc:  # ImportError | AttributeError (v11 removed msg)
            errors.append(f"{module_name}.{attr}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "No bittensor-compatible Keypair found. Install bittensor>=9 (v11 uses "
        "bittensor.sp_core.Keypair). Tried: " + "; ".join(errors)
    )


def keypair_from_ss58(ss58_address: str) -> Any:
    """Public-only keypair used for signature verification."""
    return resolve_keypair_type()(ss58_address=ss58_address)


def keypair_from_uri(uri: str) -> Any:
    return resolve_keypair_type().create_from_uri(uri)


def keypair_from_mnemonic(mnemonic: str) -> Any:
    return resolve_keypair_type().create_from_mnemonic(mnemonic)


def keypair_from_seed(seed: str) -> Any:
    create = cast(Callable[[str], Any], resolve_keypair_type().create_from_seed)
    return create(seed)


def reset_keypair_type_cache() -> None:
    """Test helper: clear the cached Keypair class."""
    global _KeypairType
    _KeypairType = None
