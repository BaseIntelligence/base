"""Lean review image must load quote helpers without the eval client package.

The review Dockerfile copies only ``keyrelease/__init__.py`` and
``keyrelease/quote.py``. Eager package exports that import
:mod:`agent_challenge.keyrelease.client` would crash measurement extraction
and empty-GetQuote RTMR3 recompute in-CVM (live residual mapped as
``quote_unavailable`` / ``stage measure failed``).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def test_quote_submodule_imports_without_client_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review image ships no client.py; package init must stay quote-safe."""

    client_mod = "agent_challenge.keyrelease.client"
    # Drop any cached package/client modules so the lean path is exercised.
    for name in list(sys.modules):
        if name == "agent_challenge.keyrelease" or name.startswith("agent_challenge.keyrelease."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    client_path = (
        Path(__file__).resolve().parents[1] / "src" / "agent_challenge" / "keyrelease" / "client.py"
    )
    assert client_path.is_file()

    real_import_module = importlib.import_module

    def block_client(name: str, package: str | None = None):  # type: ignore[no-untyped-def]
        if name == client_mod or name.endswith(".client") and "keyrelease" in name:
            raise ImportError("simulated lean review image: client module absent")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", block_client)

    # Package import must succeed even when client is unimportable.
    pkg = importlib.import_module("agent_challenge.keyrelease")
    assert pkg is not None

    # Direct quote import used by the measured runtime must work.
    quote = importlib.import_module("agent_challenge.keyrelease.quote")
    assert callable(quote.runtime_event_digest)
    assert callable(quote.validate_rtmr3_event_log)
    assert callable(quote.parse_tdx_quote_v4)

    # Package-root client symbols remain available only when client loads.
    with pytest.raises(AttributeError):
        _ = pkg.GoldenKeyReleaseClient


def test_package_root_still_exports_client_symbols_when_present() -> None:
    """Host/eval environments with client.py keep the package re-exports."""

    from agent_challenge.keyrelease import (
        DEFAULT_KEY_RELEASE_TIMEOUT,
        KEY_RELEASE_TAG,
        GoldenKeyReleaseClient,
        KeyReleaseDenied,
        KeyReleaseError,
        key_release_report_data,
    )

    assert isinstance(KEY_RELEASE_TAG, (bytes, str)) and KEY_RELEASE_TAG
    assert DEFAULT_KEY_RELEASE_TIMEOUT > 0
    assert callable(GoldenKeyReleaseClient)
    assert issubclass(KeyReleaseDenied, KeyReleaseError)
    assert callable(key_release_report_data)
