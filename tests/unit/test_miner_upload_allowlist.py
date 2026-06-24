from __future__ import annotations

import time
from typing import Any

import pytest

from base.security.miner_auth import (
    MinerAuthError,
    MinerUploadVerifier,
)

ALLOWLISTED_HOTKEY = "5F9owJrcZtgw9WsvD1K4QrRXmhSY9Fsr8kJytFHg4cGGnhCe"
OTHER_HOTKEY = "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy"


class FakeNonceStore:
    async def reserve(self, **kwargs: Any) -> None:
        pass


def _headers(hotkey: str) -> dict[str, str]:
    return {
        "x-hotkey": hotkey,
        "x-signature": "0x00",
        "x-nonce": "n1",
        "x-timestamp": str(int(time.time())),
    }


async def test_allowlisted_hotkey_bypasses_metagraph_without_chain() -> None:
    verifier = MinerUploadVerifier(
        netuid=100,
        nonce_store=FakeNonceStore(),
        metagraph_cache=None,
        require_registered_hotkey=True,
        extra_registered_hotkeys={ALLOWLISTED_HOTKEY},
        signature_verifier=lambda hotkey, message, signature: True,
    )

    identity = await verifier.verify(
        method="POST",
        path="/v1/challenges/prism/submissions",
        headers=_headers(ALLOWLISTED_HOTKEY),
        body=b"zip",
        challenge_slug="prism",
    )

    assert identity.hotkey == ALLOWLISTED_HOTKEY
    assert identity.uid is None


async def test_non_allowlisted_hotkey_raises_when_metagraph_unavailable() -> None:
    verifier = MinerUploadVerifier(
        netuid=100,
        nonce_store=FakeNonceStore(),
        metagraph_cache=None,
        require_registered_hotkey=True,
        extra_registered_hotkeys={ALLOWLISTED_HOTKEY},
        signature_verifier=lambda hotkey, message, signature: True,
    )

    with pytest.raises(MinerAuthError, match="metagraph unavailable"):
        await verifier.verify(
            method="POST",
            path="/v1/challenges/prism/submissions",
            headers=_headers(OTHER_HOTKEY),
            body=b"zip",
            challenge_slug="prism",
        )
