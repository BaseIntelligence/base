"""S7: HttpChallengeResultForwarder embeds result.constation_bundle from store."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from base.master.challenge_work_source import HttpChallengeResultForwarder
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.proof import build_execution_proof

MANIFEST = "a" * 64


class _Reg:
    async def get(self, slug: str) -> Any:
        class R:
            internal_base_url = "http://prism.test"

        return R()

    async def get_token(self, slug: str) -> str:
        return "tok"


class _CaptureTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import json

        self.bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json={"status": "accepted"})


def _minimal_proof() -> dict[str, Any]:
    import bittensor as bt

    signer = KeypairRequestSigner(bt.Keypair.create_from_uri("//WorkerAlice"))
    proof = build_execution_proof(
        signer=signer, manifest_sha256=MANIFEST, unit_id="wu-1"
    )
    return proof.model_dump(mode="json")


@pytest.mark.asyncio
async def test_forwarder_embeds_bundle_from_lookup() -> None:
    transport = _CaptureTransport()
    bundle = {"digest": "sha256:" + ("1" * 64), "nonce": "n1", "work_unit_id": "wu-1"}

    async def lookup(wu: str) -> dict[str, Any] | None:
        return bundle if wu == "wu-1" else None

    fwd = HttpChallengeResultForwarder(
        _Reg(), transport=transport, retries=1, bundle_lookup=lookup
    )
    await fwd.forward_result(
        challenge_slug="prism",
        work_unit_id="wu-1",
        submission_ref="hk",
        result_payload={"execution_proof": _minimal_proof(), "executed": 1},
    )
    assert transport.bodies, "expected POST body"
    body = transport.bodies[0]
    assert body["result"]["constation_bundle"] == bundle


@pytest.mark.asyncio
async def test_forwarder_does_not_overwrite_existing_bundle() -> None:
    transport = _CaptureTransport()
    existing = {"digest": "existing"}

    async def lookup(_wu: str) -> dict[str, Any]:
        return {"digest": "from-store"}

    fwd = HttpChallengeResultForwarder(
        _Reg(), transport=transport, retries=1, bundle_lookup=lookup
    )
    await fwd.forward_result(
        challenge_slug="prism",
        work_unit_id="wu-1",
        submission_ref="hk",
        result_payload={
            "execution_proof": _minimal_proof(),
            "constation_bundle": existing,
        },
    )
    assert transport.bodies[0]["result"]["constation_bundle"] == existing
