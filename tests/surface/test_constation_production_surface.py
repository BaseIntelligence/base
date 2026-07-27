"""T12 surface core: S1/S2/S4/S5/S7 + B1s/B2s (composition, no live Lium)."""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from base.attestation.payload import derive_attestation_key
from base.compute.attestation_nonce import (
    NonceBinding,
    NonceConsumeHit,
    NonceConsumeMiss,
    NonceConsumeReason,
)
from base.compute.constation_triangle import evaluate_digest_triangle
from base.compute.constation_types import ConstationFailCode
from base.compute.digest_allowlist import DigestRecord, ImageVariant
from base.config.settings import ConstationSettings, Settings
from base.master.challenge_work_source import HttpChallengeResultForwarder
from base.master.constation.attestation_keys import load_attestation_verify_key
from base.master.constation.bundle_seal import seal_constation_bundle
from base.master.constation.orchestrator import ConstationOrchestrationResult
from tests.surface.constation_surface_helpers import (
    BUILD_SECRET,
    COMMIT,
    DIGEST,
    DIGEST_BAD,
    HOTKEY,
    MANIFEST,
    POD,
    TREE,
    WIRE,
    WORK_UNIT,
    CaptureTransport,
    ChallengeRegistry,
    FakeRunner,
    fail_record,
    make_orchestrator,
    minimal_proof,
    ok_record,
    orchestration_request,
)


@pytest.mark.asyncio
async def test_s1_honest_path_seal_store_forward_nests_bundle() -> None:
    """S1: orchestrator seals+puts; forwarder nests store bundle under result."""
    orch, _nonces, store, runner = make_orchestrator()
    result = await orch.run(orchestration_request())

    assert isinstance(result, ConstationOrchestrationResult)
    assert result.ok is True
    assert result.reason is ConstationFailCode.OK
    assert runner.run_calls == 1
    stored = store.get(WORK_UNIT)
    assert stored is not None
    assert stored["work_unit_id"] == WORK_UNIT
    assert stored["digest"] == DIGEST
    assert stored["nonce"] == result.end_phase_nonce
    assert "signed_attestation" in stored
    assert stored["expected_sealed_manifest_hashes"] == dict(MANIFEST)

    allow = DigestRecord(
        commit_sha=COMMIT,
        tree_sha=TREE,
        variant=ImageVariant.CUDA,
        digest=DIGEST,
        sealed_manifest_hashes=dict(MANIFEST),
    )
    sealed = seal_constation_bundle(
        allowlist_record=allow,
        run_record=ok_record(),
        nonce=result.end_phase_nonce or "n",
        signed_attestation=dict(WIRE),
    )
    assert sealed["digest"] == DIGEST
    assert sealed["work_unit_id"] == WORK_UNIT

    transport = CaptureTransport()

    async def lookup(wu: str) -> dict[str, Any] | None:
        return store.get(wu)

    fwd = HttpChallengeResultForwarder(
        ChallengeRegistry(), transport=transport, retries=1, bundle_lookup=lookup
    )
    await fwd.forward_result(
        challenge_slug="prism",
        work_unit_id=WORK_UNIT,
        submission_ref=HOTKEY,
        result_payload={"execution_proof": minimal_proof(), "executed": 1},
    )
    assert transport.bodies, "expected POST body"
    assert transport.bodies[0]["result"]["constation_bundle"] == stored


@pytest.mark.asyncio
async def test_s2_adversarial_triangle_fail_no_ok_bundle_put() -> None:
    """S2: triangle mismatch + runner fail → store stays empty."""
    tri = evaluate_digest_triangle(
        required=DIGEST,
        lium_declared=DIGEST,
        sidecar=DIGEST_BAD,
    )
    assert tri.ok is False
    assert tri.fail_code is ConstationFailCode.REQUIRED_DIGEST_MISMATCH

    fake = FakeRunner(
        outcome=fail_record(reason=ConstationFailCode.REQUIRED_DIGEST_MISMATCH)
    )
    orch, _nonces, store, runner = make_orchestrator(fake=fake)
    result = await orch.run(orchestration_request())

    assert result.ok is False
    assert result.reason is ConstationFailCode.REQUIRED_DIGEST_MISMATCH
    assert store.get(WORK_UNIT) is None
    assert result.bundle is None
    assert runner.run_calls == 1


@pytest.mark.asyncio
async def test_s4_nonce_issue_seal_consume_once_ok() -> None:
    """S4/B2: after seal, end-phase nonce is first-consumable exactly once."""
    orch, nonces, store, _runner = make_orchestrator()
    result = await orch.run(orchestration_request())

    assert result.ok is True
    assert result.end_phase_nonce is not None
    stored = store.get(WORK_UNIT)
    assert stored is not None
    assert stored["nonce"] == result.end_phase_nonce

    binding = NonceBinding(work_unit_id=WORK_UNIT, miner_hotkey=HOTKEY, pod_id=POD)
    first = nonces.consume(result.end_phase_nonce, binding)
    assert isinstance(first, NonceConsumeHit), (
        f"end-phase nonce not first-consumable after seal (got {first!r})"
    )
    second = nonces.consume(result.end_phase_nonce, binding)
    assert isinstance(second, NonceConsumeMiss)
    assert second.reason is NonceConsumeReason.ALREADY_CONSUMED


def test_s5_allowlist_sealed_hashes_required() -> None:
    """S5: DigestRecord rejects empty sealed hashes; seal carries them."""
    with pytest.raises(ValueError, match="sealed_manifest_hashes must be non-empty"):
        DigestRecord(
            commit_sha=COMMIT,
            tree_sha=TREE,
            variant=ImageVariant.CUDA,
            digest=DIGEST,
            sealed_manifest_hashes={},
        )

    allow = DigestRecord(
        commit_sha=COMMIT,
        tree_sha=TREE,
        variant=ImageVariant.CUDA,
        digest=DIGEST,
        sealed_manifest_hashes=dict(MANIFEST),
    )
    out = seal_constation_bundle(
        allowlist_record=allow,
        run_record=ok_record(),
        nonce="surface-nonce",
        signed_attestation=dict(WIRE),
    )
    assert out["expected_sealed_manifest_hashes"] == dict(MANIFEST)
    assert out["reported_sealed_manifest_hashes"] == dict(MANIFEST)


@pytest.mark.asyncio
async def test_s7_forwarder_embed_from_lookup() -> None:
    """S7: forwarder embeds result.constation_bundle from lookup."""
    transport = CaptureTransport()
    bundle = {
        "digest": DIGEST,
        "nonce": "n-surface",
        "work_unit_id": WORK_UNIT,
    }

    async def lookup(wu: str) -> dict[str, Any] | None:
        return bundle if wu == WORK_UNIT else None

    fwd = HttpChallengeResultForwarder(
        ChallengeRegistry(), transport=transport, retries=1, bundle_lookup=lookup
    )
    await fwd.forward_result(
        challenge_slug="prism",
        work_unit_id=WORK_UNIT,
        submission_ref=HOTKEY,
        result_payload={"execution_proof": minimal_proof(), "executed": 1},
    )
    assert transport.bodies
    assert transport.bodies[0]["result"]["constation_bundle"] == bundle


def test_b1s_verify_key_wiring_smoke() -> None:
    """B1s: load_attestation_verify_key + main AST wires key kwarg."""
    key = derive_attestation_key(BUILD_SECRET)
    settings = Settings(
        constation=ConstationSettings(attestation_verify_key_hex=key.hex())
    )
    assert load_attestation_verify_key(settings) == key
    assert load_attestation_verify_key(Settings()) is None

    import base.cli_app.main as main_mod

    source = inspect.getsource(main_mod)
    tree = ast.parse(source)
    hits: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name != "build_constation_router":
            continue
        hits.append(node)
    assert hits, "build_constation_router call missing from main"
    for call in hits:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "attestation_verify_key" in kw_names, (
            "build_constation_router must receive attestation_verify_key= "
            f"(got keywords {sorted(kw_names)})"
        )


def test_b2s_orchestrator_source_never_calls_consume() -> None:
    """B2s: orchestrator module must not call nonce.consume."""
    from base.master.constation import orchestrator as orch_mod

    src = inspect.getsource(orch_mod)
    assert ".consume(" not in src
    assert "nonce_service.consume" not in src
