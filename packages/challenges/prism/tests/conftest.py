from __future__ import annotations

import base64
import hmac
import io
import time
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.auth import canonical_submission_message
from prism_challenge.config import PrismSettings

VALID_CODE = """
import torch
from prism_challenge.evaluator.interface import TrainingRecipe

class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.linear = torch.nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.linear(self.embedding(tokens))

def build_model(ctx):
    return TinyModel(ctx.vocab_size)

def get_recipe(ctx):
    return TrainingRecipe(learning_rate=0.0003, batch_size=4)
"""

VALID_ARCH_CODE = VALID_CODE

VALID_TRAIN_CODE = """
from architecture import build_model

def train(ctx):
    build_model(ctx)
    return None
"""


def two_script_bundle(
    *,
    arch_code: str = VALID_ARCH_CODE,
    train_code: str = VALID_TRAIN_CODE,
    prism_yaml: str | None = None,
    arch_name: str = "architecture.py",
    train_name: str = "training.py",
    extra_files: dict[str, str] | None = None,
) -> str:
    """Build a base64-encoded two-script submission bundle (architecture + training)."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        if prism_yaml is not None:
            archive.writestr("prism.yaml", prism_yaml)
        archive.writestr(arch_name, arch_code)
        archive.writestr(train_name, train_code)
        for name, content in (extra_files or {}).items():
            archive.writestr(name, content)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def signed_headers(
    secret: str, body: bytes, hotkey: str = "hk", nonce: str = "n1"
) -> dict[str, str]:
    timestamp = str(int(time.time()))
    message = canonical_submission_message(
        hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
    )
    signature = hmac.new(secret.encode(), message, sha256).hexdigest()
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
    }


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        # The OpenRouter LLM gate is enabled by default in production, but the unit env has no
        # API key; disable it here so pipeline tests exercise their own subject. The gate itself
        # is covered in the test_*llm* suites. llm_review_required is pinned False so the disabled
        # gate keeps the allow path (the production default is now fail-closed True).
        # Most pipeline tests use minimal single-process training doubles; the multi-GPU static
        # contract (production default: reject) is an isolation knob here and is exercised
        # explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


# --- Wave 4 constation auto-inject for legacy tests (todo 22) ------------------------------------
# Production P1: no bundle ⇒ no score. Legacy tests that never heard of constation omit the
# kwargs entirely; we inject a valid bundle so they keep exercising finalization/audit.
# Tests that pass ``constation_bundle=None`` (or ``constation_infra_fault=...``) explicitly
# exercise the fail-closed / break-glass paths and are left alone.


@pytest.fixture(autouse=True)
def _auto_constation_for_legacy_ingestion(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    # Production-path tests must exercise real fail-closed without the legacy seam.
    modname = getattr(getattr(request, "module", None), "__name__", "") or ""
    if "prod_constation" in modname or request.node.get_closest_marker("no_auto_constation"):
        return

    from prism_challenge.constation import CheckOutcome, ConstationBundle
    import prism_challenge.ingestion as ingestion_mod

    original = ingestion_mod.ingest_work_unit_result

    def _ok(**_k: object) -> CheckOutcome:
        return CheckOutcome(ok=True, reason="ok")

    def _bundle() -> ConstationBundle:
        man = {"legacy-test-harness.py": "a" * 64}
        digest = "sha256:" + ("11" * 32)
        return ConstationBundle(
            commit_sha="a" * 40,
            tree_sha="b" * 40,
            variant="cuda",
            digest=digest,
            work_unit_id="legacy-wu",
            miner_hotkey="legacy-hk",
            pod_id="legacy-pod",
            nonce="legacy-nonce",
            signed_attestation={"legacy": True},
            expected_sealed_manifest_hashes=dict(man),
            reported_sealed_manifest_hashes=dict(man),
            lium_declared_digest=digest,
            constation_gap_budget_seconds=30.0,
            constation_observed_max_gap_seconds=1.0,
        )

    async def _wrapped(**kwargs):  # type: ignore[no-untyped-def]
        if (
            "constation_bundle" not in kwargs
            and "constation_infra_fault" not in kwargs
            and kwargs.get("check_allowlist") is None
        ):
            kwargs = dict(kwargs)
            kwargs["constation_bundle"] = _bundle()
            kwargs["check_allowlist"] = _ok
            kwargs["check_nonce"] = _ok
            kwargs["verify_constation_signature"] = lambda _s: _ok()
        return await original(**kwargs)

    monkeypatch.setattr(ingestion_mod, "ingest_work_unit_result", _wrapped)
    # Patch the bound name on the requesting test module (from-import sites).
    mod = getattr(request, "module", None)
    if mod is not None and getattr(mod, "ingest_work_unit_result", None) is not None:
        monkeypatch.setattr(mod, "ingest_work_unit_result", _wrapped, raising=False)
