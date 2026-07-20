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
