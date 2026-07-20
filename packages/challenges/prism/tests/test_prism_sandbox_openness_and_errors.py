from __future__ import annotations

import json
import time
from pathlib import Path

import anyio
import pytest
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code
from prism_challenge.evaluator.static_instantiation import check_build_model_static

# This suite covers the sandbox OPENNESS items (legitimate arbitrary PyTorch passes) and the static
# CLEAN-ERROR handling that must reject bad/hostile submissions before any GPU work
# (architecture.md section 4.1; VAL-CONTRACT-023..028, 032).

CTX = PrismContext(vocab_size=256, sequence_length=16)


# --- VAL-CONTRACT-024: `from __future__ import annotations` allowed ---


@pytest.mark.parametrize(
    "statement",
    ["from __future__ import annotations", "from __future__ import annotations, division"],
)
def test_sandbox_openness_allows_future_imports(statement: str) -> None:
    code = statement + "\nimport torch\n\ndef build_model(ctx):\n    return torch.nn.Linear(4, 4)\n"
    report = inspect_code(code, require_contract=False)
    assert "function:build_model" in report.ast_fingerprint


# --- VAL-CONTRACT-023: module/function docstrings allowed ---


def test_sandbox_openness_allows_module_and_function_docstrings() -> None:
    code = (
        '"""Module docstring describing the architecture."""\n'
        "import torch\n\n"
        "def build_model(ctx):\n"
        '    """Build the model."""\n'
        "    return torch.nn.Linear(4, 4)\n\n"
        "def get_recipe(ctx):\n"
        '    """Return the recipe."""\n'
        "    return {}\n"
    )
    # require_contract=True exercises validate_miner_contract (the strictest path).
    report = inspect_code(code)
    assert "function:build_model" in report.ast_fingerprint


# --- VAL-CONTRACT-025: normal top-level defs / constants / dataclasses / helpers allowed ---


def test_sandbox_openness_allows_top_level_defs_constants_dataclasses() -> None:
    code = (
        '"""Top-level structure."""\n'
        "from __future__ import annotations\n\n"
        "import torch\n"
        "from dataclasses import dataclass\n\n"
        "HIDDEN_DIM = 16\n"
        "LABELS = (1, 2, 3)\n\n"
        "@dataclass\n"
        "class ModelConfig:\n"
        "    vocab: int\n"
        "    dim: int = HIDDEN_DIM\n\n"
        "def _make_head(cfg: ModelConfig) -> torch.nn.Module:\n"
        "    return torch.nn.Linear(cfg.dim, cfg.vocab)\n\n"
        "def build_model(ctx):\n"
        "    return _make_head(ModelConfig(vocab=ctx.vocab_size))\n\n"
        "def get_recipe(ctx):\n"
        "    return {}\n"
    )
    report = inspect_code(code)
    assert "function:build_model" in report.ast_fingerprint
    assert "function:_make_head" in report.ast_fingerprint


# --- VAL-CONTRACT-027: build_model returning a non-nn.Module (or None) rejected ---


@pytest.mark.parametrize(
    "body",
    [
        "    return None",
        "    return torch.zeros(3)",
        "    return (torch.nn.Linear(2, 2),)",
        "    return {'model': 1}",
        "    return 42",
    ],
)
def test_static_build_model_non_module_rejected(body: str) -> None:
    arch = "import torch\n\ndef build_model(ctx):\n" + body + "\n"
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    assert raised.value.evidence[0].rule_id == "prism:build-model-return-type"
    assert "torch.nn.Module" in str(raised.value)


# --- VAL-CONTRACT-028: build_model raising during forced-seed instantiation handled cleanly ---


def test_static_build_model_raising_rejected_cleanly() -> None:
    arch = "import torch\n\ndef build_model(ctx):\n    raise ValueError('bad config')\n"
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    assert raised.value.evidence[0].rule_id == "prism:build-model-instantiation"
    assert "ValueError" in str(raised.value)
    assert "bad config" in str(raised.value)


def test_static_build_model_shape_error_rejected_cleanly() -> None:
    arch = (
        "import torch\n\n"
        "def build_model(ctx):\n"
        "    a = torch.zeros(2, 3)\n"
        "    b = torch.zeros(4, 5)\n"
        "    return torch.nn.Linear(2, 2) if (a @ b).sum() else None\n"
    )
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    assert raised.value.evidence[0].rule_id == "prism:build-model-instantiation"


# --- VAL-CONTRACT-032: hostile build_model construction is time/resource-bounded ---


def test_static_build_model_infinite_loop_is_time_bounded() -> None:
    arch = "import torch\n\ndef build_model(ctx):\n    while True:\n        pass\n"
    start = time.monotonic()
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static(
            {"architecture.py": arch}, "architecture.py", ctx=CTX, timeout_seconds=2.0
        )
    elapsed = time.monotonic() - start
    assert raised.value.evidence[0].rule_id == "prism:build-model-resource"
    assert elapsed < 15.0, f"construction was not bounded promptly ({elapsed:.1f}s)"


def test_static_build_model_memory_balloon_is_resource_bounded() -> None:
    arch = "import torch\n\ndef build_model(ctx):\n    big = [0] * (10 ** 13)\n    return big\n"
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static(
            {"architecture.py": arch}, "architecture.py", ctx=CTX, timeout_seconds=20.0
        )
    assert raised.value.evidence[0].rule_id == "prism:build-model-resource"


# --- Regression: a legitimate model instantiates and reports its real parameter count ---


def test_static_build_model_valid_returns_param_count() -> None:
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self, vocab):\n"
        "        super().__init__()\n"
        "        self.emb = nn.Embedding(vocab, 16)\n"
        "        self.head = nn.Linear(16, vocab)\n\n"
        "    def forward(self, tokens):\n"
        "        return self.head(self.emb(tokens))\n\n"
        "def build_model(ctx):\n"
        "    return Net(ctx.vocab_size)\n"
    )
    count = check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    assert count == 256 * 16 + (16 * 256 + 256)


def test_static_build_model_large_model_not_false_rejected_by_memory_cap() -> None:
    """~138M params fits promote (350M) but not default explore (124M)."""
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "class Big(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.layers = nn.ModuleList([nn.Linear(2048, 2048) for _ in range(33)])\n\n"
        "    def forward(self, x):\n"
        "        for layer in self.layers:\n"
        "            x = layer(x)\n"
        "        return x\n\n"
        "def build_model(ctx):\n"
        "    return Big()\n"
    )
    promote_ctx = PrismContext(
        vocab_size=256,
        sequence_length=16,
        max_parameters=350_000_000,
        param_ladder_stage="promote",
    )
    count = check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=promote_ctx)
    assert count == 33 * (2048 * 2048 + 2048)


def test_static_build_model_resolves_sibling_imports() -> None:
    arch = (
        "import torch\n"
        "from layers import make_head\n\n"
        "def build_model(ctx):\n"
        "    return make_head(ctx.vocab_size)\n"
    )
    layers = "import torch\n\ndef make_head(vocab):\n    return torch.nn.Linear(8, vocab)\n"
    count = check_build_model_static(
        {"architecture.py": arch, "layers.py": layers}, "architecture.py", ctx=CTX
    )
    assert count == 8 * 256 + 256


# --- Pipeline (black-box) behaviour through process-next, like the validator ---


def _make_client(tmp_path: Path, **overrides: object) -> TestClient:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        # Single-process training doubles; the multi-GPU static contract (default reject) is
        # exercised explicitly in test_prism_distributed_contract.py.
        distributed_contract_policy="off",
        **overrides,
    )
    return TestClient(create_app(settings))


def _submit(client: TestClient, code: str, *, nonce: str) -> str:
    payload = {"code": code, "filename": "bundle.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, hotkey="hk", nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text


def _submission_row(client: TestClient, submission_id: str) -> dict:
    repository = client.app.state.repository

    async def fetch() -> dict:
        async with repository.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status, error FROM submissions WHERE id=?", (submission_id,)
            )
        return dict(rows[0])

    return anyio.run(fetch)


def _gpu_work_counts(client: TestClient, submission_id: str) -> tuple[int, int]:
    repository = client.app.state.repository

    async def fetch() -> tuple[int, int]:
        async with repository.database.connect() as conn:
            leases = await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM gpu_leases WHERE submission_id=?", (submission_id,)
            )
            jobs = await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM eval_jobs WHERE submission_id=? AND level != 'l1'",
                (submission_id,),
            )
        return int(leases[0]["n"]), int(jobs[0]["n"])

    return anyio.run(fetch)


# --- VAL-CONTRACT-023/024/025: openness bundle is accepted (advances, not rejected) ---

OPEN_ARCH = (
    '"""Architecture module docstring."""\n'
    "from __future__ import annotations\n\n"
    "import torch\n"
    "from dataclasses import dataclass\n\n"
    "HIDDEN_DIM = 16\n\n"
    "@dataclass\n"
    "class ModelConfig:\n"
    "    vocab: int\n"
    "    dim: int = HIDDEN_DIM\n\n"
    "def _make_head(cfg: ModelConfig) -> torch.nn.Module:\n"
    '    """Helper that builds the head."""\n'
    "    return torch.nn.Linear(cfg.dim, cfg.vocab)\n\n"
    "class TinyNet(torch.nn.Module):\n"
    '    """A tiny net."""\n\n'
    "    def __init__(self, cfg: ModelConfig) -> None:\n"
    "        super().__init__()\n"
    "        self.emb = torch.nn.Embedding(cfg.vocab, cfg.dim)\n"
    "        self.head = _make_head(cfg)\n\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.emb(tokens))\n\n"
    "def build_model(ctx):\n"
    '    """Build the model."""\n'
    "    return TinyNet(ModelConfig(vocab=ctx.vocab_size))\n"
)

OPEN_TRAIN = (
    '"""Training module docstring."""\n'
    "from __future__ import annotations\n\n"
    "from architecture import build_model\n\n"
    "def train(ctx) -> None:\n"
    '    """Train loop."""\n'
    "    build_model(ctx)\n"
    "    return None\n"
)


def test_pipeline_openness_bundle_not_rejected(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        code = two_script_bundle(arch_code=OPEN_ARCH, train_code=OPEN_TRAIN)
        submission_id = _submit(client, code, nonce="openness-1")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] != "rejected", row
        assert row["status"] in {"pending", "running"}, row


# --- VAL-CONTRACT-026: unparseable (SyntaxError) script rejected cleanly before GPU ---


def test_pipeline_unparseable_training_rejected_before_gpu(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        bad_train = "def train(ctx):\n    return (((\n"
        code = two_script_bundle(train_code=bad_train)
        submission_id = _submit(client, code, nonce="syntax-1")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] == "rejected", row
        assert "cannot parse" in str(row["error"]), row
        assert "training.py" in str(row["error"]), row
        leases, jobs = _gpu_work_counts(client, submission_id)
        assert (leases, jobs) == (0, 0)


# --- VAL-CONTRACT-027: build_model non-nn.Module rejected before GPU ---


def test_pipeline_build_model_non_module_rejected_before_gpu(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        arch = "import torch\n\ndef build_model(ctx):\n    return None\n"
        code = two_script_bundle(arch_code=arch)
        submission_id = _submit(client, code, nonce="nonmodule-1")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] == "rejected", row
        assert "torch.nn.Module" in str(row["error"]), row
        leases, jobs = _gpu_work_counts(client, submission_id)
        assert (leases, jobs) == (0, 0)


# --- VAL-CONTRACT-028: build_model raising during instantiation handled cleanly ---


def test_pipeline_build_model_raises_rejected_before_gpu(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        arch = "import torch\n\ndef build_model(ctx):\n    raise RuntimeError('boom in init')\n"
        code = two_script_bundle(arch_code=arch)
        submission_id = _submit(client, code, nonce="raise-1")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] == "rejected", row
        assert "instantiation" in str(row["error"]), row
        assert "boom in init" in str(row["error"]), row
        leases, jobs = _gpu_work_counts(client, submission_id)
        assert (leases, jobs) == (0, 0)
        assert client.get("/health").status_code == 200


# --- VAL-CONTRACT-032: hostile build_model is bounded at the static phase ---


def test_pipeline_hostile_build_model_bounded_before_gpu(tmp_path: Path) -> None:
    with _make_client(tmp_path, static_instantiation_timeout_seconds=2.0) as client:
        arch = "import torch\n\ndef build_model(ctx):\n    while True:\n        pass\n"
        code = two_script_bundle(arch_code=arch)
        submission_id = _submit(client, code, nonce="hostile-1")
        start = time.monotonic()
        _process(client)
        elapsed = time.monotonic() - start
        row = _submission_row(client, submission_id)
        assert row["status"] == "rejected", row
        assert "budget" in str(row["error"]) or "resource" in str(row["error"]), row
        leases, jobs = _gpu_work_counts(client, submission_id)
        assert (leases, jobs) == (0, 0)
        assert elapsed < 25.0, f"static phase was not bounded promptly ({elapsed:.1f}s)"
        assert client.get("/health").status_code == 200
