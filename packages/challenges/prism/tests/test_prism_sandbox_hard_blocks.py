from __future__ import annotations

import json

import anyio
import pytest
from conftest import signed_headers, two_script_bundle

from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code

# The static AST sandbox (architecture.md section 4.1) must hard-block every escape vector below
# on BOTH submission scripts, and every rejection must happen at the static phase (before any GPU
# work). These unit checks drive ``inspect_code`` exactly as the worker static gate does
# (``require_contract=False``); the pipeline check at the bottom proves no GPU lease/job is created.

IMPORT_TORCH = "import torch\n"


def _violation(code: str) -> SandboxViolation:
    with pytest.raises(SandboxViolation) as raised:
        inspect_code(code, require_contract=False)
    return raised.value


def _fn(body: str) -> str:
    return IMPORT_TORCH + "\n\ndef use(model, ctx, data):\n" + body + "\n"


# --- VAL-CONTRACT-010: process/system imports ---


@pytest.mark.parametrize("module", ["os", "sys", "subprocess"])
def test_sandbox_blocks_process_system_imports(module: str) -> None:
    violation = _violation(f"import {module}\n")
    assert violation.evidence[0].rule_id == "prism:no-process"


# --- VAL-CONTRACT-011: network imports ---


@pytest.mark.parametrize(
    "statement",
    [
        "import socket",
        "import requests",
        "import urllib",
        "import httpx",
        "import aiohttp",
        "from http import client",
    ],
)
def test_sandbox_blocks_network_imports(statement: str) -> None:
    violation = _violation(statement + "\n")
    assert violation.evidence[0].rule_id == "prism:no-network"


# --- VAL-CONTRACT-012: pickle / torch.load of an external untrusted path ---


def test_sandbox_blocks_pickle_import() -> None:
    violation = _violation("import pickle\n")
    assert violation.evidence[0].rule_id == "prism:no-deserialization"


def test_sandbox_blocks_pickle_loads_call() -> None:
    violation = _violation(_fn("    return pickle.loads(data)"))
    assert violation.evidence[0].rule_id == "prism:no-deserialization"


def test_sandbox_blocks_torch_load_external_path() -> None:
    violation = _violation(_fn("    return torch.load('/tmp/external_weights.pt')"))
    assert violation.evidence[0].rule_id == "prism:no-deserialization"


def test_sandbox_allows_torch_checkpoint_io_from_trusted_dir() -> None:
    code = (
        IMPORT_TORCH
        + "\n\ndef load_checkpoint(model, checkpoint_dir, ctx):\n"
        + "    payload = torch.load(checkpoint_dir / 'model.pt', weights_only=True)\n"
        + "    model.load_state_dict(payload['state_dict'])\n"
        + "    return None\n\n"
        + "def save_checkpoint(model, checkpoint_dir, ctx):\n"
        + "    torch.save({'state_dict': model.state_dict()}, checkpoint_dir / 'model.pt')\n"
        + "    return None\n"
    )
    report = inspect_code(code, require_contract=False)
    assert "function:load_checkpoint" in report.ast_fingerprint
    assert "function:save_checkpoint" in report.ast_fingerprint


# --- VAL-CONTRACT-013: ctypes / native FFI ---


@pytest.mark.parametrize("module", ["ctypes", "cffi"])
def test_sandbox_blocks_ctypes_and_ffi(module: str) -> None:
    violation = _violation(f"import {module}\n")
    assert violation.evidence[0].rule_id == "prism:no-ffi"


# --- VAL-CONTRACT-014: dynamic importlib / __import__ ---


def test_sandbox_blocks_importlib_import() -> None:
    violation = _violation("import importlib\n")
    assert violation.evidence[0].rule_id == "prism:no-dynamic-import"


def test_sandbox_blocks_importlib_import_module_call() -> None:
    violation = _violation(_fn("    return importlib.import_module('os')"))
    assert violation.evidence[0].rule_id == "prism:no-dynamic-import"


def test_sandbox_blocks_builtin_dunder_import_call() -> None:
    violation = _violation(_fn("    return __import__('os')"))
    assert violation.evidence[0].rule_id == "prism:no-dynamic-import"


# --- VAL-CONTRACT-015: builtins eval / exec / compile ---


@pytest.mark.parametrize(
    "body",
    [
        "    return eval('1 + 1')",
        "    exec('x = 1')\n    return None",
        "    return compile('1', '<s>', 'eval')",
    ],
)
def test_sandbox_blocks_eval_exec_compile(body: str) -> None:
    violation = _violation(_fn(body))
    assert violation.evidence[0].rule_id == "prism:no-dynamic-code"


# --- VAL-CONTRACT-016: attribute-escape access ---


@pytest.mark.parametrize(
    "body",
    [
        "    return model.__globals__",
        "    return model.__reduce__()",
        "    return model.__reduce_ex__(2)",
        "    return type(model).__subclasses__()",
        "    return (1).__class__",
        "    return type(model).__mro__",
        "    return model.__builtins__",
    ],
)
def test_sandbox_blocks_attribute_escapes(body: str) -> None:
    violation = _violation(_fn(body))
    assert "forbidden attribute" in str(violation)


# --- VAL-CONTRACT-017: filesystem writes outside artifacts_dir ---


def test_sandbox_blocks_open_for_write() -> None:
    violation = _violation(_fn("    return open('/etc/cron.d/x', 'w')"))
    assert violation.evidence[0].rule_id == "prism:no-filesystem"


@pytest.mark.parametrize("module", ["pathlib", "shutil", "tempfile", "glob"])
def test_sandbox_blocks_filesystem_imports(module: str) -> None:
    violation = _violation(f"import {module}\n")
    assert violation.evidence[0].rule_id == "prism:no-filesystem"


def test_sandbox_blocks_torch_save_to_external_path() -> None:
    body = "    torch.save(model.state_dict(), '/etc/evil.pt')\n    return None"
    violation = _violation(_fn(body))
    assert violation.evidence[0].rule_id == "prism:no-filesystem"


# --- VAL-CONTRACT-031: torch.hub / native-compile escapes inside the allowed torch namespace ---


@pytest.mark.parametrize(
    "body",
    [
        "    return torch.hub.load('pytorch/vision', 'resnet18')",
        "    return torch.hub.download_url_to_file('http://x/y', '/tmp/z')",
        "    return torch.utils.cpp_extension.load(name='x', sources=[])",
        "    return torch.utils.cpp_extension.load_inline(name='x', cpp_sources='')",
    ],
)
def test_sandbox_blocks_torch_hub_and_native_compile(body: str) -> None:
    violation = _violation(_fn(body))
    assert violation.evidence[0].rule_id == "prism:no-torch-escape"


def test_sandbox_allows_legitimate_torch_namespaces() -> None:
    code = (
        IMPORT_TORCH
        + "import torch.nn.functional as F\n"
        + "from torch import nn\n\n"
        + "def use(model, ctx, data):\n"
        + "    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)\n"
        + "    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
        + "    loader = torch.utils.data.DataLoader(data)\n"
        + "    return F.cross_entropy(model(ctx), ctx), loader, opt\n"
    )
    report = inspect_code(code, require_contract=False)
    assert "function:use" in report.ast_fingerprint


# --- VAL-CONTRACT-033: getattr / string-built indirection to a blocked symbol ---


@pytest.mark.parametrize(
    "body",
    [
        "    return getattr(torch, 'lo' + 'ad')(data)",
        "    return getattr(torch, 'load')(data)",
        "    return getattr(__builtins__, 'ev' + 'al')('1')",
        "    return vars()[data]",
        "    return getattr(torch, f'lo{data}ad')",
    ],
)
def test_sandbox_blocks_getattr_indirection(body: str) -> None:
    violation = _violation(_fn(body))
    assert violation.evidence[0].rule_id == "prism:no-dynamic-attr"


def test_sandbox_allows_constant_safe_getattr() -> None:
    report = inspect_code(_fn("    return getattr(model, 'forward')"), require_contract=False)
    assert "function:use" in report.ast_fingerprint


# --- Regression: benign arbitrary torch code still passes the sandbox ---


def test_sandbox_allows_benign_arbitrary_torch_model() -> None:
    code = (
        IMPORT_TORCH
        + "from torch import nn\n\n"
        + "class Net(nn.Module):\n"
        + "    def __init__(self, vocab):\n"
        + "        super().__init__()\n"
        + "        self.emb = nn.Embedding(vocab, 16)\n"
        + "        self.head = nn.Linear(16, vocab)\n\n"
        + "    def forward(self, tokens):\n"
        + "        return self.head(self.emb(tokens))\n\n"
        + "def build_model(ctx):\n"
        + "    return Net(ctx.vocab_size)\n"
    )
    report = inspect_code(code, require_contract=False)
    assert "function:build_model" in report.ast_fingerprint


# --- VAL-CONTRACT-018: hard-block rejection happens BEFORE any GPU work (pipeline) ---


def _submit(client, code: str, *, nonce: str) -> str:
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


def _process(client) -> None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text


def _submission_row(client, submission_id: str) -> dict:
    repository = client.app.state.repository

    async def fetch() -> dict:
        async with repository.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status, error FROM submissions WHERE id=?", (submission_id,)
            )
        return dict(rows[0])

    return anyio.run(fetch)


def _gpu_work_counts(client, submission_id: str) -> tuple[int, int]:
    """Count GPU work for a submission: active/queued gpu leases and GPU-level eval jobs.

    Every submission gets an ``l1`` static-tracking ``eval_jobs`` placeholder at creation time;
    GPU work is recorded as a separate eval job whose ``level`` is the container execution backend
    (never ``l1``). VAL-CONTRACT-018 forbids GPU leases and GPU eval jobs for hard-block rejections.
    """
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


def test_sandbox_hard_block_in_training_rejects_before_gpu(client) -> None:
    # torch.hub.load is a native/network escape inside the allowed torch namespace: it passes the
    # deterministic LLM safety prefilter (no os.system/open/eval token) so the AST sandbox is the
    # gate that must reject it, proving BOTH scripts are statically checked before any GPU work.
    malicious_train = (
        "import torch\n"
        "from architecture import build_model\n\n"
        "def train(ctx):\n"
        "    build_model(ctx)\n"
        "    return torch.hub.load('pytorch/vision', 'resnet18')\n"
    )
    code = two_script_bundle(train_code=malicious_train)
    submission_id = _submit(client, code, nonce="hardblock-train-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] == "rejected", row
    assert "torch.hub" in str(row["error"]) or "torch escape" in str(row["error"]), row
    leases, jobs = _gpu_work_counts(client, submission_id)
    assert leases == 0, f"expected no gpu lease, found {leases}"
    assert jobs == 0, f"expected no GPU eval job, found {jobs}"
