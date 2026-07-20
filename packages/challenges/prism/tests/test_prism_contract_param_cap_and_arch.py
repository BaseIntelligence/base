from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code
from prism_challenge.evaluator.static_instantiation import (
    PARAM_CAP_RULE,
    check_build_model_static,
)

# Param-cap enforcement (forced-seed instantiation) + architecture-agnostic acceptance +
# tokenizer acceptance (architecture.md sections 2, 4.1, 5; VAL-CONTRACT-009, 019, 020, 021, 022).

CTX = PrismContext(vocab_size=256, sequence_length=16)


# --- Representative architectures: only `torch` (+ allowlisted stdlib), sub-cap, two-script ---

TRANSFORMER_ARCH = (
    "import torch\n"
    "from torch import nn\n\n"
    "class TransformerLM(nn.Module):\n"
    "    def __init__(self, vocab, dim=32):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True)\n"
    "        self.encoder = nn.TransformerEncoder(layer, num_layers=2)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.encoder(self.emb(tokens)))\n\n"
    "def build_model(ctx):\n"
    "    return TransformerLM(ctx.vocab_size)\n"
)

RNN_ARCH = (
    "import torch\n"
    "from torch import nn\n\n"
    "class LstmLM(nn.Module):\n"
    "    def __init__(self, vocab, dim=32):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.rnn = nn.LSTM(dim, dim, num_layers=2, batch_first=True)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        out, _ = self.rnn(self.emb(tokens))\n"
    "        return self.head(out)\n\n"
    "def build_model(ctx):\n"
    "    return LstmLM(ctx.vocab_size)\n"
)

SSM_ARCH = (
    "import torch\n"
    "from torch import nn\n\n"
    "class SsmBlock(nn.Module):\n"
    "    def __init__(self, dim):\n"
    "        super().__init__()\n"
    "        self.in_proj = nn.Linear(dim, dim)\n"
    "        self.out_proj = nn.Linear(dim, dim)\n"
    "        self.log_decay = nn.Parameter(torch.zeros(dim))\n\n"
    "    def forward(self, x):\n"
    "        u = self.in_proj(x)\n"
    "        decay = torch.sigmoid(self.log_decay)\n"
    "        state = torch.zeros(x.shape[0], x.shape[2])\n"
    "        outputs = []\n"
    "        for t in range(x.shape[1]):\n"
    "            state = state * decay + u[:, t]\n"
    "            outputs.append(state)\n"
    "        return self.out_proj(torch.stack(outputs, dim=1))\n\n"
    "class SsmLM(nn.Module):\n"
    "    def __init__(self, vocab, dim=32):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.block = SsmBlock(dim)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.block(self.emb(tokens)))\n\n"
    "def build_model(ctx):\n"
    "    return SsmLM(ctx.vocab_size)\n"
)

CNN_ARCH = (
    "import torch\n"
    "from torch import nn\n\n"
    "class ConvMixerLM(nn.Module):\n"
    "    def __init__(self, vocab, dim=32):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)\n"
    "        self.mix = nn.Linear(dim, dim)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        x = self.emb(tokens).transpose(1, 2)\n"
    "        x = self.conv(x).transpose(1, 2)\n"
    "        return self.head(self.mix(x))\n\n"
    "def build_model(ctx):\n"
    "    return ConvMixerLM(ctx.vocab_size)\n"
)

ARCHITECTURES = {
    "transformer": TRANSFORMER_ARCH,
    "rnn_lstm": RNN_ARCH,
    "ssm": SSM_ARCH,
    "cnn_mlp_mixer": CNN_ARCH,
}

PLAIN_TRAIN = (
    "from architecture import build_model\n\n"
    "def train(ctx):\n"
    "    build_model(ctx)\n"
    "    return None\n"
)


# --- VAL-CONTRACT-019 / VAL-RESLAB-003: model > explore 124M rejected at static gate ---


def test_contract_param_cap_rejects_oversize_model() -> None:
    # 50304 * 2500 = 125,760,000 > 124,000,000 explore ladder default
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "def build_model(ctx):\n"
        "    return nn.Embedding(50304, 2500)\n"
    )
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    assert raised.value.evidence[0].rule_id == PARAM_CAP_RULE
    assert "125,760,000" in str(raised.value) or "125760000" in str(raised.value)


# --- VAL-CONTRACT-020 / VAL-RESLAB-003: model just under explore 124M passes ---


def test_contract_param_cap_accepts_just_under_model() -> None:
    # 50304 * 2460 = 123,747,840 < 124,000,000 explore ladder default
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "def build_model(ctx):\n"
        "    return nn.Embedding(50304, 2460)\n"
    )
    count = check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    assert count == 50304 * 2460
    assert count <= CTX.max_params


def test_contract_param_cap_logic_uses_override_cap() -> None:
    arch = "import torch\n\ndef build_model(ctx):\n    return torch.nn.Linear(64, 64)\n"
    # 64 * 64 + 64 = 4160 params
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static(
            {"architecture.py": arch}, "architecture.py", ctx=CTX, max_parameters=1000
        )
    assert raised.value.evidence[0].rule_id == PARAM_CAP_RULE
    # Same model with a generous cap passes and returns the real count.
    count = check_build_model_static(
        {"architecture.py": arch}, "architecture.py", ctx=CTX, max_parameters=10_000
    )
    assert count == 64 * 64 + 64


# --- VAL-CONTRACT-009: every architecture family passes the contract + sandbox + param gate ---


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_contract_arch_agnostic_static_instantiation(name: str) -> None:
    arch = ARCHITECTURES[name]
    report = inspect_code(arch, require_contract=False)
    assert "function:build_model" in report.ast_fingerprint
    inspect_code(PLAIN_TRAIN, require_contract=False, allowed_import_roots={"architecture"})
    count = check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    assert 0 < count <= CTX.max_params


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


# --- VAL-CONTRACT-019: oversize model rejected before any GPU work (pipeline) ---


def test_pipeline_param_cap_oversize_rejected_before_gpu(tmp_path: Path) -> None:
    with _make_client(tmp_path, max_parameters=1_000_000) as client:
        arch = (
            "import torch\n"
            "from torch import nn\n\n"
            "def build_model(ctx):\n"
            "    return nn.Embedding(50304, 64)\n"  # 3,219,456 params > 1M cap
        )
        code = two_script_bundle(arch_code=arch, train_code=PLAIN_TRAIN)
        submission_id = _submit(client, code, nonce="paramcap-over-1")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] == "rejected", row
        assert "parameter cap" in str(row["error"]).lower(), row
        assert _gpu_work_counts(client, submission_id) == (0, 0)
        assert client.get("/health").status_code == 200


# --- VAL-CONTRACT-020: sub-cap model passes the param gate (pipeline) ---


def test_pipeline_param_cap_just_under_not_rejected(tmp_path: Path) -> None:
    with _make_client(tmp_path, max_parameters=5_000_000) as client:
        arch = (
            "import torch\n"
            "from torch import nn\n\n"
            "def build_model(ctx):\n"
            "    return nn.Embedding(50304, 64)\n"  # 3,219,456 params < 5M cap
        )
        code = two_script_bundle(arch_code=arch, train_code=PLAIN_TRAIN)
        submission_id = _submit(client, code, nonce="paramcap-under-1")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] != "rejected", row
        assert row["status"] in {"pending", "running"}, row


# --- VAL-CONTRACT-009: every architecture family clears the pipeline static gate ---


@pytest.mark.parametrize("name", sorted(ARCHITECTURES))
def test_pipeline_arch_agnostic_not_rejected(name: str, tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        code = two_script_bundle(arch_code=ARCHITECTURES[name], train_code=PLAIN_TRAIN)
        submission_id = _submit(client, code, nonce=f"arch-{name}")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] != "rejected", row
        assert row["status"] in {"pending", "running"}, row


# --- VAL-CONTRACT-021: a submission with its own in-code tokenizer is accepted ---

IN_CODE_TOKENIZER_TRAIN = (
    "from architecture import build_model\n\n"
    "class ByteTokenizer:\n"
    "    vocab_size = 256\n\n"
    "    def encode(self, text):\n"
    "        return list(text.encode('utf-8'))\n\n"
    "    def decode(self, ids):\n"
    "        return bytes(ids).decode('utf-8', errors='ignore')\n\n"
    "def train(ctx):\n"
    "    tokenizer = ByteTokenizer()\n"
    "    tokenizer.encode('hello')\n"
    "    build_model(ctx)\n"
    "    return None\n"
)


def test_pipeline_in_code_tokenizer_accepted(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        code = two_script_bundle(train_code=IN_CODE_TOKENIZER_TRAIN)
        submission_id = _submit(client, code, nonce="tok-incode-1")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] != "rejected", row
        assert row["status"] in {"pending", "running"}, row


# --- VAL-CONTRACT-022: pre-staged reference tokenizers (gpt2 / llama) are accepted ---


@pytest.mark.parametrize("tokenizer", ["gpt2", "llama"])
def test_pipeline_reference_tokenizer_accepted(tokenizer: str, tmp_path: Path) -> None:
    train = (
        "from architecture import build_model\n\n"
        "def train(ctx):\n"
        f"    tok = ctx.reference_tokenizer({tokenizer!r})\n"
        "    build_model(ctx)\n"
        "    return None\n"
    )
    with _make_client(tmp_path) as client:
        code = two_script_bundle(train_code=train)
        submission_id = _submit(client, code, nonce=f"tok-ref-{tokenizer}")
        _process(client)
        row = _submission_row(client, submission_id)
        assert row["status"] != "rejected", row
        assert row["status"] in {"pending", "running"}, row
