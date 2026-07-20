from __future__ import annotations

import base64
import io
import json
import zipfile

import anyio
import pytest
from conftest import signed_headers
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.distributed_contract import (
    DISTRIBUTED_CONTRACT_RULE,
    RANK0_GUARD_RULE,
    SINGLE_NODE_RULE,
    check_distributed_contract,
    enforce_single_node_bound,
)
from prism_challenge.evaluator.sandbox import SandboxViolation

# The multi-GPU static contract (architecture.md section 8): the miner training.py MUST reference
# the distributed primitives (init_process_group, device binding, a DDP/FSDP wrap, per-rank data
# sharding, a rank-0 write guard, destroy_process_group) and guard its checkpoint/manifest writes
# with a rank-0 condition. The single-node bound rejects gpu_count > 8 / multi-node configs. These
# checks run at the static phase, BEFORE any GPU work (VAL-GPU-006/007/008/016).

COMPLIANT_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 8)
        self.head = nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.head(self.embed(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

COMPLIANT_TRAIN = """
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from architecture import build_model


def train(ctx):
    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(ctx.local_rank)
    model = build_model(ctx)
    model = DDP(model)
    sampler = DistributedSampler(range(64), num_replicas=ctx.world_size, rank=ctx.rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in sampler:
        optimizer.zero_grad()
    if ctx.rank == 0:
        torch.save(model.state_dict(), ctx.artifacts_dir + "/ckpt.pt")
    dist.barrier()
    dist.destroy_process_group()
    return None
"""

# All primitives present except the writes are NOT guarded by rank == 0 (VAL-GPU-008 negative).
_GUARDED_WRITE = (
    "    if ctx.rank == 0:\n"
    '        torch.save(model.state_dict(), ctx.artifacts_dir + "/ckpt.pt")\n'
)
_UNGUARDED_WRITE = '    torch.save(model.state_dict(), ctx.artifacts_dir + "/ckpt.pt")\n'
UNGUARDED_WRITE_TRAIN = COMPLIANT_TRAIN.replace(_GUARDED_WRITE, _UNGUARDED_WRITE)

# No distributed primitives at all (the legacy single-process loop).
NON_DISTRIBUTED_TRAIN = """
import torch
from architecture import build_model


def train(ctx):
    model = build_model(ctx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
    return None
"""

# Otherwise compliant but the device is hardcoded to cuda:0 instead of set_device(local_rank)
# (VAL-GPU-018: the hardcode must not fail the contract).
HARDCODED_DEVICE_TRAIN = """
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from architecture import build_model


def train(ctx):
    dist.init_process_group(backend="gloo")
    model = build_model(ctx)
    model = model.to(torch.device("cuda:0"))
    model = DDP(model)
    sampler = DistributedSampler(range(64), num_replicas=ctx.world_size, rank=ctx.rank)
    if ctx.rank == 0:
        torch.save(model.state_dict(), ctx.artifacts_dir + "/ckpt.pt")
    dist.destroy_process_group()
    return None
"""

# FSDP variant of the compliant script (VAL-GPU-006 accepts DDP OR FSDP).
FSDP_TRAIN = """
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DistributedSampler
from architecture import build_model


def train(ctx):
    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(ctx.local_rank)
    model = FSDP(build_model(ctx))
    sampler = DistributedSampler(range(64), num_replicas=ctx.world_size, rank=ctx.rank)
    if ctx.local_rank == 0:
        torch.save(model.state_dict(), ctx.artifacts_dir + "/ckpt.pt")
    dist.destroy_process_group()
    return None
"""


# --- Unit-level static contract check (VAL-GPU-006/007/008) ---


def test_distributed_contract_compliant_passes() -> None:
    report = check_distributed_contract(
        COMPLIANT_TRAIN, artifact_path="training.py", policy="reject"
    )
    assert report.compliant is True
    assert report.missing == ()
    assert report.unguarded_writes == 0


def test_distributed_contract_fsdp_variant_passes() -> None:
    report = check_distributed_contract(FSDP_TRAIN, artifact_path="training.py", policy="reject")
    assert report.compliant is True


def test_distributed_contract_hardcoded_device_passes() -> None:
    report = check_distributed_contract(
        HARDCODED_DEVICE_TRAIN, artifact_path="training.py", policy="reject"
    )
    assert report.compliant is True


def test_distributed_contract_non_distributed_rejected() -> None:
    with pytest.raises(SandboxViolation) as raised:
        check_distributed_contract(
            NON_DISTRIBUTED_TRAIN, artifact_path="training.py", policy="reject"
        )
    assert raised.value.evidence[0].rule_id == DISTRIBUTED_CONTRACT_RULE
    assert "init_process_group" in str(raised.value)


@pytest.mark.parametrize(
    "removed",
    [
        'dist.init_process_group(backend="gloo")',
        "    model = DDP(model)\n",
        "    dist.destroy_process_group()\n",
    ],
)
def test_distributed_contract_missing_single_primitive_rejected(removed: str) -> None:
    code = COMPLIANT_TRAIN.replace(removed, "")
    with pytest.raises(SandboxViolation) as raised:
        check_distributed_contract(code, artifact_path="training.py", policy="reject")
    assert raised.value.evidence[0].rule_id == DISTRIBUTED_CONTRACT_RULE


def test_distributed_contract_unguarded_write_rejected() -> None:
    with pytest.raises(SandboxViolation) as raised:
        check_distributed_contract(
            UNGUARDED_WRITE_TRAIN, artifact_path="training.py", policy="reject"
        )
    assert raised.value.evidence[0].rule_id == RANK0_GUARD_RULE


def test_distributed_contract_rank0_guarded_write_passes() -> None:
    report = check_distributed_contract(
        COMPLIANT_TRAIN, artifact_path="training.py", policy="reject"
    )
    assert report.compliant is True


def test_distributed_contract_policy_flag_does_not_raise() -> None:
    report = check_distributed_contract(
        NON_DISTRIBUTED_TRAIN, artifact_path="training.py", policy="flag"
    )
    assert report.compliant is False
    assert "init_process_group" in report.missing


def test_distributed_contract_policy_off_skips() -> None:
    report = check_distributed_contract(
        NON_DISTRIBUTED_TRAIN, artifact_path="training.py", policy="off"
    )
    assert report.compliant is True


# --- Single-node bound (VAL-GPU-016) ---


@pytest.mark.parametrize("gpu_count", [1, 2, 4, 8])
def test_single_node_bound_accepts_one_to_eight(gpu_count: int) -> None:
    enforce_single_node_bound(gpu_count, max_gpu_count=8)


def test_single_node_bound_none_is_ok() -> None:
    enforce_single_node_bound(None, max_gpu_count=8)


def test_single_node_bound_rejects_over_eight() -> None:
    with pytest.raises(SandboxViolation) as raised:
        enforce_single_node_bound(9, max_gpu_count=8)
    assert raised.value.evidence[0].rule_id == SINGLE_NODE_RULE


def test_single_node_bound_rejects_multi_node() -> None:
    with pytest.raises(SandboxViolation) as raised:
        enforce_single_node_bound(2, num_nodes=2, max_gpu_count=8)
    assert raised.value.evidence[0].rule_id == SINGLE_NODE_RULE


# --- Pipeline-level behavior (black-box, like the validator: curl + default reject policy) ---


def _enforcing_client(tmp_path) -> TestClient:
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism-distributed.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        plagiarism_enabled=False,
        base_gpu_targets="[]",
        distributed_contract_policy="reject",
    )
    return TestClient(create_app(settings))


def _zip_bundle(files: dict[str, str]) -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _submit(client: TestClient, code: str, *, nonce: str, metadata: dict | None = None) -> str:
    payload: dict = {"code": code, "filename": "bundle.zip"}
    if metadata is not None:
        payload["metadata"] = metadata
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


def _row(client: TestClient, submission_id: str) -> dict:
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


def test_distributed_contract_pipeline_noncompliant_rejected_before_gpu(tmp_path) -> None:
    with _enforcing_client(tmp_path) as client:
        code = _zip_bundle(
            {"architecture.py": COMPLIANT_ARCH, "training.py": NON_DISTRIBUTED_TRAIN}
        )
        submission_id = _submit(client, code, nonce="dist-reject-1")
        _process(client)
        row = _row(client, submission_id)
        assert row["status"] == "rejected", row
        assert "distributed" in str(row["error"]).lower(), row
        leases, jobs = _gpu_work_counts(client, submission_id)
        assert leases == 0
        assert jobs == 0


def test_distributed_contract_pipeline_compliant_advances(tmp_path) -> None:
    with _enforcing_client(tmp_path) as client:
        code = _zip_bundle({"architecture.py": COMPLIANT_ARCH, "training.py": COMPLIANT_TRAIN})
        submission_id = _submit(client, code, nonce="dist-accept-1")
        _process(client)
        row = _row(client, submission_id)
        assert row["status"] != "rejected", row
        assert row["status"] in {"pending", "running"}, row


def test_single_node_bound_pipeline_gpu_count_over_eight_rejected(tmp_path) -> None:
    with _enforcing_client(tmp_path) as client:
        code = _zip_bundle({"architecture.py": COMPLIANT_ARCH, "training.py": COMPLIANT_TRAIN})
        submission_id = _submit(client, code, nonce="dist-bound-1", metadata={"gpu_count": 9})
        _process(client)
        row = _row(client, submission_id)
        assert row["status"] == "rejected", row
        assert "single-node" in str(row["error"]).lower() or "gpu_count" in str(row["error"])
        leases, jobs = _gpu_work_counts(client, submission_id)
        assert leases == 0
        assert jobs == 0
