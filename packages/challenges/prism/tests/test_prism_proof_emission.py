"""ExecutionProof emission in the work-unit result payload (VAL-PRISM-001, and 002 end-to-end).

Offline, no GPU: drives a real prism finalization via the CPU re-exec mock (the same seam
``test_validator_dispatch`` uses) and asserts that a successful finalization emits exactly one
version-1 ExecutionProof IN the dispatch result payload, that its ``manifest_sha256`` equals the
independently computed hash of the on-disk ``prism_run_manifest.v2.json``, and that the worker
signature verifies. Also pins the flag-OFF regression (no proof, legacy payload).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from prism_challenge.app import create_app
from prism_challenge.auth import verify_hotkey_signature
from prism_challenge.config import PrismSettings, WorkerPlaneConfig
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run
from prism_challenge.models import SubmissionCreate
from prism_challenge.proof import (
    PROOF_PAYLOAD_KEY,
    ExecutionProof,
    execution_proof_signing_payload,
    verify_execution_proof,
)
from prism_challenge.validator_dispatch import dispatch_assignment

BROKER_URL = "http://broker-val:8082"
WORKER_KEY = "//WorkerEmitter"

TINY_ARCH = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

TINY_TRAIN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

_SHARD_LINE = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


def _stage_train(root: Path, *, lines: int = 64) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD_LINE.format(i=i) for i in range(lines)), encoding="utf-8"
    )
    return data_dir


def _bundle() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _settings(tmp_path: Path, *, worker_plane: WorkerPlaneConfig) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'coord.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp_path / "artifacts",
        worker_plane=worker_plane,
    )


def _payload() -> dict[str, str]:
    # Gateway-free assignment payload (deterministic admission path).
    return {"task": "primary"}


async def _seed(settings: PrismSettings, hotkey: str) -> str:
    app = create_app(settings)
    await app.state.database.init()
    sub = await app.state.repository.create_submission(
        hotkey, SubmissionCreate(code=_bundle(), filename="project.zip")
    )
    await app.state.database.close()
    return sub.id


def _on_disk_manifest(settings: PrismSettings) -> Path:
    matches = list(Path(settings.base_eval_artifact_root).rglob("prism_run_manifest.v2.json"))
    assert len(matches) == 1, f"expected exactly one manifest, found {matches}"
    return matches[0]


async def test_finalization_emits_version1_proof_in_result_payload(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    # Provider env the worker agent would inject (tier-1 provenance).
    monkeypatch.setenv("PRISM_PROVIDER_NAME", "lium")
    monkeypatch.setenv("PRISM_EXECUTOR_ID", "ex-77")
    monkeypatch.setenv("PRISM_POD_ID", "pod-77")
    monkeypatch.setenv("PRISM_MINER_HOTKEY", "miner-owner")
    monkeypatch.setenv("PRISM_IMAGE_DIGEST", "sha256:" + "d" * 64)

    settings = _settings(
        tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY)
    )
    submission_id = await _seed(settings, "hk-owner")

    result = await dispatch_assignment(
        work_unit_id=submission_id,
        payload=_payload(),
        broker_url=BROKER_URL,
        settings=settings,
    )

    assert result["executed"] == 1
    # Exactly one proof, IN the result payload.
    assert PROOF_PAYLOAD_KEY in result
    proof_json = result[PROOF_PAYLOAD_KEY]
    proof = ExecutionProof.model_validate(proof_json)
    assert proof.version == 1
    assert isinstance(proof.tier, int)
    assert len(proof.manifest_sha256) == 64
    assert proof.manifest_sha256 == proof.manifest_sha256.lower()
    assert proof.worker_signature.worker_pubkey
    assert proof.worker_signature.sig

    # VAL-PRISM-002 end-to-end: manifest_sha256 equals the independent hash of the on-disk bytes,
    # computed both ways.
    manifest_path = _on_disk_manifest(settings)
    disk_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_digest = hashlib.sha256(
        json.dumps(reloaded, sort_keys=True, indent=2).encode("utf-8")
    ).hexdigest()
    assert proof.manifest_sha256 == disk_digest == canonical_digest

    # Tier 1 provenance echoed from the injected env.
    assert proof.tier == 1
    assert proof.image_digest == "sha256:" + "d" * 64
    assert proof.provider is not None
    assert proof.provider.name == "lium"
    assert proof.provider.pod_id == "pod-77"
    assert proof.provider.miner_hotkey == "miner-owner"

    # Signature verifies against the pinned message for THIS unit and not another.
    assert verify_execution_proof(proof, unit_id=submission_id) is True
    assert verify_execution_proof(proof, unit_id="some-other-unit") is False
    payload = execution_proof_signing_payload(
        manifest_sha256=proof.manifest_sha256, unit_id=submission_id
    )
    assert verify_hotkey_signature(
        proof.worker_signature.worker_pubkey, payload, proof.worker_signature.sig
    )


async def test_no_proof_emitted_when_worker_plane_disabled(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path, worker_plane=WorkerPlaneConfig(enabled=False))
    submission_id = await _seed(settings, "hk-owner")

    result = await dispatch_assignment(
        work_unit_id=submission_id,
        payload=_payload(),
        broker_url=BROKER_URL,
        settings=settings,
    )

    assert result["executed"] == 1
    assert PROOF_PAYLOAD_KEY not in result


async def test_no_proof_when_enabled_without_signing_key(tmp_path, monkeypatch):
    data_dir = _stage_train(tmp_path)
    monkeypatch.setattr(
        "prism_challenge.evaluator.container.DockerExecutor.run",
        cpu_reexec_run(train_data_dir=data_dir),
    )
    settings = _settings(tmp_path, worker_plane=WorkerPlaneConfig(enabled=True, signing_key=None))
    submission_id = await _seed(settings, "hk-owner")

    result = await dispatch_assignment(
        work_unit_id=submission_id,
        payload=_payload(),
        broker_url=BROKER_URL,
        settings=settings,
    )

    assert result["executed"] == 1
    assert PROOF_PAYLOAD_KEY not in result
