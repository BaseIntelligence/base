from __future__ import annotations

import base64
import io
import json
import zipfile

import anyio
import pytest
from conftest import signed_headers

from prism_challenge.db import loads
from prism_challenge.evaluator.components import component_fingerprints, project_components
from prism_challenge.evaluator.interface import SubmissionContractError
from prism_challenge.evaluator.source_similarity import (
    snapshot_from_named_sources,
    snapshot_from_submission,
)

ARCH_CODE = """
import torch

class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, 8)
        self.linear = torch.nn.Linear(8, vocab_size)

    def forward(self, tokens):
        return self.linear(self.embedding(tokens))

def build_model(ctx):
    return TinyModel(ctx.vocab_size)
"""

TRAIN_CODE = """
import torch
from architecture import build_model

def train(ctx):
    model = build_model(ctx)
    return None
"""


def _snapshot(files: dict[str, str]):
    return snapshot_from_named_sources(tuple(files.items()))


# --- Unit-level contract resolution (VAL-CONTRACT-001..008, 029, 030) ---


def test_contract_two_script_resolves_distinct_roles() -> None:
    components = project_components(
        _snapshot({"architecture.py": ARCH_CODE, "training.py": TRAIN_CODE})
    )
    assert components.architecture_entrypoint == "architecture.py"
    assert components.training_entrypoint == "training.py"
    assert components.build_model_symbol == "build_model"
    assert components.train_symbol == "train"
    fingerprints = component_fingerprints(components)
    assert fingerprints.arch_fingerprint != fingerprints.training_hash


def test_contract_prism_yaml_optional_default_entrypoints_inferred() -> None:
    components = project_components(
        _snapshot({"architecture.py": ARCH_CODE, "training.py": TRAIN_CODE})
    )
    assert components.architecture_entrypoint == "architecture.py"
    assert components.training_entrypoint == "training.py"


def test_contract_prism_yaml_default_entrypoints_explicitly_honored() -> None:
    manifest = (
        "architecture:\n  entrypoint: architecture.py\ntraining:\n  entrypoint: training.py\n"
    )
    components = project_components(
        _snapshot({"prism.yaml": manifest, "architecture.py": ARCH_CODE, "training.py": TRAIN_CODE})
    )
    assert components.architecture_entrypoint == "architecture.py"
    assert components.training_entrypoint == "training.py"
    assert components.build_model_symbol == "build_model"
    assert components.train_symbol == "train"


def test_contract_missing_architecture_rejected() -> None:
    with pytest.raises(SubmissionContractError) as excinfo:
        project_components(_snapshot({"training.py": TRAIN_CODE}))
    assert "architecture.py" in str(excinfo.value)


def test_contract_missing_training_rejected() -> None:
    with pytest.raises(SubmissionContractError) as excinfo:
        project_components(_snapshot({"architecture.py": ARCH_CODE}))
    assert "training.py" in str(excinfo.value)


def test_contract_architecture_without_build_model_rejected() -> None:
    arch = "import torch\n\ndef make(ctx):\n    return torch.nn.Linear(4, 4)\n"
    with pytest.raises(SubmissionContractError) as excinfo:
        project_components(_snapshot({"architecture.py": arch, "training.py": TRAIN_CODE}))
    assert "build_model" in str(excinfo.value)


def test_contract_training_without_train_rejected() -> None:
    train = "import torch\n\ndef run(ctx):\n    return None\n"
    with pytest.raises(SubmissionContractError) as excinfo:
        project_components(_snapshot({"architecture.py": ARCH_CODE, "training.py": train}))
    assert "train" in str(excinfo.value)


def test_contract_single_module_reexport_no_longer_satisfies() -> None:
    combined = ARCH_CODE + "\ndef train(ctx):\n    return None\n"
    with pytest.raises(SubmissionContractError) as excinfo:
        project_components(_snapshot({"architecture.py": combined}))
    assert "training.py" in str(excinfo.value)


def test_contract_prism_yaml_non_default_entrypoints_resolved() -> None:
    manifest = (
        "architecture:\n  entrypoint: model.py::make_model\ntraining:\n  entrypoint: loop.py::run\n"
    )
    arch = "import torch\n\ndef make_model(ctx):\n    return torch.nn.Linear(4, 4)\n"
    loop = "def run(ctx):\n    return None\n"
    components = project_components(
        _snapshot({"prism.yaml": manifest, "model.py": arch, "loop.py": loop})
    )
    assert components.architecture_entrypoint == "model.py"
    assert components.training_entrypoint == "loop.py"
    assert components.build_model_symbol == "make_model"
    assert components.train_symbol == "run"
    fingerprints = component_fingerprints(components)
    assert fingerprints.arch_fingerprint != fingerprints.training_hash


def test_contract_prism_yaml_declared_entrypoint_missing_no_silent_fallback() -> None:
    manifest = (
        "architecture:\n  entrypoint: model.py::make_model\ntraining:\n  entrypoint: loop.py::run\n"
    )
    with pytest.raises(SubmissionContractError) as excinfo:
        project_components(
            _snapshot(
                {"prism.yaml": manifest, "architecture.py": ARCH_CODE, "training.py": TRAIN_CODE}
            )
        )
    assert "model.py" in str(excinfo.value)


def test_contract_malformed_bundle_rejected_cleanly() -> None:
    payload = base64.b64encode(b"this is not a zip archive").decode("ascii")
    with pytest.raises(ValueError):
        snapshot_from_submission(payload, "submission.zip", {})


# --- Pipeline-level behavior (black-box, like the validator) ---


def _zip_bundle(files: dict[str, str]) -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _submit(client, code: str, *, filename: str = "bundle.zip", nonce: str) -> str:
    payload = {"code": code, "filename": filename}
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


def _source_files(client, submission_id: str) -> list[dict]:
    repository = client.app.state.repository

    async def fetch() -> list[dict]:
        async with repository.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT files FROM submission_sources WHERE submission_id=?", (submission_id,)
            )
        return [loads(str(row["files"])) for row in rows]

    return anyio.run(fetch)


def test_contract_two_script_submission_lands_pending(client) -> None:
    code = _zip_bundle({"architecture.py": ARCH_CODE, "training.py": TRAIN_CODE})
    submission_id = _submit(client, code, nonce="pending-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] != "rejected", row
    assert row["status"] in {"pending", "running"}, row


def test_contract_two_script_roles_persisted_distinct(client) -> None:
    code = _zip_bundle({"architecture.py": ARCH_CODE, "training.py": TRAIN_CODE})
    submission_id = _submit(client, code, nonce="roles-1")
    _process(client)
    files_payloads = _source_files(client, submission_id)
    assert files_payloads, "submission_sources row must exist"
    files = files_payloads[0]
    by_name = {item["path"].rsplit("/", 1)[-1]: item["sha256"] for item in files}
    assert "architecture.py" in by_name
    assert "training.py" in by_name
    assert by_name["architecture.py"] != by_name["training.py"]


def test_contract_single_module_submission_rejected(client) -> None:
    combined = ARCH_CODE + "\ndef train(ctx):\n    return None\n"
    code = _zip_bundle({"architecture.py": combined})
    submission_id = _submit(client, code, nonce="single-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] == "rejected", row
    assert "training.py" in str(row["error"])


def test_contract_missing_training_submission_rejected(client) -> None:
    code = _zip_bundle({"architecture.py": ARCH_CODE})
    submission_id = _submit(client, code, nonce="missing-train-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] == "rejected", row
    assert "training.py" in str(row["error"])


def test_contract_missing_architecture_submission_rejected(client) -> None:
    code = _zip_bundle({"training.py": TRAIN_CODE})
    submission_id = _submit(client, code, nonce="missing-arch-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] == "rejected", row
    assert "architecture.py" in str(row["error"])


def test_contract_malformed_bundle_submission_rejected(client) -> None:
    code = base64.b64encode(b"definitely not a zip").decode("ascii")
    submission_id = _submit(client, code, nonce="malformed-1")
    _process(client)
    row = _submission_row(client, submission_id)
    assert row["status"] == "rejected", row
