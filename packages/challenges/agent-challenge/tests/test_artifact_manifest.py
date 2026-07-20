from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from agent_challenge.artifacts import (
    ArtifactReadError,
    ArtifactReadSession,
    ArtifactValidationError,
    build_zip_manifest,
    load_zip_manifest,
    store_zip_bytes,
)


def zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in entries.items():
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def entries_by_path(metadata):
    assert metadata.manifest is not None
    return {entry.normalized_path: entry for entry in metadata.manifest.entries}


def test_safe_zip_persists_manifest_and_reads_only_text_files(tmp_path: Path) -> None:
    archive_bytes = zip_bytes(
        {
            "./agent.py": "class Agent:\n    pass\n",
            "README.md": "hello docs\n",
            "data/blob.bin": b"\x00\x01\x02",
        }
    )

    metadata = store_zip_bytes(zip_bytes=archive_bytes, artifact_root=str(tmp_path))

    assert Path(metadata.artifact_path).read_bytes() == archive_bytes
    assert metadata.manifest_path is not None
    assert metadata.manifest is not None
    stored_manifest = json.loads(Path(metadata.manifest_path).read_text(encoding="utf-8"))
    assert stored_manifest == metadata.manifest.to_dict()
    assert load_zip_manifest(metadata.manifest_path) == metadata.manifest
    assert metadata.manifest.zip_sha256 == hashlib.sha256(archive_bytes).hexdigest()
    assert metadata.manifest.zip_size_bytes == len(archive_bytes)
    assert metadata.manifest.artifact_reference == metadata.artifact_path
    assert metadata.manifest.extraction_root is None
    assert [entry.normalized_path for entry in metadata.manifest.entries] == [
        "README.md",
        "agent.py",
        "data/blob.bin",
    ]

    manifest_entries = entries_by_path(metadata)
    agent_entry = manifest_entries["agent.py"]
    readme_entry = manifest_entries["README.md"]
    binary_entry = manifest_entries["data/blob.bin"]
    assert agent_entry.original_path == "./agent.py"
    assert agent_entry.size == len("class Agent:\n    pass\n")
    assert agent_entry.sha256 == hashlib.sha256(b"class Agent:\n    pass\n").hexdigest()
    assert agent_entry.content_type in {"text/x-python", "text/plain"}
    assert agent_entry.is_text is True
    assert agent_entry.is_binary is False
    assert agent_entry.is_python is True
    assert agent_entry.read_eligible is True
    assert readme_entry.is_text is True
    assert readme_entry.is_python is False
    assert binary_entry.content_type == "application/octet-stream"
    assert binary_entry.is_text is False
    assert binary_entry.is_binary is True
    assert binary_entry.read_eligible is False

    read_session = ArtifactReadSession.from_artifact_metadata(
        metadata,
        per_read_max_bytes=32,
        total_read_budget=30,
    )
    assert read_session.read_text("agent.py") == "class Agent:\n    pass\n"
    assert read_session.read_text("README.md", offset=6, limit=4) == "docs"
    with pytest.raises(ArtifactReadError) as exc_info:
        read_session.read_text("data/blob.bin")
    assert exc_info.value.reason_code == "binary_file"


def test_invalid_zip_does_not_create_manifest_or_read_session(tmp_path: Path) -> None:
    with pytest.raises(ArtifactValidationError) as exc_info:
        store_zip_bytes(zip_bytes=zip_bytes({"../agent.py": "bad"}), artifact_root=str(tmp_path))

    assert exc_info.value.reason_code == "parent_path"
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(ArtifactValidationError) as manifest_exc:
        build_zip_manifest(
            zip_bytes=zip_bytes({"/agent.py": "bad"}),
            artifact_reference="agent.zip",
        )
    assert manifest_exc.value.reason_code == "absolute_path"


def test_read_session_rejects_unknown_traversal_and_budget_violations(tmp_path: Path) -> None:
    metadata = store_zip_bytes(
        zip_bytes=zip_bytes({"agent.py": "class Agent:\n    pass\n", "README.md": "hello"}),
        artifact_root=str(tmp_path),
    )
    read_session = ArtifactReadSession.from_artifact_metadata(
        metadata,
        per_read_max_bytes=4,
        total_read_budget=5,
    )

    with pytest.raises(ArtifactReadError) as unknown_exc:
        read_session.read_text("missing.py")
    assert unknown_exc.value.reason_code == "unknown_path"

    with pytest.raises(ArtifactReadError) as traversal_exc:
        read_session.read_text("../agent.py")
    assert traversal_exc.value.reason_code == "unsafe_path"

    with pytest.raises(ArtifactReadError) as absolute_exc:
        read_session.read_text("/agent.py")
    assert absolute_exc.value.reason_code == "unsafe_path"

    with pytest.raises(ArtifactReadError) as per_read_exc:
        read_session.read_text("agent.py", limit=5)
    assert per_read_exc.value.reason_code == "per_read_limit_exceeded"

    assert read_session.read_text("agent.py", limit=4) == "clas"
    with pytest.raises(ArtifactReadError) as budget_exc:
        read_session.read_text("README.md", limit=2)
    assert budget_exc.value.reason_code == "total_read_budget_exceeded"


@pytest.mark.parametrize(
    ("entries", "reason_code"),
    [
        ({"main.py": "class Agent:\n    pass\n"}, "missing_entrypoint"),
        ({"agent.py": "def Agent():\n    pass\n"}, "missing_agent_class"),
        ({"agent.py": "class Agent(:\n    pass\n"}, "invalid_entrypoint"),
    ],
)
def test_zip_entrypoint_contract_is_enforced(
    entries: dict[str, str],
    reason_code: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(ArtifactValidationError) as exc_info:
        store_zip_bytes(zip_bytes=zip_bytes(entries), artifact_root=str(tmp_path))

    assert exc_info.value.reason_code == reason_code


def test_build_zip_manifest_computes_hash_when_not_provided() -> None:
    archive_bytes = zip_bytes({"agent.py": "class Agent:\n    pass\n"})

    manifest = build_zip_manifest(zip_bytes=archive_bytes, artifact_reference="agent.zip")

    assert manifest.zip_sha256 == hashlib.sha256(archive_bytes).hexdigest()


def test_build_zip_manifest_reuses_provided_zip_sha256() -> None:
    archive_bytes = zip_bytes({"agent.py": "class Agent:\n    pass\n"})
    precomputed = hashlib.sha256(archive_bytes).hexdigest()

    manifest = build_zip_manifest(
        zip_bytes=archive_bytes,
        artifact_reference="agent.zip",
        zip_sha256=precomputed,
    )

    assert manifest.zip_sha256 == precomputed


def test_store_zip_bytes_manifest_hash_matches_metadata(tmp_path: Path) -> None:
    archive_bytes = zip_bytes({"agent.py": "class Agent:\n    pass\n"})

    metadata = store_zip_bytes(zip_bytes=archive_bytes, artifact_root=str(tmp_path))

    assert metadata.manifest is not None
    assert metadata.zip_sha256 == metadata.manifest.zip_sha256
    assert metadata.zip_sha256 == hashlib.sha256(archive_bytes).hexdigest()
