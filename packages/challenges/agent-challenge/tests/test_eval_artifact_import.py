"""Fail-closed guest-side eval artifact import (fetch + dual-hash verify)."""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.canonical.eval_wire import agent_artifact_sha256_hex
from agent_challenge.evaluation.artifact_import import (
    ArtifactImportError,
    ArtifactProof,
    fetch_eval_artifact,
    materialize_agent_artifact,
    verify_zip_bytes,
)
from agent_challenge.submissions.artifacts import compute_package_tree_sha_from_zip_bytes


def _zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in entries.items():
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def _valid_agent_zip() -> bytes:
    return _zip_bytes({"agent.py": "class Agent:\n    pass\n", "README.md": "ok\n"})


def test_verify_zip_bytes_mismatch_raises() -> None:
    zip_bytes = _valid_agent_zip()
    tree_sha = compute_package_tree_sha_from_zip_bytes(zip_bytes)
    wrong_agent_hash = "0" * 64

    with pytest.raises(ArtifactImportError) as exc_info:
        verify_zip_bytes(
            zip_bytes,
            expected_agent_hash=wrong_agent_hash,
            expected_package_tree_sha=tree_sha,
        )

    assert exc_info.value.reason_code == "digest_mismatch"


def test_verify_zip_bytes_tree_mismatch_raises() -> None:
    zip_bytes = _valid_agent_zip()
    agent_hash = agent_artifact_sha256_hex(zip_bytes)
    wrong_tree = "f" * 64

    with pytest.raises(ArtifactImportError) as exc_info:
        verify_zip_bytes(
            zip_bytes,
            expected_agent_hash=agent_hash,
            expected_package_tree_sha=wrong_tree,
        )

    assert exc_info.value.reason_code == "tree_mismatch"


def test_materialize_writes_and_returns_guest_hashes(tmp_path: Path) -> None:
    zip_bytes = _valid_agent_zip()
    expected_agent_hash = hashlib.sha256(zip_bytes).hexdigest()
    expected_tree = compute_package_tree_sha_from_zip_bytes(zip_bytes)

    zip_dest = tmp_path / "agent.zip"
    package_dest = tmp_path / "package"

    proof = materialize_agent_artifact(zip_bytes, zip_dest=zip_dest, package_dest=package_dest)

    assert isinstance(proof, ArtifactProof)
    assert zip_dest.is_file()
    assert zip_dest.read_bytes() == zip_bytes
    assert package_dest.is_dir()
    assert (package_dest / "agent.py").is_file()
    assert proof.agent_hash == expected_agent_hash
    assert proof.package_tree_sha == expected_tree
    assert proof.zip_size_bytes == len(zip_bytes)
    assert proof.zip_path == zip_dest
    assert proof.package_root == package_dest


def test_fetch_eval_artifact_non_200_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        status = 403
        headers: dict[str, str] = {}

        def read(self, _n: int = -1) -> bytes:
            return b"denied"

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def _urlopen(_req: Any, timeout: float | None = None) -> _Resp:  # noqa: ARG001
        return _Resp()

    monkeypatch.setattr(
        "agent_challenge.evaluation.artifact_import.urlopen",
        _urlopen,
    )

    with pytest.raises(ArtifactImportError) as exc_info:
        fetch_eval_artifact(
            "https://example.test/artifact.zip",
            "secret-token-value",
            timeout=5.0,
        )

    assert exc_info.value.reason_code == "fetch_failed"


def test_fetch_eval_artifact_does_not_log_token(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token = "super-secret-bearer-token-xyz"
    body = b"PK\x03\x04fake-zip-bytes-for-fetch"

    class _Resp:
        status = 200
        headers = {"Content-Length": str(len(body))}

        def read(self, n: int = -1) -> bytes:
            if n < 0:
                return body
            return body[:n]

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def _urlopen(req: Any, timeout: float | None = None) -> _Resp:  # noqa: ARG001
        auth = req.get_header("Authorization")
        assert auth == f"Bearer {token}"
        return _Resp()

    monkeypatch.setattr(
        "agent_challenge.evaluation.artifact_import.urlopen",
        _urlopen,
    )

    with caplog.at_level(logging.DEBUG):
        result = fetch_eval_artifact(
            "https://example.test/artifact.zip?sig=abc",
            token,
            timeout=5.0,
        )

    assert result == body
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert token not in joined
    assert token not in str(caplog.text)
