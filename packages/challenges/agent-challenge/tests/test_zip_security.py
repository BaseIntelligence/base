from __future__ import annotations

import base64
import io
import json
import stat
import zipfile

import pytest

from agent_challenge.artifacts import ArtifactValidationError, store_base64_zip


def zip_bytes(
    entries: list[tuple[str, bytes | str]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for filename, contents in entries:
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def symlink_zip() -> bytes:
    buffer = io.BytesIO()
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(info, "target")
    return buffer.getvalue()


def encrypted_flag_zip() -> bytes:
    payload = bytearray(zip_bytes([("agent.py", "ok")]))
    local_header = payload.index(b"PK\x03\x04")
    central_header = payload.index(b"PK\x01\x02")
    payload[local_header + 6 : local_header + 8] = (
        int.from_bytes(payload[local_header + 6 : local_header + 8], "little") | 0x1
    ).to_bytes(2, "little")
    payload[central_header + 8 : central_header + 10] = (
        int.from_bytes(payload[central_header + 8 : central_header + 10], "little") | 0x1
    ).to_bytes(2, "little")
    return bytes(payload)


def assert_rejected(archive_bytes: bytes, tmp_path, reason_code: str) -> None:
    with pytest.raises(ArtifactValidationError) as exc_info:
        store_base64_zip(
            encoded_zip=base64.b64encode(archive_bytes).decode("ascii"),
            artifact_root=str(tmp_path),
        )
    assert exc_info.value.reason_code == reason_code


@pytest.mark.parametrize(
    ("archive_bytes", "reason_code"),
    [
        (zip_bytes([("/agent.py", "ok")]), "absolute_path"),
        (zip_bytes([("../agent.py", "ok")]), "parent_path"),
        (zip_bytes([("agent.py", "ok"), ("./agent.py", "ok")]), "duplicate_path"),
        (symlink_zip(), "symlink_entry"),
        (encrypted_flag_zip(), "encrypted_entry"),
        (zip_bytes([("nested.zip", "ok")]), "nested_archive"),
        (zip_bytes([("/".join(["d"] * 13) + "/agent.py", "ok")]), "path_too_deep"),
        (zip_bytes([("a" * 181, "ok")]), "filename_too_long"),
        (zip_bytes([(f"file-{index}.txt", "") for index in range(201)]), "too_many_files"),
        (
            zip_bytes(
                [("large.txt", b"0" * (20 * 1024 * 1024 + 1))],
                compression=zipfile.ZIP_DEFLATED,
            ),
            "uncompressed_size_too_large",
        ),
        (
            zip_bytes([("ratio.txt", b"0" * 20_000)], compression=zipfile.ZIP_DEFLATED),
            "compression_ratio_too_high",
        ),
        (b"0" * 1_048_577, "zip_too_large"),
    ],
)
def test_rejects_unsafe_zip_shapes(archive_bytes: bytes, reason_code: str, tmp_path):
    assert_rejected(archive_bytes, tmp_path, reason_code)


def test_stores_zip_without_extracting_contents(tmp_path):
    archive_bytes = zip_bytes([("agent.py", "class Agent:\n    pass\n")])

    metadata = store_base64_zip(
        encoded_zip=base64.b64encode(archive_bytes).decode("ascii"),
        artifact_root=str(tmp_path),
    )

    assert metadata.zip_size_bytes == len(archive_bytes)
    assert (tmp_path / metadata.zip_sha256 / "agent.zip").read_bytes() == archive_bytes
    manifest_path = tmp_path / metadata.zip_sha256 / "manifest.json"
    assert manifest_path.exists()
    assert (
        json.loads(manifest_path.read_text(encoding="utf-8"))["entries"][0]["normalized_path"]
        == "agent.py"
    )
    assert not (tmp_path / metadata.zip_sha256 / "agent.py").exists()
