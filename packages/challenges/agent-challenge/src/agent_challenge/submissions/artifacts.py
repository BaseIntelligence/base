from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import io
import json
import mimetypes
import stat
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

MAX_ZIP_BYTES = 1_048_576
MAX_FILES = 200
MAX_PATH_DEPTH = 12
MAX_FILENAME_LENGTH = 180
MAX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100
NESTED_ARCHIVE_SUFFIXES = (".zip", ".jar", ".whl", ".egg")
DEFAULT_ARTIFACT_READ_MAX_BYTES = 64_000
DEFAULT_ARTIFACT_READ_TOTAL_BUDGET = 256_000
_TEXT_CONTROL_BYTES = frozenset(range(0, 9)) | frozenset(range(14, 32)) | {127}


@dataclass(frozen=True)
class ZipManifestEntry:
    normalized_path: str
    original_path: str
    size: int
    sha256: str
    content_type: str | None
    is_text: bool
    is_binary: bool
    is_python: bool
    read_eligible: bool
    artifact_reference: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ZipArtifactManifest:
    zip_sha256: str
    zip_size_bytes: int
    artifact_reference: str
    extraction_root: str | None
    entries: tuple[ZipManifestEntry, ...]
    package_tree_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "zip_sha256": self.zip_sha256,
            "zip_size_bytes": self.zip_size_bytes,
            "artifact_reference": self.artifact_reference,
            "extraction_root": self.extraction_root,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if self.package_tree_sha is not None:
            payload["package_tree_sha"] = self.package_tree_sha
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZipArtifactManifest:
        entries = tuple(ZipManifestEntry(**entry) for entry in data.get("entries", []))
        package_tree_sha = data.get("package_tree_sha")
        if package_tree_sha is not None:
            package_tree_sha = str(package_tree_sha)
        return cls(
            zip_sha256=data["zip_sha256"],
            zip_size_bytes=int(data["zip_size_bytes"]),
            artifact_reference=data["artifact_reference"],
            extraction_root=data.get("extraction_root"),
            entries=entries,
            package_tree_sha=package_tree_sha,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class ArtifactMetadata:
    zip_sha256: str
    zip_size_bytes: int
    artifact_path: str
    manifest: ZipArtifactManifest | None = None
    manifest_path: str | None = None
    package_tree_sha: str | None = None


class ArtifactValidationError(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


class ArtifactReadError(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message


def store_base64_zip(
    *,
    encoded_zip: str,
    artifact_root: str,
    max_zip_bytes: int = MAX_ZIP_BYTES,
) -> ArtifactMetadata:
    try:
        zip_bytes = base64.b64decode(encoded_zip, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ArtifactValidationError(
            "invalid_base64",
            "artifact_zip_base64 must be valid base64",
        ) from exc
    return store_zip_bytes(
        zip_bytes=zip_bytes,
        artifact_root=artifact_root,
        max_zip_bytes=max_zip_bytes,
    )


def store_zip_uri(
    *,
    artifact_uri: str,
    artifact_root: str,
    max_zip_bytes: int = MAX_ZIP_BYTES,
) -> ArtifactMetadata:
    artifact_root_path = Path(artifact_root).expanduser().resolve()
    source_path = Path(artifact_uri).expanduser().resolve()
    if not source_path.exists() or not source_path.is_file():
        raise ArtifactValidationError(
            "artifact_uri_not_found",
            "artifact_uri must point to an existing zip file on the challenge host",
        )
    if artifact_root_path not in source_path.parents and source_path != artifact_root_path:
        raise ArtifactValidationError(
            "artifact_uri_outside_root",
            "artifact_uri must be inside CHALLENGE_ARTIFACT_ROOT",
        )
    zip_size_bytes = source_path.stat().st_size
    if zip_size_bytes > max_zip_bytes:
        raise ArtifactValidationError("zip_too_large", "artifact zip exceeds 1MB")
    zip_bytes = source_path.read_bytes()
    return store_zip_bytes(
        zip_bytes=zip_bytes,
        artifact_root=artifact_root,
        max_zip_bytes=max_zip_bytes,
    )


def store_zip_bytes(
    *,
    zip_bytes: bytes,
    artifact_root: str,
    max_zip_bytes: int = MAX_ZIP_BYTES,
) -> ArtifactMetadata:
    zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    root = Path(artifact_root).expanduser().resolve()
    target_dir = (root / zip_sha256).resolve()
    if root not in target_dir.parents:
        raise ArtifactValidationError("invalid_artifact_target", "invalid artifact storage target")
    target_path = target_dir / "agent.zip"
    manifest = build_zip_manifest(
        zip_bytes=zip_bytes,
        artifact_reference=str(target_path),
        max_zip_bytes=max_zip_bytes,
        zip_sha256=zip_sha256,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        temporary_path = target_dir / "agent.zip.tmp"
        temporary_path.write_bytes(zip_bytes)
        temporary_path.replace(target_path)
    manifest_path = target_dir / "manifest.json"
    if not manifest_path.exists():
        temporary_manifest_path = target_dir / "manifest.json.tmp"
        temporary_manifest_path.write_text(manifest.to_json(), encoding="utf-8")
        temporary_manifest_path.replace(manifest_path)
    return ArtifactMetadata(
        zip_sha256=zip_sha256,
        zip_size_bytes=len(zip_bytes),
        artifact_path=str(target_path),
        manifest=manifest,
        manifest_path=str(manifest_path),
        package_tree_sha=manifest.package_tree_sha,
    )


def compute_package_tree_sha_from_entries(
    entries: list[tuple[str, bytes]] | tuple[tuple[str, bytes], ...],
) -> str:
    """Canonical content-addressed SHA-256 of a package tree.

    Algorithm (AGATE package_tree_sha / VAL-AGATE-001):
    1. Take each regular file as (normalized relative POSIX path, raw bytes).
    2. Sort by relative path ascending (bytewise UTF-8 / POSIX path order).
    3. For each path in order, hash:
         SHA256( path_utf8 + b"\\0" + content_bytes )
       and append that 32-byte digest to a running buffer.
    4. Return hex(SHA256(concatenated leaf digests)).

    Empty trees are refused (agent packages must include at least agent.py).
    """

    if not entries:
        raise ArtifactValidationError("empty_package_tree", "package tree has no files")
    normalized: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for relpath, content in entries:
        path = str(relpath).replace("\\", "/").strip()
        if not path or path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise ArtifactValidationError("unsafe_path", "package tree path is unsafe")
        if path in seen:
            raise ArtifactValidationError(
                "duplicate_path",
                "package tree contains duplicate normalized paths",
            )
        seen.add(path)
        if not isinstance(content, (bytes, bytearray)):
            raise ArtifactValidationError(
                "invalid_package_entry",
                "package tree entry content must be bytes",
            )
        normalized.append((path, bytes(content)))
    normalized.sort(key=lambda item: item[0])
    hasher = hashlib.sha256()
    for relpath, content in normalized:
        leaf = hashlib.sha256(relpath.encode("utf-8") + b"\0" + content).digest()
        hasher.update(leaf)
    return hasher.hexdigest()


def compute_package_tree_sha_from_zip_bytes(zip_bytes: bytes) -> str:
    """Compute package_tree_sha from ZIP member paths + contents (files only)."""

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            pairs: list[tuple[str, bytes]] = []
            for member in archive.infolist():
                if member.is_dir():
                    continue
                normalized_path = _normalized_member_path(member.filename)
                with archive.open(member) as source:
                    content = source.read()
                pairs.append((normalized_path, content))
            return compute_package_tree_sha_from_entries(pairs)
    except zipfile.BadZipFile as exc:
        raise ArtifactValidationError(
            "invalid_zip",
            "artifact zip must contain a zip",
        ) from exc


def package_tree_sha_from_directory(root: str | Path) -> str:
    """Recompute package_tree_sha of an extracted package directory (guest path)."""

    root_path = Path(root).expanduser().resolve(strict=True)
    if not root_path.is_dir():
        raise ArtifactValidationError("not_a_directory", "package root must be a directory")
    pairs: list[tuple[str, bytes]] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root_path).as_posix()
        pairs.append((rel, path.read_bytes()))
    return compute_package_tree_sha_from_entries(pairs)


def build_zip_manifest(
    *,
    zip_bytes: bytes,
    artifact_reference: str,
    max_zip_bytes: int = MAX_ZIP_BYTES,
    extraction_root: str | None = None,
    zip_sha256: str | None = None,
) -> ZipArtifactManifest:
    _validate_zip_bytes(zip_bytes, max_zip_bytes=max_zip_bytes)
    if zip_sha256 is None:
        zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
    entries: list[ZipManifestEntry] = []
    content_pairs: list[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            normalized_path = _normalized_member_path(member.filename)
            if member.is_dir():
                continue
            with archive.open(member) as source:
                content = source.read()
            content_pairs.append((normalized_path, content))
            is_text = _is_probably_text(content)
            entries.append(
                ZipManifestEntry(
                    normalized_path=normalized_path,
                    original_path=member.filename,
                    size=member.file_size,
                    sha256=hashlib.sha256(content).hexdigest(),
                    content_type=_guess_content_type(normalized_path, is_text),
                    is_text=is_text,
                    is_binary=not is_text,
                    is_python=normalized_path.lower().endswith(".py"),
                    read_eligible=is_text,
                    artifact_reference=artifact_reference,
                )
            )
        _validate_entrypoint(archive, entries)
    package_tree_sha = compute_package_tree_sha_from_entries(content_pairs)
    return ZipArtifactManifest(
        zip_sha256=zip_sha256,
        zip_size_bytes=len(zip_bytes),
        artifact_reference=artifact_reference,
        extraction_root=extraction_root,
        entries=tuple(sorted(entries, key=lambda entry: entry.normalized_path)),
        package_tree_sha=package_tree_sha,
    )


def load_zip_manifest(manifest_path: str | Path) -> ZipArtifactManifest:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return ZipArtifactManifest.from_dict(data)


class ArtifactReadSession:
    def __init__(
        self,
        *,
        zip_path: str | Path,
        manifest: ZipArtifactManifest,
        per_read_max_bytes: int = DEFAULT_ARTIFACT_READ_MAX_BYTES,
        total_read_budget: int = DEFAULT_ARTIFACT_READ_TOTAL_BUDGET,
    ) -> None:
        self.zip_path = Path(zip_path).expanduser().resolve(strict=True)
        self.manifest = manifest
        self.per_read_max_bytes = max(per_read_max_bytes, 1)
        self.total_read_budget = max(total_read_budget, 1)
        self.bytes_read = 0
        self._entries = {entry.normalized_path: entry for entry in manifest.entries}

    @classmethod
    def from_artifact_metadata(
        cls,
        metadata: ArtifactMetadata,
        **kwargs: object,
    ) -> ArtifactReadSession:
        if metadata.manifest is None:
            if metadata.manifest_path is None:
                raise ArtifactReadError("missing_manifest", "artifact manifest is required")
            manifest = load_zip_manifest(metadata.manifest_path)
        else:
            manifest = metadata.manifest
        return cls(zip_path=metadata.artifact_path, manifest=manifest, **kwargs)

    def read_text(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        normalized_path = self._normalize_read_path(path)
        entry = self._entries.get(normalized_path)
        if entry is None:
            raise ArtifactReadError("unknown_path", "path is not listed in the artifact manifest")
        if not entry.read_eligible or entry.is_binary:
            raise ArtifactReadError("binary_file", "path is not eligible for text reads")
        if offset < 0:
            raise ArtifactReadError("invalid_offset", "read offset must be non-negative")
        remaining_size = max(entry.size - offset, 0)
        requested_limit = remaining_size if limit is None else limit
        if requested_limit < 0:
            raise ArtifactReadError("invalid_limit", "read limit must be non-negative")
        if requested_limit > self.per_read_max_bytes:
            raise ArtifactReadError("per_read_limit_exceeded", "read exceeds per-read byte limit")
        if offset >= entry.size or requested_limit == 0:
            return ""
        readable_bytes = min(requested_limit, entry.size - offset)
        if self.bytes_read + readable_bytes > self.total_read_budget:
            raise ArtifactReadError("total_read_budget_exceeded", "read exceeds total byte budget")
        data = self._read_member_bytes(entry, offset=offset, limit=readable_bytes)
        self.bytes_read += len(data)
        if self.bytes_read > self.total_read_budget:
            raise ArtifactReadError("total_read_budget_exceeded", "read exceeds total byte budget")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactReadError("binary_file", "path is not valid utf-8 text") from exc

    def _read_member_bytes(self, entry: ZipManifestEntry, *, offset: int, limit: int) -> bytes:
        if limit == 0:
            return b""
        with zipfile.ZipFile(self.zip_path) as archive:
            with archive.open(entry.original_path) as source:
                if offset:
                    source.read(offset)
                return source.read(limit)

    def _normalize_read_path(self, path: str) -> str:
        try:
            return _normalized_member_path(path)
        except ArtifactValidationError as exc:
            raise ArtifactReadError("unsafe_path", "read path is unsafe") from exc


def extract_zip_to_directory(
    *,
    zip_path: str | Path,
    target_directory: str | Path,
    max_zip_bytes: int = MAX_ZIP_BYTES,
) -> Path:
    source_path = Path(zip_path).expanduser().resolve(strict=True)
    target_root = Path(target_directory).expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    zip_bytes = source_path.read_bytes()
    _validate_zip_bytes(zip_bytes, max_zip_bytes=max_zip_bytes)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            normalized_path = _normalized_member_path(member.filename)
            target_path = (target_root / normalized_path).resolve()
            if target_path != target_root and target_root not in target_path.parents:
                raise ArtifactValidationError("unsafe_path", "artifact zip contains unsafe paths")
            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                target.write(source.read())
    return target_root


def _validate_zip_bytes(zip_bytes: bytes, *, max_zip_bytes: int) -> None:
    if not zip_bytes:
        raise ArtifactValidationError("empty_zip", "artifact zip must not be empty")
    if len(zip_bytes) > max_zip_bytes:
        raise ArtifactValidationError("zip_too_large", "artifact zip exceeds 1MB")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            _validate_zip_members(archive)
    except zipfile.BadZipFile as exc:
        raise ArtifactValidationError(
            "invalid_zip",
            "artifact_zip_base64 must contain a zip",
        ) from exc


def _validate_zip_members(archive: zipfile.ZipFile) -> None:
    normalized_paths: set[str] = set()
    file_count = 0
    total_uncompressed = 0
    for member in archive.infolist():
        normalized_path = _normalized_member_path(member.filename)
        if normalized_path in normalized_paths:
            raise ArtifactValidationError(
                "duplicate_path",
                "artifact zip contains duplicate normalized paths",
            )
        normalized_paths.add(normalized_path)

        if member.flag_bits & 0x1:
            raise ArtifactValidationError(
                "encrypted_entry",
                "artifact zip contains encrypted entries",
            )
        if _is_symlink(member):
            raise ArtifactValidationError("symlink_entry", "artifact zip contains symlinks")
        if member.compress_size > MAX_ZIP_BYTES:
            raise ArtifactValidationError("zip_too_large", "artifact zip exceeds 1MB")

        if member.is_dir():
            continue
        file_count += 1
        if file_count > MAX_FILES:
            raise ArtifactValidationError("too_many_files", "artifact zip contains too many files")
        if normalized_path.lower().endswith(NESTED_ARCHIVE_SUFFIXES):
            raise ArtifactValidationError(
                "nested_archive",
                "artifact zip contains nested archives",
            )
        total_uncompressed += member.file_size
        if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
            raise ArtifactValidationError(
                "uncompressed_size_too_large",
                "artifact zip uncompressed content exceeds 20MB",
            )
        if member.compress_size == 0 and member.file_size > 0:
            raise ArtifactValidationError(
                "compression_ratio_too_high",
                "artifact zip compression ratio exceeds 100:1",
            )
        if member.compress_size and member.file_size / member.compress_size > MAX_COMPRESSION_RATIO:
            raise ArtifactValidationError(
                "compression_ratio_too_high",
                "artifact zip compression ratio exceeds 100:1",
            )


def _validate_entrypoint(
    archive: zipfile.ZipFile,
    entries: list[ZipManifestEntry],
) -> None:
    entry = next((item for item in entries if item.normalized_path == "agent.py"), None)
    if entry is None:
        raise ArtifactValidationError(
            "missing_entrypoint",
            "artifact zip must include agent.py at the archive root",
        )
    if not entry.is_text or not entry.is_python:
        raise ArtifactValidationError(
            "invalid_entrypoint",
            "agent.py must be a readable Python source file",
        )
    source = archive.read(entry.original_path).decode("utf-8")
    try:
        module = ast.parse(source, filename="agent.py")
    except SyntaxError as exc:
        raise ArtifactValidationError(
            "invalid_entrypoint",
            "agent.py must parse as valid Python",
        ) from exc
    if not any(isinstance(node, ast.ClassDef) and node.name == "Agent" for node in module.body):
        raise ArtifactValidationError(
            "missing_agent_class",
            "agent.py must define a top-level Agent class",
        )


def _normalized_member_path(filename: str) -> str:
    if "\x00" in filename or "\\" in filename:
        raise ArtifactValidationError("unsafe_path", "artifact zip contains unsafe paths")
    path = PurePosixPath(filename)
    if path.is_absolute():
        raise ArtifactValidationError("absolute_path", "artifact zip contains absolute paths")
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        raise ArtifactValidationError("empty_path", "artifact zip contains empty paths")
    if ".." in parts:
        raise ArtifactValidationError("parent_path", "artifact zip contains parent paths")
    if len(parts) > MAX_PATH_DEPTH:
        raise ArtifactValidationError("path_too_deep", "artifact zip paths exceed max depth")
    if any(len(part) > MAX_FILENAME_LENGTH for part in parts):
        raise ArtifactValidationError("filename_too_long", "artifact zip filenames are too long")
    return "/".join(parts)


def _is_symlink(member: zipfile.ZipInfo) -> bool:
    mode = member.external_attr >> 16
    return stat.S_IFMT(mode) == stat.S_IFLNK


def _guess_content_type(path: str, is_text: bool) -> str | None:
    content_type, _encoding = mimetypes.guess_type(path)
    if content_type is not None:
        return content_type
    return "text/plain" if is_text else "application/octet-stream"


def _is_probably_text(content: bytes) -> bool:
    if b"\0" in content:
        return False
    try:
        content.decode("utf-8")
    except UnicodeDecodeError:
        return False
    if not content:
        return True
    control_count = sum(byte in _TEXT_CONTROL_BYTES for byte in content)
    return control_count / len(content) <= 0.05
