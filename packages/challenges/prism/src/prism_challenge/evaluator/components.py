from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from .interface import (
    ARCHITECTURE_FACTORY_NAME,
    DEFAULT_ARCHITECTURE_ENTRYPOINT,
    DEFAULT_TRAINING_ENTRYPOINT,
    TRAINING_ENTRYPOINT_NAME,
    SubmissionContractError,
)
from .source_similarity import SourceFile, SourceSnapshot

MANIFEST_NAMES = {"prism.yaml", "prism.yml"}
PROJECT_KINDS = {"full", "architecture_only", "training_for_arch"}
DEFAULT_PROJECT_KIND = "full"

# Miner-declared human-readable architecture name: a module-level ``ARCHITECTURE_NAME = "..."``
# constant in the architecture entrypoint, parsed deterministically via AST (no code execution) and
# moderated deterministically. The moderation MUST be byte-for-byte identical across validators for
# consensus, so it is a pure function with no randomness and no LLM (architecture-lab API contract).
ARCHITECTURE_NAME_CONSTANT = "ARCHITECTURE_NAME"
ARCHITECTURE_NAME_MAX_LENGTH = 48
# Allowed non-alphanumeric characters: spaces are handled separately by the whitespace collapse.
_ARCHITECTURE_NAME_ALLOWED_SPECIAL = frozenset("-_.()[]/+&")


@dataclass(frozen=True)
class PrismProjectComponents:
    kind: str
    architecture_id: str | None
    entrypoint: str
    architecture_files: tuple[SourceFile, ...]
    training_files: tuple[SourceFile, ...]
    manifest: dict[str, Any]
    architecture_entrypoint: str = ""
    training_entrypoint: str = ""
    build_model_symbol: str = ARCHITECTURE_FACTORY_NAME
    train_symbol: str = TRAINING_ENTRYPOINT_NAME


@dataclass(frozen=True)
class _RoleResolution:
    entrypoint: str
    symbol: str
    section: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrismComponentFingerprints:
    family_hash: str
    arch_fingerprint: str
    behavior_fingerprint: str
    training_hash: str


def project_components(snapshot: SourceSnapshot) -> PrismProjectComponents:
    """Resolve a bundle into the two-script contract: a distinct architecture role exposing
    ``build_model`` and a distinct training role exposing ``train``.

    A ``prism.yaml`` manifest is optional. When absent, the default entrypoints
    (``architecture.py``/``training.py``) and default symbols are inferred. When present, declared
    entrypoints are honored exactly with NO silent fallback to the defaults. A single combined
    module no longer satisfies the contract.
    """
    manifest_file = _manifest_file(snapshot)
    manifest: dict[str, Any] = {}
    architecture_id: str | None = None
    manifest_root = ""
    kind = DEFAULT_PROJECT_KIND
    arch_role = _RoleResolution(DEFAULT_ARCHITECTURE_ENTRYPOINT, ARCHITECTURE_FACTORY_NAME)
    train_role = _RoleResolution(DEFAULT_TRAINING_ENTRYPOINT, TRAINING_ENTRYPOINT_NAME)

    if manifest_file is not None:
        loaded = yaml.safe_load(manifest_file.content) or {}
        if not isinstance(loaded, dict):
            raise SubmissionContractError("prism.yaml must contain a mapping")
        manifest = loaded
        kind = str(manifest.get("kind") or DEFAULT_PROJECT_KIND)
        if kind not in PROJECT_KINDS:
            raise SubmissionContractError(f"unsupported Prism project kind: {kind}")
        architecture_id = (
            str(manifest["architecture_id"]) if manifest.get("architecture_id") else None
        )
        manifest_root = str(PurePosixPath(manifest_file.path).parent)
        if manifest_root == ".":
            manifest_root = ""
        arch_role = _role_from_manifest(
            manifest, "architecture", DEFAULT_ARCHITECTURE_ENTRYPOINT, ARCHITECTURE_FACTORY_NAME
        )
        train_role = _role_from_manifest(
            manifest, "training", DEFAULT_TRAINING_ENTRYPOINT, TRAINING_ENTRYPOINT_NAME
        )

    explicit = manifest_file is not None
    architecture_entrypoint = _resolve_role_path(
        snapshot, arch_role.entrypoint, manifest_root, role="architecture", explicit=explicit
    )
    training_entrypoint = _resolve_role_path(
        snapshot, train_role.entrypoint, manifest_root, role="training", explicit=explicit
    )
    if architecture_entrypoint == training_entrypoint:
        raise SubmissionContractError(
            "submission contract violation: architecture and training must be two distinct scripts "
            "(the single-module re-export idiom no longer satisfies the contract)"
        )
    _require_symbol(snapshot, architecture_entrypoint, arch_role.symbol, role="architecture")
    _require_symbol(snapshot, training_entrypoint, train_role.symbol, role="training")

    architecture_files = _component_files(
        snapshot, arch_role.section, manifest_root, architecture_entrypoint
    )
    training_files = _component_files(
        snapshot, train_role.section, manifest_root, training_entrypoint
    )
    return PrismProjectComponents(
        kind=kind,
        architecture_id=architecture_id,
        entrypoint=architecture_entrypoint,
        architecture_files=architecture_files,
        training_files=training_files,
        manifest=manifest,
        architecture_entrypoint=architecture_entrypoint,
        training_entrypoint=training_entrypoint,
        build_model_symbol=arch_role.symbol,
        train_symbol=train_role.symbol,
    )


def _role_from_manifest(
    manifest: dict[str, Any], name: str, default_entrypoint: str, default_symbol: str
) -> _RoleResolution:
    section = _section(manifest, name)
    raw_entrypoint = section.get("entrypoint")
    entrypoint = str(raw_entrypoint) if raw_entrypoint else default_entrypoint
    symbol = default_symbol
    if "::" in entrypoint:
        entrypoint, _, declared_symbol = entrypoint.partition("::")
        if declared_symbol:
            symbol = declared_symbol
    declared = section.get("factory") or section.get("function") or section.get("entry")
    if declared:
        symbol = str(declared)
    return _RoleResolution(entrypoint=entrypoint, symbol=symbol, section=section)


def _resolve_role_path(
    snapshot: SourceSnapshot, path: str, manifest_root: str, *, role: str, explicit: bool
) -> str:
    resolved = _find_path(snapshot, path, manifest_root)
    if resolved is not None:
        return resolved
    if explicit:
        raise SubmissionContractError(
            f"submission contract violation: declared {role} entrypoint {path!r} not found "
            "(declared entrypoints are honored with no silent fallback)"
        )
    raise SubmissionContractError(
        f"submission contract violation: missing {path} ({role} script is required)"
    )


def component_fingerprints(
    components: PrismProjectComponents,
) -> PrismComponentFingerprints:
    arch_fingerprint = _files_fingerprint(components.architecture_files)
    training_hash = _files_fingerprint(components.training_files)
    return PrismComponentFingerprints(
        family_hash=arch_fingerprint,
        arch_fingerprint=arch_fingerprint,
        behavior_fingerprint=arch_fingerprint,
        training_hash=training_hash,
    )


def architecture_name(components: PrismProjectComponents) -> str | None:
    """Parse + moderate the miner-declared ``ARCHITECTURE_NAME`` for the architecture entrypoint.

    Returns the deterministically moderated display name, or ``None`` when the constant is absent,
    is not a plain string literal, or moderates away to empty.
    """
    entrypoint = components.architecture_entrypoint or components.entrypoint
    content = next(
        (file.content for file in components.architecture_files if file.path == entrypoint),
        None,
    )
    if content is None and components.architecture_files:
        content = components.architecture_files[0].content
    if content is None:
        return None
    return moderate_architecture_name(parse_architecture_name(content))


def parse_architecture_name(content: str) -> str | None:
    """Extract the module-level ``ARCHITECTURE_NAME`` string literal via AST (no code execution).

    Only a plain top-level string-literal assignment (``ARCHITECTURE_NAME = "..."`` or an annotated
    ``ARCHITECTURE_NAME: str = "..."``) is honored; anything else (missing, non-string, computed
    expression) yields ``None``.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == ARCHITECTURE_NAME_CONSTANT
            for target in targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        return None
    return None


def moderate_architecture_name(raw: str | None) -> str | None:
    """Deterministically moderate a raw architecture name (architecture-lab API contract).

    Trims and collapses internal whitespace runs to single spaces, strips control / non-printable
    characters, keeps only letters / digits / spaces / ``-_.()[]/+&`` (dropping any other
    character), truncates to 48 characters, and returns ``None`` when the result is empty. Pure and
    deterministic so it is identical across validators for consensus.
    """
    if not isinstance(raw, str):
        return None
    # Collapse all whitespace runs to single ASCII spaces (also converts tabs/newlines so word
    # boundaries survive the printable + charset filters below) and trim.
    collapsed = " ".join(raw.split())
    # Strip control / non-printable characters (ASCII space is printable and preserved).
    printable = "".join(ch for ch in collapsed if ch.isprintable())
    # Keep only the allowed charset; dropping disallowed chars can leave whitespace runs, so the
    # collapse is re-applied before enforcing the max length.
    filtered = "".join(
        ch
        for ch in printable
        if ch == " "
        or ch in _ARCHITECTURE_NAME_ALLOWED_SPECIAL
        or ("a" <= ch <= "z")
        or ("A" <= ch <= "Z")
        or ("0" <= ch <= "9")
    )
    normalized = " ".join(filtered.split())
    truncated = normalized[:ARCHITECTURE_NAME_MAX_LENGTH].strip()
    return truncated or None


def _manifest_file(snapshot: SourceSnapshot) -> SourceFile | None:
    return next(
        (file for file in snapshot.files if PurePosixPath(file.path).name in MANIFEST_NAMES),
        None,
    )


def _section(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    value = manifest.get(name) or {}
    if not isinstance(value, dict):
        raise SubmissionContractError(f"prism.yaml {name} section must be a mapping")
    return value


def _find_path(snapshot: SourceSnapshot, path: str, manifest_root: str) -> str | None:
    candidates = [path.lstrip("/")]
    if manifest_root:
        candidates.append(f"{manifest_root}/{path.lstrip('/')}")
    existing = {file.path for file in snapshot.files}
    for candidate in candidates:
        normalized = PurePosixPath(candidate).as_posix()
        if normalized in existing:
            return normalized
    for candidate in candidates:
        normalized = PurePosixPath(candidate).as_posix()
        basename = PurePosixPath(normalized).name
        match = next(
            (file.path for file in snapshot.files if PurePosixPath(file.path).name == basename),
            None,
        )
        if match:
            return match
    return None


def _resolve_path(snapshot: SourceSnapshot, path: str, manifest_root: str) -> str:
    resolved = _find_path(snapshot, path, manifest_root)
    if resolved is None:
        raise SubmissionContractError(f"Prism project entrypoint not found: {path}")
    return resolved


def _top_level_functions(content: str, path: str) -> set[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        raise SubmissionContractError(
            f"submission contract violation: cannot parse {path} ({exc.msg})"
        ) from exc
    return {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _require_symbol(snapshot: SourceSnapshot, path: str, symbol: str, *, role: str) -> None:
    file = next((item for item in snapshot.files if item.path == path), None)
    if file is None or not file.content.strip():
        raise SubmissionContractError(
            f"submission contract violation: {role} script {path} is empty"
        )
    if symbol not in _top_level_functions(file.content, path):
        raise SubmissionContractError(
            f"submission contract violation: {role} script {path} is missing required "
            f"function {symbol}"
        )


def _component_files(
    snapshot: SourceSnapshot,
    section: dict[str, Any],
    manifest_root: str,
    entrypoint: str,
) -> tuple[SourceFile, ...]:
    paths = {entrypoint}
    extra = section.get("files") or []
    if isinstance(extra, str):
        extra = [extra]
    if not isinstance(extra, list):
        raise SubmissionContractError("prism.yaml component files must be a list")
    for item in extra:
        paths.add(_resolve_path(snapshot, str(item), manifest_root))
    files = tuple(file for file in snapshot.files if file.path in paths)
    if not files:
        raise SubmissionContractError(f"Prism component has no files for entrypoint: {entrypoint}")
    return files


def _files_fingerprint(files: tuple[SourceFile, ...]) -> str:
    root = _shared_root(files)
    parts: list[str] = []
    for file in sorted(files, key=lambda item: item.path):
        content = _normalized_python(file.content) if file.path.endswith(".py") else file.content
        path = file.path.removeprefix(f"{root}/") if root else file.path
        parts.append(f"{path}\n{content}")
    return hashlib.sha256("\n---\n".join(parts).encode()).hexdigest()


def _shared_root(files: tuple[SourceFile, ...]) -> str:
    roots = {file.path.split("/", 1)[0] for file in files if "/" in file.path}
    if len(roots) != 1:
        return ""
    root = next(iter(roots))
    return root if all(file.path.startswith(f"{root}/") for file in files) else ""


def _normalized_python(content: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content
    return ast.dump(tree, annotate_fields=False, include_attributes=False)
