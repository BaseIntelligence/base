from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from agent_challenge.core.models import PythonAstFeature
from agent_challenge.submissions.artifacts import (
    ArtifactReadError,
    ArtifactReadSession,
    ZipArtifactManifest,
    ZipManifestEntry,
)

AST_STATUS_OK = "ok"
AST_STATUS_PARTIAL = "partial"
AST_STATUS_SYNTAX_ERROR = "syntax_error"
AST_STATUS_READ_ERROR = "read_error"
AST_STATUS_UNSUPPORTED = "unsupported"

AstExtractionStatus = Literal["ok", "partial", "unsupported"]
AstFileStatus = Literal["ok", "syntax_error", "read_error"]

_SKIP_PATH_PARTS = {
    ".cache",
    ".eggs",
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "site-packages",
    "venv",
}
_SKIP_SUFFIXES = (".dist-info", ".egg-info", "_pb2.py", "_pb2_grpc.py")
_SHINGLE_SIZE = 3


@dataclass(frozen=True)
class PythonAstSyntaxError:
    message: str
    line: int | None
    offset: int | None
    text_hash: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PythonAstFileResult:
    file_path: str
    status: AstFileStatus
    artifact_hash: str
    file_hash: str
    artifact_reference: str
    ast_hash: str | None = None
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    call_shingles: tuple[str, ...] = ()
    name_shingles: tuple[str, ...] = ()
    docstring_count: int = 0
    string_literal_count: int = 0
    imports: tuple[str, ...] = ()
    has_module_docstring: bool = False
    has_function_docstring: bool = False
    has_class_docstring: bool = False
    syntax_error: PythonAstSyntaxError | None = None
    read_error_code: str | None = None
    read_error_message: str | None = None

    @property
    def parser_error(self) -> str | None:
        if self.syntax_error is not None:
            return self.syntax_error.message
        return self.read_error_code

    @property
    def syntax_line(self) -> int | None:
        return self.syntax_error.line if self.syntax_error is not None else None

    @property
    def syntax_offset(self) -> int | None:
        return self.syntax_error.offset if self.syntax_error is not None else None

    @property
    def docstring_present(self) -> bool:
        return self.docstring_count > 0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["call_shingles"] = list(self.call_shingles)
        data["name_shingles"] = list(self.name_shingles)
        data["imports"] = list(self.imports)
        if self.syntax_error is not None:
            data["syntax_error"] = self.syntax_error.to_dict()
        data["parser_error"] = self.parser_error
        data["syntax_line"] = self.syntax_line
        data["syntax_offset"] = self.syntax_offset
        data["docstring_present"] = self.docstring_present
        return data

    def feature_records(self) -> tuple[dict[str, object], ...]:
        metadata = {
            "artifact_hash": self.artifact_hash,
            "artifact_reference": self.artifact_reference,
            "file_hash": self.file_hash,
            "status": self.status,
        }
        if self.status != AST_STATUS_OK:
            error_metadata = dict(metadata)
            if self.syntax_error is not None:
                error_metadata["syntax_error"] = self.syntax_error.to_dict()
            if self.read_error_code is not None:
                error_metadata["read_error_code"] = self.read_error_code
                error_metadata["read_error_message"] = self.read_error_message or ""
            return (
                _feature_record(
                    self.file_path,
                    "parser_status",
                    "parser_status",
                    self.status,
                    error_metadata,
                ),
            )

        return (
            _feature_record(
                self.file_path,
                "parser_status",
                "parser_status",
                self.status,
                metadata,
            ),
            _feature_record(self.file_path, "ast_hash", "ast_hash", self.ast_hash or "", metadata),
            _feature_record(
                self.file_path,
                "function_count",
                "count",
                str(self.function_count),
                metadata,
            ),
            _feature_record(
                self.file_path,
                "class_count",
                "count",
                str(self.class_count),
                metadata,
            ),
            _feature_record(
                self.file_path,
                "import_count",
                "count",
                str(self.import_count),
                metadata,
            ),
            _feature_record(
                self.file_path,
                "docstrings",
                "json",
                _stable_json(
                    {
                        "count": self.docstring_count,
                        "has_class_docstring": self.has_class_docstring,
                        "has_function_docstring": self.has_function_docstring,
                        "has_module_docstring": self.has_module_docstring,
                    }
                ),
                metadata,
            ),
            _feature_record(
                self.file_path,
                "call_shingles",
                "json",
                _stable_json(list(self.call_shingles)),
                metadata,
            ),
            _feature_record(
                self.file_path,
                "name_shingles",
                "json",
                _stable_json(list(self.name_shingles)),
                metadata,
            ),
        )


@dataclass(frozen=True)
class PythonAstExtractionReport:
    status: AstExtractionStatus
    artifact_hash: str
    artifact_reference: str
    python_file_count: int
    parsed_file_count: int
    syntax_error_count: int
    read_error_count: int
    files: tuple[PythonAstFileResult, ...]
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_hash": self.artifact_hash,
            "artifact_reference": self.artifact_reference,
            "files": [file_result.to_dict() for file_result in self.files],
            "parsed_file_count": self.parsed_file_count,
            "python_file_count": self.python_file_count,
            "read_error_count": self.read_error_count,
            "reason": self.reason,
            "status": self.status,
            "syntax_error_count": self.syntax_error_count,
        }

    def feature_records(self) -> tuple[dict[str, object], ...]:
        records: list[dict[str, object]] = [
            _feature_record(
                "",
                "artifact_status",
                "artifact_status",
                self.status,
                {
                    "artifact_hash": self.artifact_hash,
                    "artifact_reference": self.artifact_reference,
                    "parsed_file_count": self.parsed_file_count,
                    "python_file_count": self.python_file_count,
                    "read_error_count": self.read_error_count,
                    "reason": self.reason,
                    "syntax_error_count": self.syntax_error_count,
                },
            )
        ]
        for file_result in self.files:
            records.extend(file_result.feature_records())
        return tuple(records)


def extract_python_ast_features(
    manifest: ZipArtifactManifest,
    read_session: ArtifactReadSession | None = None,
    *,
    zip_path: str | Path | None = None,
) -> PythonAstExtractionReport:
    if read_session is None:
        if zip_path is None:
            raise ValueError("zip_path is required when read_session is not provided")
        read_session = ArtifactReadSession(zip_path=zip_path, manifest=manifest)

    entries = tuple(entry for entry in manifest.entries if _is_eligible_python_entry(entry))
    if not entries:
        return PythonAstExtractionReport(
            status=AST_STATUS_UNSUPPORTED,
            artifact_hash=manifest.zip_sha256,
            artifact_reference=manifest.artifact_reference,
            python_file_count=0,
            parsed_file_count=0,
            syntax_error_count=0,
            read_error_count=0,
            files=(),
            reason="no_python_files",
        )

    results = tuple(_extract_file(entry, manifest, read_session) for entry in entries)
    parsed_file_count = sum(result.status == AST_STATUS_OK for result in results)
    syntax_error_count = sum(result.status == AST_STATUS_SYNTAX_ERROR for result in results)
    read_error_count = sum(result.status == AST_STATUS_READ_ERROR for result in results)
    status: AstExtractionStatus = AST_STATUS_OK if parsed_file_count == len(results) else "partial"
    return PythonAstExtractionReport(
        status=status,
        artifact_hash=manifest.zip_sha256,
        artifact_reference=manifest.artifact_reference,
        python_file_count=len(results),
        parsed_file_count=parsed_file_count,
        syntax_error_count=syntax_error_count,
        read_error_count=read_error_count,
        files=results,
    )


def build_python_ast_feature_rows(
    *,
    analysis_run_id: int,
    report: PythonAstExtractionReport,
) -> list[PythonAstFeature]:
    return [
        PythonAstFeature(
            analysis_run_id=analysis_run_id,
            file_path=str(record["file_path"]),
            feature_key=str(record["feature_key"]),
            feature_type=str(record["feature_type"]),
            feature_value=str(record["feature_value"]),
            line_start=None,
            line_end=None,
            metadata_json=str(record["metadata_json"]),
        )
        for record in report.feature_records()
    ]


def _extract_file(
    entry: ZipManifestEntry,
    manifest: ZipArtifactManifest,
    read_session: ArtifactReadSession,
) -> PythonAstFileResult:
    try:
        source = read_session.read_text(entry.normalized_path, limit=entry.size)
    except ArtifactReadError as exc:
        return PythonAstFileResult(
            file_path=entry.normalized_path,
            status=AST_STATUS_READ_ERROR,
            artifact_hash=manifest.zip_sha256,
            file_hash=entry.sha256,
            artifact_reference=entry.artifact_reference,
            read_error_code=exc.reason_code,
            read_error_message=exc.message,
        )

    try:
        tree = ast.parse(source, filename=entry.normalized_path)
    except SyntaxError as exc:
        return PythonAstFileResult(
            file_path=entry.normalized_path,
            status=AST_STATUS_SYNTAX_ERROR,
            artifact_hash=manifest.zip_sha256,
            file_hash=entry.sha256,
            artifact_reference=entry.artifact_reference,
            syntax_error=PythonAstSyntaxError(
                message=exc.msg,
                line=exc.lineno,
                offset=exc.offset,
                text_hash=_hash_text(exc.text) if exc.text is not None else None,
            ),
        )

    features = _valid_file_features(tree)
    return PythonAstFileResult(
        file_path=entry.normalized_path,
        status=AST_STATUS_OK,
        artifact_hash=manifest.zip_sha256,
        file_hash=entry.sha256,
        artifact_reference=entry.artifact_reference,
        ast_hash=features["ast_hash"],
        function_count=features["function_count"],
        class_count=features["class_count"],
        import_count=features["import_count"],
        call_shingles=features["call_shingles"],
        name_shingles=features["name_shingles"],
        docstring_count=features["docstring_count"],
        string_literal_count=features["string_literal_count"],
        imports=features["imports"],
        has_module_docstring=features["has_module_docstring"],
        has_function_docstring=features["has_function_docstring"],
        has_class_docstring=features["has_class_docstring"],
    )


def _valid_file_features(tree: ast.Module) -> dict[str, Any]:
    counter = _AstCounter()
    counter.visit(tree)
    normalized = _normalize_node(tree, _NameRoles())
    return {
        "ast_hash": _hash_text(_stable_json(normalized)),
        "call_shingles": _shingles(counter.call_tokens),
        "class_count": counter.class_count,
        "docstring_count": counter.docstring_count,
        "function_count": counter.function_count,
        "has_class_docstring": counter.class_docstring_count > 0,
        "has_function_docstring": counter.function_docstring_count > 0,
        "has_module_docstring": ast.get_docstring(tree, clean=False) is not None,
        "import_count": counter.import_count,
        "imports": tuple(sorted(counter.imports)),
        "name_shingles": _shingles(counter.name_tokens),
        "string_literal_count": counter.string_literal_count,
    }


class _AstCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_count = 0
        self.class_count = 0
        self.import_count = 0
        self.function_docstring_count = 0
        self.class_docstring_count = 0
        self.docstring_count = 0
        self.string_literal_count = 0
        self.imports: list[str] = []
        self.call_tokens: list[str] = []
        self.name_tokens: list[str] = []

    def visit_Module(self, node: ast.Module) -> None:
        self.docstring_count += _has_docstring(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_count += 1
        has_docstring = _has_docstring(node)
        self.function_docstring_count += has_docstring
        self.docstring_count += has_docstring
        self.name_tokens.append("function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_count += 1
        has_docstring = _has_docstring(node)
        self.function_docstring_count += has_docstring
        self.docstring_count += has_docstring
        self.name_tokens.append("async_function")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_count += 1
        has_docstring = _has_docstring(node)
        self.class_docstring_count += has_docstring
        self.docstring_count += has_docstring
        self.name_tokens.append("class")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.import_count += len(node.names)
        self.imports.extend(alias.name for alias in node.names)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.import_count += len(node.names)
        module = "." * node.level + (node.module or "")
        self.imports.extend(f"{module}:{alias.name}" for alias in node.names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.call_tokens.append(_call_token(node.func))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.name_tokens.append(type(node.ctx).__name__)
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        self.name_tokens.append("arg")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.string_literal_count += 1
        self.generic_visit(node)


class _NameRoles:
    def __init__(self) -> None:
        self._roles: dict[str, str] = {}
        self._counts: Counter[str] = Counter()

    def role(self, name: str, kind: str) -> str:
        key = f"{kind}:{name}"
        if key not in self._roles:
            self._counts[kind] += 1
            self._roles[key] = f"{kind}_{self._counts[kind]}"
        return self._roles[key]


def _normalize_node(node: ast.AST | list[ast.AST] | Any, roles: _NameRoles) -> Any:
    if isinstance(node, list):
        return [_normalize_node(child, roles) for child in _sorted_nodes(node)]
    if not isinstance(node, ast.AST):
        return node
    if isinstance(node, ast.Module):
        return {"body": _normalize_node(node.body, roles), "type": "Module"}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {
            "args": _normalize_node(node.args, roles),
            "body": _normalize_node(node.body, roles),
            "decorator_list": _normalize_node(node.decorator_list, roles),
            "name": roles.role(node.name, "function"),
            "returns": _normalize_node(node.returns, roles),
            "type": type(node).__name__,
        }
    if isinstance(node, ast.ClassDef):
        return {
            "bases": _normalize_node(node.bases, roles),
            "body": _normalize_node(node.body, roles),
            "decorator_list": _normalize_node(node.decorator_list, roles),
            "name": roles.role(node.name, "class"),
            "type": "ClassDef",
        }
    if isinstance(node, ast.arg):
        return {"arg": roles.role(node.arg, "arg"), "type": "arg"}
    if isinstance(node, ast.Name):
        return {"ctx": type(node.ctx).__name__, "id": roles.role(node.id, "var"), "type": "Name"}
    if isinstance(node, ast.Attribute):
        return {
            "attr": node.attr,
            "ctx": type(node.ctx).__name__,
            "type": "Attribute",
            "value": _normalize_node(node.value, roles),
        }
    if isinstance(node, ast.Import):
        return {"names": _normalized_aliases(node.names), "type": "Import"}
    if isinstance(node, ast.ImportFrom):
        return {
            "level": node.level,
            "module": node.module or "",
            "names": _normalized_aliases(node.names),
            "type": "ImportFrom",
        }
    if isinstance(node, ast.alias):
        return {"asname": bool(node.asname), "name": node.name, "type": "alias"}
    if isinstance(node, ast.Constant):
        return {"type": "Constant", "value": _constant_value(node.value)}

    fields: dict[str, Any] = {"type": type(node).__name__}
    for field_name, value in ast.iter_fields(node):
        if field_name in {"col_offset", "ctx", "end_col_offset", "end_lineno", "lineno"}:
            continue
        if field_name == "type_comment":
            continue
        fields[field_name] = _normalize_node(value, roles)
    return fields


def _sorted_nodes(nodes: list[ast.AST]) -> list[ast.AST]:
    imports = [node for node in nodes if isinstance(node, (ast.Import, ast.ImportFrom))]
    others = [node for node in nodes if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return sorted(imports, key=lambda node: _stable_json(_normalize_import_for_sort(node))) + others


def _normalize_import_for_sort(node: ast.AST) -> dict[str, object]:
    if isinstance(node, ast.Import):
        return {"names": _normalized_aliases(node.names), "type": "Import"}
    if isinstance(node, ast.ImportFrom):
        return {
            "level": node.level,
            "module": node.module or "",
            "names": _normalized_aliases(node.names),
            "type": "ImportFrom",
        }
    return {"type": type(node).__name__}


def _normalized_aliases(names: list[ast.alias]) -> list[dict[str, object]]:
    aliases = ({"asname": bool(alias.asname), "name": alias.name} for alias in names)
    return sorted(aliases, key=lambda item: (str(item["name"]), bool(item["asname"])))


def _constant_value(value: object) -> str:
    if isinstance(value, str):
        return "str"
    if isinstance(value, bytes):
        return "bytes"
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float | complex):
        return "number"
    return type(value).__name__


def _call_token(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return "call:name"
    if isinstance(node, ast.Attribute):
        return f"call:attr:{node.attr}"
    if isinstance(node, ast.Call):
        return "call:nested"
    return f"call:{type(node).__name__}"


def _shingles(tokens: list[str]) -> tuple[str, ...]:
    if not tokens:
        return ()
    if len(tokens) < _SHINGLE_SIZE:
        return tuple(sorted(set(tokens)))
    return tuple(
        sorted(
            {
                "|".join(tokens[index : index + _SHINGLE_SIZE])
                for index in range(len(tokens) - _SHINGLE_SIZE + 1)
            }
        )
    )


def _is_eligible_python_entry(entry: ZipManifestEntry) -> bool:
    if not entry.is_python or not entry.read_eligible or entry.is_binary:
        return False
    parts = entry.normalized_path.split("/")
    lowered_parts = {part.lower() for part in parts}
    if lowered_parts & _SKIP_PATH_PARTS:
        return False
    lowered_path = entry.normalized_path.lower()
    return not any(
        lowered_path.endswith(suffix) or any(part.lower().endswith(suffix) for part in parts)
        for suffix in _SKIP_SUFFIXES
    )


def _has_docstring(node: ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    return int(ast.get_docstring(node, clean=False) is not None)


def _feature_record(
    file_path: str,
    feature_key: str,
    feature_type: str,
    feature_value: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "feature_key": f"{file_path}:{feature_key}" if file_path else feature_key,
        "feature_type": feature_type,
        "feature_value": feature_value,
        "file_path": file_path,
        "line_end": None,
        "line_start": None,
        "metadata_json": _stable_json(metadata),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
