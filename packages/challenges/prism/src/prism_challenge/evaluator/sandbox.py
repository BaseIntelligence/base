from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

from .schemas import DeterministicEvidence

ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "math",
    "prism_challenge",
    "torch",
    "typing",
}
NETWORK_IMPORT_ROOTS = {
    "aiohttp",
    "ftplib",
    "http",
    "httpx",
    "requests",
    "socket",
    "smtplib",
    "urllib",
}
FILESYSTEM_IMPORT_ROOTS = {"glob", "pathlib", "shutil", "tempfile"}
PROCESS_IMPORT_ROOTS = {"multiprocessing", "os", "pty", "signal", "subprocess", "sys"}
# Native FFI escapes, dynamic-import machinery, and pickle-family deserialization roots each get a
# specific rule id (architecture.md section 4.1) even though none are on the import allowlist.
FFI_IMPORT_ROOTS = {"_ctypes", "cffi", "ctypes"}
DYNAMIC_IMPORT_ROOTS = {"importlib"}
DESERIALIZATION_IMPORT_ROOTS = {
    "_pickle",
    "cloudpickle",
    "cpickle",
    "dill",
    "joblib",
    "marshal",
    "pickle",
    "shelve",
}
FORBIDDEN_CALL_RULES = {
    "__import__": "prism:no-dynamic-import",
    "compile": "prism:no-dynamic-code",
    "eval": "prism:no-dynamic-code",
    "exec": "prism:no-dynamic-code",
    "input": "prism:no-process",
    "open": "prism:no-filesystem",
}
# getattr/setattr/delattr indirection and namespace introspection.
INDIRECTION_CALL_NAMES = {"getattr", "setattr", "delattr"}
NAMESPACE_INTROSPECTION_CALLS = {"globals", "locals", "vars"}
# Paths the harness controls; torch.load/torch.save are only allowed against these roots so a
# legitimate checkpoint round-trip works while external/untrusted load+save are blocked.
TRUSTED_PATH_ROOTS = {"artifacts_dir", "checkpoint_dir", "resume_checkpoint_dir"}
# Network / native-code escapes reachable THROUGH the otherwise-allowed torch namespace.
BLOCKED_TORCH_PREFIXES = ("torch.hub", "torch.utils.cpp_extension")
DESERIALIZATION_CALL_ATTRS = {"Unpickler", "load", "loads"}
FORBIDDEN_ATTRS = {
    "__bases__",
    "__base__",
    "__builtins__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__import__",
    "__mro__",
    "__reduce__",
    "__reduce_ex__",
    "__subclasses__",
    "__subclasshook__",
    "system",
    "popen",
    "spawn",
    "fork",
    "remove",
    "unlink",
    "rmdir",
}
FORBIDDEN_ATTR_RULES = {
    "__bases__": "prism:no-attribute-escape",
    "__base__": "prism:no-attribute-escape",
    "__builtins__": "prism:no-dynamic-code",
    "__class__": "prism:no-dynamic-code",
    "__closure__": "prism:no-attribute-escape",
    "__code__": "prism:no-attribute-escape",
    "__dict__": "prism:no-dynamic-code",
    "__getattribute__": "prism:no-attribute-escape",
    "__globals__": "prism:no-dynamic-code",
    "__import__": "prism:no-forbidden-import",
    "__mro__": "prism:no-dynamic-code",
    "__reduce__": "prism:no-attribute-escape",
    "__reduce_ex__": "prism:no-attribute-escape",
    "__subclasses__": "prism:no-dynamic-code",
    "__subclasshook__": "prism:no-attribute-escape",
    "fork": "prism:no-process",
    "popen": "prism:no-process",
    "remove": "prism:no-filesystem",
    "rmdir": "prism:no-filesystem",
    "spawn": "prism:no-process",
    "system": "prism:no-process",
    "unlink": "prism:no-filesystem",
}
# Symbol names that must not be reached through getattr/string-built indirection.
BLOCKED_INDIRECTION_NAMES = (
    FORBIDDEN_ATTRS
    | set(FORBIDDEN_CALL_RULES)
    | {
        "Unpickler",
        "attrgetter",
        "cpp_extension",
        "hub",
        "import_module",
        "itemgetter",
        "load",
        "loads",
        "methodcaller",
        "save",
    }
)


@dataclass(frozen=True)
class SandboxReport:
    tree: ast.AST
    ast_fingerprint: set[str]
    imports: set[str]
    deterministic_evidence: tuple[DeterministicEvidence, ...] = ()


class SandboxViolation(ValueError):
    def __init__(
        self,
        message: str,
        evidence: DeterministicEvidence | tuple[DeterministicEvidence, ...] | None = None,
    ) -> None:
        super().__init__(message)
        if evidence is None:
            self.evidence: tuple[DeterministicEvidence, ...] = ()
        elif isinstance(evidence, DeterministicEvidence):
            self.evidence = (evidence,)
        else:
            self.evidence = evidence

    def evidence_payload(self) -> list[dict[str, Any]]:
        return [item.model_dump() for item in self.evidence]


def inspect_code(
    code: str,
    *,
    require_contract: bool = False,
    allowed_import_roots: set[str] | None = None,
    artifact_path: str = "model.py",
) -> SandboxReport:
    # ``require_contract`` is retained for call-site compatibility but no longer enforces the
    # legacy single-module ``build_model``/``get_recipe`` contract: the two-script contract is the
    # only loading path now and is validated in ``components.py::project_components``.
    _ = require_contract
    tree = ast.parse(code)
    imports: set[str] = set()
    fingerprint: set[str] = set()
    # A torch.load/save path is trusted only when anchored at a harness-provided dir whose name was
    # NOT rebound by the submission; a shadowed trusted name no longer refers to the harness dir.
    trusted_path_roots = TRUSTED_PATH_ROOTS - _shadowed_trusted_roots(tree)
    for node in ast.walk(tree):
        fingerprint.add(type(node).__name__)
        if isinstance(node, ast.ClassDef):
            fingerprint.add(f"class_base:{','.join(_node_name(base) for base in node.bases)}")
        elif isinstance(node, ast.FunctionDef):
            fingerprint.add(f"function:{node.name}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_RULES:
                raise SandboxViolation(
                    f"forbidden call: {node.func.id}",
                    _evidence(
                        code,
                        node,
                        artifact_path=artifact_path,
                        rule_id=FORBIDDEN_CALL_RULES[node.func.id],
                        explanation=f"call {node.func.id} is not allowed in Prism submissions",
                    ),
                )
            _check_builtin_indirection(node, code, artifact_path=artifact_path)
            fingerprint.add(f"call:{node.func.id}")
        elif isinstance(node, ast.Call):
            _check_dotted_call(
                node, code, artifact_path=artifact_path, trusted_roots=trusted_path_roots
            )
            fingerprint.add(f"call:{_node_name(node.func)}")
        if isinstance(node, ast.Attribute):
            _check_attribute(node, code, artifact_path=artifact_path)
    blocked = imports - ALLOWED_IMPORT_ROOTS - (allowed_import_roots or set())
    if blocked:
        evidence = tuple(
            _import_evidence(code, tree, root, artifact_path=artifact_path)
            for root in sorted(blocked)
        )
        raise SandboxViolation(f"forbidden imports: {', '.join(sorted(blocked))}", evidence)
    return SandboxReport(tree=tree, ast_fingerprint=fingerprint, imports=imports)


def _import_evidence(
    code: str, tree: ast.AST, root: str, *, artifact_path: str
) -> DeterministicEvidence:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] == root for alias in node.names):
                return _evidence(
                    code,
                    node,
                    artifact_path=artifact_path,
                    rule_id=_import_rule_id(root),
                    explanation=f"import root {root} is not allowed in Prism submissions",
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] == root:
                return _evidence(
                    code,
                    node,
                    artifact_path=artifact_path,
                    rule_id=_import_rule_id(root),
                    explanation=f"import root {root} is not allowed in Prism submissions",
                )
    return _synthetic_evidence(
        rule_id=_import_rule_id(root),
        artifact_path=artifact_path,
        ast_node="Import",
        basis=root,
        explanation=f"import root {root} is not allowed in Prism submissions",
    )


def _import_rule_id(root: str) -> str:
    if root in NETWORK_IMPORT_ROOTS:
        return "prism:no-network"
    if root in FILESYSTEM_IMPORT_ROOTS:
        return "prism:no-filesystem"
    if root in PROCESS_IMPORT_ROOTS:
        return "prism:no-process"
    if root in FFI_IMPORT_ROOTS:
        return "prism:no-ffi"
    if root in DYNAMIC_IMPORT_ROOTS:
        return "prism:no-dynamic-import"
    if root in DESERIALIZATION_IMPORT_ROOTS:
        return "prism:no-deserialization"
    return "prism:no-forbidden-import"


def _check_attribute(node: ast.Attribute, code: str, *, artifact_path: str) -> None:
    if node.attr in FORBIDDEN_ATTRS:
        raise SandboxViolation(
            f"forbidden attribute: {node.attr}",
            _evidence(
                code,
                node,
                artifact_path=artifact_path,
                rule_id=FORBIDDEN_ATTR_RULES[node.attr],
                explanation=f"attribute {node.attr} is not allowed in Prism submissions",
            ),
        )
    dotted = _node_name(node)
    for prefix in BLOCKED_TORCH_PREFIXES:
        if dotted == prefix or dotted.startswith(f"{prefix}."):
            raise SandboxViolation(
                f"forbidden torch escape: {dotted}",
                _evidence(
                    code,
                    node,
                    artifact_path=artifact_path,
                    rule_id="prism:no-torch-escape",
                    explanation=(
                        f"{dotted} reaches the network or native code through the torch namespace"
                    ),
                ),
            )


def _check_dotted_call(
    node: ast.Call, code: str, *, artifact_path: str, trusted_roots: set[str]
) -> None:
    dotted = _node_name(node.func)
    if "." not in dotted:
        return
    root, _, _ = dotted.partition(".")
    last = dotted.rsplit(".", 1)[-1]
    for prefix in BLOCKED_TORCH_PREFIXES:
        if dotted == prefix or dotted.startswith(f"{prefix}."):
            raise _sandbox_call_violation(
                code,
                node,
                artifact_path=artifact_path,
                rule_id="prism:no-torch-escape",
                message=f"forbidden torch escape: {dotted}",
                explanation=(
                    f"{dotted} reaches the network or native code through the torch namespace"
                ),
            )
    if root in DYNAMIC_IMPORT_ROOTS:
        raise _sandbox_call_violation(
            code,
            node,
            artifact_path=artifact_path,
            rule_id="prism:no-dynamic-import",
            message=f"forbidden dynamic import: {dotted}",
            explanation=f"{dotted} performs a dynamic import",
        )
    if root == "operator" and last in {"attrgetter", "itemgetter", "methodcaller"}:
        raise _sandbox_call_violation(
            code,
            node,
            artifact_path=artifact_path,
            rule_id="prism:no-dynamic-attr",
            message=f"forbidden indirection: {dotted}",
            explanation=f"{dotted} is a string-built attribute indirection",
        )
    if root in DESERIALIZATION_IMPORT_ROOTS and last in DESERIALIZATION_CALL_ATTRS:
        raise _sandbox_call_violation(
            code,
            node,
            artifact_path=artifact_path,
            rule_id="prism:no-deserialization",
            message=f"forbidden deserialization: {dotted}",
            explanation=f"{dotted} deserializes untrusted data",
        )
    if dotted == "torch.load" and not _call_path_is_trusted(
        node, index=0, trusted_roots=trusted_roots
    ):
        raise _sandbox_call_violation(
            code,
            node,
            artifact_path=artifact_path,
            rule_id="prism:no-deserialization",
            message="forbidden deserialization: torch.load of an external/untrusted path",
            explanation="torch.load may only read from the harness checkpoint/artifacts path",
        )
    if dotted == "torch.save" and not _call_path_is_trusted(
        node, index=1, trusted_roots=trusted_roots
    ):
        raise _sandbox_call_violation(
            code,
            node,
            artifact_path=artifact_path,
            rule_id="prism:no-filesystem",
            message="forbidden filesystem write: torch.save outside artifacts_dir",
            explanation="torch.save may only write under the harness checkpoint/artifacts path",
        )


def _check_builtin_indirection(node: ast.Call, code: str, *, artifact_path: str) -> None:
    name = node.func.id if isinstance(node.func, ast.Name) else ""
    if name in NAMESPACE_INTROSPECTION_CALLS:
        raise _sandbox_call_violation(
            code,
            node,
            artifact_path=artifact_path,
            rule_id="prism:no-dynamic-attr",
            message=f"forbidden indirection: {name}()",
            explanation=f"{name}() namespace introspection is not allowed",
        )
    if name not in INDIRECTION_CALL_NAMES:
        return
    if node.args and _references_builtins(node.args[0]):
        raise _sandbox_call_violation(
            code,
            node,
            artifact_path=artifact_path,
            rule_id="prism:no-dynamic-attr",
            message=f"forbidden indirection: {name} on the builtins namespace",
            explanation=f"{name} reaches the builtins namespace",
        )
    if len(node.args) >= 2:
        attr = _const_str(node.args[1])
        if attr is None:
            raise _sandbox_call_violation(
                code,
                node,
                artifact_path=artifact_path,
                rule_id="prism:no-dynamic-attr",
                message=f"forbidden indirection: {name} with a dynamic attribute name",
                explanation=f"{name} uses a string-built (non-literal) attribute name",
            )
        if attr in BLOCKED_INDIRECTION_NAMES:
            raise _sandbox_call_violation(
                code,
                node,
                artifact_path=artifact_path,
                rule_id="prism:no-dynamic-attr",
                message=f"forbidden indirection: {name} to blocked symbol {attr!r}",
                explanation=f"{name} reaches the blocked symbol {attr!r}",
            )


def _sandbox_call_violation(
    code: str,
    node: ast.AST,
    *,
    artifact_path: str,
    rule_id: str,
    message: str,
    explanation: str,
) -> SandboxViolation:
    return SandboxViolation(
        message,
        _evidence(
            code,
            node,
            artifact_path=artifact_path,
            rule_id=rule_id,
            explanation=explanation,
        ),
    )


def _call_path_is_trusted(node: ast.Call, *, index: int, trusted_roots: set[str]) -> bool:
    if len(node.args) <= index:
        return False
    return _path_is_trusted(node.args[index], trusted_roots)


def _shadowed_trusted_roots(tree: ast.AST) -> set[str]:
    """Trusted-path roots that the submission rebinds (shadows) anywhere in the code unit.

    The name-only trusted check would otherwise trust ``torch.load(artifacts_dir)`` even after
    ``artifacts_dir = "/some/external/path"`` (or ``ctx.artifacts_dir = ...``). Any assignment-style
    bind of a trusted root (``Store`` context on a matching ``Name``/``Attribute``) means that name
    no longer reliably refers to the harness dir, so it must stop being treated as trusted. A
    dynamic ``setattr(obj, "artifacts_dir", ...)`` / ``delattr(obj, "artifacts_dir")`` rebind is not
    a ``Store`` node, so the trust model would otherwise be static-name-only; detect it here too so
    a trusted root reached through a ``setattr``-laundered attribute also drops out of the trusted
    set for the whole code unit.
    """
    shadowed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id in TRUSTED_PATH_ROOTS:
                shadowed.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if node.attr in TRUSTED_PATH_ROOTS:
                shadowed.add(node.attr)
        elif isinstance(node, ast.Call):
            rebound = _setattr_rebound_trusted_root(node)
            if rebound is not None:
                shadowed.add(rebound)
    return shadowed


def _setattr_rebound_trusted_root(node: ast.Call) -> str | None:
    """The trusted-path root rebound by a ``setattr``/``delattr`` call, if any.

    ``setattr(ctx, "artifacts_dir", "/ext")`` / ``delattr(ctx, "artifacts_dir")`` dynamically
    rebinds a trusted root without a ``Store`` AST node, so a later
    ``torch.load(ctx.artifacts_dir)`` must not be trusted on the strength of the static name alone.
    Only a constant-string attribute that names a trusted root is matched (a non-literal attribute
    name is already blocked as a dynamic indirection by ``_check_builtin_indirection``).
    """
    if not (isinstance(node.func, ast.Name) and node.func.id in {"setattr", "delattr"}):
        return None
    if len(node.args) < 2:
        return None
    attr = _const_str(node.args[1])
    if attr in TRUSTED_PATH_ROOTS:
        return attr
    return None


def _path_is_trusted(node: ast.AST, trusted_roots: set[str]) -> bool:
    """A torch.load/save path is trusted only when ANCHORED at a non-shadowed trusted root.

    The path expression must START at a harness-provided trusted dir (a ``Name``/``Attribute`` whose
    name is a non-shadowed trusted root), following ``a / "x"`` / ``a + "/x"`` / ``a[...]`` path
    builders down to their leftmost base. A path that merely MENTIONS a trusted name inside an
    unrelated sub-expression (e.g. ``"/etc/passwd" + str(len(artifacts_dir))``) or that anchors at a
    constant/other variable is NOT trusted, so a variable-built or shadowed path cannot launder an
    external load/save past the gate.
    """
    current = node
    while True:
        if isinstance(current, ast.Name):
            return current.id in trusted_roots
        if isinstance(current, ast.Attribute):
            return current.attr in trusted_roots
        if isinstance(current, ast.BinOp):
            current = current.left
            continue
        if isinstance(current, ast.Subscript):
            current = current.value
            continue
        return False


def _references_builtins(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"__builtins__", "builtins"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"__builtins__", "builtins"}
    return False


def _const_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _const_str(node.left)
        right = _const_str(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    return None


def _evidence(
    code: str,
    node: ast.AST,
    *,
    artifact_path: str,
    rule_id: str,
    explanation: str,
) -> DeterministicEvidence:
    snippet = ast.get_source_segment(code, node) or _line_at(code, getattr(node, "lineno", 1))
    return DeterministicEvidence(
        rule_id=rule_id,
        artifact_path=artifact_path,
        line=getattr(node, "lineno", None),
        ast_node=type(node).__name__,
        snippet_hash=_sha256(snippet),
        explanation=explanation,
    )


def _synthetic_evidence(
    *, rule_id: str, artifact_path: str, ast_node: str, basis: str, explanation: str
) -> DeterministicEvidence:
    return DeterministicEvidence(
        rule_id=rule_id,
        artifact_path=artifact_path,
        ast_node=ast_node,
        snippet_hash=_sha256(basis),
        explanation=explanation,
    )


def _line_at(code: str, line: int) -> str:
    lines = code.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _node_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _node_name(node.value)
    return type(node).__name__
