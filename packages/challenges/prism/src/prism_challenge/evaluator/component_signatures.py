from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from .components import PrismComponentFingerprints, PrismProjectComponents
from .schemas import ArchitectureGraph
from .source_similarity import SourceFile

HOOK_NAMES = (
    "configure_optimizer",
    "inference_logits",
    "infer",
    "compute_loss",
    "train_step",
)


@dataclass(frozen=True)
class ComponentSemanticSignature:
    project_kind: str
    family_hash: str
    arch_fingerprint: str
    behavior_fingerprint: str
    architecture_graph_hash: str
    training_hash: str
    hook_metadata: dict[str, Any]
    architecture_graph: dict[str, Any]
    training_graph: dict[str, Any]
    mermaid: str
    architecture_summary: str
    training_summary: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "project_kind": self.project_kind,
            "family_hash": self.family_hash,
            "arch_fingerprint": self.arch_fingerprint,
            "behavior_fingerprint": self.behavior_fingerprint,
            "architecture_graph_hash": self.architecture_graph_hash,
            "training_hash": self.training_hash,
            "hook_metadata": self.hook_metadata,
            "architecture_graph": self.architecture_graph,
            "training_graph": self.training_graph,
            "mermaid": self.mermaid,
            "architecture_summary": self.architecture_summary,
            "training_summary": self.training_summary,
        }


def build_semantic_signature(
    components: PrismProjectComponents,
    fingerprints: PrismComponentFingerprints,
) -> ComponentSemanticSignature:
    architecture_graph = _graph_for_files(components.architecture_files)
    training_graph = _graph_for_files(components.training_files)
    hook_metadata = _hook_metadata(components)
    return ComponentSemanticSignature(
        project_kind=components.kind,
        family_hash=fingerprints.family_hash,
        arch_fingerprint=fingerprints.arch_fingerprint,
        behavior_fingerprint=_graph_hash(architecture_graph),
        architecture_graph_hash=_graph_hash(architecture_graph),
        training_hash=fingerprints.training_hash,
        hook_metadata=hook_metadata,
        architecture_graph=architecture_graph,
        training_graph=training_graph,
        mermaid=_architecture_mermaid(architecture_graph),
        architecture_summary=_summary("architecture", architecture_graph),
        training_summary=_summary("training", training_graph),
    )


def semantic_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def _graph_for_files(files: tuple[SourceFile, ...]) -> dict[str, Any]:
    classes: set[str] = set()
    functions: set[str] = set()
    imports: set[str] = set()
    calls: set[str] = set()
    modules: set[str] = set()
    parameterized_blocks: list[dict[str, Any]] = []
    interface: dict[str, Any] = {"required": [], "optional_hooks": []}
    for file in files:
        if not file.path.endswith(".py"):
            continue
        modules.add(_stable_name(file.path))
        try:
            tree = ast.parse(file.content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.add(_normalized_symbol(node.name))
                parameterized_blocks.extend(_class_parameterized_blocks(node))
            elif isinstance(node, ast.FunctionDef):
                functions.add(node.name)
                if node.name in {"build_model", "get_recipe"}:
                    interface["required"].append(node.name)
                elif node.name in HOOK_NAMES:
                    interface["optional_hooks"].append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Call):
                name = _node_name(node.func)
                if name:
                    calls.add(_normalized_call(name))
    graph = ArchitectureGraph(
        modules=sorted(modules),
        classes=sorted(classes),
        functions=sorted(functions),
        imports=sorted(imports),
        calls=sorted(calls),
        parameterized_blocks=sorted(parameterized_blocks, key=lambda item: item["name"]),
        tokenizer_constraints={"source": "submission_contract"},
        dynamic_routing={"enabled": _has_dynamic_routing(calls)},
        interface={
            "required": sorted(set(interface["required"])),
            "optional_hooks": sorted(set(interface["optional_hooks"])),
        },
    )
    return graph.model_dump()


def _class_parameterized_blocks(node: ast.ClassDef) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        call = child.value
        if not isinstance(call, ast.Call):
            continue
        call_name = _node_name(call.func)
        if not call_name or not any(
            token in call_name
            for token in ("Embedding", "Linear", "Conv", "Attention", "LayerNorm")
        ):
            continue
        target = ".".join(_node_name(target) for target in child.targets if _node_name(target))
        blocks.append(
            {
                "name": _safe_label(target or call_name),
                "kind": _normalized_call(call_name),
                "args": len(call.args),
                "keywords": sorted(keyword.arg for keyword in call.keywords if keyword.arg),
            }
        )
    return blocks


def _has_dynamic_routing(calls: set[str]) -> bool:
    routing_tokens = ("topk", "where", "gather", "scatter", "router", "moe")
    return any(any(token in call.lower() for token in routing_tokens) for call in calls)


def _hook_metadata(components: PrismProjectComponents) -> dict[str, Any]:
    defined: dict[str, list[str]] = {name: [] for name in HOOK_NAMES}
    for file in (*components.architecture_files, *components.training_files):
        if not file.path.endswith(".py"):
            continue
        try:
            tree = ast.parse(file.content)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in defined:
                defined[node.name].append(file.path)
    present = {name: paths for name, paths in defined.items() if paths}
    inference_hook = "inference_logits" if present.get("inference_logits") else None
    if inference_hook is None and present.get("infer"):
        inference_hook = "infer"
    return {
        "present": sorted(present),
        "locations": present,
        "inference_hook": inference_hook,
        "custom_optimizer": bool(present.get("configure_optimizer")),
        "custom_loss": bool(present.get("compute_loss")),
        "custom_train_step": bool(present.get("train_step")),
    }


def _architecture_mermaid(graph: dict[str, Any]) -> str:
    lines = ["flowchart LR"]
    classes = list(graph.get("classes") or [])[:8]
    functions = [
        item for item in graph.get("functions") or [] if item in {"build_model", "get_recipe"}
    ]
    if not classes and not functions:
        lines.append('    Code["Python project"]')
        return "\n".join(lines)
    for index, cls in enumerate(classes):
        lines.append(f'    C{index}["{_safe_label(cls)}"]')
    for index, fn in enumerate(functions):
        lines.append(f'    F{index}["{_safe_label(fn)}"]')
    if classes and functions:
        for index in range(len(functions)):
            lines.append(f"    F{index} --> C0")
    return "\n".join(lines)


def _summary(scope: str, graph: dict[str, Any]) -> str:
    classes = ", ".join((graph.get("classes") or [])[:8]) or "none"
    functions = ", ".join((graph.get("functions") or [])[:12]) or "none"
    calls = ", ".join((graph.get("calls") or [])[:16]) or "none"
    return f"{scope}: classes={classes}; functions={functions}; calls={calls}"


def _semantic_tokens(graph: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for key in ("classes", "functions", "imports", "calls"):
        for item in graph.get(key) or []:
            out.add(f"{key}:{item}")
    return out


def _graph_hash(graph: dict[str, Any]) -> str:
    return ArchitectureGraph.model_validate(graph).sha256()


def _stable_name(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".py")


def _normalized_symbol(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "", value.lower())
    for token in ("tiny", "small", "model", "module", "block", "net", "network"):
        lowered = lowered.replace(token, "")
    return lowered or value.lower()


def _normalized_call(value: str) -> str:
    parts = value.split(".")
    if len(parts) > 1 and parts[0] in {"self", "torch", "nn"}:
        return ".".join(parts[-2:])
    return value


def _safe_label(value: str) -> str:
    return value.replace('"', "'")[:40]


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _node_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return _node_name(node.func)
    if isinstance(node, ast.Subscript):
        return _node_name(node.value)
    return ""
