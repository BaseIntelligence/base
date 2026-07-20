"""Multi-GPU static contract checks (architecture.md section 8).

The miner owns the training loop and therefore owns multi-GPU scaling. The challenge launches
``torchrun --standalone --nnodes=1 --nproc-per-node=<gpus>`` (scored run: nproc=1) and the
training script MUST use the distributed primitives so the same script degrades cleanly to
``world_size==1`` and scales to a single node of up to 8 GPUs. With only one physical GPU the
distributed code path is validated statically here (Gate A) and functionally with a gloo
multi-rank test elsewhere (Gate B).

Gate A statically verifies (over ``training.py`` only) that the script references the required
distributed primitives -- ``init_process_group``, a device binding (``set_device(local_rank)`` or
an explicit cuda device), a DDP/FSDP model wrap, per-rank data sharding (``DistributedSampler`` or
rank-strided windows), a rank-0 write guard, and ``destroy_process_group`` -- and that any
checkpoint/manifest write is guarded by a ``rank == 0`` condition. The single-node bound rejects a
``gpu_count > 8`` or multi-node request before any launch. Enforcement is policy-driven
(``reject``/``flag``/``off``); the production default rejects so a non-distributed script never
reaches the GPU.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from .sandbox import SandboxViolation, _node_name, _synthetic_evidence

logger = logging.getLogger(__name__)

DISTRIBUTED_CONTRACT_RULE = "prism:distributed-contract"
RANK0_GUARD_RULE = "prism:distributed-rank0-guard"
SINGLE_NODE_RULE = "prism:single-node-bound"

DEFAULT_MAX_GPU_COUNT = 8

REQUIRED_PRIMITIVES = (
    "init_process_group",
    "device_binding",
    "ddp_or_fsdp",
    "data_sharding",
    "rank0_guard",
    "destroy_process_group",
)
_PRIMITIVE_LABELS = {
    "init_process_group": "init_process_group",
    "device_binding": "set_device(local_rank)/device binding",
    "ddp_or_fsdp": "DDP or FSDP model wrap",
    "data_sharding": "per-rank data sharding (DistributedSampler or rank-strided windows)",
    "rank0_guard": "rank-0 write guard",
    "destroy_process_group": "destroy_process_group",
}

# Names a compliant script uses to identify the rank / world size, by AST node kind.
_RANK_NAME_IDS = {"rank", "local_rank", "global_rank", "RANK", "LOCAL_RANK", "GLOBAL_RANK"}
_RANK_ATTRS = {"rank", "local_rank", "global_rank"}
_RANK_CALLS = {"get_rank", "get_local_rank"}
_WORLD_NAME_IDS = {"world_size", "WORLD_SIZE", "num_replicas", "world"}
_WORLD_ATTRS = {"world_size", "num_replicas"}
_PARALLEL_WRAP_NAMES = {"DistributedDataParallel", "FullyShardedDataParallel"}
_PARALLEL_WRAP_ALIASES = {"DDP", "FSDP"}
_DEVICE_CUDA_FUNCS = {"device", "to"}


@dataclass(frozen=True)
class DistributedContractReport:
    compliant: bool
    missing: tuple[str, ...] = ()
    unguarded_writes: int = 0
    present: frozenset[str] = field(default_factory=frozenset)


def check_distributed_contract(
    code: str,
    *,
    artifact_path: str = "training.py",
    policy: str = "reject",
) -> DistributedContractReport:
    """Statically verify the multi-GPU contract over the training script.

    Returns a :class:`DistributedContractReport`. When the script is non-compliant the behavior
    depends on ``policy``: ``reject`` raises a :class:`SandboxViolation` (the worker turns it into
    a ``rejected`` submission before any GPU work), ``flag`` logs and returns the report without
    raising, and ``off`` skips the check entirely.
    """
    if policy == "off":
        return DistributedContractReport(compliant=True)

    tree = ast.parse(code)
    present = _detect_primitives(tree)
    unguarded_writes = _count_unguarded_writes(tree)
    missing = tuple(name for name in REQUIRED_PRIMITIVES if name not in present)
    compliant = not missing and unguarded_writes == 0
    report = DistributedContractReport(
        compliant=compliant,
        missing=missing,
        unguarded_writes=unguarded_writes,
        present=frozenset(present),
    )
    if compliant or policy == "flag":
        if not compliant:
            logger.warning(
                "training script %s violates the multi-GPU contract: "
                "missing=%s unguarded_writes=%d",
                artifact_path,
                missing,
                unguarded_writes,
            )
        return report

    non_guard_missing = [name for name in missing if name != "rank0_guard"]
    if non_guard_missing:
        labels = ", ".join(_PRIMITIVE_LABELS[name] for name in non_guard_missing)
        raise SandboxViolation(
            f"training script is missing required distributed primitives: {labels}",
            _synthetic_evidence(
                rule_id=DISTRIBUTED_CONTRACT_RULE,
                artifact_path=artifact_path,
                ast_node="Module",
                basis=f"{artifact_path}:missing:{','.join(non_guard_missing)}",
                explanation=(
                    "the multi-GPU contract requires the training loop to use the distributed "
                    "primitives (init_process_group, device binding, DDP/FSDP wrap, per-rank data "
                    "sharding, rank-0 write guard, destroy_process_group)"
                ),
            ),
        )
    raise SandboxViolation(
        "training script must guard checkpoint/manifest writes with a rank == 0 condition "
        "(missing rank-0 write guard)",
        _synthetic_evidence(
            rule_id=RANK0_GUARD_RULE,
            artifact_path=artifact_path,
            ast_node="Module",
            basis=f"{artifact_path}:rank0-guard",
            explanation="checkpoint/manifest writes must run only on rank 0",
        ),
    )


def enforce_single_node_bound(
    gpu_count: object,
    *,
    num_nodes: object = None,
    max_gpu_count: int = DEFAULT_MAX_GPU_COUNT,
    artifact_path: str = "prism.yaml",
) -> None:
    """Reject a multi-node config or a ``gpu_count`` above the single-node bound (max 8).

    ``gpu_count``/``num_nodes`` may be ``None`` (no explicit request) or any value coercible to an
    int. A non-coercible value is ignored here (the scheduler enforces its own bounds). Raises a
    :class:`SandboxViolation` for a multi-node request or ``gpu_count > max_gpu_count``.
    """
    nodes = _as_int(num_nodes)
    if nodes is not None and nodes > 1:
        raise SandboxViolation(
            f"multi-node distributed config is rejected (num_nodes={nodes}); single-node only",
            _single_node_evidence(artifact_path, f"num_nodes={nodes}"),
        )
    count = _as_int(gpu_count)
    if count is not None and count > max_gpu_count:
        raise SandboxViolation(
            f"requested gpu_count {count} exceeds the single-node bound of {max_gpu_count}",
            _single_node_evidence(artifact_path, f"gpu_count={count}"),
        )


def _single_node_evidence(artifact_path: str, basis: str):
    return _synthetic_evidence(
        rule_id=SINGLE_NODE_RULE,
        artifact_path=artifact_path,
        ast_node="Module",
        basis=f"{artifact_path}:{basis}",
        explanation="only a single node of up to 8 GPUs is supported",
    )


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (str, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _detect_primitives(tree: ast.AST) -> set[str]:
    present: set[str] = set()
    wrap_aliases = _parallel_wrap_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            _inspect_call(node, present, wrap_aliases)
        elif isinstance(node, ast.Name):
            if node.id in _PARALLEL_WRAP_NAMES or node.id in wrap_aliases:
                present.add("ddp_or_fsdp")
            if node.id == "DistributedSampler":
                present.add("data_sharding")
        elif isinstance(node, ast.Attribute):
            if node.attr in _PARALLEL_WRAP_NAMES:
                present.add("ddp_or_fsdp")
            if node.attr == "DistributedSampler":
                present.add("data_sharding")
        elif isinstance(node, ast.Subscript) and _is_rank_strided(node.slice):
            present.add("data_sharding")
        elif isinstance(node, ast.If) and _is_rank0_test(node.test):
            present.add("rank0_guard")
    return present


def _inspect_call(node: ast.Call, present: set[str], wrap_aliases: set[str]) -> None:
    last = _node_name(node.func).rsplit(".", 1)[-1]
    if last == "init_process_group":
        present.add("init_process_group")
    elif last == "destroy_process_group":
        present.add("destroy_process_group")
    elif last in {"set_device", "cuda"}:
        present.add("device_binding")
    elif last in _DEVICE_CUDA_FUNCS and _has_cuda_arg(node):
        present.add("device_binding")
    if last == "DistributedSampler":
        present.add("data_sharding")
    elif last == "range" and _is_rank_strided(node):
        present.add("data_sharding")
    if last in _PARALLEL_WRAP_NAMES or last in _PARALLEL_WRAP_ALIASES or last in wrap_aliases:
        present.add("ddp_or_fsdp")


def _parallel_wrap_aliases(tree: ast.AST) -> set[str]:
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _PARALLEL_WRAP_NAMES:
                    aliases.add(alias.asname or alias.name)
    return aliases


def _has_cuda_arg(node: ast.Call) -> bool:
    for arg in (*node.args, *(kw.value for kw in node.keywords)):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "cuda" in arg.value:
            return True
    return False


def _is_rank_strided(node: ast.AST) -> bool:
    return _refs_rank(node) and _refs_world(node)


def _refs_rank(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _RANK_NAME_IDS:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _RANK_ATTRS:
            return True
    return False


def _refs_world(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in _WORLD_NAME_IDS:
            return True
        if isinstance(sub, ast.Attribute) and sub.attr in _WORLD_ATTRS:
            return True
    return False


def _is_rank0_test(test: ast.AST) -> bool:
    if isinstance(test, ast.BoolOp):
        return any(_is_rank0_test(value) for value in test.values)
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        left, right = test.left, test.comparators[0]
        return (_is_rank_ref(left) and _is_zero(right)) or (_is_rank_ref(right) and _is_zero(left))
    return False


def _is_rank_ref(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _RANK_NAME_IDS
    if isinstance(node, ast.Attribute):
        return node.attr in _RANK_ATTRS
    if isinstance(node, ast.Call):
        return _node_name(node.func).rsplit(".", 1)[-1] in _RANK_CALLS
    return False


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0 and not isinstance(node.value, bool)


def _is_torch_save(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _node_name(node.func) == "torch.save"


def _count_unguarded_writes(tree: ast.AST) -> int:
    counter = _UnguardedWriteCounter()
    counter.visit(tree)
    return counter.unguarded


class _UnguardedWriteCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.unguarded = 0
        self._guard_depth = 0

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        guarded = _is_rank0_test(node.test)
        if guarded:
            self._guard_depth += 1
        for child in node.body:
            self.visit(child)
        if guarded:
            self._guard_depth -= 1
        for child in node.orelse:
            self.visit(child)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_torch_save(node) and self._guard_depth == 0:
            self.unguarded += 1
        self.generic_visit(node)
