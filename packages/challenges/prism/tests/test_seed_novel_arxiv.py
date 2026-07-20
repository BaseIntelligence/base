"""Novel arXiv-class seed package gates (VAL-ARXEVAL-002/003).

Targeted lab surface only: ≥3 shortlist keepers (DeepLoop + gated-delta + hybrid
attn×SSM) pack under the shared harness, pass AST sandbox (no blocked natives),
forced-seed param count under explore 124M (~1–5M thrash preferred), multi-GPU
single-node static contract, and Imp baselines still pack.
"""

from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import pytest

from prism_challenge.evaluator.distributed_contract import (
    DEFAULT_MAX_GPU_COUNT,
    check_distributed_contract,
    enforce_single_node_bound,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code
from prism_challenge.evaluator.static_instantiation import check_build_model_static
from prism_challenge.seed_packaging import (
    REQUIRED_ENTRY_SCRIPTS,
    SEED_FAMILIES,
    collect_seed_files,
    get_family,
    list_families,
    package_all_families,
    package_seed_zip,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
# Explore ladder hard cap (VAL-RESLAB-003 / VAL-ARXEVAL-002).
MAX_PARAMS = 124_000_000
# Soft thrash band preferred for fair-eval packages (~1–5M).
SOFT_THRASH_MAX = 5_000_000

# Shortlist keepers that MUST ship (DeepLoop mandatory when shortlist kept it).
NOVEL_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("deeploop-tiny-1m", "deeploop-tiny", "deeploop"),
    ("gated-delta-tiny-1m", "gated-delta-tiny", "gated_delta"),
    ("hybrid-attn-ssm-tiny-1m", "hybrid-attn-ssm-tiny", "hybrid_attn_ssm"),
)
IMP_FAMILIES: tuple[tuple[str, str], ...] = (
    ("transformer-tiny-1m", "tiny-1m"),
    ("mamba-tiny-1m", "mamba-tiny"),
)

_BLOCKED_IMPORT_ROOTS = frozenset(
    {
        "mamba_ssm",
        "causal_conv1d",
        "flash_attn",
        "flash_linear_attn",
        "ctypes",
        "cffi",
        "importlib",
    }
)


def _imported_roots(source: str) -> set[str]:
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".", 1)[0])
            for alias in node.names:
                if node.module and "cpp_extension" in f"{node.module}.{alias.name}":
                    roots.add("cpp_extension")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain: list[str] = []
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                chain.append(cur.id)
            dotted = ".".join(reversed(chain))
            if dotted.startswith("torch.utils.cpp_extension") or dotted == "cpp_extension":
                roots.add("cpp_extension")
    return roots


def _read_scripts(root: Path) -> tuple[str, str]:
    return (
        (root / "architecture.py").read_text(encoding="utf-8"),
        (root / "training.py").read_text(encoding="utf-8"),
    )


# --- VAL-ARXEVAL-002: ≥3 novel packs under contract --------------------------------------------


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    NOVEL_FAMILIES,
    ids=[row[0] for row in NOVEL_FAMILIES],
)
def test_novel_seed_package_complete_and_submit_shaped(
    family_id: str, dirname: str, arch_family: str
) -> None:
    """Each shortlist keeper packs as a two-script zip under the shared harness."""
    root = EXAMPLES / dirname
    assert root.is_dir(), f"missing seed tree {root}"
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert (root / name).is_file(), f"{family_id} missing {name}"
    assert (root / "prism.yaml").is_file()
    assert (root / "README.md").is_file()
    assert family_id in SEED_FAMILIES
    assert family_id in list_families()
    family = get_family(family_id)
    assert family.architecture_family == arch_family
    assert family.source_dir.resolve() == root.resolve()

    files = collect_seed_files(root)
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert name in files

    out = package_seed_zip(family_id, REPO_ROOT / "dist" / "seed-packages-novel-test")
    assert out.zip_path.is_file()
    assert out.size_bytes > 0
    assert len(out.content_sha256) == 64
    with zipfile.ZipFile(out.zip_path) as archive:
        names = set(archive.namelist())
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert name in names
    assert "prism.yaml" in names or "prism.yml" in names
    assert not any(n.endswith((".pem", ".key", ".env", ".pt", ".bin")) for n in names)


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    NOVEL_FAMILIES,
    ids=[row[0] for row in NOVEL_FAMILIES],
)
def test_novel_seed_passes_ast_sandbox_pure_torch(
    family_id: str, dirname: str, arch_family: str
) -> None:
    """AST allowlist pass with no blocked native extension imports."""
    del arch_family  # param kept for table symmetry
    root = EXAMPLES / dirname
    arch, train = _read_scripts(root)
    imported = _imported_roots(arch) | _imported_roots(train)
    blocked_hit = sorted(imported & _BLOCKED_IMPORT_ROOTS)
    assert blocked_hit == [], f"{family_id} blocked imports: {blocked_hit}"
    assert "cpp_extension" not in imported

    arch_report = inspect_code(arch, require_contract=False)
    assert "function:build_model" in arch_report.ast_fingerprint
    train_report = inspect_code(
        train, require_contract=False, allowed_import_roots={"architecture"}
    )
    assert "function:train" in train_report.ast_fingerprint


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    NOVEL_FAMILIES,
    ids=[row[0] for row in NOVEL_FAMILIES],
)
@pytest.mark.parametrize("vocab_size", [4096, 50257])
def test_novel_seed_under_param_cap_forced_seed(
    family_id: str, dirname: str, arch_family: str, vocab_size: int
) -> None:
    """Forced-seed static instantiation reports ≤ explore 124M (prefer ≤5M thrash)."""
    del arch_family
    root = EXAMPLES / dirname
    arch, _train = _read_scripts(root)
    ctx = PrismContext(vocab_size=vocab_size, sequence_length=128, seed=1337)
    count = check_build_model_static(
        {"architecture.py": arch},
        "architecture.py",
        ctx=ctx,
        max_parameters=MAX_PARAMS,
    )
    assert 0 < count <= MAX_PARAMS, f"{family_id} count={count}"
    assert count <= ctx.max_params
    # Soft thrash band: keep lab packages under ~5M for fair serial CUDA thrash.
    if vocab_size == 4096:
        assert count <= SOFT_THRASH_MAX, (
            f"{family_id} vocab={vocab_size} count={count} exceeds soft thrash {SOFT_THRASH_MAX}"
        )


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    NOVEL_FAMILIES,
    ids=[row[0] for row in NOVEL_FAMILIES],
)
def test_novel_seed_multi_gpu_static_contract_safe(
    family_id: str, dirname: str, arch_family: str
) -> None:
    """Single-node ≤8 multi-GPU static contract (same outer spirit as Imp seeds)."""
    del family_id, arch_family
    root = EXAMPLES / dirname
    _arch, train = _read_scripts(root)
    report = check_distributed_contract(train, artifact_path="training.py", policy="reject")
    assert report.compliant is True
    assert report.missing == ()
    assert report.unguarded_writes == 0
    present = set(report.present)
    for primitive in (
        "init_process_group",
        "device_binding",
        "ddp_or_fsdp",
        "data_sharding",
        "rank0_guard",
        "destroy_process_group",
    ):
        assert primitive in present
    for gpu_count in (1, 2, 4, 8):
        enforce_single_node_bound(gpu_count, num_nodes=1, max_gpu_count=DEFAULT_MAX_GPU_COUNT)
    with pytest.raises(SandboxViolation):
        enforce_single_node_bound(9, max_gpu_count=DEFAULT_MAX_GPU_COUNT)


def test_deeploop_class_is_among_novel_packages() -> None:
    """DeepLoop-class MUST be one of the novel packs when shortlist kept it."""
    assert "deeploop-tiny-1m" in SEED_FAMILIES
    arch = (EXAMPLES / "deeploop-tiny" / "architecture.py").read_text(encoding="utf-8")
    tree = ast.parse(arch)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "DeepLoopBlock" in class_names
    assert "DeepLoopLM" in class_names
    assert "for loop_idx in range(self.loops)" in arch or "for _ in range(self.loops)" in arch


def test_novel_family_knobs_documented_in_registry() -> None:
    for family_id, _dirname, _af in NOVEL_FAMILIES:
        family = get_family(family_id)
        for key in ("param_counting", "step_throughput", "stability", "tokenizer"):
            assert key in family.knobs
            assert family.knobs[key].strip()


def test_at_least_three_novel_families_registered() -> None:
    novel_ids = {row[0] for row in NOVEL_FAMILIES}
    registered_novel = novel_ids & set(SEED_FAMILIES)
    assert len(registered_novel) >= 3


# --- VAL-ARXEVAL-003: Imp baselines still pack -------------------------------------------------


@pytest.mark.parametrize(
    ("family_id", "dirname"),
    IMP_FAMILIES,
    ids=[row[0] for row in IMP_FAMILIES],
)
def test_imp_baseline_still_packs(family_id: str, dirname: str) -> None:
    """transformer-tiny-1m and mamba-tiny-1m still pack with unchanged outer contract."""
    root = EXAMPLES / dirname
    assert root.is_dir()
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert (root / name).is_file()
    assert family_id in SEED_FAMILIES
    out = package_seed_zip(family_id, REPO_ROOT / "dist" / "seed-packages-imp-reassert")
    assert out.zip_path.is_file()
    assert out.size_bytes > 0
    assert len(out.content_sha256) == 64
    with zipfile.ZipFile(out.zip_path) as archive:
        names = set(archive.namelist())
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert name in names
    assert "prism.yaml" in names or "prism.yml" in names


def test_multi_family_harness_packs_novels_and_imps() -> None:
    """Shared packaging harness packs all novels + both Imp baselines together."""
    out_dir = REPO_ROOT / "dist" / "seed-packages-arxiv-multi-test"
    packed = package_all_families(out_dir)
    by_id = {item.family_id: item for item in packed}
    for family_id, _dirname, _af in NOVEL_FAMILIES:
        assert family_id in by_id
        item = by_id[family_id]
        assert item.zip_path.is_file()
        assert "architecture.py" in item.entry_names
        assert "training.py" in item.entry_names
    for family_id, _dirname in IMP_FAMILIES:
        assert family_id in by_id
        item = by_id[family_id]
        assert item.zip_path.is_file()
    # Digests must differ across families (distinct content surface).
    digests = {item.content_sha256 for item in packed}
    assert len(digests) == len(packed)
