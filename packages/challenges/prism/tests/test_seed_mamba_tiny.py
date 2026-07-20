"""Mamba/SSM pure-torch seed package gates (VAL-SEED-004/005/006/007/008/009).

Targeted lab-seed surface only: inventory + packaging harness symmetry with Transformer,
AST sandbox (no blocked mamba_ssm), forced-seed param cap ≤124M explore, multi-GPU single-node
static contract, dual-family zip outer shape, and documented family knobs.
"""

from __future__ import annotations

import ast
import re
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
MAMBA_ROOT = REPO_ROOT / "examples" / "mamba-tiny"
TINY_ROOT = REPO_ROOT / "examples" / "tiny-1m"
FAMILY_ID = "mamba-tiny-1m"
TRANSFORMER_FAMILY_ID = "transformer-tiny-1m"
# Dual ladder explore default (VAL-RESLAB-003); seeds stay under 124M explore.
MAX_PARAMS = 124_000_000
# Static lab path must not import these blocked native / FFI / extension surfaces.
_BLOCKED_IMPORT_ROOTS = frozenset(
    {
        "mamba_ssm",
        "causal_conv1d",
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
            # from torch.utils import cpp_extension-style attribute imports
            for alias in node.names:
                if node.module and "cpp_extension" in f"{node.module}.{alias.name}":
                    roots.add("cpp_extension")
    # Detect torch.utils.cpp_extension attribute use without requiring a real import root hop.
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


def _read_scripts() -> tuple[str, str]:
    return (
        (MAMBA_ROOT / "architecture.py").read_text(encoding="utf-8"),
        (MAMBA_ROOT / "training.py").read_text(encoding="utf-8"),
    )


def test_mamba_seed_package_exists_and_is_submit_shaped() -> None:
    """VAL-SEED-004: inventory + packaging harness entry produce a two-script zip."""
    assert MAMBA_ROOT.is_dir()
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert (MAMBA_ROOT / name).is_file(), f"missing {name}"
    assert (MAMBA_ROOT / "prism.yaml").is_file()
    assert FAMILY_ID in SEED_FAMILIES
    assert FAMILY_ID in list_families()
    family = get_family(FAMILY_ID)
    assert family.architecture_family == "mamba"
    assert family.source_dir.resolve() == MAMBA_ROOT.resolve()

    files = collect_seed_files(MAMBA_ROOT)
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert name in files

    out = package_seed_zip(FAMILY_ID, REPO_ROOT / "dist" / "seed-packages-test")
    assert out.zip_path.is_file()
    assert out.size_bytes > 0
    assert len(out.content_sha256) == 64
    with zipfile.ZipFile(out.zip_path) as archive:
        names = set(archive.namelist())
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert name in names
    assert not any(n.endswith((".pem", ".key", ".env", ".pt", ".bin")) for n in names)


def test_mamba_seed_passes_ast_sandbox_without_native_mamba_ssm() -> None:
    """VAL-SEED-005: allowlisted pure-torch path; no blocked mamba_ssm for static pass."""
    arch, train = _read_scripts()
    imported = _imported_roots(arch) | _imported_roots(train)
    blocked_hit = sorted(imported & _BLOCKED_IMPORT_ROOTS)
    assert blocked_hit == [], f"blocked native import roots present: {blocked_hit}"
    assert "cpp_extension" not in imported

    arch_report = inspect_code(arch, require_contract=False)
    assert "function:build_model" in arch_report.ast_fingerprint
    train_report = inspect_code(
        train, require_contract=False, allowed_import_roots={"architecture"}
    )
    assert "function:train" in train_report.ast_fingerprint

    # Explicitly prove a blocked native import would fail the same gate.
    with pytest.raises(SandboxViolation):
        inspect_code("import mamba_ssm\n", require_contract=False)


@pytest.mark.parametrize("vocab_size", [4096, 50257])
def test_mamba_seed_under_param_cap_forced_seed(vocab_size: int) -> None:
    """VAL-SEED-006: forced-seed static instantiation reports ≤ 124M explore params."""
    arch, _train = _read_scripts()
    ctx = PrismContext(vocab_size=vocab_size, sequence_length=128, seed=1337)
    count = check_build_model_static(
        {"architecture.py": arch},
        "architecture.py",
        ctx=ctx,
        max_parameters=MAX_PARAMS,
    )
    assert 0 < count <= MAX_PARAMS
    assert count <= ctx.max_params


def test_mamba_seed_multi_gpu_static_contract_safe() -> None:
    """VAL-SEED-007 (Mamba half): single-node ≤8 multi-GPU static contract."""
    _arch, train = _read_scripts()
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
    with pytest.raises(SandboxViolation):
        enforce_single_node_bound(2, num_nodes=2, max_gpu_count=DEFAULT_MAX_GPU_COUNT)


def test_shared_harness_dual_family_zips_same_outer_shape() -> None:
    """VAL-SEED-008: packaging harness yields both family zips with identical outer contract."""
    assert TRANSFORMER_FAMILY_ID in list_families()
    assert FAMILY_ID in list_families()
    out_dir = REPO_ROOT / "dist" / "seed-packages-dual-test"
    packed = package_all_families(out_dir)
    by_id = {item.family_id: item for item in packed}
    assert TRANSFORMER_FAMILY_ID in by_id
    assert FAMILY_ID in by_id

    # Same required entry scripts + optional manifests shape; digests of contents differ.
    for item in (by_id[TRANSFORMER_FAMILY_ID], by_id[FAMILY_ID]):
        assert item.zip_path.is_file()
        assert item.size_bytes > 0
        assert len(item.content_sha256) == 64
        for required in REQUIRED_ENTRY_SCRIPTS:
            assert required in item.entry_names
            assert required in item.file_digests
        # No miner secrets of known high-risk suffixes in either zip.
        assert not any(
            name.endswith((".pem", ".key", ".env", ".pt", ".bin", ".safetensors"))
            for name in item.entry_names
        )
    assert by_id[TRANSFORMER_FAMILY_ID].content_sha256 != by_id[FAMILY_ID].content_sha256

    # Outer contract: required entry member names exist in both archives.
    for item in packed:
        with zipfile.ZipFile(item.zip_path) as archive:
            names = set(archive.namelist())
        for required in REQUIRED_ENTRY_SCRIPTS:
            assert required in names
        assert "prism.yaml" in names or "prism.yml" in names


def test_seed_family_differences_documented() -> None:
    """VAL-SEED-009: operator docs record both families and lab-relevant knobs."""
    mamba_readme = (MAMBA_ROOT / "README.md").read_text(encoding="utf-8")
    assert "mamba-tiny-1m" in mamba_readme
    assert "transformer-tiny-1m" in mamba_readme
    assert (
        "124_000_000" in mamba_readme
        or "124,000,000" in mamba_readme
        or "124M" in mamba_readme
        or "124m" in mamba_readme.lower()
    )
    assert "mamba_ssm" in mamba_readme  # pure-torch caveat vs blocked native dep
    for needle in ("Param", "throughput", "Multi-GPU", "Stability"):
        assert re.search(needle, mamba_readme, re.IGNORECASE)

    miner_guide = (REPO_ROOT / "docs" / "miner" / "README.md").read_text(encoding="utf-8")
    assert "mamba-tiny-1m" in miner_guide
    assert "transformer-tiny-1m" in miner_guide
    assert "Lab seed families" in miner_guide

    family = get_family(FAMILY_ID)
    for key in (
        "param_counting",
        "step_throughput",
        "stability",
        "tokenizer",
        "pure_torch_caveat",
    ):
        assert key in family.knobs
        assert family.knobs[key].strip()


def test_mamba_and_transformer_both_roots_present_for_lab() -> None:
    # Precondition: transformer seed ready path still exists (feature precondition).
    assert TINY_ROOT.is_dir()
    assert (TINY_ROOT / "architecture.py").is_file()
    assert (TINY_ROOT / "training.py").is_file()
    assert MAMBA_ROOT.is_dir()
