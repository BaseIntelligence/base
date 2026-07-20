"""Transformer tiny-1m seed package gates (VAL-SEED-001/002/003/007).

Targeted lab-seed surface only: inventory + packaging harness, AST sandbox, forced-seed
param cap ≤124M explore, and multi-GPU single-node static contract. No GPU re-exec thrash.
"""

from __future__ import annotations

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
    package_seed_zip,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_ROOT = REPO_ROOT / "examples" / "tiny-1m"
FAMILY_ID = "transformer-tiny-1m"
# Dual ladder explore default (VAL-RESLAB-003); seeds stay under 124M explore.
MAX_PARAMS = 124_000_000


def _read_scripts() -> tuple[str, str]:
    return (
        (TINY_ROOT / "architecture.py").read_text(encoding="utf-8"),
        (TINY_ROOT / "training.py").read_text(encoding="utf-8"),
    )


def test_transformer_seed_package_is_complete_and_submit_shaped() -> None:
    """VAL-SEED-001: inventory + packaging harness entry produce a two-script zip."""
    assert TINY_ROOT.is_dir()
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert (TINY_ROOT / name).is_file(), f"missing {name}"
    assert (TINY_ROOT / "prism.yaml").is_file()
    assert FAMILY_ID in SEED_FAMILIES
    assert FAMILY_ID in list_families()
    family = get_family(FAMILY_ID)
    assert family.architecture_family == "transformer"
    assert family.source_dir.resolve() == TINY_ROOT.resolve()

    files = collect_seed_files(TINY_ROOT)
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
    # Packaging surface excludes secrets; only researched text/code suffixes.
    assert not any(n.endswith((".pem", ".key", ".env", ".pt", ".bin")) for n in names)


def test_transformer_seed_passes_ast_sandbox_hard_blocks() -> None:
    """VAL-SEED-002: both scripts accept under the Prism AST allowlist."""
    arch, train = _read_scripts()
    arch_report = inspect_code(arch, require_contract=False)
    assert "function:build_model" in arch_report.ast_fingerprint
    train_report = inspect_code(
        train, require_contract=False, allowed_import_roots={"architecture"}
    )
    assert "function:train" in train_report.ast_fingerprint


@pytest.mark.parametrize("vocab_size", [4096, 50257])
def test_transformer_seed_under_param_cap_forced_seed(vocab_size: int) -> None:
    """VAL-SEED-003: forced-seed static instantiation reports ≤ 124M explore params."""
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


def test_transformer_seed_multi_gpu_static_contract_safe() -> None:
    """VAL-SEED-007: single-node ≤8 multi-GPU static contract for Transformer seed."""
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


def test_transformer_family_knobs_documented_in_registry() -> None:
    family = get_family(FAMILY_ID)
    for key in ("param_counting", "step_throughput", "stability", "tokenizer"):
        assert key in family.knobs
        assert family.knobs[key].strip()
