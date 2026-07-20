"""Frontier-inspired seed package gates (VAL-FRNTEVAL-002/003).

Targeted lab surface: >=3 shortlist keepers (MLA + DeepSeekMoE-style fine-grain
MoE + Kimi/KDA) pack under the shared harness, pass AST sandbox (no blocked natives),
forced-seed param count under explore 124M (~3-15M thrash band), multi-GPU
single-node static contract, open-arch shape, and Imp + deeploop controls still pack.

Honesty: these are mechanism downscales of DeepSeek/Kimi-class ideas, not full
frontier V4/K3 weights.
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
# Explore ladder hard cap (VAL-RESLAB-003 / VAL-FRNTEVAL-002).
MAX_PARAMS = 124_000_000
# Frontier thrash band is ~3-15M (wider than the prior arXiv soft 5M band).
SOFT_THRASH_MAX = 15_000_000
SOFT_THRASH_MIN = 500_000

# Shortlist keepers that MUST ship (MLA + MoE + KDA).
FRONTIER_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("mla-tiny-1m", "mla-tiny", "mla"),
    ("ds-moe-tiny-1m", "ds-moe-tiny", "ds_moe"),
    ("kda-tiny-1m", "kda-tiny", "kda"),
)
CONTROL_FAMILIES: tuple[tuple[str, str], ...] = (
    ("transformer-tiny-1m", "tiny-1m"),
    ("mamba-tiny-1m", "mamba-tiny"),
    ("deeploop-tiny-1m", "deeploop-tiny"),
)

_BLOCKED_IMPORT_ROOTS = frozenset(
    {
        "mamba_ssm",
        "causal_conv1d",
        "flash_attn",
        "flash_linear_attn",
        "fla",
        "ctypes",
        "cffi",
        "importlib",
    }
)

# Per-family signature symbols that prove the intended mechanism (not a rename of prior packs).
_FAMILY_SIGNATURES: dict[str, tuple[str, ...]] = {
    "mla-tiny-1m": ("MultiHeadLatentAttention", "MLALM", "kv_lora_rank"),
    "ds-moe-tiny-1m": ("MoEFeedForward", "DSMoELM", "top_k"),
    "kda-tiny-1m": ("KimiDeltaAttention", "KDALM", "channel_gate"),
}


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


# --- VAL-FRNTEVAL-002: >=3 frontier packs under contract ---------------------------------------


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    FRONTIER_FAMILIES,
    ids=[row[0] for row in FRONTIER_FAMILIES],
)
def test_frontier_seed_package_complete_and_submit_shaped(
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

    out = package_seed_zip(family_id, REPO_ROOT / "dist" / "seed-packages-frontier-test")
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
    FRONTIER_FAMILIES,
    ids=[row[0] for row in FRONTIER_FAMILIES],
)
def test_frontier_seed_passes_ast_sandbox_pure_torch(
    family_id: str, dirname: str, arch_family: str
) -> None:
    """AST allowlist pass with no blocked native extension imports."""
    del arch_family
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
    FRONTIER_FAMILIES,
    ids=[row[0] for row in FRONTIER_FAMILIES],
)
@pytest.mark.parametrize("vocab_size", [4096, 50257])
def test_frontier_seed_under_param_cap_forced_seed(
    family_id: str, dirname: str, arch_family: str, vocab_size: int
) -> None:
    """Forced-seed static instantiation reports <= explore 124M (prefer <=15M thrash)."""
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
    # Soft thrash band for frontier packs (~3-15M attracted; allow from 0.5M up).
    if vocab_size == 4096:
        assert count <= SOFT_THRASH_MAX, (
            f"{family_id} vocab={vocab_size} count={count} exceeds soft thrash {SOFT_THRASH_MAX}"
        )
        assert count >= SOFT_THRASH_MIN, (
            f"{family_id} vocab={vocab_size} count={count} below thrash floor {SOFT_THRASH_MIN}"
        )


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    FRONTIER_FAMILIES,
    ids=[row[0] for row in FRONTIER_FAMILIES],
)
def test_frontier_seed_multi_gpu_static_contract_safe(
    family_id: str, dirname: str, arch_family: str
) -> None:
    """Single-node <=8 multi-GPU static contract (same outer spirit as Imp seeds)."""
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


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    FRONTIER_FAMILIES,
    ids=[row[0] for row in FRONTIER_FAMILIES],
)
def test_frontier_seed_forward_logits_shape_open_arch(
    family_id: str, dirname: str, arch_family: str
) -> None:
    """Open-arch style: build_model forward yields [B,T,V] logits (host pure-torch)."""
    del arch_family
    import importlib.util
    import sys

    import torch

    root = EXAMPLES / dirname
    arch_path = root / "architecture.py"
    module_name = f"_frontier_seed_{dirname.replace('-', '_')}"
    # Isolate each package load so sibling examples do not share a cached module.
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, arch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    ctx = PrismContext(vocab_size=512, sequence_length=16, seed=7)
    model = module.build_model(ctx)
    tokens = torch.randint(0, ctx.vocab_size, (2, 8))
    logits = model(tokens)
    assert tuple(logits.shape) == (2, 8, ctx.vocab_size), f"{family_id} logits={logits.shape}"
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    ("family_id", "dirname", "arch_family"),
    FRONTIER_FAMILIES,
    ids=[row[0] for row in FRONTIER_FAMILIES],
)
def test_frontier_seed_mechanism_signature_distinct(
    family_id: str, dirname: str, arch_family: str
) -> None:
    """Each pack exposes its named mechanism classes (not a re-label of prior arxiv packs)."""
    del arch_family
    root = EXAMPLES / dirname
    arch = (root / "architecture.py").read_text(encoding="utf-8")
    sigs = _FAMILY_SIGNATURES[family_id]
    for needle in sigs:
        assert needle in arch, f"{family_id} missing mechanism signature {needle!r}"
    tree = ast.parse(arch)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    # First two signatures are class names.
    assert sigs[0] in class_names
    assert sigs[1] in class_names


def test_at_least_three_frontier_families_registered() -> None:
    frontier_ids = {row[0] for row in FRONTIER_FAMILIES}
    registered = frontier_ids & set(SEED_FAMILIES)
    assert len(registered) >= 3


def test_frontier_family_knobs_documented_in_registry() -> None:
    for family_id, _dirname, _af in FRONTIER_FAMILIES:
        family = get_family(family_id)
        for key in ("param_counting", "step_throughput", "stability", "tokenizer"):
            assert key in family.knobs
            assert family.knobs[key].strip()
        # Honesty knob documents downscale (not full frontier weights).
        assert "honesty" in family.knobs
        assert (
            "downscale" in family.knobs["honesty"].lower()
            or "not full" in family.knobs["honesty"].lower()
        )


def test_kda_is_not_a_copy_of_gated_delta() -> None:
    """KDA pack must be Kimi/channel-gate class, not a rename of gated-delta-tiny."""
    kda = (EXAMPLES / "kda-tiny" / "architecture.py").read_text(encoding="utf-8")
    gd = (EXAMPLES / "gated-delta-tiny" / "architecture.py").read_text(encoding="utf-8")
    assert kda != gd
    assert "KimiDeltaAttention" in kda
    assert "channel_gate" in kda
    # Must not simply re-export GatedDeltaRecurrence as the primary block.
    assert "class GatedDeltaRecurrence" not in kda


def test_readme_honesty_disclaimer_present() -> None:
    for _family_id, dirname, _af in FRONTIER_FAMILIES:
        readme = (EXAMPLES / dirname / "README.md").read_text(encoding="utf-8").lower()
        has_downscale_note = any(
            token in readme for token in ("full", "frontier", "downscale", "mechanism")
        )
        assert "not" in readme and has_downscale_note


# --- VAL-FRNTEVAL-003: controls still pack -----------------------------------------------------


@pytest.mark.parametrize(
    ("family_id", "dirname"),
    CONTROL_FAMILIES,
    ids=[row[0] for row in CONTROL_FAMILIES],
)
def test_control_family_still_packs(family_id: str, dirname: str) -> None:
    """transformer, mamba, and deeploop controls still pack under the outer contract."""
    root = EXAMPLES / dirname
    assert root.is_dir()
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert (root / name).is_file()
    assert family_id in SEED_FAMILIES
    out = package_seed_zip(family_id, REPO_ROOT / "dist" / "seed-packages-frontier-controls")
    assert out.zip_path.is_file()
    assert out.size_bytes > 0
    assert len(out.content_sha256) == 64
    with zipfile.ZipFile(out.zip_path) as archive:
        names = set(archive.namelist())
    for name in REQUIRED_ENTRY_SCRIPTS:
        assert name in names
    assert "prism.yaml" in names or "prism.yml" in names


def test_multi_family_harness_packs_frontier_and_controls() -> None:
    """Shared packaging harness packs all frontier packs + required controls together."""
    out_dir = REPO_ROOT / "dist" / "seed-packages-frontier-multi-test"
    packed = package_all_families(out_dir)
    by_id = {item.family_id: item for item in packed}
    for family_id, _dirname, _af in FRONTIER_FAMILIES:
        assert family_id in by_id
        item = by_id[family_id]
        assert item.zip_path.is_file()
        assert "architecture.py" in item.entry_names
        assert "training.py" in item.entry_names
    for family_id, _dirname in CONTROL_FAMILIES:
        assert family_id in by_id
        item = by_id[family_id]
        assert item.zip_path.is_file()
    digests = {item.content_sha256 for item in packed}
    assert len(digests) == len(packed)
