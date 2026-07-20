"""Open-arch research-lab norm regression (VAL-RESLAB-010).

Locks:
1. Novel ``nn.Module`` architectures under AST + dual param ladder are **expected**,
   not second-class / family-hard-blocked.
2. Seeds ``transformer-tiny-1m`` + ``mamba-tiny-1m`` still pack under the shared harness.
3. DeepLoop-class looped-depth modules are allowed when they meet the forward contract.
4. Emission scoring stays architecture-agnostic (no family-specific emission shortcuts).
"""

from __future__ import annotations

import ast
import inspect
import zipfile
from pathlib import Path

import pytest

from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code
from prism_challenge.evaluator.scoring import (
    PrequentialBpbScore,
    score_prequential_bpb,
)
from prism_challenge.evaluator.static_instantiation import check_build_model_static
from prism_challenge.seed_packaging import (
    REQUIRED_ENTRY_SCRIPTS,
    SEED_FAMILIES,
    list_families,
    package_all_families,
    package_seed_zip,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_ROOT = REPO_ROOT / "examples" / "tiny-1m"
MAMBA_ROOT = REPO_ROOT / "examples" / "mamba-tiny"
EXPLORE_CAP = 124_000_000
CTX = PrismContext(vocab_size=256, sequence_length=16, seed=1337)

PLAIN_TRAIN = (
    "from architecture import build_model\n\n"
    "def train(ctx):\n"
    "    build_model(ctx)\n"
    "    return None\n"
)

# --- Novel / non-baseline nn.Module families expected under AST + cap ---------------------------

NOVEL_DEEPLOOP = (
    "import torch\n"
    "from torch import nn\n\n"
    "class DeepLoopBlock(nn.Module):\n"
    '    """LightDeepLoop-class: shared-weight residual looped over depth."""\n'
    "    def __init__(self, dim: int, loops: int = 3):\n"
    "        super().__init__()\n"
    "        self.loops = loops\n"
    "        self.norm = nn.LayerNorm(dim)\n"
    "        self.fc1 = nn.Linear(dim, dim * 2)\n"
    "        self.fc2 = nn.Linear(dim * 2, dim)\n\n"
    "    def forward(self, x):\n"
    "        for _ in range(self.loops):\n"
    "            h = self.fc1(self.norm(x))\n"
    "            h = torch.nn.functional.gelu(h)\n"
    "            x = x + self.fc2(h)\n"
    "        return x\n\n"
    "class DeepLoopLM(nn.Module):\n"
    "    def __init__(self, vocab: int, dim: int = 32):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.loop = DeepLoopBlock(dim, loops=4)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.loop(self.emb(tokens)))\n\n"
    "def build_model(ctx):\n"
    "    return DeepLoopLM(ctx.vocab_size)\n"
)

NOVEL_GATED_MLP = (
    "import torch\n"
    "from torch import nn\n\n"
    "class GatedMlpLM(nn.Module):\n"
    "    def __init__(self, vocab: int, dim: int = 32):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.gate = nn.Linear(dim, dim)\n"
    "        self.up = nn.Linear(dim, dim)\n"
    "        self.down = nn.Linear(dim, dim)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        x = self.emb(tokens)\n"
    "        x = self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))\n"
    "        return self.head(x)\n\n"
    "def build_model(ctx):\n"
    "    return GatedMlpLM(ctx.vocab_size)\n"
)

NOVEL_PURE_TORCH_SSM = (
    "import torch\n"
    "from torch import nn\n\n"
    "class PureTorchSsmLM(nn.Module):\n"
    "    def __init__(self, vocab: int, dim: int = 24):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.in_proj = nn.Linear(dim, dim)\n"
    "        self.out_proj = nn.Linear(dim, dim)\n"
    "        self.log_decay = nn.Parameter(torch.zeros(dim))\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        x = self.in_proj(self.emb(tokens))\n"
    "        decay = torch.sigmoid(self.log_decay)\n"
    "        state = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)\n"
    "        outs = []\n"
    "        for t in range(x.shape[1]):\n"
    "            state = state * decay + x[:, t]\n"
    "            outs.append(state)\n"
    "        y = self.out_proj(torch.stack(outs, dim=1))\n"
    "        return self.head(y)\n\n"
    "def build_model(ctx):\n"
    "    return PureTorchSsmLM(ctx.vocab_size)\n"
)

NOVEL_HYBRID_ATTN_MLP = (
    "import torch\n"
    "from torch import nn\n\n"
    "class HybridTinyLM(nn.Module):\n"
    "    def __init__(self, vocab: int, dim: int = 32, heads: int = 4):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)\n"
    "        self.ff = nn.Sequential(\n"
    "            nn.Linear(dim, dim * 2),\n"
    "            nn.GELU(),\n"
    "            nn.Linear(dim * 2, dim),\n"
    "        )\n"
    "        self.norm = nn.LayerNorm(dim)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        x = self.emb(tokens)\n"
    "        a, _ = self.attn(x, x, x, need_weights=False)\n"
    "        x = self.norm(x + a)\n"
    "        return self.head(x + self.ff(x))\n\n"
    "def build_model(ctx):\n"
    "    return HybridTinyLM(ctx.vocab_size)\n"
)

NOVEL_ARCHITECTURES = {
    "deeploop_looped_depth": NOVEL_DEEPLOOP,
    "gated_mlp": NOVEL_GATED_MLP,
    "pure_torch_ssm": NOVEL_PURE_TORCH_SSM,
    "hybrid_attn_mlp": NOVEL_HYBRID_ATTN_MLP,
}

# Names that must never appear as hard family eligibility filters on emission scoring source.
_FORBIDDEN_FAMILY_EMISSION_BRANCHES = (
    "architecture_family",
    "family_id",
    "is_mamba",
    "is_transformer",
    "if family",
    "family ==",
    "FAMILY_",
)


def _minimal_manifest(
    *,
    bpb: float = 2.5,
    heldout_delta: float = 0.4,
    covered_bytes: int = 10_000,
    tokens: int = 2_000,
    step0: float = 5.0,
) -> dict:
    """Challenge-owned v2 manifest shape expected by ``score_prequential_bpb``."""
    import math

    sum_nll_nats = (bpb * covered_bytes) * math.log(2.0)
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": "open-arch-probe",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": [3.0, 2.5, 2.0, 1.8],
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "sum_neg_log2_likelihood_bits": bpb * covered_bytes,
            "cumulative_codelength_bits": bpb * covered_bytes,
            "covered_bytes": covered_bytes,
            "total_bytes_covered": covered_bytes,
            "predicted_tokens": tokens,
            "tokens_seen": tokens,
            "prequential_bpb": bpb,
            "bits_per_byte": bpb,
            "step0_loss": step0,
            "consumed_batches": 4,
            "heldout_delta": heldout_delta,
            "held_out_delta": heldout_delta,
            "val_bpb_trained": 3.0,
            "val_bpb_random_init": 3.0 + heldout_delta,
            "train_heldout_gap": 0.1,
            "train_bpb_converged": 2.4,
            "gap_basis": "converged",
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


# --- VAL-RESLAB-010: novel / DeepLoop-class modules admitted under AST + cap --------------------


@pytest.mark.parametrize("name", sorted(NOVEL_ARCHITECTURES))
def test_open_arch_novel_modules_pass_ast_and_param_cap(name: str) -> None:
    """Novel nn.Module families under AST + explore cap are first-class (not family-blocked)."""
    arch = NOVEL_ARCHITECTURES[name]
    report = inspect_code(arch, require_contract=False)
    assert "function:build_model" in report.ast_fingerprint
    inspect_code(PLAIN_TRAIN, require_contract=False, allowed_import_roots={"architecture"})
    count = check_build_model_static(
        {"architecture.py": arch},
        "architecture.py",
        ctx=CTX,
        max_parameters=EXPLORE_CAP,
    )
    assert 0 < count <= EXPLORE_CAP
    assert count <= CTX.max_params


def test_open_arch_deeploop_class_is_explicitly_allowed() -> None:
    """LightDeepLoop-class looped modules meet forward contract and pack under explore cap."""
    arch = NOVEL_DEEPLOOP
    # AST must allow pure-torch looped residual depth (no special family allowlist).
    report = inspect_code(arch, require_contract=False)
    assert "function:build_model" in report.ast_fingerprint
    tree = ast.parse(arch)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "DeepLoopBlock" in class_names
    assert "DeepLoopLM" in class_names
    # Source encodes shared-weight depth loops (research-lab explicitly in-scope).
    assert "for _ in range(self.loops)" in arch or "for _ in range(" in arch
    count = check_build_model_static(
        {"architecture.py": arch},
        "architecture.py",
        ctx=CTX,
        max_parameters=EXPLORE_CAP,
    )
    assert 0 < count <= EXPLORE_CAP


def test_open_arch_no_hard_family_allowlist_in_static_gate() -> None:
    """Static admission does not branch on seed family ids (architecture-agnostic gate)."""
    # All novel names clear the same gate without registering a family id.
    for name, arch in NOVEL_ARCHITECTURES.items():
        count = check_build_model_static(
            {"architecture.py": arch},
            "architecture.py",
            ctx=CTX,
            max_parameters=EXPLORE_CAP,
        )
        assert count > 0, name


# --- Seeds still pack ------------------------------------------------------------------------


def test_open_arch_seed_families_still_registered_and_pack() -> None:
    """Seeds tiny-1m + mamba-tiny remain packable under the shared harness."""
    assert "transformer-tiny-1m" in SEED_FAMILIES
    assert "mamba-tiny-1m" in SEED_FAMILIES
    assert "transformer-tiny-1m" in list_families()
    assert "mamba-tiny-1m" in list_families()
    assert TINY_ROOT.is_dir() and MAMBA_ROOT.is_dir()

    out_dir = REPO_ROOT / "dist" / "seed-packages-open-arch-test"
    packed = package_all_families(out_dir)
    by_id = {item.family_id: item for item in packed}
    assert "transformer-tiny-1m" in by_id
    assert "mamba-tiny-1m" in by_id

    for family_id in ("transformer-tiny-1m", "mamba-tiny-1m"):
        item = by_id[family_id]
        assert item.zip_path.is_file()
        assert item.size_bytes > 0
        assert len(item.content_sha256) == 64
        for required in REQUIRED_ENTRY_SCRIPTS:
            assert required in item.entry_names
        with zipfile.ZipFile(item.zip_path) as archive:
            names = set(archive.namelist())
        for required in REQUIRED_ENTRY_SCRIPTS:
            assert required in names

    # Individual family pack still works (regression surface).
    tf = package_seed_zip("transformer-tiny-1m", out_dir)
    mb = package_seed_zip("mamba-tiny-1m", out_dir)
    assert tf.zip_path.is_file() and mb.zip_path.is_file()
    assert tf.content_sha256 != mb.content_sha256


def test_open_arch_seeds_still_under_explore_cap() -> None:
    """Both default lab seeds stay under the 124M explore ladder and pass AST."""
    for root in (TINY_ROOT, MAMBA_ROOT):
        arch = (root / "architecture.py").read_text(encoding="utf-8")
        train = (root / "training.py").read_text(encoding="utf-8")
        assert "function:build_model" in inspect_code(arch, require_contract=False).ast_fingerprint
        assert (
            "function:train"
            in inspect_code(
                train, require_contract=False, allowed_import_roots={"architecture"}
            ).ast_fingerprint
        )
        count = check_build_model_static(
            {"architecture.py": arch},
            "architecture.py",
            ctx=PrismContext(vocab_size=50257, sequence_length=128, seed=1337),
            max_parameters=EXPLORE_CAP,
        )
        assert 0 < count <= EXPLORE_CAP


# --- Architecture-agnostic emission score path (no family shortcuts) ---------------------------


def test_open_arch_emission_score_has_no_family_branch() -> None:
    """score_prequential_bpb source must not family-hard-code emission shortcuts."""
    source = inspect.getsource(score_prequential_bpb)
    lowered = source.lower()
    for needle in _FORBIDDEN_FAMILY_EMISSION_BRANCHES:
        assert needle.lower() not in lowered, f"family shortcut in score path: {needle!r}"
    # Score signature accepts only the challenge-owned manifest (+ skip_heldout).
    sig = inspect.signature(score_prequential_bpb)
    assert "manifest" in sig.parameters
    assert "family" not in sig.parameters
    assert "architecture_family" not in sig.parameters


def test_open_arch_emission_score_identical_for_same_metrics_across_families() -> None:
    """Equal challenge metrics yield equal emission ranks regardless of seed family label."""
    manifest = _minimal_manifest(bpb=2.1, heldout_delta=0.55)
    score_a = score_prequential_bpb(manifest)
    score_b = score_prequential_bpb(dict(manifest))  # same metrics, recomputed
    assert isinstance(score_a, PrequentialBpbScore)
    assert score_a.final_score == pytest.approx(score_b.final_score)
    assert score_a.bpb == pytest.approx(score_b.bpb)
    assert score_a.heldout_delta == pytest.approx(score_b.heldout_delta)
    assert score_a.emission_crown_eligible is True
    assert score_a.primary_metric == "heldout_delta"
    # Injecting a fictitious family label on the manifest must not change rank.
    tagged = {
        **manifest,
        "metrics": {
            **manifest["metrics"],
            "architecture_family": "mamba",
            "family_id": "mamba-tiny-1m",
        },
    }
    score_tagged = score_prequential_bpb(tagged)
    assert score_tagged.final_score == pytest.approx(score_a.final_score)
    assert score_tagged.bpb == pytest.approx(score_a.bpb)


def test_open_arch_heldout_primary_not_family_contingent() -> None:
    """Better held-out always beats better bpb alone — regardless of imputed family."""
    better_hold = score_prequential_bpb(_minimal_manifest(bpb=3.0, heldout_delta=0.8))
    better_bpb = score_prequential_bpb(_minimal_manifest(bpb=1.0, heldout_delta=0.1))
    assert better_hold.final_score > better_bpb.final_score


# --- Docs: open-arch shown as expected (not second-class) --------------------------------------


def test_open_arch_docs_norm_explicit() -> None:
    """Public docs state novel nn.Module / DeepLoop-class modules are expected under AST+cap."""
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "README.md",
            REPO_ROOT / "docs" / "overview.md",
            REPO_ROOT / "docs" / "submissions.md",
            REPO_ROOT / "docs" / "miner" / "README.md",
            REPO_ROOT / "docs" / "scoring.md",
        )
    )
    lower = docs.lower()
    assert "research lab" in lower
    assert "new architecture" in lower or "new architectures" in lower
    assert "nn.module" in lower or "torch.nn.module" in lower
    assert "expected" in lower or "first-class" in lower or "welcome" in lower
    assert "ast" in lower
    assert "deeploop" in lower or "looped" in lower
    assert "tiny-1m" in lower or "transformer-tiny-1m" in lower
    assert "mamba-tiny" in lower or "mamba-tiny-1m" in lower
    # Architecture-agnostic / no family-only emission claim.
    assert "architecture-agnostic" in lower or "architecture agnostic" in lower
    assert "family-hard" not in lower  # must not document family-hard blocks as policy


def test_open_arch_hostile_native_extension_still_blocked() -> None:
    """Open-arch does not weaken AST: blocked native extensions still fail closed."""
    with pytest.raises(SandboxViolation):
        inspect_code("import mamba_ssm\n", require_contract=False)
    with pytest.raises(SandboxViolation):
        inspect_code("import ctypes\n", require_contract=False)
