"""Dense 1B miner example: unique params in 850M–1B, no MoE modules."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prismlib.params import enforce_param_range

FLOOR = 850_000_000
CAP = 1_000_000_000
EXAMPLE = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "external-miner"
    / "examples"
    / "dense-1b"
)
BANNED_CLASSES = ("FineGrainedMoE", "LoopMoE", "DeltaMoEBlock", "MoE")
BANNED_NAMES = ("n_experts", "moe_top_k", "expert_hidden", "shared_expert_hidden", "loop_bias")


def _example_sources():
    for name in ("model.py", "entry.py", "kernels.py", "ddp_worker.py"):
        yield name, (EXAMPLE / name).read_text(encoding="utf-8")


def test_example_sources_have_no_moe():
    assert EXAMPLE.is_dir(), EXAMPLE
    for name, src in _example_sources():
        tree = ast.parse(src, filename=name)
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        for banned in BANNED_CLASSES:
            assert banned not in classes, f"{name} defines {banned}"
        assigned = {
            t.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        assigned |= {
            k.value if isinstance(k, ast.Constant) else getattr(k, "s", None)
            for n in ast.walk(tree)
            if isinstance(n, ast.Dict)
            for k in n.keys
        }
        for marker in BANNED_NAMES:
            assert marker not in assigned, f"{name} assigns {marker}"


def test_dense_defaults_in_param_range():
    import importlib.util
    import types

    pkg = types.ModuleType("dense1b_local")
    pkg.__path__ = [str(EXAMPLE)]
    sys.modules["dense1b_local"] = pkg
    kspec = importlib.util.spec_from_file_location("dense1b_local.kernels", EXAMPLE / "kernels.py")
    kmod = importlib.util.module_from_spec(kspec)
    sys.modules["dense1b_local.kernels"] = kmod
    kspec.loader.exec_module(kmod)
    mspec = importlib.util.spec_from_file_location("dense1b_local.model", EXAMPLE / "model.py")
    mmod = importlib.util.module_from_spec(mspec)
    sys.modules["dense1b_local.model"] = mmod
    mspec.loader.exec_module(mmod)

    import torch

    with torch.device("meta"):
        model = mmod.DenseTransformer(dict(mmod.DEFAULTS), use_te=False)
    n = mmod.unique_n_params(model)
    assert enforce_param_range(n, FLOOR, CAP) == n
    names = " ".join(type(m).__name__ for m in model.modules())
    for banned in ("MoE", "Expert", "Router"):
        assert banned not in names, names
    print(f"dense-1b n_params={n}")


def test_b200_profile_microbatch_and_pin():
    import importlib.util
    import types

    pkg = types.ModuleType("dense1b_local")
    pkg.__path__ = [str(EXAMPLE)]
    sys.modules["dense1b_local"] = pkg
    kspec = importlib.util.spec_from_file_location("dense1b_local.kernels", EXAMPLE / "kernels.py")
    kmod = importlib.util.module_from_spec(kspec)
    sys.modules["dense1b_local.kernels"] = kmod
    kspec.loader.exec_module(kmod)
    mspec = importlib.util.spec_from_file_location("dense1b_local.model", EXAMPLE / "model.py")
    mmod = importlib.util.module_from_spec(mspec)
    sys.modules["dense1b_local.model"] = mmod
    mspec.loader.exec_module(mmod)

    b200 = {"gpu_type": "NVIDIA B200", "gpu_count": 1}
    rtx = {"gpu_type": "NVIDIA GeForce RTX 5090", "gpu_count": 1}
    wide = {"gpu_type": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "gpu_count": 2}
    assert mmod.is_b200_class(b200, 1)
    assert not mmod.is_b200_class(rtx, 1)
    assert not mmod.is_b200_class(wide, 2)
    assert mmod.is_96gb_class(b200, 1)
    entry = (EXAMPLE / "entry.py").read_text(encoding="utf-8")
    assert "B200_MICRO_BATCH = 8" in entry
    assert "PEAK_FLOPS_B200 = 2250.0e12" in entry
    assert "stream.device = device" in entry
    assert "input_ids.to(target" in entry


if __name__ == "__main__":
    test_example_sources_have_no_moe()
    test_dense_defaults_in_param_range()
    test_b200_profile_microbatch_and_pin()
    print("dense-1b example OK")
