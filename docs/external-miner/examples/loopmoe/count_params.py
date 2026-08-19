"""Print unique LoopMoE params (tied embeddings counted once)."""

from __future__ import annotations

try:
    from nemo_automodel.components.models.loopmoe.model import DEFAULTS, LoopMoE
except ImportError:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("loopmoe_model", root / "model.py")
    # model.py does `from . import kernels` — load as a package.
    import sys
    import types

    pkg = types.ModuleType("loopmoe_local")
    pkg.__path__ = [str(root)]
    sys.modules["loopmoe_local"] = pkg
    kspec = importlib.util.spec_from_file_location("loopmoe_local.kernels", root / "kernels.py")
    kmod = importlib.util.module_from_spec(kspec)
    sys.modules["loopmoe_local.kernels"] = kmod
    kspec.loader.exec_module(kmod)
    mspec = importlib.util.spec_from_file_location("loopmoe_local.model", root / "model.py")
    mmod = importlib.util.module_from_spec(mspec)
    sys.modules["loopmoe_local.model"] = mmod
    mspec.loader.exec_module(mmod)
    DEFAULTS, LoopMoE = mmod.DEFAULTS, mmod.LoopMoE


def unique_params(model):
    seen = set()
    n = 0
    for p in model.parameters():
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        n += p.numel()
    return n


def main():
    model = LoopMoE(dict(DEFAULTS), use_te=False)
    n = unique_params(model)
    embed = model.tok_emb.weight.numel()
    print(f"n_params={n} ({n / 1e6:.1f}M unique total)")
    print(f"n_embed={embed} ({embed / 1e6:.1f}M) body={n - embed}")
    assert 850_000_000 <= n <= 1_000_000_000, n


if __name__ == "__main__":
    main()
