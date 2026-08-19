"""Print unique dense-1B params (tied embeddings counted once)."""

from __future__ import annotations

try:
    from nemo_automodel.components.models.dense1b.model import (
        DEFAULTS,
        DenseTransformer,
        unique_n_params,
    )
except ImportError:
    import importlib.util
    import sys
    import types
    from pathlib import Path

    root = Path(__file__).resolve().parent
    pkg = types.ModuleType("dense1b_local")
    pkg.__path__ = [str(root)]
    sys.modules["dense1b_local"] = pkg
    kspec = importlib.util.spec_from_file_location("dense1b_local.kernels", root / "kernels.py")
    kmod = importlib.util.module_from_spec(kspec)
    sys.modules["dense1b_local.kernels"] = kmod
    kspec.loader.exec_module(kmod)
    mspec = importlib.util.spec_from_file_location("dense1b_local.model", root / "model.py")
    mmod = importlib.util.module_from_spec(mspec)
    sys.modules["dense1b_local.model"] = mmod
    mspec.loader.exec_module(mmod)
    DEFAULTS, DenseTransformer, unique_n_params = (
        mmod.DEFAULTS,
        mmod.DenseTransformer,
        mmod.unique_n_params,
    )


def main():
    try:
        import torch

        with torch.device("meta"):
            model = DenseTransformer(dict(DEFAULTS), use_te=False)
    except Exception:
        model = DenseTransformer(dict(DEFAULTS), use_te=False)
    n = unique_n_params(model)
    embed = model.tok_emb.weight.numel()
    print(f"n_params={n} ({n / 1e6:.1f}M unique total)")
    print(f"n_embed={embed} ({embed / 1e6:.1f}M) body={n - embed}")
    assert 850_000_000 <= n <= 1_000_000_000, n


if __name__ == "__main__":
    main()
