"""Prism custom-arch loader (trust_remote_code).

Supports:
  1. Legacy seam: ``architecture.build_model(ctx)`` + ``checkpoint.pt``
  2. AutoModel novelty: sources under ``sources/`` (apply patch offline)

This is intentionally permissive — novel arches are not limited to stock
GPT-2 ``transformers`` configs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

try:
    from configuration_prism import PrismConfig
except ImportError:  # package-style local import
    from .configuration_prism import PrismConfig


def _load_architecture_module(root: Path):
    path = root / "architecture.py"
    if not path.is_file():
        raise FileNotFoundError(f"architecture.py missing under {root}")
    spec = importlib.util.spec_from_file_location("prism_architecture", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load architecture.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PrismCustomModel(PreTrainedModel):
    config_class = PrismConfig
    _no_split_modules = []

    def __init__(self, config: PrismConfig, inner: Optional[nn.Module] = None):
        super().__init__(config)
        self.inner = inner if inner is not None else nn.Identity()
        self.post_init()

    def forward(self, *args: Any, **kwargs: Any):
        return self.inner(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        trust = kwargs.pop("trust_remote_code", True)
        config = kwargs.pop("config", None)
        if config is None:
            config = PrismConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust, **kwargs
            )
        root = Path(pretrained_model_name_or_path)
        # Hub download may leave us with a cache dir; prefer local folder layout.
        if not (root / "architecture.py").is_file():
            # Fall back to empty shell when only config is present.
            return cls(config)

        mod = _load_architecture_module(root)
        ctx = {
            "device": "cpu",
            "dtype": torch.float32,
            "seed": 0,
            "vocab_size": getattr(config, "vocab_size", 50257),
        }
        if hasattr(mod, "build_model"):
            inner = mod.build_model(ctx)
        elif hasattr(mod, "Model"):
            inner = mod.Model(**{k: v for k, v in ctx.items() if k in ("vocab_size",)})
        else:
            raise AttributeError(
                "architecture.py must define build_model(ctx) or Model for Hub reload"
            )
        model = cls(config, inner=inner)
        ckpt_name = getattr(config, "checkpoint_file", "checkpoint.pt") or "checkpoint.pt"
        ckpt = root / ckpt_name
        if ckpt.is_file():
            blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
            if isinstance(state, dict):
                try:
                    model.inner.load_state_dict(state, strict=False)
                except Exception:
                    # Novel arches / tied keys — best-effort; sources remain authoritative.
                    pass
        return model
