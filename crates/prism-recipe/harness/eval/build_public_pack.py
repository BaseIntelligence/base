#!/usr/bin/env python3
"""Alias for the public HF eval-assets pack builder.

Delegates to `build_private_pack.py` (historical filename). Prefer this
entrypoint — the default pack is public held-out HF data (`PACK_TIER=public`).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_builder():
    path = Path(__file__).resolve().with_name("build_private_pack.py")
    spec = importlib.util.spec_from_file_location("prism_build_eval_pack", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    sys.exit(_load_builder().main())
