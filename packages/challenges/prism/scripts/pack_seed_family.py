#!/usr/bin/env python
"""CLI entry for :mod:`prism_challenge.seed_packaging` (lab dual-family seed zips)."""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src tree importable when run as a bare checkout script (same pattern as
# scripts/staging_e2e.py). Production installs already expose prism_challenge on path.
_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from prism_challenge.seed_packaging import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
