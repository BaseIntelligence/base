#!/usr/bin/env python
"""CLI entrypoint for the one-time locked FineWeb-Edu prep job.

Network is required (prep runs OUTSIDE the eval sandbox). Example:

    python scripts/prepare_fineweb_edu.py --output-dir /data/fineweb-edu --limit 200000
"""

from __future__ import annotations

import sys

from prism_challenge.evaluator.data_prep import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
