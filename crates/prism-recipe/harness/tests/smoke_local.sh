#!/usr/bin/env bash
# Local end-to-end smoke for the PRISM harness package (CPU-friendly).
# Exit 0 = pass, 1 = fail, 2 = skipped (torch/transformers missing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$HERE/tests/smoke_local.py" "$@"
