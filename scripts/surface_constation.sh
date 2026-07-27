#!/usr/bin/env bash
# T12: production constation surface scenarios S1–S8 + B1s/B2s.
# Exit non-zero on any failure. No live Lium.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/var/tmp/uv-cache}"
# Prism optional surface pieces (S3/S6) + local surface helpers.
export PYTHONPATH="${ROOT}/packages/challenges/prism/src:${ROOT}/tests/surface${PYTHONPATH:+:${PYTHONPATH}}"

SURFACE_NODEIDS=(
  "tests/surface/test_constation_production_surface.py::test_s1_honest_path_seal_store_forward_nests_bundle"
  "tests/surface/test_constation_production_surface.py::test_s2_adversarial_triangle_fail_no_ok_bundle_put"
  "tests/surface/test_constation_production_surface_prism_http.py::test_s3_missing_bundle_prism_behavior"
  "tests/surface/test_constation_production_surface.py::test_s4_nonce_issue_seal_consume_once_ok"
  "tests/surface/test_constation_production_surface.py::test_s5_allowlist_sealed_hashes_required"
  "tests/surface/test_constation_production_surface_prism_http.py::test_s6_legacy_gate_no_elevation_without_constation"
  "tests/surface/test_constation_production_surface.py::test_s7_forwarder_embed_from_lookup"
  "tests/surface/test_constation_production_surface_prism_http.py::test_s8_challenge_route_smoke"
  "tests/surface/test_constation_production_surface.py::test_b1s_verify_key_wiring_smoke"
  "tests/surface/test_constation_production_surface.py::test_b2s_orchestrator_source_never_calls_consume"
)

echo "==> surface_constation: UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "==> running ${#SURFACE_NODEIDS[@]} node ids"

exec uv run pytest -q --tb=short "${SURFACE_NODEIDS[@]}"
