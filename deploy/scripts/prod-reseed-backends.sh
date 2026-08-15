#!/usr/bin/env bash
# Re-seed in-memory gateway challenge backends (idempotent).
# The registry is wiped on every gateway restart; this heals 503s until
# BASE_GATEWAY_BACKENDS boot-seed is in the running image.
set -euo pipefail
ROOT="${BASE_ROOT:-/opt/base}"
cd "$ROOT"
exec "$ROOT/deploy/scripts/register-challenge-backends.sh" \
  --gateway-url http://127.0.0.1:8080 \
  --prism-url http://prism-challenge:8092 \
  --design-url http://design-challenge:8093
