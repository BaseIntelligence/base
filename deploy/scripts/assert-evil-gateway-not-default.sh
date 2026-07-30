#!/usr/bin/env bash
# Fail if evil-gateway would start on default or master-only compose paths.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE="$ROOT/docker-compose.yml"
test -f "$COMPOSE"

# Default config must not list evil-gateway as a started service.
default_svcs=$(docker compose -f "$COMPOSE" config --services 2>/dev/null || true)
if echo "$default_svcs" | grep -qx 'evil-gateway'; then
  echo "FAIL: evil-gateway appears in default compose services" >&2
  exit 1
fi

# Master profile must not pull evil-gateway.
master_svcs=$(docker compose -f "$COMPOSE" --profile master config --services 2>/dev/null || true)
if echo "$master_svcs" | grep -qx 'evil-gateway'; then
  echo "FAIL: evil-gateway appears under --profile master" >&2
  exit 1
fi

# evil-gateway profile must expose the service.
evil_svcs=$(docker compose -f "$COMPOSE" --profile evil-gateway config --services 2>/dev/null || true)
if ! echo "$evil_svcs" | grep -qx 'evil-gateway'; then
  echo "FAIL: evil-gateway missing under --profile evil-gateway" >&2
  exit 1
fi

echo "PASS: evil-gateway isolated (default/master clean; evil-gateway profile only)"
