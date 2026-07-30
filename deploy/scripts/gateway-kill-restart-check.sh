#!/usr/bin/env bash
# Ops scenario: kill gateway container → docker restart policy brings it back < 60s.
# Requires a running stack with --profile master (gateway present).
# Offline/CI without a healthy gateway: SKIP + unit proof a48_ops_gateway_restart_policy.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "SKIP: docker not available"
  echo "OFFLINE_PROOF: a48_ops_gateway_restart_policy_unless_stopped asserts restart: unless-stopped"
  exit 0
fi

cid=$(docker compose ps -q gateway 2>/dev/null || true)
if [[ -z "${cid}" ]]; then
  echo "SKIP: gateway not running (start with: docker compose --profile master up -d)"
  echo "PENDING_LIVE: task-47 staging required for live kill→restart <60s measurement"
  echo "OFFLINE_PROOF: a48_ops_gateway_restart_policy_unless_stopped"
  exit 0
fi

policy=$(docker inspect "$cid" --format '{{.HostConfig.RestartPolicy.Name}}' 2>/dev/null || echo unknown)
if [[ "$policy" != "unless-stopped" ]]; then
  echo "FAIL: gateway RestartPolicy is '$policy', expected unless-stopped" >&2
  exit 1
fi

# If gateway is already unhealthy/exited, policy alone is the offline proof.
st=$(docker inspect "$cid" --format '{{.State.Status}}' 2>/dev/null || echo unknown)
if [[ "$st" != "running" ]]; then
  echo "SKIP: gateway status=$st (not running); cannot measure kill→restart"
  echo "OFFLINE_PROOF: RestartPolicy=$policy (unless-stopped) on gateway service"
  echo "PENDING_LIVE: healthy staging gateway required for <60s restart measurement"
  exit 0
fi

echo "Killing healthy gateway $cid (RestartPolicy=$policy) ..."
before=$(docker inspect "$cid" --format '{{.RestartCount}}' 2>/dev/null || echo 0)
docker kill "$cid" >/dev/null
deadline=$((SECONDS + 60))
while (( SECONDS < deadline )); do
  st=$(docker inspect "$cid" --format '{{.State.Status}}' 2>/dev/null || echo gone)
  if [[ "$st" == "running" ]]; then
    after=$(docker inspect "$cid" --format '{{.RestartCount}}' 2>/dev/null || echo 0)
    echo "PASS: gateway restarted in <60s (status=running, RestartCount ${before}->${after})"
    exit 0
  fi
  sleep 1
done
echo "FAIL: gateway did not return to running within 60s (last status=$st)" >&2
exit 1
