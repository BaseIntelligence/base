#!/usr/bin/env bash
# After docker build (local or CI), record image digests keyed by commit SHA.
#
#   record-image-digests.sh [--out deploy/digests/<sha>.json] [image:tag ...]
#
# Default images: validator:0.1.0 gateway:0.1.0 updater:0.1.0 agent-challenge:0.1.0 base-agent:0.1.0
# Writes JSON:
#   { "commit_sha", "created_at", "images": { "validator": { "id", "digest", "repo_digest", "tag" } } }
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=lib-promote.sh
source "${SCRIPT_DIR}/lib-promote.sh"

require_cmd docker
require_cmd python3

OUT=""
IMAGES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) IMAGES+=("$1"); shift ;;
  esac
done

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  IMAGES=(validator:0.1.0 gateway:0.1.0 updater:0.1.0 agent-challenge:0.1.0 base-agent:0.1.0)
fi

COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
SHORT="$(git -C "$ROOT" rev-parse --short HEAD)"
if [[ -z "$OUT" ]]; then
  OUT="${ROOT}/deploy/digests/${COMMIT}.json"
fi
mkdir -p "$(dirname "$OUT")"

python3 - "$OUT" "$COMMIT" "$SHORT" "${IMAGES[@]}" <<'PY'
import json, subprocess, sys
from datetime import datetime, timezone

out_path = sys.argv[1]
commit = sys.argv[2]
short = sys.argv[3]
images = sys.argv[4:]

def inspect(tag: str) -> dict:
    raw = subprocess.check_output(
        ["docker", "image", "inspect", tag, "--format", "{{json .}}"],
        text=True,
    )
    data = json.loads(raw)
    image_id = data.get("Id", "")
    # Prefer RepoDigests entry; else use Id as local digest pin.
    repo_digests = data.get("RepoDigests") or []
    repo_digest = repo_digests[0] if repo_digests else ""
    digest = ""
    if "@sha256:" in repo_digest:
        digest = "sha256:" + repo_digest.split("@sha256:", 1)[1]
    elif image_id.startswith("sha256:"):
        digest = image_id
    name = tag.split(":")[0]
    # Service key = last path segment. Legacy local tags base-<svc> map to <svc>
    # for validator/gateway/updater only; base-agent keeps its full name.
    base = name.rsplit("/", 1)[-1]
    legacy = {"base-validator": "validator", "base-gateway": "gateway", "base-updater": "updater"}
    service = legacy.get(base, base)
    pinned = f"{name}@{digest}" if digest else ""
    return {
        "service": service,
        "tag": tag,
        "repository": name,
        "id": image_id,
        "digest": digest,
        "repo_digest": repo_digest,
        "image": pinned,
    }

payload = {
    "commit_sha": commit,
    "commit_short": short,
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "images": {},
}
for tag in images:
    try:
        info = inspect(tag)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"docker inspect failed for {tag}: {e}") from e
    if not info["digest"]:
        raise SystemExit(f"no digest for {tag}")
    payload["images"][info["service"]] = info

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(out_path)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
