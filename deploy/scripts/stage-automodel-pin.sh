#!/usr/bin/env bash
# Stage the frozen NeMo AutoModel pin for live Prism recipe 2.0 intake.
#
# Clones NVIDIA-NeMo/Automodel at the commit frozen in crates/prism-automodel,
# verifies the tree content SHA-256, and prints the PRISM_AUTOMODEL_PIN_DIR
# operators must mount/export on the prism-challenge host.
#
# Usage:
#   ./deploy/scripts/stage-automodel-pin.sh [--dir DIR] [--verify-only]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PIN_DIR="${PRISM_AUTOMODEL_PIN_DIR:-/var/lib/prism/automodel-pin}"
VERIFY_ONLY=0
GIT_URL="https://github.com/NVIDIA-NeMo/Automodel"
# Keep in sync with crates/prism-automodel/src/pin.rs
GIT_COMMIT="d02f49cb314554715aabb97e8dba6599c9f6e9e0"
WANT_SHA="f8af64ef572e2e3634dcbae7b351fdcd3c8d458caf2fe974aff26d301a11d838"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) PIN_DIR="$2"; shift 2 ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help)
      sed -n '1,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

hash_tree() {
  local pin="$1"
  python3 - <<'PY' "$pin"
import hashlib, os, sys
root = sys.argv[1]
files = []
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d != ".git" and not d.startswith(".")]
    for name in filenames:
        if name.startswith("."):
            continue
        abs_path = os.path.join(dirpath, name)
        if not os.path.isfile(abs_path):
            continue
        rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
        files.append(rel)
files.sort()
h = hashlib.sha256()
for rel in files:
    data = open(os.path.join(root, rel), "rb").read()
    h.update(rel.encode())
    h.update(b"\0")
    h.update(str(len(data)).encode())
    h.update(b"\0")
    h.update(data)
    h.update(b"\0")
print(h.hexdigest())
print(len(files), file=sys.stderr)
PY
}

if [[ "$VERIFY_ONLY" -eq 0 ]]; then
  mkdir -p "$(dirname "$PIN_DIR")"
  if [[ ! -d "$PIN_DIR/.git" ]]; then
    rm -rf "$PIN_DIR"
    git clone --filter=blob:none --no-checkout "$GIT_URL" "$PIN_DIR"
  fi
  git -C "$PIN_DIR" fetch --depth 1 origin "$GIT_COMMIT"
  git -C "$PIN_DIR" checkout --force "$GIT_COMMIT"
fi

[[ -d "$PIN_DIR" ]] || { echo "FAIL: pin dir missing: $PIN_DIR" >&2; exit 1; }
GOT_COMMIT="$(git -C "$PIN_DIR" rev-parse HEAD)"
[[ "$GOT_COMMIT" == "$GIT_COMMIT" ]] || {
  echo "FAIL: commit mismatch got=$GOT_COMMIT want=$GIT_COMMIT" >&2
  exit 1
}
GOT_SHA="$(hash_tree "$PIN_DIR")"
[[ "$GOT_SHA" == "$WANT_SHA" ]] || {
  echo "FAIL: content sha mismatch got=$GOT_SHA want=$WANT_SHA" >&2
  exit 1
}

echo "OK: AutoModel pin staged"
echo "  dir=$PIN_DIR"
echo "  commit=$GOT_COMMIT"
echo "  content_sha256=$GOT_SHA"
echo "Export for prism-challenge:"
echo "  export PRISM_AUTOMODEL_PIN_DIR=$PIN_DIR"
# Hint for local compose overrides (do not bake floating tags).
echo "Compose volume example:"
echo "  - $PIN_DIR:/var/lib/prism/automodel-pin:ro"
echo "  environment: PRISM_AUTOMODEL_PIN_DIR: /var/lib/prism/automodel-pin"
