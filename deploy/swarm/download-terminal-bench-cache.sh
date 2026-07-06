#!/usr/bin/env bash
#
# download-terminal-bench-cache.sh — download + stage the pinned terminal-bench-2.1
# task definitions for the agent-challenge own_runner eval plane.
#
# WHY THIS EXISTS
#   own_runner reads task defs ONLY from a local cache and fails closed on any
#   digest mismatch (src/agent_challenge/evaluation/own_runner/taskdefs.py). The
#   runner image does NOT bake the ~89 task trees; they must be provisioned onto
#   named volumes. acquire-agent-challenge-cache.sh COPIES + VERIFIES an
#   already-acquired cache but deliberately does not download. THIS script is the
#   missing "download" half: it clones the byte-exact public source pinned in the
#   frozen digest manifest and lays it out as a --source dir acquire can consume.
#
#   Pipeline (see provision_agent_challenge_cache in install-swarm.sh):
#     download-terminal-bench-cache.sh --dest DIR --apply   # this script
#     acquire-agent-challenge-cache.sh --source DIR --apply # copy + digest-verify
#
# SOURCE OF TRUTH
#   The repo + revision are read from golden/dataset-digest.json
#   (public_sources.byte_exact.{repo,revision,branch_at_freeze}) so the download
#   always tracks the exact frozen pin the eval plane digest-verifies against.
#
# CACHE LAYOUT PRODUCED
#   <dest>/<task_id>/task.toml (+ instruction.md, environment/, tests/ ...), one
#   dir per manifest task. The byte-exact repo stores task dirs at repo root; the
#   only transform is stripping each task-root .gitignore (harbor strips it at
#   packaging time, and the frozen per-task digest is computed over the tree
#   WITHOUT it — nested .gitignore files are retained).
#
# SAFETY MODEL (mirrors acquire-agent-challenge-cache.sh)
#   * DEFAULT MODE IS DRY-RUN: with no --apply it prints what it would do and
#     changes nothing.
#   * Idempotent: --apply wipes and rebuilds <dest> from a fresh clone.
#   * No secrets are read, printed, or required.
#
set -euo pipefail

DEST=""
GOLDEN_FILE=""
APPLY=false

log() { printf '[download-cache] %s\n' "$*" >&2; }
die() { printf '[download-cache][FATAL] %s\n' "$*" >&2; exit 1; }

usage() {
  cat >&2 <<USAGE
Usage: $0 --dest <dir> [--golden <dataset-digest.json>] [--apply]

Required:
  --dest DIR      Output staging dir for the task cache (fed to acquire --source).

Options:
  --golden FILE   Frozen dataset-digest.json (default: <repo>/golden/dataset-digest.json).
  --apply         Actually clone + stage (default: dry-run).
  -h, --help      Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --golden) GOLDEN_FILE="$2"; shift 2 ;;
    --apply) APPLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) log "unknown argument: $1"; usage; exit 2 ;;
  esac
done

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${GOLDEN_FILE}" ]]; then
  GOLDEN_FILE="${_script_dir}/../../golden/dataset-digest.json"
fi

[[ -n "${DEST}" ]] || { log "ERROR: --dest is required"; usage; exit 2; }
[[ -f "${GOLDEN_FILE}" ]] || die "digest manifest not found: ${GOLDEN_FILE}"
command -v git >/dev/null 2>&1 || die "git is required but not found on PATH"
command -v python3 >/dev/null 2>&1 || die "python3 is required but not found on PATH"

# Read the byte-exact pin straight from the frozen manifest (single source of truth).
_read_pin() {
  python3 - "$GOLDEN_FILE" "$1" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
be = doc["public_sources"]["byte_exact"]
keys = {
    "repo": be["repo"],
    "revision": be["revision"],
    "branch": be.get("branch_at_freeze", ""),
    "task_count": str(doc["task_count"]),
}
sys.stdout.write(keys[sys.argv[2]])
PY
}

REPO="$(_read_pin repo)"
REV="$(_read_pin revision)"
BRANCH="$(_read_pin branch)"
WANT="$(_read_pin task_count)"

log "golden   : ${GOLDEN_FILE}"
log "repo     : ${REPO}"
log "revision : ${REV}"
log "branch   : ${BRANCH}"
log "tasks    : ${WANT} expected"
log "dest     : ${DEST}"

if [[ "${APPLY}" != "true" ]]; then
  log "DRY-RUN: would clone ${REPO}@${REV} and stage ${WANT} task dirs into ${DEST}"
  log "         (pass --apply to execute). Then run:"
  log "         acquire-agent-challenge-cache.sh --source ${DEST} --apply"
  exit 0
fi

workdir="$(mktemp -d -t tbench-cache-XXXXXX)"
cleanup() { rm -rf "${workdir}"; }
trap cleanup EXIT

src="${workdir}/src"
# Shallow-clone when the branch tip IS the pinned revision (the common case);
# otherwise full-clone and check the exact revision out.
if git ls-remote "${REPO}" "refs/heads/${BRANCH}" 2>/dev/null | grep -q "^${REV}[[:space:]]"; then
  log "branch tip == pinned revision; shallow clone"
  git clone --depth 1 --branch "${BRANCH}" "${REPO}" "${src}" >/dev/null 2>&1
else
  log "branch tip != pinned revision; full clone + checkout"
  git clone "${REPO}" "${src}" >/dev/null 2>&1
  git -C "${src}" checkout --quiet "${REV}"
fi

head="$(git -C "${src}" rev-parse HEAD)"
[[ "${head}" == "${REV}" ]] || die "cloned HEAD ${head} != pinned revision ${REV}"

rm -rf "${DEST:?}"
mkdir -p "${DEST}"
n=0
for d in "${src}"/*/; do
  name="$(basename "${d}")"
  [[ -f "${d}task.toml" ]] || continue
  cp -a "${d%/}" "${DEST}/${name}"
  # Strip ONLY the task-root .gitignore (harbor strips it at packaging time; the
  # frozen per-task digest is computed over the tree without it). Nested
  # .gitignore files (e.g. environment/isos/.gitignore) are intentionally kept.
  rm -f "${DEST}/${name}/.gitignore"
  n=$((n + 1))
done

log "staged ${n} task dirs into ${DEST} (expected ${WANT})"
[[ "${n}" == "${WANT}" ]] || die "staged ${n} task dirs but manifest expects ${WANT}"
log "DONE: ${DEST} ready. Next: acquire-agent-challenge-cache.sh --source ${DEST} --apply"
