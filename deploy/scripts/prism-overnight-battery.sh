#!/usr/bin/env bash
# Launch a full overnight Prism battery on real Lium with the public HF pack.
# Usage: ./deploy/scripts/prism-overnight-battery.sh [--evidence DIR] [--skip-pack] [--skip-recreate]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVIDENCE="${EVIDENCE:-/tmp/prism-overnight-$(date -u +%Y%m%dT%H%M%SZ)}"
PACK_DIR="${PRISM_EVAL_ASSETS_DIR:-/tmp/prism-eval-assets}"
ARTIFACT_DIR="${PRISM_ARTIFACT_DIR:-/tmp/prism-artifacts}"
OVERRIDE="${PRISM_OVERNIGHT_OVERRIDE:-/tmp/prism-overnight-compose.override.yml}"
PRISM_URL="${PRISM_URL:-http://127.0.0.1:28092}"
GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8080}"
HOTKEY="${PRISM_OVERNIGHT_HOTKEY:-343d50f1222b260aaa48e0dfd72c94b935bd14c87ec1c88bd90934193c72f534}"
SKIP_PACK=0
SKIP_RECREATE=0
POLL_S="${POLL_S:-30}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence) EVIDENCE="$2"; shift 2 ;;
    --skip-pack) SKIP_PACK=1; shift ;;
    --skip-recreate) SKIP_RECREATE=1; shift ;;
    -h|--help)
      sed -n '1,4p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$EVIDENCE" "$ARTIFACT_DIR"
SUMMARY="$EVIDENCE/SUMMARY.md"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$SUMMARY"; }

write_summary_header() {
  cat > "$SUMMARY" <<EOF
# Prism overnight battery

- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Evidence: \`$EVIDENCE\`
- Pack: \`$PACK_DIR\`
- Artifacts: \`$ARTIFACT_DIR\`
- Prism: \`$PRISM_URL\`

## Status

- state: starting
- submission_id: (pending)

## Watch

\`\`\`bash
tail -f $SUMMARY
docker logs -f base-prism-challenge-1
curl -sS $PRISM_URL/v1/submissions/\$SUB_ID
\`\`\`

EOF
}

write_summary_header

# --- pack -----------------------------------------------------------------
if [[ "$SKIP_PACK" -eq 0 ]]; then
  log "building / refreshing public pack → $PACK_DIR"
  export PRISM_EVAL_ASSETS_DIR="$PACK_DIR"
  export PACK_TIER=public
  if [[ -d /tmp/natural-packs/g5/natural ]]; then
    export G5_NATURAL_SRC=/tmp/natural-packs/g5/natural
  fi
  # Reuse existing G1/G2 if present and only refresh natural + tier when pack exists.
  if [[ -f "$PACK_DIR/g1/fresh.jsonl" && -f "$PACK_DIR/g1/domains/prose.jsonl" ]]; then
    log "G1/G2 already present — merging natural + tier.json"
    mkdir -p "$PACK_DIR/g5"
    if [[ -n "${G5_NATURAL_SRC:-}" ]]; then
      rm -rf "$PACK_DIR/g5/natural"
      cp -a "$G5_NATURAL_SRC" "$PACK_DIR/g5/natural"
    fi
    printf '%s\n' '{"tier":"public","kind":"hf_held_out_public"}' > "$PACK_DIR/tier.json"
  else
    python3 "$ROOT/crates/prism-recipe/harness/eval/build_public_pack.py" | tee "$EVIDENCE/pack-build.log"
  fi
fi

# Pack sanity
for need in \
  g1/domains/prose.jsonl g1/domains/math.jsonl g1/domains/code.jsonl g1/fresh.jsonl \
  g2/hellaswag.jsonl g5/natural/natural_mcq.jsonl tier.json
do
  [[ -f "$PACK_DIR/$need" ]] || { log "FAIL: missing $PACK_DIR/$need"; exit 1; }
done
MCQ_N=$(wc -l < "$PACK_DIR/g5/natural/natural_mcq.jsonl")
PACKED=$(tar -cz -C "$PACK_DIR" . | wc -c)
log "pack OK: natural_mcq rows=$MCQ_N packed_bytes=$PACKED tier=$(cat "$PACK_DIR/tier.json")"
if [[ "$MCQ_N" -lt 16 ]]; then
  log "WARN: natural_mcq looks like tiny fixtures ($MCQ_N rows) — G5 natural will be weak"
fi

# --- compose override -----------------------------------------------------
cat > "$OVERRIDE" <<EOF
services:
  prism-challenge:
    image: prism-challenge:0.1.0-overnight-public
    volumes:
      - ${PACK_DIR}:/tmp/prism-eval-assets:ro
      - ${ARTIFACT_DIR}:/tmp/prism-artifacts
      - ${ROOT}/crates/prism-recipe/harness:/opt/prism/harness:ro
      - /root/gbase/deploy/secrets/prism:/run/base/prism:ro
      - /root/gbase/deploy/secrets/lium:/run/base/lium:ro
      - /root/gbase/deploy/secrets/openrouter:/run/base/openrouter:ro
      - /root/gbase/deploy/secrets/github:/run/base/github:ro
      - /root/gbase/deploy/secrets/prism_sk:/run/base/challenge_sk:ro
    environment:
      PRISM_FORCE_SIM: "false"
      PRISM_FLOW: "v3"
      PRISM_EVAL_ASSETS_DIR: "/tmp/prism-eval-assets"
      PRISM_ARTIFACT_DIR: "/tmp/prism-artifacts"
      PRISM_TEST_EVAL_CAPS: "0"
      PRISM_TEST_TRAIN_MINUTES: "0"
      PRISM_TEST_MAX_PARAMS: "350000000"
      PRISM_MAX_CONCURRENT_EVALS: "1"
      PRISM_TRAIN_HOURS_CAP: "6"
      PRISM_PLAYGROUND_INFER_SCRIPT: "/opt/prism/harness/playground_infer.py"
      PRISM_ADMIN_TOKENS_FILE: "/run/base/prism/admin_tokens"
EOF
log "wrote $OVERRIDE"

COMPOSE=(
  docker compose
  -f /root/gbase/docker-compose.yml
  -f /root/gbase/deploy/compose/role-master.yml
  -f /root/gbase/deploy/compose/env-staging.yml
  -f /root/gbase/deploy/compose/env-local.yml
  -f "$OVERRIDE"
  --profile master
)

if [[ "$SKIP_RECREATE" -eq 0 ]]; then
  log "recreating prism-challenge with overnight env (no short-train knobs)"
  # Drop the previous short-train az override by not including it.
  "${COMPOSE[@]}" up -d --no-deps --force-recreate prism-challenge
fi

ENV_SNAP="$EVIDENCE/prism-env.txt"
for i in $(seq 1 60); do
  if docker exec base-prism-challenge-1 true 2>/dev/null; then
    break
  fi
  sleep 1
done
docker exec base-prism-challenge-1 env | grep -E 'PRISM_|LIUM_' | sort > "$ENV_SNAP" || true
log "env snapshot → $ENV_SNAP"
if grep -qE '^PRISM_TEST_TRAIN_MINUTES=([1-9]|0*[1-9][0-9]+)' "$ENV_SNAP"; then
  log "FAIL: PRISM_TEST_TRAIN_MINUTES still short-trains — refuse overnight"
  cat "$ENV_SNAP" | grep TRAIN || true
  exit 1
fi
if ! grep -q '^PRISM_FORCE_SIM=false$' "$ENV_SNAP"; then
  log "FAIL: PRISM_FORCE_SIM is not false"
  exit 1
fi
if ! grep -q '^PRISM_TEST_EVAL_CAPS=0$' "$ENV_SNAP"; then
  log "FAIL: need PRISM_TEST_EVAL_CAPS=0 for full battery"
  exit 1
fi
if grep -qE '^PRISM_EVAL_N_ITEMS=' "$ENV_SNAP"; then
  log "WARN: PRISM_EVAL_N_ITEMS override present — prefer unset for production item counts"
fi

# Gateway may still 503 backends — hit prism directly.
for i in $(seq 1 30); do
  if curl -sfS "$PRISM_URL/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl -sfS "$PRISM_URL/health" >/dev/null || { log "FAIL: prism /health"; exit 1; }

# --- submit baseline ------------------------------------------------------
ARCH="$ROOT/crates/prism-recipe/baselines/transformer_pp/architecture.py"
TRAIN="$ROOT/crates/prism-recipe/baselines/transformer_pp/training.py"
BODY=$(python3 - <<PY
import json, pathlib
arch = pathlib.Path("$ARCH").read_text()
train = pathlib.Path("$TRAIN").read_text()
print(json.dumps({
  "miner_hotkey": "$HOTKEY",
  "architecture_py": arch,
  "training_py": train,
}))
PY
)
HTTP=$(curl -sS -o "$EVIDENCE/submit.json" -w '%{http_code}' \
  -X POST "$PRISM_URL/v1/submissions" \
  -H 'content-type: application/json' \
  -d "$BODY" || true)
log "submit HTTP $HTTP → $EVIDENCE/submit.json"
SUB_ID=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("submission_id") or (d.get("submission") or {}).get("id") or d.get("id") or "")' "$EVIDENCE/submit.json" 2>/dev/null || true)
if [[ -z "$SUB_ID" ]]; then
  log "FAIL: no submission_id (gateway/prism reject?) — see submit.json"
  exit 1
fi
log "submission_id=$SUB_ID"
echo "$SUB_ID" > "$EVIDENCE/submission_id"

# --- poll -----------------------------------------------------------------
POLL_PID_FILE="$EVIDENCE/poll.pid"
(
  set +e
  while true; do
    curl -sS "$PRISM_URL/v1/submissions/$SUB_ID" > "$EVIDENCE/status.json" 2>/dev/null || true
    STATE=$(python3 - <<PY 2>/dev/null
import json
try:
    d=json.load(open("$EVIDENCE/status.json"))
except Exception:
    print("unknown"); raise SystemExit
sub=d.get("submission") or d
print(sub.get("status") or sub.get("state") or d.get("status") or "unknown")
PY
)
    STAGE=$(python3 - <<PY 2>/dev/null
import json
try:
    d=json.load(open("$EVIDENCE/status.json"))
except Exception:
    print(""); raise SystemExit
sub=d.get("submission") or d
print(sub.get("stage") or sub.get("current_stage") or d.get("stage") or "")
PY
)
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    # rewrite status block
    python3 - <<PY
from pathlib import Path
p = Path("$SUMMARY")
text = p.read_text() if p.exists() else ""
marker = "## Status\n"
tail = text.split(marker, 1)
head = tail[0] + marker if tail else marker
rest = ""
if len(tail) > 1 and "\n## Watch" in tail[1]:
    rest = "\n## Watch" + tail[1].split("\n## Watch", 1)[1]
elif len(tail) > 1:
    # keep Watch section if present later
    idx = tail[1].find("\n## ")
    rest = tail[1][idx:] if idx >= 0 else ""
body = f"""
- updated: {TS}
- state: {STATE}
- stage: {STAGE}
- submission_id: `{SUB_ID}`
- poll_pid: {__import__('os').getpid()}
- evidence: `{EVIDENCE}`
"""
p.write_text(head + body + rest)
PY
    case "$STATE" in
      scored|failed|rejected|error|terminal|complete|completed|Score*|Ineligible*|Eligible*)
        log "terminal state=$STATE stage=$STAGE"
        curl -sS "$PRISM_URL/v1/submissions/$SUB_ID" > "$EVIDENCE/final.json" || true
        # metrics dump if present
        python3 - <<'PY' > "$EVIDENCE/metrics-keys.txt" 2>/dev/null || true
import json
from pathlib import Path
d=json.load(open("'"$EVIDENCE"'/final.json"))
bat=(d.get("battery") or d.get("metrics") or {})
flat=bat.get("metrics") if isinstance(bat, dict) else {}
if not flat and isinstance(d.get("org_metrics"), dict):
    flat=d["org_metrics"]
keys=sorted(flat.keys()) if isinstance(flat, dict) else []
print("\n".join(keys))
for k in ("org.g1.bits_per_byte_prose","org.g1.bits_per_byte_math","org.g1.bits_per_byte_code","org.g1.bits_per_byte_fresh_crawl","org.g8.mup_lr_stability","org.g6.auc_log_tokens"):
    print(f"HAS {k}: {k in (flat or {})}")
PY
        exit 0
        ;;
    esac
    sleep "$POLL_S"
  done
) >> "$EVIDENCE/poll.log" 2>&1 &
echo $! > "$POLL_PID_FILE"
log "poller pid=$(cat "$POLL_PID_FILE") log=$EVIDENCE/poll.log"
log "overnight launched — leave poller running; wall clock is multi-hour"
echo "EVIDENCE=$EVIDENCE"
echo "SUB_ID=$SUB_ID"
echo "POLL_PID=$(cat "$POLL_PID_FILE")"
echo "SUMMARY=$SUMMARY"
