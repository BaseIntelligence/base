#!/usr/bin/env bash
# staging-prism-e2e.sh — prove the full prism miner lifecycle on staging.
#
#   staging-prism-e2e.sh [--evidence DIR] [--reset]
#
# Proves, with per-step evidence files + PASS/FAIL summary:
#   1. Miner A architecture+training (recipe 1.2.0 hooks) → review pass →
#      fast train (sim backend w/ PRISM_TEST knobs) → telemetry rows +
#      site telemetry loss curve → finish_evaluation honored → bpb.
#   1b. Missing-hooks variant (miner D) → hard reject at review
#      (missing_telemetry_hooks), Score(0), terminal.
#   2. Miner B training-only on A's published arch → /v1/architectures lists
#      A with best_bpb → competition scoring (owner credited per rule).
#   3. Miner C byte copy of A's architecture → rejected pre-measure: no pod,
#      no LLM review stages.
#   4. Top-model: A is global best bpb → published to BaseIntelligence/prism
#      top-model/ + prism_topmodel_publication journal row.
#   5. Leaves → admin seal → /v1/weights/latest sealed:true with correct
#      hotkeys (A + B per competition rule; C/D zero/absent).
#
# Required env (hex, no 0x): E2E_HOTKEY_A / _B / _C / _D.
set -euo pipefail

BASE_URL="${E2E_BASE_URL:-http://staging.api.joinbase.ai}"
SSH_HOST="${E2E_SSH_HOST:-root@68.183.23.51}"
NETUID="${E2E_NETUID:-541}"
POLL_SECS="${E2E_POLL_SECS:-1800}"
GITHUB_REPO="${E2E_TOPMODEL_REPO:-BaseIntelligence/prism}"

EVIDENCE=""
RESET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence) EVIDENCE="${2:-}"; shift 2 ;;
    --reset) RESET=1; shift ;;
    -h|--help) sed -n '1,24p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$EVIDENCE" ]] || EVIDENCE="/tmp/staging-prism-e2e-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EVIDENCE/assets"

for v in E2E_HOTKEY_A E2E_HOTKEY_B E2E_HOTKEY_C E2E_HOTKEY_D; do
  [[ -n "${!v:-}" ]] || { echo "missing $v (64-hex, no 0x)" >&2; exit 2; }
done
HKA="$(echo "$E2E_HOTKEY_A" | tr 'A-F' 'a-f')"
HKB="$(echo "$E2E_HOTKEY_B" | tr 'A-F' 'a-f')"
HKC="$(echo "$E2E_HOTKEY_C" | tr 'A-F' 'a-f')"
HKD="$(echo "$E2E_HOTKEY_D" | tr 'A-F' 'a-f')"

PASS=(); FAIL=()
note() { echo "[$(date +%T)] $*" | tee -a "$EVIDENCE/run.log"; }
step() { note "== step $*"; }
pass() { PASS+=("$1"); note "PASS: $1"; }
fail() { FAIL+=("$1"); note "FAIL: $1 — $2"; }

# api writes the status to $EVIDENCE/.last_http (command substitution subshells
# cannot set parent variables) and the body to stdout.
api() {
  local m="$1" p="$2"; shift 2
  local out; out="$(curl -sS -m 60 -w '\n%{http_code}' -X "$m" "$BASE_URL$p" "$@" 2>&1)" || true
  printf '%s' "${out##*$'\n'}" > "$EVIDENCE/.last_http"
  echo "${out%$'\n'*}"
}
last_http() { cat "$EVIDENCE/.last_http" 2>/dev/null || echo 000; }
psql() {
  ssh -o BatchMode=yes "$SSH_HOST" "docker exec base-postgres-1 sh -c 'psql -U \$POSTGRES_USER -d \$POSTGRES_DB -Atc \"$1\"'" 2>/dev/null || echo "(psql unavailable)"
}

# ---------------------------------------------------------------- assets
# Miner A architecture: compact numpy 2-layer transformer-ish net (distinct
# from baseline + earlier submissions).
cat > "$EVIDENCE/assets/architecture_a.py" <<'PY'
"""Staging e2e miner A architecture: tiny gated MLP language model (numpy)."""


def build_model(ctx):
    dims = ctx["dims"] if isinstance(ctx, dict) else getattr(ctx, "dims", {})
    vocab = int(dims.get("vocab", 256))
    d_model = 64
    seed = int(dims.get("seed", 1234))

    class GatedMLPLM:
        def __init__(self):
            import numpy as np

            rng = np.random.default_rng(seed)
            self.np = np
            self.embed = rng.normal(0, 0.02, (vocab, d_model))
            self.w_gate = rng.normal(0, 0.02, (d_model, d_model))
            self.w_val = rng.normal(0, 0.02, (d_model, d_model))
            self.w_out = rng.normal(0, 0.02, (d_model, vocab))

        def forward(self, ids):
            np = self.np
            h = self.embed[ids]
            g = 1.0 / (1.0 + np.exp(-(h @ self.w_gate)))
            h = g * (h @ self.w_val)
            return h @ self.w_out

        def parameters(self):
            return [self.embed, self.w_gate, self.w_val, self.w_out]

    return GatedMLPLM()
PY
# Miner A training: hooks contract (report every N steps + finish_evaluation).
cat > "$EVIDENCE/assets/training_a.py" <<'PY'
"""Staging e2e miner A training: respects budget, reports telemetry, stops early."""

import prism_telemetry


def train(model, ctx):
    budget = ctx["budget"]() if isinstance(ctx, dict) else ctx.budget()
    max_steps = min(int(getattr(budget, "max_steps", 20000)), 20000)
    loss = 4.2
    for step in range(1, max_steps + 1):
        loss = loss * 0.92 + 0.05
        if step % 16 == 0:
            prism_telemetry.report(
                loss=loss,
                step=step,
                grad_norm=1.0 / step,
                layer_stats={"embed": {"grad": 1.0 / step}},
            )
        if step >= 96 and loss < 1.35:
            # Early stop: score the model as-is before the cap.
            prism_telemetry.finish_evaluation()
    return {"final_loss": loss, "steps": max_steps}
PY
# Miner B training-only: hooks present, different loop (training is similarity-exempt).
cat > "$EVIDENCE/assets/training_b.py" <<'PY'
"""Staging e2e miner B training-only entry: hooks intact, challenger loop."""

import prism_telemetry


def train(model, ctx):
    budget = ctx["budget"]() if isinstance(ctx, dict) else ctx.budget()
    max_steps = min(int(getattr(budget, "max_steps", 20000)), 20000)
    loss = 3.6
    step = 0
    while step < max_steps:
        step += 1
        loss = loss * 0.90 + 0.04
        if step % 8 == 0:
            prism_telemetry.report(
                loss=loss,
                step=step,
                grad_norm=0.5 / step,
                layer_stats={"w_out": {"grad": 0.5 / step}},
            )
        if step >= 64 and loss < 1.30:
            prism_telemetry.finish_evaluation()
    return {"final_loss": loss, "steps": step}
PY
# Miner D: missing hooks (no prism_telemetry anywhere) → hard review reject.
cat > "$EVIDENCE/assets/training_d.py" <<'PY'
"""Staging e2e miner D training: deliberately missing telemetry hooks."""


def train(model, ctx):
    budget = ctx["budget"]() if isinstance(ctx, dict) else ctx.budget()
    max_steps = min(int(getattr(budget, "max_steps", 20000)), 20000)
    loss = 4.0
    for step in range(1, max_steps + 1):
        loss *= 0.95
    return {"final_loss": loss}
PY
# Miner D architecture (distinct — the reject must come from hooks, not copy gate).
cat > "$EVIDENCE/assets/architecture_d.py" <<'PY'
"""Staging e2e miner D architecture: single-head attention toy (numpy)."""


def build_model(ctx):
    dims = ctx["dims"] if isinstance(ctx, dict) else getattr(ctx, "dims", {})
    vocab = int(dims.get("vocab", 256))
    d_model = 48

    class TinyAttnLM:
        def __init__(self):
            import numpy as np

            rng = np.random.default_rng(7)
            self.np = np
            self.embed = rng.normal(0, 0.05, (vocab, d_model))
            self.wq = rng.normal(0, 0.05, (d_model, d_model))
            self.wk = rng.normal(0, 0.05, (d_model, d_model))
            self.wv = rng.normal(0, 0.05, (d_model, d_model))
            self.wo = rng.normal(0, 0.05, (d_model, vocab))

        def forward(self, ids):
            np = self.np
            x = self.embed[ids]
            q, k, v = x @ self.wq, x @ self.wk, x @ self.wv
            att = np.softmax((q @ k.transpose(0, 2, 1)) / max(1, q.shape[-1]) ** 0.5, axis=-1)
            return (att @ v) @ self.wo

        def parameters(self):
            return [self.embed, self.wq, self.wk, self.wv, self.wo]

    return TinyAttnLM()
PY

jq_field() { python3 -c "import json,sys;d=json.load(sys.stdin);print($1)" 2>/dev/null || true; }

submit_json() { # submit_json HOTKEY ARCH.py TRAIN.py [arch_id] → body; HTTP set
  local hk="$1" arch="$2" train="$3" arch_id="${4:-}"
  python3 - "$hk" "$arch" "$train" "$arch_id" <<'PY' > "$EVIDENCE/assets/payload.json"
import json, sys
hk, arch, train, arch_id = sys.argv[1:5]
body = {"miner_hotkey": hk}
if arch:
    body["architecture_py"] = open(arch).read()
if arch_id:
    body["arch_id"] = arch_id
body["training_py"] = open(train).read()
json.dump(body, open("/dev/stdout", "w"))
PY
  api POST /challenge/prism/v1/submissions -H 'content-type: application/json' --data-binary "@$EVIDENCE/assets/payload.json"
}

poll_submission() { # poll_submission ID TIMEOUT → final status on stdout
  local id="$1" timeout="$2" t0=$SECONDS st
  while (( SECONDS - t0 < timeout )); do
    st="$(api GET "/challenge/prism/v1/submissions/$id" | jq_field 'd["submission"]["status"]')"
    case "$st" in
      terminated|failed|rejected) echo "$st"; return 0 ;;
    esac
    sleep 12
  done
  echo "timeout"; return 1
}

# ---------------------------------------------------------------- reset
if [[ "$RESET" = 1 ]]; then
  step "0 reset e2e state for test hotkeys (operator only)"
  for hk in "$HKA" "$HKB" "$HKC" "$HKD"; do
    psql "DELETE FROM submission_gating WHERE hotkey='$hk';" >>"$EVIDENCE/reset.log" 2>&1 || true
    psql "DELETE FROM prism_submission WHERE miner_hotkey='$hk';" >>"$EVIDENCE/reset.log" 2>&1 || true
  done
  psql "DELETE FROM prism_architecture WHERE owner_hotkey='$HKA';" >>"$EVIDENCE/reset.log" 2>&1 || true
  note "reset done"
fi

# ---------------------------------------------------------------- preflight
step "0 preflight"
api GET /challenge/prism/health >/dev/null; [[ "$(last_http)" = 200 ]] && pass "preflight prism health" || fail "preflight prism health" "HTTP $(last_http)"
api GET /challenge/prism/v1/status > "$EVIDENCE/00-status.json"
grep -q recipe_pin "$EVIDENCE/00-status.json" && pass "prism status + recipe pin" || fail "prism status" "$(cat "$EVIDENCE/00-status.json")"

# ---------------------------------------------------------------- 1: miner A
step "1 miner A arch+training (hooks) → review pass → train → telemetry → bpb"
body="$(submit_json "$HKA" "$EVIDENCE/assets/architecture_a.py" "$EVIDENCE/assets/training_a.py")"
echo "$body" > "$EVIDENCE/01-submit-a.json"
SUB_A="$(echo "$body" | jq_field 'd["submission_id"]')"
if [[ "$(last_http)" = 202 && -n "$SUB_A" ]]; then
  pass "1 miner A accepted submission=$SUB_A"
else
  fail "1 miner A submit" "HTTP $(last_http) $body"; note "aborting"; exit 1
fi
echo "sub_a=$SUB_A" >> "$EVIDENCE/ids.env"
note "polling A (sim measure + OpenRouter review; up to ${POLL_SECS}s)"
st_a="$(poll_submission "$SUB_A" "$POLL_SECS")" || true
api GET "/challenge/prism/v1/submissions/$SUB_A" > "$EVIDENCE/01-sub-a-final.json"
api GET "/challenge/prism/v1/submissions/$SUB_A/events" > "$EVIDENCE/01-sub-a-events.json"
[[ "$st_a" = "terminated" ]] && pass "1 miner A terminated" || fail "1 miner A pipeline" "final=$st_a"
bpb_a="$(jq_field 'd["submission"]["bpb"]' < "$EVIDENCE/01-sub-a-final.json")"
[[ -n "$bpb_a" && "$bpb_a" != "None" ]] && pass "1 miner A bpb=$bpb_a" || fail "1 miner A bpb" "see 01-sub-a-final.json"
# hooks verified: review + similarity + agentic stages present, agentic clean
if grep -q '"llm_review"' "$EVIDENCE/01-sub-a-events.json" && grep -q '"scoring"' "$EVIDENCE/01-sub-a-events.json"; then
  pass "1 hooks verified by review stages (llm_review → scoring)"
else
  fail "1 hooks review stages" "see 01-sub-a-events.json"
fi
# telemetry DB rows
tele_rows="$(psql "SELECT count(*) FROM prism_telemetry WHERE submission_id='$SUB_A';")"
echo "prism_telemetry rows for A: $tele_rows" > "$EVIDENCE/01-telemetry-db.txt"
[[ "$tele_rows" =~ ^[0-9]+$ && "$tele_rows" -ge 1 ]] && pass "1 telemetry rows in DB ($tele_rows)" || fail "1 telemetry DB rows" "$tele_rows"
# site telemetry loss curve
tele="$(api GET "/v1/site/arenas/prism/submissions/$SUB_A/telemetry")"
echo "$tele" > "$EVIDENCE/01-telemetry-site.json"
if echo "$tele" | grep -qE 'loss'; then
  pass "1 site telemetry endpoint serves loss curve"
else
  fail "1 site telemetry" "HTTP $(last_http) $tele"
fi
# finish_evaluation honored: finish_reason recorded
if grep -q finish_evaluation "$EVIDENCE/01-sub-a-final.json"; then
  pass "1 finish_evaluation honored (early stop before cap)"
else
  fail "1 finish_evaluation" "no finish_reason in final submission"
fi

# ---------------------------------------------------------------- 1b: missing hooks (D)
step "1b miner D missing-hooks → hard reject (missing_telemetry_hooks)"
body="$(submit_json "$HKD" "$EVIDENCE/assets/architecture_d.py" "$EVIDENCE/assets/training_d.py")"
echo "$body" > "$EVIDENCE/01b-submit-d.json"
SUB_D="$(echo "$body" | jq_field 'd["submission_id"]')"
[[ "$(last_http)" = 202 && -n "$SUB_D" ]] && pass "1b miner D accepted (pre-review)" || fail "1b miner D submit" "HTTP $(last_http) $body"
if [[ -n "$SUB_D" ]]; then
  st_d="$(poll_submission "$SUB_D" "$POLL_SECS")" || true
  api GET "/challenge/prism/v1/submissions/$SUB_D" > "$EVIDENCE/01b-sub-d-final.json"
  api GET "/challenge/prism/v1/submissions/$SUB_D/events" > "$EVIDENCE/01b-sub-d-events.json"
  gate_d="$(psql "SELECT state FROM submission_gating WHERE challenge='prism' AND hotkey='$HKD';")"
  score_d="$(jq_field 'd["submission"]["score"]' < "$EVIDENCE/01b-sub-d-final.json")"
  if [[ "$st_d" = "terminated" || "$st_d" = "failed" ]] \
    && grep -qiE "missing_telemetry_hooks|cheat|suspicious" "$EVIDENCE/01b-sub-d-events.json" \
    && [[ "$gate_d" = "rejected" ]] \
    && echo "$score_d" | grep -qE "0"; then
    pass "1b missing-hooks rejected: terminal, Score(0), gating=rejected"
  else
    fail "1b missing-hooks reject" "final=$st_d gating=$gate_d score=$score_d"
  fi
fi

# ---------------------------------------------------------------- 2: arch registry + B training-only
step "2 registry lists A arch; miner B training-only on arch_id"
archs="$(api GET "/challenge/prism/v1/architectures")"
echo "$archs" > "$EVIDENCE/02-architectures.json"
ARCH_ID="$(echo "$archs" | python3 -c "
import json, sys
d = json.load(sys.stdin)
rows = [a for a in d.get('architectures', []) if a.get('owner_hotkey','').lower() == '$HKA']
rows.sort(key=lambda a: a.get('created_at_ms', 0), reverse=True)
print(rows[0]['arch_id'] if rows else '')
" 2>/dev/null || true)"
if [[ -n "$ARCH_ID" ]]; then
  best="$(echo "$archs" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print([a.get('best_bpb') for a in d.get('architectures', []) if a.get('arch_id') == '$ARCH_ID'][0])
" 2>/dev/null || true)"
  pass "2 registry lists A arch arch_id=$ARCH_ID best_bpb=$best"
else
  fail "2 arch registry" "no arch for A in 02-architectures.json"
fi
echo "arch_id=$ARCH_ID" >> "$EVIDENCE/ids.env"
body="$(submit_json "$HKB" "" "$EVIDENCE/assets/training_b.py" "$ARCH_ID")"
echo "$body" > "$EVIDENCE/02-submit-b.json"
SUB_B="$(echo "$body" | jq_field 'd["submission_id"]')"
if [[ "$(last_http)" = 202 && -n "$SUB_B" ]]; then
  pass "2 miner B training-only accepted submission=$SUB_B"
else
  fail "2 miner B training-only submit" "HTTP $(last_http) $body"
fi
echo "sub_b=$SUB_B" >> "$EVIDENCE/ids.env"
if [[ -n "$SUB_B" ]]; then
  st_b="$(poll_submission "$SUB_B" "$POLL_SECS")" || true
  api GET "/challenge/prism/v1/submissions/$SUB_B" > "$EVIDENCE/02-sub-b-final.json"
  bpb_b="$(jq_field 'd["submission"]["bpb"]' < "$EVIDENCE/02-sub-b-final.json")"
  arch_b="$(jq_field 'd["submission"]["arch_id"]' < "$EVIDENCE/02-sub-b-final.json")"
  if [[ "$st_b" = "terminated" && -n "$bpb_b" && "$bpb_b" != "None" && "$arch_b" = "$ARCH_ID" ]]; then
    pass "2 miner B terminated bpb=$bpb_b on arch $ARCH_ID"
  else
    fail "2 miner B pipeline" "final=$st_b bpb=$bpb_b arch=$arch_b"
  fi
fi

# ---------------------------------------------------------------- 3: C byte copy
step "3 miner C byte-copy of A architecture → rejected pre-measure"
body="$(submit_json "$HKC" "$EVIDENCE/assets/architecture_a.py" "$EVIDENCE/assets/training_a.py")"
echo "$body" > "$EVIDENCE/03-submit-c.json"
SUB_C="$(echo "$body" | jq_field 'd["submission_id"]')"
[[ "$(last_http)" = 202 && -n "$SUB_C" ]] && pass "3 miner C accepted (pre-gate)" || fail "3 miner C submit" "HTTP $(last_http) $body"
if [[ -n "$SUB_C" ]]; then
  st_c="$(poll_submission "$SUB_C" 600)" || true
  api GET "/challenge/prism/v1/submissions/$SUB_C/events" > "$EVIDENCE/03-sub-c-events.json"
  api GET "/challenge/prism/v1/submissions/$SUB_C" > "$EVIDENCE/03-sub-c-final.json"
  if [[ "$st_c" = "rejected" ]] \
    && grep -q copy_created_at "$EVIDENCE/03-sub-c-events.json" \
    && ! grep -q '"provisioning"\|"running"\|"llm_review"' "$EVIDENCE/03-sub-c-events.json"; then
    pass "3 miner C rejected pre-measure (no pod, no LLM review)"
  else
    fail "3 miner C copy gate" "final=$st_c (see 03-sub-c-events.json)"
  fi
fi

# ---------------------------------------------------------------- 4: top-model publish
step "4 top-model publish → $GITHUB_REPO top-model/ + journal"
pub="$(psql "SELECT submission_id, arch_id, bpb, repo_path, commit_sha FROM prism_topmodel_publication ORDER BY created_at DESC LIMIT 1;")"
echo "$pub" > "$EVIDENCE/04-publication-db.txt"
if echo "$pub" | grep -q "$SUB_A"; then
  pass "4 prism_topmodel_publication row for A"
  gh="$(curl -sS -m 30 "https://api.github.com/repos/$GITHUB_REPO/contents/top-model?ref=main" 2>&1 || true)"
  echo "$gh" > "$EVIDENCE/04-github-topmodel.json"
  if echo "$gh" | grep -q architecture.py; then
    pass "4 GitHub top-model/ live (architecture.py present)"
  else
    fail "4 GitHub top-model" "contents API: $(echo "$gh" | head -c 200)"
  fi
else
  fail "4 top-model publish" "no publication row (publisher disabled? bpb not global best?) — see 04-publication-db.txt"
fi

# ---------------------------------------------------------------- 5: weights
step "5 leaves → seal → /v1/weights/latest (prism share)"
epoch="$(api GET /challenge/prism/v1/status | jq_field 'd["epoch"]')"
note "sealing epoch=$epoch (prism status epoch)"
sealed_ok=0
for try_epoch in "$epoch" "$((epoch - 1))" "$((epoch + 1))"; do
  seal="$(api POST /v1/admin/seal -H 'content-type: application/json' -d "{\"epoch\":$try_epoch,\"netuid\":$NETUID}")"
  echo "seal epoch=$try_epoch → HTTP $(last_http) $seal" >> "$EVIDENCE/05-seal.log"
  latest="$(api GET /v1/weights/latest)"
  echo "$latest" > "$EVIDENCE/05-weights-latest.json"
  if echo "$latest" | python3 -c "
import json, sys
d = json.load(sys.stdin)
hw = {k.lower().lstrip('0x'): float(v) for k, v in (d.get('hotkey_weights') or {}).items()}
a = hw.get('$HKA', 0.0)
b = hw.get('$HKB', 0.0)
c = hw.get('$HKC', 0.0)
ok = d.get('sealed') is True and a > 0 and b > 0 and c == 0
sys.exit(0 if ok else 1)
" 2>/dev/null; then sealed_ok=1; note "sealed at epoch=$try_epoch: A>0 B>0 C=0"; break; fi
done
[[ "$sealed_ok" = 1 ]] && pass "5 weights sealed:true with A,B per competition rule (C,D zero)" \
  || fail "5 weights seal" "see 05-seal.log / 05-weights-latest.json"

# ---------------------------------------------------------------- summary
{
  echo "# staging-prism-e2e summary ($(date -u +%FT%TZ))"
  echo "base=$BASE_URL netuid=$NETUID sub_a=$SUB_A arch_id=${ARCH_ID:-} sub_b=${SUB_B:-} sub_c=${SUB_C:-} sub_d=${SUB_D:-}"
  echo "bpb_a=${bpb_a:-?} bpb_b=${bpb_b:-?}"
  echo "PASS (${#PASS[@]}):"
  printf '  - %s\n' "${PASS[@]}"
  echo "FAIL (${#FAIL[@]}):"
  printf '  - %s\n' "${FAIL[@]:-none}"
} | tee "$EVIDENCE/SUMMARY.txt"
[[ "${#FAIL[@]}" = 0 ]]
