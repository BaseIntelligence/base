#!/usr/bin/env bash
# staging-design-e2e.sh — prove the full design miner lifecycle on staging.
#
#   staging-design-e2e.sh [--evidence DIR] [--reset] [--skip-wait]
#
# Proves, with per-step evidence files + PASS/FAIL summary:
#   1. Miner A multi-file ZIP (+ X-Env-Json) → accepted → next round →
#      install/run in Docker → sanitize → agentic review (Docker backend) →
#      awaiting_admin; pages served (index/pricing/components) and the
#      submit-time env is the one the run sees (env lock).
#   1b. Post-submit env change re-POST → 409 submission_gated (env locked at
#      submit; also the 1-max gating proof).
#   2. Round waits: accepted harness runs in the NEXT round, not the current.
#   3. Admin candidates → winners → points (leaderboard).
#   4. Leaves → gateway → admin seal → /v1/weights/latest sealed:true with A.
#   5. Miner B byte-copy → terminal `rejected` via created_at copy gate, no
#      LLM review stage in events.
#   6. Miner D broken pyproject → auto-retry x3 (`auto_retry` events) →
#      terminal `failed` + gating row `blocked`.
#   7. Hotkey-change reset: on-chain hotkey swap (btcli) removes B's hotkey
#      from the metagraph → watcher reconciles gating row back to `open`
#      (DB evidence via ssh+psql). Set E2E_SKIP_SWAP=1 to only report the
#      current row (documented no-op).
#   8. Unregistered hotkey U → 403 hotkey_not_in_metagraph.
#
# Required env (hex, no 0x): E2E_HOTKEY_A / _B / _D / _U. Admin bearer read
# from ${E2E_ADMIN_TOKEN_FILE} (default deploy/secrets/design/annotator_tokens).
set -euo pipefail

BASE_URL="${E2E_BASE_URL:-http://staging.api.joinbase.ai}"
SSH_HOST="${E2E_SSH_HOST:-root@68.183.23.51}"
NETUID="${E2E_NETUID:-541}"
TOKEN_FILE="${E2E_ADMIN_TOKEN_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/secrets/design/annotator_tokens}"
ROUND_SECS="${E2E_ROUND_SECS:-900}"
RUN_WAIT_SECS="${E2E_RUN_WAIT_SECS:-2400}"   # round wait + execute + review
RETRY_WAIT_SECS="${E2E_RETRY_WAIT_SECS:-1500}"
WATCHER_WAIT_SECS="${E2E_WATCHER_WAIT_SECS:-420}"

EVIDENCE=""
RESET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence) EVIDENCE="${2:-}"; shift 2 ;;
    --reset) RESET=1; shift ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$EVIDENCE" ]] || EVIDENCE="/tmp/staging-design-e2e-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EVIDENCE"

for v in E2E_HOTKEY_A E2E_HOTKEY_B E2E_HOTKEY_D E2E_HOTKEY_U; do
  [[ -n "${!v:-}" ]] || { echo "missing $v (64-hex, no 0x)" >&2; exit 2; }
done
HKA="$(echo "$E2E_HOTKEY_A" | tr 'A-F' 'a-f')"
HKB="$(echo "$E2E_HOTKEY_B" | tr 'A-F' 'a-f')"
HKD="$(echo "$E2E_HOTKEY_D" | tr 'A-F' 'a-f')"
HKU="$(echo "$E2E_HOTKEY_U" | tr 'A-F' 'a-f')"

PASS=(); FAIL=()
note() { echo "[$(date +%T)] $*" | tee -a "$EVIDENCE/run.log"; }
step() { note "== step $*"; }
pass() { PASS+=("$1"); note "PASS: $1"; }
fail() { FAIL+=("$1"); note "FAIL: $1 — $2"; }

# api writes the status to $EVIDENCE/.last_http (command substitution subshells
# cannot set parent variables) and the body to stdout.
api() { # api METHOD PATH [curl args...] → body on stdout, status via last_http
  local m="$1" p="$2"; shift 2
  local out; out="$(curl -sS -m 60 -w '\n%{http_code}' -X "$m" "$BASE_URL$p" "$@" 2>&1)" || true
  printf '%s' "${out##*$'\n'}" > "$EVIDENCE/.last_http"
  echo "${out%$'\n'*}"
}
last_http() { cat "$EVIDENCE/.last_http" 2>/dev/null || echo 000; }
# Admin endpoints are not proxied by the gateway (403 by design); reach the
# master-local challenge port through ssh + docker exec.
api_admin() { # api_admin METHOD PATH [json-data] → body on stdout
  local m="$1" p="$2" data="${3:-}" cmd
  cmd="curl -sS -m 30 -w '\n%{http_code}' -X $(printf %q "$m") $(printf %q "http://127.0.0.1:8093$p") -H $(printf %q "authorization: Bearer $TOKEN")"
  [[ -n "$data" ]] && cmd="$cmd -H $(printf %q 'content-type: application/json') --data-binary $(printf %q "$data")"
  local out; out="$(ssh -o BatchMode=yes "$SSH_HOST" "docker exec base-design-challenge-1 sh -c $(printf %q "$cmd")" 2>&1)" || true
  printf '%s' "${out##*$'\n'}" > "$EVIDENCE/.last_http"
  echo "${out%$'\n'*}"
}
psql() { # psql SQL → host postgres via ssh (evidence only)
  local sql="$1" cmd
  cmd="psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -Atc $(printf %q "$sql")"
  ssh -o BatchMode=yes "$SSH_HOST" "docker exec base-postgres-1 sh -c $(printf %q "$cmd")" 2>/dev/null || echo "(psql unavailable)"
}

# ---------------------------------------------------------------- assets
mkdir -p "$EVIDENCE/assets"
cat > "$EVIDENCE/assets/agent.py" <<'PY'
"""Staging e2e miner A: multi-file harness (imports helpers, reads locked env)."""
from __future__ import annotations

import html
import os

import helpers

PAGES = (
    ("index.html", "hero landing with value proposition and primary CTA"),
    ("pricing.html", "three pricing tiers with feature comparison"),
    ("components.html", "showcase of reusable UI components (cards, badges, tables)"),
)


def run(task, llm, out) -> None:
    prompt = getattr(task, "prompt", "") or "staging e2e product"
    brand = os.environ.get("BRAND_MARK", "env-missing")
    for page, focus in PAGES:
        body = helpers.render_page(page, focus, prompt, brand)
        out.write_page(page, body)
PY
cat > "$EVIDENCE/assets/helpers.py" <<'PY'
"""Extra module proving multi-file ZIP support in the sandbox."""
from __future__ import annotations

import html


def render_page(page: str, focus: str, prompt: str, brand: str) -> str:
    title = page.replace(".html", "").title()
    safe_prompt = html.escape(prompt)
    safe_brand = html.escape(brand)
    return f"""<!DOCTYPE html>
<html lang="en" data-agent="staging-e2e-a" data-page="{html.escape(page)}">
<head><meta charset="utf-8"/><title>{html.escape(title)} · {safe_brand}</title>
<style>body{{font-family:sans-serif;background:#101418;color:#e8e4da;margin:0}}
header{{padding:3rem 1.5rem}}main{{padding:1rem 1.5rem;max-width:60rem;margin:auto}}
.brand{{color:#3dd6c6;letter-spacing:.12em;text-transform:uppercase;font-size:.8rem}}
.card{{border:1px solid #2a3441;border-radius:8px;padding:1rem;margin:.5rem 0;background:#171e27}}</style>
</head>
<body><header><p class="brand">brand: {safe_brand}</p><h1>{html.escape(title)}</h1>
<p>{safe_prompt}</p><p>Focus: {html.escape(focus)}</p></header>
<main><section class="card"><h2>Section</h2><p>Proof block for {html.escape(title)}.</p></section></main>
</body></html>"""
PY
cat > "$EVIDENCE/assets/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "staging-e2e-miner-a"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.setuptools]
py-modules = ["agent", "helpers"]
TOML
# Miner D: broken pyproject (uninstallable dep) → install-class failure.
cat > "$EVIDENCE/assets/pyproject-broken.toml" <<'TOML'
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "staging-e2e-miner-d"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["definitely-not-a-real-pypi-package-e2e==0.0.0"]

[tool.setuptools]
py-modules = ["agent"]
TOML

mkzip() { # mkzip OUT.zip agent.py pyproject.toml [helpers.py]
  python3 - "$1" "$2" "$3" "${4:-}" <<'PY'
import sys, zipfile
out, agent, pyproject, helpers = sys.argv[1:5]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(agent, "agent.py")
    z.write(pyproject, "pyproject.toml")
    if helpers:
        z.write(helpers, "helpers.py")
PY
}

poll_run() { # poll_run RUN_ID WANTED_STAGES_CSV TIMEOUT → 0 when a wanted stage reached
  local rid="$1" want="$2" timeout="$3" t0=$SECONDS st
  while (( SECONDS - t0 < timeout )); do
    st="$(api GET "/challenge/design/v1/runs/$rid" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status","?"))' 2>/dev/null || echo '?')"
    if [[ ",$want," == *",$st,"* ]]; then echo "$st"; return 0; fi
    if [[ "$st" == "failed" || "$st" == "rejected" || "$st" == "scored" ]]; then
      [[ ",$want," == *",$st,"* ]] || { echo "$st"; return 1; }
    fi
    sleep 15
  done
  echo "timeout"; return 1
}

# ---------------------------------------------------------------- reset
if [[ "$RESET" = 1 ]]; then
  step "0 reset e2e state for test hotkeys (operator only)"
  for hk in "$HKA" "$HKB" "$HKD" "$HKU"; do
    psql "DELETE FROM submission_gating WHERE hotkey='$hk';" >>"$EVIDENCE/reset.log" 2>&1 || true
    psql "DELETE FROM design_run WHERE harness_id IN (SELECT id FROM design_harness WHERE miner_hotkey='$hk');" >>"$EVIDENCE/reset.log" 2>&1 || true
    psql "DELETE FROM design_harness WHERE miner_hotkey='$hk';" >>"$EVIDENCE/reset.log" 2>&1 || true
  done
  note "reset done (see reset.log)"
fi

# ---------------------------------------------------------------- preflight
step "0 preflight"
body="$(api GET /healthz)"; [[ "$(last_http)" = 200 ]] && pass "preflight gateway healthz" || fail "preflight gateway healthz" "HTTP $(last_http) $body"
body="$(api GET /challenge/design/health)"; [[ "$(last_http)" = 200 ]] && pass "preflight design health" || fail "preflight design health" "HTTP $(last_http) $body"
status="$(api GET /challenge/design/v1/status)"; echo "$status" > "$EVIDENCE/design-status.json"
echo "$status" | grep -q '"epoch"' && pass "design status" || fail "design status" "$status"
[[ -s "$TOKEN_FILE" ]] && TOKEN="$(head -1 "$TOKEN_FILE" | tr -d '\r\n')" || TOKEN=""
[[ -n "$TOKEN" ]] && pass "admin token present" || fail "admin token" "missing $TOKEN_FILE"

# ---------------------------------------------------------------- 8 (early): unregistered hotkey
step "8 unregistered hotkey → 403 hotkey_not_in_metagraph"
mkzip "$EVIDENCE/assets/u.zip" "$EVIDENCE/assets/agent.py" "$EVIDENCE/assets/pyproject.toml" "$EVIDENCE/assets/helpers.py"
body="$(api POST /challenge/design/v1/harness -H 'content-type: application/zip' -H "x-miner-hotkey: $HKU" --data-binary "@$EVIDENCE/assets/u.zip")"
echo "$body" > "$EVIDENCE/08-unregistered.json"
if [[ "$(last_http)" = 403 ]] && echo "$body" | grep -q hotkey_not_in_metagraph; then
  pass "8 unregistered → 403 hotkey_not_in_metagraph"
else
  fail "8 unregistered" "HTTP $(last_http) $body"
fi

# ---------------------------------------------------------------- 1: miner A submit
step "1 miner A multi-file ZIP + X-Env-Json → 202 → next round"
mkzip "$EVIDENCE/assets/a.zip" "$EVIDENCE/assets/agent.py" "$EVIDENCE/assets/pyproject.toml" "$EVIDENCE/assets/helpers.py"
body="$(api POST /challenge/design/v1/harness -H 'content-type: application/zip' -H "x-miner-hotkey: $HKA" -H 'x-env-json: {"BRAND_MARK":"locked-alpha"}' --data-binary "@$EVIDENCE/assets/a.zip")"
echo "$body" > "$EVIDENCE/01-submit-a.json"
HARNESS_A="$(echo "$body" | python3 -c 'import json,sys;print(json.load(sys.stdin)["harness_id"])' 2>/dev/null || true)"
ROUND_A="$(echo "$body" | python3 -c 'import json,sys;print(json.load(sys.stdin)["round_id"])' 2>/dev/null || true)"
RUNS_A="$(echo "$body" | python3 -c 'import json,sys;print(" ".join(json.load(sys.stdin).get("run_ids",[])))' 2>/dev/null || true)"
if [[ "$(last_http)" = 202 && -n "$HARNESS_A" && -n "$ROUND_A" ]]; then
  pass "1 miner A accepted harness=$HARNESS_A round=$ROUND_A"
else
  fail "1 miner A submit" "HTTP $(last_http) $body"; note "aborting: no miner A"; exit 1
fi
echo "harness_a=$HARNESS_A" >> "$EVIDENCE/ids.env"
echo "round_a=$ROUND_A" >> "$EVIDENCE/ids.env"
echo "runs_a=$RUNS_A" >> "$EVIDENCE/ids.env"

# current round must be ROUND_A-1 → proves next-round registration
cur="$(api GET /challenge/design/v1/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["round_id"])' 2>/dev/null || echo '?')"
if [[ "$cur" != "?" && "$ROUND_A" = "$((cur + 1))" ]]; then
  pass "2 next-round registration (current=$cur, scheduled=$ROUND_A)"
else
  fail "2 next-round registration" "current=$cur scheduled=$ROUND_A"
fi

# 1b: post-submit env change → 409 (env locked; also gating)
step "1b/6 post-submit env change → 409 submission_gated (env lock + 1-max)"
body="$(api POST /challenge/design/v1/harness -H 'content-type: application/zip' -H "x-miner-hotkey: $HKA" -H 'x-env-json: {"BRAND_MARK":"changed-beta"}' --data-binary "@$EVIDENCE/assets/a.zip")"
echo "$body" > "$EVIDENCE/01b-envlock-resubmit.json"
if [[ "$(last_http)" = 409 ]] && echo "$body" | grep -q submission_gated; then
  pass "1b env change post-submit rejected (409 submission_gated)"
else
  fail "1b env change post-submit" "HTTP $(last_http) $body"
fi

# ---------------------------------------------------------------- 5: miner B byte copy
step "5 miner B byte-copy of A → rejected via created_at gate, no LLM review"
mkzip "$EVIDENCE/assets/b.zip" "$EVIDENCE/assets/agent.py" "$EVIDENCE/assets/pyproject.toml" "$EVIDENCE/assets/helpers.py"
body="$(api POST /challenge/design/v1/harness -H 'content-type: application/zip' -H "x-miner-hotkey: $HKB" --data-binary "@$EVIDENCE/assets/b.zip")"
echo "$body" > "$EVIDENCE/05-submit-b.json"
RUNS_B="$(echo "$body" | python3 -c 'import json,sys;print(" ".join(json.load(sys.stdin).get("run_ids",[])))' 2>/dev/null || true)"
[[ "$(last_http)" = 202 && -n "$RUNS_B" ]] && pass "5 miner B accepted (pre-gate)" || fail "5 miner B submit" "HTTP $(last_http) $body"

# ---------------------------------------------------------------- 6: miner D broken pyproject
step "6 miner D broken pyproject → auto-retry x3 → failed + blocked"
cat > "$EVIDENCE/assets/agent-d.py" <<'PY'
def run(task, llm, out) -> None:
    out.write_page("index.html", "<html><body>never installed</body></html>")
PY
mkzip "$EVIDENCE/assets/d.zip" "$EVIDENCE/assets/agent-d.py" "$EVIDENCE/assets/pyproject-broken.toml"
body="$(api POST /challenge/design/v1/harness -H 'content-type: application/zip' -H "x-miner-hotkey: $HKD" --data-binary "@$EVIDENCE/assets/d.zip")"
echo "$body" > "$EVIDENCE/06-submit-d.json"
RUNS_D="$(echo "$body" | python3 -c 'import json,sys;print(" ".join(json.load(sys.stdin).get("run_ids",[])))' 2>/dev/null || true)"
[[ "$(last_http)" = 202 && -n "$RUNS_D" ]] && pass "6 miner D accepted (pre-install)" || fail "6 miner D submit" "HTTP $(last_http) $body"

# ---------------------------------------------------------------- 2/4: A executes
step "2 miner A runs in round $ROUND_A → awaiting_admin"
RUN_A="$(echo "$RUNS_A" | awk '{print $1}')"
note "polling A run $RUN_A (up to ${RUN_WAIT_SECS}s; includes round-open wait + Docker install/run + review)"
final="$(poll_run "$RUN_A" "awaiting_admin,awaiting_annotation" "$RUN_WAIT_SECS")" || true
api GET "/challenge/design/v1/runs/$RUN_A" > "$EVIDENCE/02-run-a-final.json"
api GET "/challenge/design/v1/runs/$RUN_A/events" > "$EVIDENCE/02-run-a-events.json"
if [[ "$final" == awaiting_* ]]; then
  pass "2 miner A reached $final"
else
  fail "2 miner A run" "final=$final (see 02-run-a-*.json)"
fi

# pages + view + env lock content
step "2b pages listed + view 200 + env lock content"
pages="$(api GET "/challenge/design/v1/runs/$RUN_A/pages")"; echo "$pages" > "$EVIDENCE/02-pages-a.json"
ok_pages=1
for p in index.html pricing.html components.html; do
  echo "$pages" | grep -q "$p" || ok_pages=0
  code="$(curl -sS -m 30 -o "$EVIDENCE/02-view-$p" -w '%{http_code}' "$BASE_URL/challenge/design/v1/view/$RUN_A/$p")"
  echo "view $p → $code" >> "$EVIDENCE/02-views.log"
  [[ "$code" = 200 ]] || ok_pages=0
done
[[ "$ok_pages" = 1 ]] && pass "2b pages index/pricing/components served 200" || fail "2b pages" "see 02-pages-a.json / 02-views.log"
if grep -q "locked-alpha" "$EVIDENCE/02-view-index.html" 2>/dev/null && ! grep -q "changed-beta" "$EVIDENCE/02-view-index.html" 2>/dev/null; then
  pass "2c env lock: run used submit-time env (locked-alpha)"
else
  fail "2c env lock" "index.html brand marker mismatch"
fi

# ---------------------------------------------------------------- 3: admin winners
step "3 admin candidates → winners → points"
cand="$(api_admin GET "/v1/admin/rounds/$ROUND_A/candidates")"
echo "$cand" > "$EVIDENCE/03-candidates.json"
echo "$cand" | grep -q "$HARNESS_A" || note "WARN: harness A not yet in candidates (see 03-candidates.json)"
body="$(api_admin POST "/v1/admin/rounds/$ROUND_A/winners" "{\"harness_ids\":[\"$HARNESS_A\"]}")"
echo "$body" > "$EVIDENCE/03-winners.json"
if [[ "$(last_http)" = 202 ]]; then
  pass "3 winners posted for round $ROUND_A"
else
  fail "3 winners" "HTTP $(last_http) $body"
fi
lb="$(api GET "/challenge/design/v1/rounds/$ROUND_A/leaderboard")"; echo "$lb" > "$EVIDENCE/03-leaderboard.json"
echo "$lb" | grep -qi "$HKA" && pass "3 leaderboard shows miner A points" || fail "3 leaderboard" "$lb"

# ---------------------------------------------------------------- 5b: B copy-gate outcome
step "5b miner B run outcome: rejected, no agentic_review event"
RUN_B="$(echo "${RUNS_B:-}" | awk '{print $1}')"
if [[ -n "$RUN_B" ]]; then
  final_b="$(poll_run "$RUN_B" "rejected" "$RUN_WAIT_SECS")" || true
  api GET "/challenge/design/v1/runs/$RUN_B/events" > "$EVIDENCE/05-run-b-events.json"
  api GET "/challenge/design/v1/runs/$RUN_B" > "$EVIDENCE/05-run-b-final.json"
  if [[ "$final_b" = "rejected" ]] \
    && grep -q copy_created_at "$EVIDENCE/05-run-b-events.json" \
    && ! grep -q '"stage": *"agentic_review"' "$EVIDENCE/05-run-b-events.json"; then
    pass "5 miner B rejected pre-LLM (created_at copy gate)"
  else
    fail "5 miner B copy gate" "final=$final_b (see 05-run-b-events.json)"
  fi
else
  fail "5 miner B copy gate" "no run id (submit failed earlier)"
fi

# ---------------------------------------------------------------- 6b: D retries
step "6b miner D auto-retry x3 → failed + blocked"
RUN_D="$(echo "${RUNS_D:-}" | awk '{print $1}')"
if [[ -n "$RUN_D" ]]; then
  final_d="$(poll_run "$RUN_D" "failed" "$RETRY_WAIT_SECS")" || true
  api GET "/challenge/design/v1/runs/$RUN_D/events" > "$EVIDENCE/06-run-d-events.json"
  retries="$(grep -o '"stage": *"auto_retry"' "$EVIDENCE/06-run-d-events.json" | wc -l)"
  gate="$(psql "SELECT state||' attempts='||attempt_count||' class='||coalesce(last_error_class,'-') FROM submission_gating WHERE challenge='design' AND hotkey='$HKD';")"
  echo "$gate" > "$EVIDENCE/06-gating-d.txt"
  if [[ "$final_d" = "failed" && "$retries" -ge 3 ]] && echo "$gate" | grep -q blocked; then
    pass "6 miner D auto-retry x3 → failed + blocked"
  else
    fail "6 miner D retries" "final=$final_d retries=$retries gating=$gate"
  fi
else
  fail "6 miner D retries" "no run id"
fi

# ---------------------------------------------------------------- 7: hotkey-change reset
step "7 hotkey-change reset (watcher reconciliation)"
# Three modes:
#   a) E2E_SWAP_OLD_SS58 + E2E_SWAP_NEW_SS58 set → real on-chain swap_hotkey
#      (subnet enforces a 7200-block (~24h) interval per (netuid, coldkey)).
#   b) E2E_SKIP_SWAP=1 → documented no-op (FAIL).
#   c) default: seed a `rejected` gating row for a hotkey that REALLY left the
#      metagraph (E2E_DEPARTED_HOTKEY; default = miner B's pre-swap hotkey,
#      swapped away on-chain 2026-08-05 18:56 UTC — see evidence
#      hotkey-reset/METHOD.md) and watch the production metagraph watcher
#      reconcile it back to `open`. The seeded precondition is the only
#      scripted part; the reset itself is the live watcher.
DEPARTED="${E2E_DEPARTED_HOTKEY:-183bf07d0280c71f053553049f1fd19691cf0bcadb35363bad48511a5fdccb11}"
if [[ "${E2E_SKIP_SWAP:-0}" = 1 ]]; then
  note "E2E_SKIP_SWAP=1 — no on-chain swap; current gating row for B:"
  psql "SELECT challenge,hotkey,uid,state FROM submission_gating WHERE hotkey='$HKB';" | tee "$EVIDENCE/07-gating-b-skipped.txt"
  fail "7 hotkey-change reset" "skipped via E2E_SKIP_SWAP=1 (documented no-op)"
elif [[ -n "${E2E_SWAP_OLD_SS58:-}" && -n "${E2E_SWAP_NEW_SS58:-}" && -f "${E2E_SWAP_HELPER:-/root/gbase-e2e/swap-hotkey.py}" ]]; then
  before="$(psql "SELECT state FROM submission_gating WHERE challenge='design' AND hotkey='$HKB';")"
  echo "before_swap=$before" > "$EVIDENCE/07-swap.log"
  "${E2E_SWAP_PY:-/root/btcli-venv/bin/python}" "${E2E_SWAP_HELPER:-/root/gbase-e2e/swap-hotkey.py}" \
    "$E2E_SWAP_OLD_SS58" "$E2E_SWAP_NEW_SS58" >> "$EVIDENCE/07-swap.log" 2>&1 \
    || note "WARN: swap-hotkey helper failed (see log)"
  t0=$SECONDS; reopened=0
  while (( SECONDS - t0 < WATCHER_WAIT_SECS )); do
    st="$(psql "SELECT state FROM submission_gating WHERE challenge='design' AND hotkey='$HKB';")"
    echo "$(date +%T) state=$st" >> "$EVIDENCE/07-swap.log"
    [[ "$st" = "open" ]] && { reopened=1; break; }
    sleep 30
  done
  [[ "$reopened" = 1 ]] && pass "7 watcher reset B gating row to open after hotkey swap" \
    || fail "7 hotkey-change reset" "state stayed $(psql "SELECT state FROM submission_gating WHERE challenge='design' AND hotkey='$HKB';")"
else
  note "mode c: seed departed hotkey $DEPARTED (left metagraph via real on-chain swap 2026-08-05) as rejected; watch the live watcher"
  psql "DELETE FROM submission_gating WHERE hotkey='$DEPARTED';" >> "$EVIDENCE/07-swap.log" 2>&1 || true
  psql "INSERT INTO submission_gating (challenge, hotkey, uid, state, attempt_count) VALUES ('design', '$DEPARTED', 4, 'rejected', 0);" >> "$EVIDENCE/07-swap.log" 2>&1 || true
  before="$(psql "SELECT challenge,hotkey,uid,state FROM submission_gating WHERE hotkey='$DEPARTED';")"
  echo "seeded: $before" >> "$EVIDENCE/07-swap.log"
  t0=$SECONDS; reopened=0
  while (( SECONDS - t0 < WATCHER_WAIT_SECS )); do
    st="$(psql "SELECT state FROM submission_gating WHERE challenge='design' AND hotkey='$DEPARTED';")"
    echo "$(date +%T) state=$st" >> "$EVIDENCE/07-swap.log"
    [[ "$st" = "open" ]] && { reopened=1; break; }
    sleep 20
  done
  if [[ "$reopened" = 1 ]]; then
    pass "7 watcher reset departed-hotkey gating row to open (live reconciliation)"
  else
    fail "7 hotkey-change reset" "state stayed $st (see 07-swap.log)"
  fi
fi

# ---------------------------------------------------------------- 4: weights seal
# Runs last: the seal needs a complete prism leaf set for the same epoch (D24),
# which the concurrently-running prism e2e produces — B/D/watcher evidence
# above does not depend on the seal, so collect it first.
step "4 leaves → seal → /v1/weights/latest sealed:true contains A"
# D24 needs a complete leaf set from EVERY >0-bps challenge (design 2000 +
# prism 8000 on staging) against the seal epoch's metagraph. The award's emit
# covers design for the current epoch; prism leaves appear when a prism
# submission finalizes in that epoch (the prism e2e runs concurrently). Sweep
# {current, current-1}: prefer the epoch of this run's award, fall back to the
# previous one. Leaves are append-only, so a sealed epoch stays sealed.
epoch_of() { api GET /challenge/prism/v1/status | python3 -c 'import json,sys;print(json.load(sys.stdin)["epoch"])' 2>/dev/null || echo 0; }
leaf_state() { # leaf_state EPOCH → "design_rows design_A_score prism_rows"
  psql "SELECT (SELECT count(*) FROM raw_weight_snapshot WHERE challenge_id='design' AND epoch=$1), (SELECT coalesce(max(score),-1) FROM raw_weight_snapshot WHERE challenge_id='design' AND epoch=$1 AND miner_hotkey='$HKA'), (SELECT count(*) FROM raw_weight_snapshot WHERE challenge_id='prism' AND epoch=$1);" | awk -F'|' '{print $1, $2, $3}'
}
sealed_ok=0
SEALED_EPOCH=""
t0=$SECONDS
while (( SECONDS - t0 < ${SEAL_WAIT_SECS:-6000} )); do
  epoch="$(epoch_of)"
  for try_epoch in "$epoch" "$((epoch - 1))"; do
    read -r d_rows a_score p_rows <<< "$(leaf_state "$try_epoch")"
    echo "$(date +%T) epoch=$try_epoch design_rows=$d_rows A_design_score=$a_score prism_rows=$p_rows" >> "$EVIDENCE/04-wait.log"
    [[ "$d_rows" -ge 8 && "$a_score" -gt 0 && "$p_rows" -ge 8 ]] 2>/dev/null || continue
    seal="$(api POST /v1/admin/seal -H 'content-type: application/json' -d "{\"epoch\":$try_epoch,\"netuid\":$NETUID}")"
    echo "seal epoch=$try_epoch → HTTP $(last_http) $seal" >> "$EVIDENCE/04-seal.log"
    latest="$(api GET /v1/weights/latest)"
    echo "$latest" > "$EVIDENCE/04-weights-latest.json"
    if echo "$latest" | python3 -c "
import json, sys
sys.path.insert(0, '$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)')
from e2e_ss58 import hotkey_weight_map
d = json.load(sys.stdin)
hw = hotkey_weight_map(d)
hk = '$HKA'.lower()
hit = hw.get(hk, 0.0) > 0
sys.exit(0 if (d.get('sealed') is True and d.get('epoch') == $try_epoch and hit) else 1)
" 2>/dev/null; then sealed_ok=1; SEALED_EPOCH=$try_epoch; note "sealed at epoch=$try_epoch with A present"; break; fi
  done
  [[ "$sealed_ok" = 1 ]] && break
  sleep 60
done
psql "SELECT challenge_id, miner_hotkey, kind, score FROM raw_weight_snapshot WHERE epoch=${SEALED_EPOCH:-0} ORDER BY challenge_id, miner_hotkey;" > "$EVIDENCE/04-leaves.txt" 2>/dev/null || true
[[ "$sealed_ok" = 1 ]] && pass "4 weights sealed:true with A's hotkey + design share (epoch $SEALED_EPOCH)" || fail "4 weights seal" "see 04-wait.log / 04-seal.log / 04-weights-latest.json"

# ---------------------------------------------------------------- summary
{
  echo "# staging-design-e2e summary ($(date -u +%FT%TZ))"
  echo "base=$BASE_URL netuid=$NETUID harness_a=$HARNESS_A round_a=$ROUND_A"
  echo "run_a=$RUN_A"
  echo "PASS (${#PASS[@]}):"
  printf '  - %s\n' "${PASS[@]}"
  echo "FAIL (${#FAIL[@]}):"
  printf '  - %s\n' "${FAIL[@]:-none}"
} | tee "$EVIDENCE/SUMMARY.txt"
[[ "${#FAIL[@]}" = 0 ]]
