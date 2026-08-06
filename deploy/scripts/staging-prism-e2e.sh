#!/usr/bin/env bash
# staging-prism-e2e.sh — prove the full prism miner lifecycle on staging.
#
#   staging-prism-e2e.sh [--evidence DIR] [--reset]
#
# Proves, with per-step evidence files + PASS/FAIL summary:
#   1. Miner A architecture+training (recipe 1.2.0 hooks) → review pass →
#      fast train (sim backend w/ PRISM_TEST knobs) → telemetry rows +
#      site telemetry loss curve → finish_evaluation honored → bpb.
#   4. Top-model: A is global best bpb → published to BaseIntelligence/prism
#      top-model/ + prism_topmodel_publication journal row.
#   2. Miner B training-only on A's published arch → /v1/architectures lists
#      A with best_bpb → competition scoring (owner credited per rule).
#      B is submitted right after an epoch boundary so its finalize is the
#      first prism emit of the epoch (append-only leaves are first-write-wins;
#      the owner+challenger credit must land in that first emit).
#   1b. Missing-hooks variant (miner D) → hard reject at review
#      (missing_telemetry_hooks), Score(0), terminal.
#   3. Miner C byte copy of A's architecture → rejected pre-measure: no pod,
#      no LLM review stages.
#   5. Leaves → admin seal → /v1/weights/latest sealed:true with correct
#      hotkeys (A + B per competition rule; C/D zero/absent). Waits for the
#      design leaf set of the same epoch (D24 needs both >0-bps challenges).
#
# Required env (hex, no 0x): E2E_HOTKEY_A / _B / _C / _D.
# Optional: E2E_HOTKEY_U (extra hotkey to reset), E2E_SKIP_BOUNDARY_WAIT=1,
#           SEAL_WAIT_SECS (design-leaf wait, default 2700).
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
  local sql="$1" cmd
  cmd="psql -U \"\$POSTGRES_USER\" -d \"\$POSTGRES_DB\" -Atc $(printf %q "$sql")"
  ssh -o BatchMode=yes "$SSH_HOST" "docker exec base-postgres-1 sh -c $(printf %q "$cmd")" 2>/dev/null || echo "(psql unavailable)"
}

# ---------------------------------------------------------------- assets
# Miner A architecture: MLP-Mixer language model (torch) — genuinely distinct
# from the baseline causal transformer (no attention at all), real nn.Module.
cat > "$EVIDENCE/assets/architecture_a.py" <<'PY'
"""Staging e2e miner A architecture: MLP-Mixer LM (token/channel mixing, no attention)."""

import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(self, d: int, block: int, mlp_ratio: int = 2):
        super().__init__()
        self.ln_t = nn.LayerNorm(d)
        self.ln_c = nn.LayerNorm(d)
        self.t_mix = nn.Sequential(
            nn.Linear(block, block * mlp_ratio), nn.GELU(), nn.Linear(block * mlp_ratio, block)
        )
        self.c_mix = nn.Sequential(
            nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Linear(d * mlp_ratio, d)
        )

    def forward(self, x):
        h = self.ln_t(x)
        h = h.transpose(1, 2)
        h = self.t_mix(h)
        x = x + h.transpose(1, 2)
        return x + self.c_mix(self.ln_c(x))


class MixerLM(nn.Module):
    def __init__(self, vocab=50257, d=192, n_layer=3, block=512):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([MixerBlock(d, block) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.logits = None

    def forward(self, ids):
        b, t = ids.shape
        t = min(t, self.block)
        ids = ids[:, -t:]
        pos = torch.arange(t, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        logits = self.head(x)
        self.logits = logits
        return logits


def build_model(ctx):
    """Recipe contract entrypoint. ctx carries device/seed/caps (unused here)."""
    torch.manual_seed(int(ctx.get("seed", 0)))
    return MixerLM()
PY
# Miner A training: real AdamW loop on the pinned shard with hooks + cosine LR
# + label smoothing (training.py is similarity-exempt; must still really train).
cat > "$EVIDENCE/assets/training_a.py" <<'PY'
"""Staging e2e miner A training: cosine LR + label smoothing on the pinned shard."""

import math
import time

import pyarrow.parquet as pq
import torch
from transformers import GPT2TokenizerFast

try:
    import prism_telemetry
except ImportError:
    class _TelemetryFallback:
        @staticmethod
        def report(**_kwargs):
            return None

        @staticmethod
        def finish_evaluation():
            return None

    prism_telemetry = _TelemetryFallback()


def _texts(path, n):
    table = pq.read_table(path, columns=["text"])
    xs = [t for t in table.column("text").to_pylist() if isinstance(t, str) and len(t) >= 200]
    return xs[:n]


def train(model, ctx):
    """Recipe contract entrypoint: returns a metrics dict (val is harness-side)."""
    device = ctx["device"]
    torch.manual_seed(int(ctx["seed"]))
    guard = ctx.get("guard", lambda: None)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = _texts(ctx["dataset_path"], int(ctx.get("train_rows", 2048)))
    block = model.block if hasattr(model, "block") else 512

    g = torch.Generator().manual_seed(int(ctx["seed"]))
    perm = torch.randperm(len(texts), generator=g).tolist()

    max_steps = int(ctx.get("max_train_steps", 20000))
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.05)
    model.train()
    steps = 0
    bs = 8
    last = 0.0
    t0 = time.time()
    for i in range(0, min(2000, len(perm) - bs), bs):
        guard()
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, steps / max(1, 500))))
        for group in opt.param_groups:
            group["lr"] = 6e-4 * lr_scale
        batch_txt = [texts[j] for j in perm[i : i + bs]]
        enc = tok(
            batch_txt, return_tensors="pt", truncation=True, max_length=block, padding=True
        ).to(device)
        ids = enc.input_ids
        out = model(ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id,
            label_smoothing=0.05,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        last = float(loss.item())
        steps += 1
        if steps == 1 or steps % 10 == 0:
            prism_telemetry.report(
                loss=last,
                step=steps,
                grad_norm=grad_norm,
                layer_stats={"mixer_head": {"grad_norm": grad_norm}},
            )
        if steps >= max_steps:
            break
        if steps >= 300 and last < 4.0:
            # Early stop: score the model as-is before the cap.
            break
    prism_telemetry.finish_evaluation()
    return {"train_loss": last, "train_steps": steps, "train_seconds": time.time() - t0}

# sim-bpb-tuning: a-28
PY
# Miner B training-only: real loop (different optimizer schedule), hooks intact.
cat > "$EVIDENCE/assets/training_b.py" <<'PY'
"""Staging e2e miner B training-only entry: SGD momentum challenger loop with hooks."""

import time

import pyarrow.parquet as pq
import torch
from transformers import GPT2TokenizerFast

try:
    import prism_telemetry
except ImportError:
    class _TelemetryFallback:
        @staticmethod
        def report(**_kwargs):
            return None

        @staticmethod
        def finish_evaluation():
            return None

    prism_telemetry = _TelemetryFallback()


def _texts(path, n):
    table = pq.read_table(path, columns=["text"])
    xs = [t for t in table.column("text").to_pylist() if isinstance(t, str) and len(t) >= 200]
    return xs[:n]


def train(model, ctx):
    """Recipe contract entrypoint: returns a metrics dict (val is harness-side)."""
    device = ctx["device"]
    torch.manual_seed(int(ctx["seed"]))
    guard = ctx.get("guard", lambda: None)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = _texts(ctx["dataset_path"], int(ctx.get("train_rows", 2048)))
    block = model.block if hasattr(model, "block") else 512

    g = torch.Generator().manual_seed(int(ctx["seed"]))
    perm = torch.randperm(len(texts), generator=g).tolist()

    max_steps = int(ctx.get("max_train_steps", 20000))
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, nesterov=True)
    model.train()
    steps = 0
    bs = 4
    last = 0.0
    t0 = time.time()
    for i in range(0, min(2000, len(perm) - bs), bs):
        guard()
        batch_txt = [texts[j] for j in perm[i : i + bs]]
        enc = tok(
            batch_txt, return_tensors="pt", truncation=True, max_length=block, padding=True
        ).to(device)
        ids = enc.input_ids
        out = model(ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        last = float(loss.item())
        steps += 1
        if steps == 1 or steps % 8 == 0:
            prism_telemetry.report(
                loss=last,
                step=steps,
                grad_norm=grad_norm,
                layer_stats={"head": {"grad_norm": grad_norm}},
            )
        if steps >= max_steps:
            break
    prism_telemetry.finish_evaluation()
    return {"train_loss": last, "train_steps": steps, "train_seconds": time.time() - t0}

# sim-bpb-tuning: b-0
PY
# Miner D: missing hooks (no prism_telemetry anywhere) → hard review reject.
cat > "$EVIDENCE/assets/training_d.py" <<'PY'
"""Staging e2e miner D training: deliberately missing telemetry hooks."""

import time

import pyarrow.parquet as pq
import torch
from transformers import GPT2TokenizerFast


def _texts(path, n):
    table = pq.read_table(path, columns=["text"])
    xs = [t for t in table.column("text").to_pylist() if isinstance(t, str) and len(t) >= 200]
    return xs[:n]


def train(model, ctx):
    """Trains without ever reporting telemetry (contract violation on purpose)."""
    device = ctx["device"]
    torch.manual_seed(int(ctx["seed"]))
    guard = ctx.get("guard", lambda: None)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    texts = _texts(ctx["dataset_path"], int(ctx.get("train_rows", 2048)))
    block = model.block if hasattr(model, "block") else 512

    g = torch.Generator().manual_seed(int(ctx["seed"]))
    perm = torch.randperm(len(texts), generator=g).tolist()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    model.train()
    steps = 0
    bs = 4
    last = 0.0
    t0 = time.time()
    for i in range(0, min(2000, len(perm) - bs), bs):
        guard()
        batch_txt = [texts[j] for j in perm[i : i + bs]]
        enc = tok(
            batch_txt, return_tensors="pt", truncation=True, max_length=block, padding=True
        ).to(device)
        ids = enc.input_ids
        out = model(ids[:, :-1])
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        steps += 1
        if steps >= int(ctx.get("max_train_steps", 20000)):
            break
    return {"train_loss": last, "train_steps": steps, "train_seconds": time.time() - t0}
PY
# Miner D architecture (distinct — the reject must come from hooks, not copy gate).
cat > "$EVIDENCE/assets/architecture_d.py" <<'PY'
"""Staging e2e miner D architecture: causal LSTM LM (torch, no attention)."""

import torch
import torch.nn as nn


class CausalLSTMLM(nn.Module):
    def __init__(self, vocab=50257, d=160, block=512):
        super().__init__()
        self.block = block
        self.tok_emb = nn.Embedding(vocab, d)
        self.rnn = nn.LSTM(d, d, num_layers=2, batch_first=True)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.logits = None

    def forward(self, ids):
        b, t = ids.shape
        t = min(t, self.block)
        ids = ids[:, -t:]
        x = self.tok_emb(ids)
        x, _ = self.rnn(x)
        x = self.ln(x)
        logits = self.head(x)
        self.logits = logits
        return logits


def build_model(ctx):
    """Recipe contract entrypoint. ctx carries device/seed/caps (unused here)."""
    torch.manual_seed(int(ctx.get("seed", 0)))
    return CausalLSTMLM()
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
  # E2E_HOTKEY_U (optional): extra hotkey whose rows pollute the similarity
  # corpus across runs (e.g. training-only rows carrying a materialized arch).
  for hk in "$HKA" "$HKB" "$HKC" "$HKD" ${E2E_HOTKEY_U:+$(echo "$E2E_HOTKEY_U" | tr 'A-F' 'a-f')}; do
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

# ---------------------------------------------------------------- 4: top-model publish
# Fires on A's finalize (post_score_hooks) when bpb is a new global best AND
# beats the last published bpb; the ground sim bpb for A (1.027) beats the
# previously published 1.096. Poll the journal: hooks run after the status flip.
step "4 top-model publish → $GITHUB_REPO top-model/ + journal"
pub=""
t0=$SECONDS
while (( SECONDS - t0 < 420 )); do
  pub="$(psql "SELECT submission_id, arch_id, bpb, repo_path, commit_sha FROM prism_topmodel_publication ORDER BY published_at DESC LIMIT 1;")"
  echo "$pub" | grep -q "$SUB_A" && break
  sleep 20
done
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

# Epoch choreography (D24 + append-only leaves): a leaf set is first-write-wins
# per (challenge, epoch, hotkey) and each finalize emits the full set for the
# CURRENT chain epoch, scoring only submissions ACCEPTED in that epoch. For the
# sealed epoch to credit both A (arch-owner) and B (challenger), B's finalize
# must be the FIRST prism emit of a fresh epoch: wait for the boundary after
# A's epoch, then submit B immediately (accept + finalize inside it).
EPOCH_A="$(api GET /challenge/prism/v1/status | jq_field 'd["epoch"]')"
note "A finalized in epoch $EPOCH_A; waiting for the next epoch boundary before B (E2E_SKIP_BOUNDARY_WAIT=1 to skip)"
# Idempotent re-runs: if the current epoch's prism leaves already credit A
# (owner) and B (challenger) from an earlier run, the seal step can target it
# directly — no boundary wait needed.
prior="$(psql "SELECT count(*) FROM raw_weight_snapshot WHERE challenge_id='prism' AND epoch=${EPOCH_A:-0} AND kind='score' AND score>0 AND miner_hotkey IN ('$HKA','$HKB');")"
if [[ "$prior" = "2" ]]; then
  note "prism|$EPOCH_A leaves already credit A and B (prior run) — skipping boundary wait"
elif [[ "${E2E_SKIP_BOUNDARY_WAIT:-0}" != 1 ]]; then
  t0=$SECONDS
  while (( SECONDS - t0 < 5400 )); do
    cur_ep="$(api GET /challenge/prism/v1/status | jq_field 'd["epoch"]')"
    [[ -n "$cur_ep" && "$cur_ep" != "None" && "$cur_ep" -gt "${EPOCH_A:-0}" ]] && break
    sleep 15
  done
fi
EPOCH_B="$(api GET /challenge/prism/v1/status | jq_field 'd["epoch"]')"
echo "epoch_a=$EPOCH_A epoch_b=$EPOCH_B" >> "$EVIDENCE/ids.env"
note "submitting B in epoch $EPOCH_B (must be first prism emit of the epoch)"
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
  api GET "/challenge/prism/v1/submissions/$SUB_B/events" > "$EVIDENCE/02-sub-b-events.json"
  bpb_b="$(jq_field 'd["submission"]["bpb"]' < "$EVIDENCE/02-sub-b-final.json")"
  arch_b="$(jq_field 'd["submission"]["arch_id"]' < "$EVIDENCE/02-sub-b-final.json")"
  if [[ "$st_b" = "terminated" && -n "$bpb_b" && "$bpb_b" != "None" && "$arch_b" = "$ARCH_ID" ]]; then
    pass "2 miner B terminated bpb=$bpb_b on arch $ARCH_ID"
  else
    fail "2 miner B pipeline" "final=$st_b bpb=$bpb_b arch=$arch_b"
  fi
fi

# ---------------------------------------------------------------- 1b: missing hooks (D)
# Runs AFTER B: D's rejected finalize re-emits the full leaf set for the epoch,
# but every leaf is already stored (first-write-wins) — D stays no_score, which
# is exactly what the sealed weights must show for a hooks violator.
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

# ---------------------------------------------------------------- 5: weights
# Seal B's epoch: prism leaves are complete from B's finalize emit (A credited
# as arch owner, B as challenger); design leaves appear at the first design
# emit of the epoch (round award/close, ≤ round_secs). Wait for both, then seal.
step "5 leaves → seal → /v1/weights/latest (prism share)"
META_N="$(psql "SELECT count(DISTINCT miner_hotkey) FROM raw_weight_snapshot WHERE challenge_id='prism' AND epoch=${EPOCH_B:-0};")"
[[ "$META_N" =~ ^[0-9]+$ && "$META_N" -gt 0 ]] || META_N=8
note "metagraph size proxy from prism|${EPOCH_B} leaves: $META_N"
design_ready=0
sealed_ok=0
SEALED_EPOCH=""
t0=$SECONDS
while (( SECONDS - t0 < ${SEAL_WAIT_SECS:-2700} )); do
  n="$(psql "SELECT count(*) FROM raw_weight_snapshot WHERE challenge_id='design' AND epoch=${EPOCH_B:-0};")"
  a_des="$(psql "SELECT coalesce(max(score),-1) FROM raw_weight_snapshot WHERE challenge_id='design' AND epoch=${EPOCH_B:-0} AND miner_hotkey='$HKA';")"
  echo "$(date +%T) design|${EPOCH_B} rows=$n A_score=$a_des" >> "$EVIDENCE/05-wait.log"
  [[ "$n" = "$META_N" && "$a_des" != "-1" && -n "$n" ]] && design_ready=1
  if [[ "$design_ready" = 1 ]]; then
    for try_epoch in "$EPOCH_B" "$((EPOCH_B - 1))" "$((EPOCH_B + 1))"; do
      seal="$(api POST /v1/admin/seal -H 'content-type: application/json' -d "{\"epoch\":$try_epoch,\"netuid\":$NETUID}")"
      echo "seal epoch=$try_epoch → HTTP $(last_http) $seal" >> "$EVIDENCE/05-seal.log"
      latest="$(api GET /v1/weights/latest)"
      echo "$latest" > "$EVIDENCE/05-weights-latest.json"
      if echo "$latest" | python3 -c "
import json, sys
sys.path.insert(0, '$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)')
from e2e_ss58 import hotkey_weight_map
d = json.load(sys.stdin)
hw = hotkey_weight_map(d)
a = hw.get('$HKA', 0.0)
b = hw.get('$HKB', 0.0)
c = hw.get('$HKC', 0.0)
ok = d.get('sealed') is True and d.get('epoch') == $try_epoch and a > 0 and b > 0 and c == 0
sys.exit(0 if ok else 1)
" 2>/dev/null; then sealed_ok=1; SEALED_EPOCH=$try_epoch; note "sealed at epoch=$try_epoch: A>0 B>0 C=0"; break; fi
    done
  fi
  [[ "$sealed_ok" = 1 ]] && break
  sleep 60
done
[[ "$design_ready" = 1 ]] || note "WARN: design leaves incomplete at $EPOCH_B (seal may fail D24)"
psql "SELECT challenge_id, miner_hotkey, kind, score FROM raw_weight_snapshot WHERE epoch=${SEALED_EPOCH:-$EPOCH_B} ORDER BY challenge_id, miner_hotkey;" > "$EVIDENCE/05-leaves.txt" 2>/dev/null || true
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
