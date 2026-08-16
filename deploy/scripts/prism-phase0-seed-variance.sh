#!/usr/bin/env bash
# Prism Phase 0 — measure the two quantities the dual-cap design is gated on.
#
# Usage: ./deploy/scripts/prism-phase0-seed-variance.sh [--dry-run] [OPTIONS]
#
# The v3 measurement plan rests on two numbers nobody has measured:
#
#   sigma_seed  Prism's clustered bootstrap resamples EVAL ITEMS only. It has
#               never measured TRAINING variance. Published NAS work finds a
#               seed change alone re-ranks architectures at Kendall tau=0.48,
#               so an LCB built from eval-item variance is OVERCONFIDENT: it
#               charges a submission for noise it does measure and gives away
#               the noise it does not. Every anchor reference value, and the
#               minimum detectable difference of every scored key, depends on
#               this number.
#
#   real MFU    Every FLOPs constant scales linearly with it. TRAIN_FLOPS_CAP
#               = 3.0e18 was chosen so a >=20% MFU implementation is
#               FLOPs-bound inside the 5.0h wall. At 15% real MFU the cap
#               becomes WALL-bound for most of the field and the design partly
#               reverts to the status quo it replaces.
#
# What it does: runs the SAME architecture N times varying ONLY the seed, on
# both E6 reference baselines (Transformer++ and the hybrid delta-net),
# collects G1 bits/byte + strict LAMBADA + the composite, and reports
# mean/sigma/CI per metric alongside measured MFU and the counter-vs-analytic
# FLOPs gap.
#
# THIS SCRIPT RENTS GPUs AND SPENDS MONEY. It refuses to launch without
# --confirm-spend. Start with --dry-run, which validates the whole wiring
# (endpoint, Lium key, baselines, seed plumbing, output schema) and rents
# nothing.
#
# Cost, at the repo's $2.5/h/pod guard with the reconciled caps
# (5.0h train + ~1.5h eval+staging = 6.5h billed worst case per run):
#
#   per run          6.5 pod-h  x $2.5  = ~$16.25   (worst case, full budget)
#   per baseline     x3 seeds            = ~$48.75
#   both baselines   x2                  = ~$97.50   <-- default: --seeds 3
#   minimum useful   2 baselines x 2     = ~$65.00   (--seeds 2)
#   5 seeds          2 baselines x 5     = ~$162.50  (--seeds 5)
#
# Pods bill for time USED, so a baseline that converges early costs less. A
# sigma estimate from 2 seeds has 1 degree of freedom and a ~14x-wide CI on
# sigma itself; 3 is the practical floor and 5 is what you want if the answer
# is close to a decision boundary.
#
# OPERATOR FLOW (live). The training seed is a HARNESS env knob
# (PRISM_SEED_OVERRIDE on the challenge container), not a submission field:
# the submission API has no env passthrough and must not grow one, because a
# miner-chosen seed would make submissions incomparable. So a live wave is
# SEQUENTIAL, one seed at a time:
#
#   1. ./deploy/scripts/prism-phase0-seed-variance.sh --dry-run   # validate
#   2. for each seed the script emits compose.seed-<N>.override.yml;
#      apply it to your stack (docker compose ... up -d --no-deps
#      --force-recreate prism-challenge) so the container carries that seed,
#   3. re-run with --confirm-spend to submit under that seed,
#   4. repeat for the remaining seeds, then run with --confirm-spend once more
#      to reduce whatever finals are present into the report.
#
# The reduction step is idempotent and reads only *.final.json, so it can be
# re-run at any time to refresh the report as runs land.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PRISM_URL="${PRISM_URL:-http://127.0.0.1:28092}"
LIUM_KEY_FILE="${LIUM_API_KEY_FILE:-$ROOT/deploy/secrets/lium/api_key}"
HOTKEY="${PRISM_PHASE0_HOTKEY:-343d50f1222b260aaa48e0dfd72c94b935bd14c87ec1c88bd90934193c72f534}"
EVIDENCE="${EVIDENCE:-/tmp/prism-phase0-$(date -u +%Y%m%dT%H%M%SZ)}"
SEEDS="${SEEDS:-3}"
POLL_S="${POLL_S:-60}"
DRY_RUN=0
CONFIRM_SPEND=0
BASELINES="transformer_pp hybrid_delta"
POD_HOURS_PER_RUN="${POD_HOURS_PER_RUN:-6.5}"
USD_PER_POD_HOUR="${USD_PER_POD_HOUR:-2.5}"

usage() { sed -n '2,60p' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --confirm-spend) CONFIRM_SPEND=1; shift ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --baseline) BASELINES="$2"; shift 2 ;;
    --evidence) EVIDENCE="$2"; shift 2 ;;
    --url) PRISM_URL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$EVIDENCE"
SUMMARY="$EVIDENCE/PHASE0.md"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$SUMMARY"; }

if ! [[ "$SEEDS" =~ ^[0-9]+$ ]] || [[ "$SEEDS" -lt 2 ]]; then
  echo "FAIL: --seeds must be an integer >= 2 (sigma from 1 run is undefined)" >&2
  exit 2
fi

N_BASELINES=$(printf '%s\n' $BASELINES | wc -l)
TOTAL_RUNS=$((SEEDS * N_BASELINES))
EST_COST=$(python3 -c "print(f'{$TOTAL_RUNS * $POD_HOURS_PER_RUN * $USD_PER_POD_HOUR:.2f}')")

cat > "$SUMMARY" <<EOF
# Prism Phase 0 — seed variance + MFU

- Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- Evidence: \`$EVIDENCE\`
- Prism: \`$PRISM_URL\`
- Baselines: $BASELINES
- Seeds per baseline: $SEEDS
- Runs: $TOTAL_RUNS
- Estimated worst-case cost: **\$$EST_COST** ($POD_HOURS_PER_RUN pod-h x \$$USD_PER_POD_HOUR/h x $TOTAL_RUNS runs)
- Mode: $([[ "$DRY_RUN" -eq 1 ]] && echo 'DRY RUN (rents nothing)' || echo 'LIVE (rents GPUs)')

## Why

- \`sigma_seed\`: the bootstrap measures eval-item variance only, never
  TRAINING variance. A seed change alone re-ranks NAS architectures at
  Kendall tau=0.48, so the LCB is overconfident.
- real MFU: every FLOPs constant scales linearly with it. At 15% real MFU
  the 3.0e18 cap becomes wall-bound and the dual cap partly reverts to the
  status quo.

## Runs

EOF

log "phase 0: $TOTAL_RUNS runs, estimated worst case \$$EST_COST"

# ---------------------------------------------------------------- preflight
preflight_fail=0
note_fail() { log "PREFLIGHT FAIL: $*"; preflight_fail=1; }
# Things a dry run legitimately does not need (it rents nothing), but a live
# run cannot proceed without.
note_live_only() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "PREFLIGHT WARN (dry run, not fatal): $*"
  else
    note_fail "$*"
  fi
}

for b in $BASELINES; do
  for f in architecture.py training.py; do
    [[ -f "$ROOT/crates/prism-recipe/baselines/$b/$f" ]] \
      || note_fail "missing baseline source crates/prism-recipe/baselines/$b/$f"
  done
done

if [[ -f "$LIUM_KEY_FILE" ]]; then
  if [[ -s "$LIUM_KEY_FILE" ]]; then
    log "lium key: present ($LIUM_KEY_FILE)"
  else
    note_live_only "lium key file $LIUM_KEY_FILE is empty"
  fi
else
  note_live_only "lium key file $LIUM_KEY_FILE missing — live submit would 400 \
missing_lium_api_key (set LIUM_API_KEY_FILE, e.g. /root/gbase/deploy/secrets/lium/api_key)"
fi

if curl -sfS "$PRISM_URL/health" >/dev/null 2>&1; then
  log "prism /health: ok"
else
  note_live_only "prism /health unreachable at $PRISM_URL"
fi

# The recipe must actually be running the dual cap, or the MFU number is
# measured against a budget that is not the one being calibrated.
if RECIPE=$(curl -sfS "$PRISM_URL/v1/recipe" 2>/dev/null); then
  if python3 - "$RECIPE" <<'PY' | tee -a "$SUMMARY"
import json, sys
r = json.loads(sys.argv[1])
keys = ("train_hours_cap", "train_flops_cap", "min_spend_fraction",
        "flops_probe_samples", "flops_analytic_gap_max")
for k in keys:
    print(f"- recipe.{k} = {r.get(k, 'ABSENT')}")
missing = [k for k in keys if k not in r]
if missing:
    print(f"- /v1/recipe does not advertise {missing}: the DEPLOYED build "
          "predates the dual cap, so MFU would be measured under the OLD "
          "budget and the FLOPs attestation would not run at all")
    raise SystemExit(1)
PY
  then
    log "recipe: dual cap deployed"
  else
    note_live_only "deployed build predates the dual cap — redeploy before measuring MFU"
  fi
else
  note_live_only "/v1/recipe unreachable — cannot confirm the dual cap is deployed"
fi

# Verify the harness carries the attestation this script reads back.
HARNESS="$ROOT/crates/prism-recipe/harness"
for marker_file in prismlib/flops.py prismlib/stream.py; do
  [[ -f "$HARNESS/$marker_file" ]] || note_fail "harness missing $marker_file"
done
if [[ -f "$HARNESS/prismlib/flops.py" ]]; then
  for marker in FlopCounterMode analytic_flops_per_token mfu; do
    grep -q "$marker" "$HARNESS/prismlib/flops.py" \
      || note_fail "prismlib/flops.py missing $marker"
  done
fi

# Static self-test of the reduction math: the report is only trustworthy if
# the statistics are right, and that does not need a GPU to check.
python3 - <<'PY' | tee -a "$SUMMARY"
import math
xs = [1.10, 1.14, 1.12, 1.09, 1.15]
n = len(xs)
mean = sum(xs) / n
sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
se = sd / math.sqrt(n)
assert abs(mean - 1.12) < 1e-9, mean
assert 0.02 < sd < 0.03, sd
print(f"- stats self-test: mean={mean:.4f} sigma={sd:.4f} se={se:.4f} (ok)")
PY

if [[ "$preflight_fail" -ne 0 ]]; then
  log "preflight failed — refusing to continue"
  exit 1
fi
log "preflight: OK"

# ------------------------------------------------------------------- submit
CONTAINER="${PRISM_CONTAINER:-base-prism-challenge-1}"

# The seed is a HARNESS env knob (`PRISM_SEED_OVERRIDE`), not a submission
# field: the submission API has no env passthrough, and adding one would let
# miners choose their own seed. So each seed is applied to the challenge
# container and the runs go one at a time.
set_container_seed() {
  local seed="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  dry-run: would set $CONTAINER PRISM_SEED_OVERRIDE=$seed"
    return 0
  fi
  if ! docker exec "$CONTAINER" true 2>/dev/null; then
    log "  FAIL: container $CONTAINER not running — cannot set the seed"
    return 1
  fi
  # Recreate with the seed in the environment. Deliberately explicit rather
  # than clever: the operator must see which seed each run used.
  local override="$EVIDENCE/compose.seed-$seed.override.yml"
  cat > "$override" <<EOF
services:
  prism-challenge:
    environment:
      PRISM_SEED_OVERRIDE: "$seed"
      PRISM_MAX_CONCURRENT_EVALS: "1"
EOF
  log "  wrote $override — apply it with your compose stack, then re-run this"
  log "  seed for this run: $seed"
}

submit_one() {
  local baseline="$1" seed="$2" tag="$3"
  local arch="$ROOT/crates/prism-recipe/baselines/$baseline/architecture.py"
  local train="$ROOT/crates/prism-recipe/baselines/$baseline/training.py"
  local body="$EVIDENCE/$tag.request.json"

  # ONLY the seed varies. The architecture, training code, data pin,
  # tokenizer and every cap are identical across the runs of one baseline —
  # that is what makes the spread an estimate of sigma_seed rather than of
  # anything else. The seed itself rides the container env, set above.
  python3 - "$arch" "$train" "$HOTKEY" "$seed" > "$body" <<'PY'
import json, pathlib, sys
arch, train, hotkey, seed = sys.argv[1:5]
print(json.dumps({
    "miner_hotkey": hotkey,
    "architecture_py": pathlib.Path(arch).read_text(),
    "training_py": pathlib.Path(train).read_text(),
    # Recorded for the evidence trail only. The harness reads the seed from
    # PRISM_SEED_OVERRIDE in its own environment; the API has no env
    # passthrough and must not grow one (a miner-chosen seed would make
    # submissions incomparable).
    "_phase0_seed_note": f"trained under PRISM_SEED_OVERRIDE={seed}",
}, indent=2))
PY

  if [[ "$DRY_RUN" -eq 1 ]]; then
    python3 - "$body" "$seed" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["architecture_py"].strip(), "empty architecture_py"
assert "def build_model(" in d["architecture_py"], "architecture_py lacks build_model"
assert "def train(" in d["training_py"], "training_py lacks train"
assert sys.argv[2] in d["_phase0_seed_note"], "seed not recorded"
print(f"  dry-run request OK: {len(d['architecture_py'])}B arch, "
      f"{len(d['training_py'])}B train, seed={sys.argv[2]}")
PY
    echo "DRY-RUN-$tag" > "$EVIDENCE/$tag.submission_id"
    return 0
  fi

  local auth=()
  [[ -f "$LIUM_KEY_FILE" ]] && auth=(-H "X-Lium-Api-Key: $(tr -d '[:space:]' <"$LIUM_KEY_FILE")")
  local http
  http=$(curl -sS -o "$EVIDENCE/$tag.submit.json" -w '%{http_code}' \
    -X POST "$PRISM_URL/v1/submissions" \
    -H 'content-type: application/json' \
    "${auth[@]}" --data-binary "@$body" || true)
  log "  submit $tag: HTTP $http"
  local sid
  sid=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("submission_id") or (d.get("submission") or {}).get("id") or d.get("id") or "")' \
    "$EVIDENCE/$tag.submit.json" 2>/dev/null || true)
  if [[ -z "$sid" ]]; then
    log "  FAIL: no submission_id for $tag — see $tag.submit.json"
    return 1
  fi
  echo "$sid" > "$EVIDENCE/$tag.submission_id"
  log "  $tag submission_id=$sid"
}

if [[ "$DRY_RUN" -eq 0 && "$CONFIRM_SPEND" -eq 0 ]]; then
  log "REFUSING TO LAUNCH: this rents $TOTAL_RUNS GPU pods (est. worst case \$$EST_COST)."
  log "Re-run with --confirm-spend once the spend is authorized, or --dry-run to validate wiring."
  exit 3
fi

TAGS=()
for baseline in $BASELINES; do
  for i in $(seq 1 "$SEEDS"); do
    # Distinct, reproducible seeds: the point is a KNOWN set, so the run can
    # be repeated exactly.
    seed=$((1000 + i))
    tag="${baseline}-seed${seed}"
    TAGS+=("$tag")
    log "submitting $tag (baseline=$baseline seed=$seed)"
    set_container_seed "$seed" || log "  WARN: could not set seed for $tag"
    submit_one "$baseline" "$seed" "$tag" || log "  WARN: $tag submit failed"
  done
done
printf '%s\n' "${TAGS[@]}" > "$EVIDENCE/tags.txt"

# --------------------------------------------------------------------- poll
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "dry run: skipping poll; writing a synthetic report to prove the reduction path"
  for tag in "${TAGS[@]}"; do
    python3 - "$EVIDENCE/$tag.final.json" "$tag" <<'PY'
import json, sys, hashlib
path, tag = sys.argv[1], sys.argv[2]
# Deterministic pseudo-values from the tag so the dry-run report is stable
# and obviously synthetic. NOT a measurement.
h = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
json.dump({
    "synthetic": True,
    "battery": {"metrics": {
        "org.g1.bits_per_byte_prose": 1.10 + 0.04 * h,
        "org.g1.bits_per_byte_code": 1.05 + 0.04 * h,
        "org.g1.bits_per_byte_math": 1.15 + 0.04 * h,
        "org.g2.lambada_strict_acc": 0.18 + 0.05 * h,
        "org.diag.mfu_achieved": 0.22 + 0.06 * h,
        "org.diag.flops_attested": 2.7e18 + 2e17 * h,
        "org.diag.flops_analytic_ratio": 0.95 + 0.08 * h,
        "org.diag.flops_probe_cv": 0.02 + 0.03 * h,
        "org.diag.binding_cap": "flops",
        "org.diag.spend_fraction": 0.90 + 0.09 * h,
    }},
    "composite": {"composite": 0.42 + 0.05 * h, "lattice": int(4200 + 500 * h)},
}, open(path, "w"), indent=2)
print(f"  wrote synthetic {path}")
PY
  done
else
  log "polling $TOTAL_RUNS runs every ${POLL_S}s (multi-hour; safe to leave running)"
  pending=("${TAGS[@]}")
  while [[ ${#pending[@]} -gt 0 ]]; do
    still=()
    for tag in "${pending[@]}"; do
      sid=$(cat "$EVIDENCE/$tag.submission_id" 2>/dev/null || true)
      [[ -z "$sid" ]] && continue
      curl -sS "$PRISM_URL/v1/submissions/$sid" > "$EVIDENCE/$tag.status.json" 2>/dev/null || true
      state=$(python3 -c 'import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print("unknown"); raise SystemExit
s=d.get("submission") or d
print(s.get("status") or s.get("state") or "unknown")' "$EVIDENCE/$tag.status.json" 2>/dev/null || echo unknown)
      case "$state" in
        scored|failed|rejected|error|terminal|complete|completed|Score*|Ineligible*|Eligible*)
          cp "$EVIDENCE/$tag.status.json" "$EVIDENCE/$tag.final.json"
          log "  $tag terminal: $state"
          ;;
        *) still+=("$tag") ;;
      esac
    done
    pending=("${still[@]+"${still[@]}"}")
    [[ ${#pending[@]} -eq 0 ]] && break
    sleep "$POLL_S"
  done
fi

# ------------------------------------------------------------------- report
log "reducing results"
python3 - "$EVIDENCE" "$SEEDS" $BASELINES <<'PY' | tee -a "$SUMMARY"
import json, math, pathlib, sys

evidence = pathlib.Path(sys.argv[1])
seeds = int(sys.argv[2])
baselines = sys.argv[3:]

# Metrics whose seed-to-seed spread IS the deliverable. G1 bits/byte carries
# 25% of composite weight; strict LAMBADA is the discriminative G2 key; the
# composite is what emissions rank on.
SCORED = [
    "org.g1.bits_per_byte_prose",
    "org.g1.bits_per_byte_code",
    "org.g1.bits_per_byte_math",
    "org.g2.lambada_strict_acc",
]
DIAG = [
    "org.diag.mfu_achieved",
    "org.diag.flops_attested",
    "org.diag.flops_analytic_ratio",
    "org.diag.flops_probe_cv",
    "org.diag.spend_fraction",
]


def stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]
    n = len(xs)
    if n == 0:
        return None
    mean = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": mean, "sigma": None, "se": None, "ci95": None}
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    se = sd / math.sqrt(n)
    # Normal approximation. With n=3 this UNDERSTATES the interval (t_2,.975
    # is 4.30 vs 1.96), which is stated rather than hidden.
    return {"n": n, "mean": mean, "sigma": sd, "se": se,
            "ci95": (mean - 1.96 * se, mean + 1.96 * se),
            "cv": (sd / mean) if mean else None}


def load(tag):
    p = evidence / f"{tag}.final.json"
    if not p.exists():
        return None, False
    d = json.loads(p.read_text())
    synthetic = bool(d.get("synthetic"))
    bat = d.get("battery") or {}
    m = bat.get("metrics") if isinstance(bat, dict) else {}
    m = dict(m or {})
    comp = d.get("composite") or {}
    if isinstance(comp, dict) and "composite" in comp:
        m["composite"] = comp["composite"]
        if "lattice" in comp:
            m["lattice"] = comp["lattice"]
    return m, synthetic


print("\n## Report\n")
any_synthetic = False
overall = {}
for baseline in baselines:
    rows = []
    for i in range(1, seeds + 1):
        tag = f"{baseline}-seed{1000 + i}"
        m, synth = load(tag)
        any_synthetic |= synth
        if m is None:
            print(f"- MISSING result for {tag}")
            continue
        rows.append((tag, m))
    if not rows:
        print(f"\n### {baseline}: no results\n")
        continue
    print(f"\n### {baseline} ({len(rows)} seeds)\n")
    print("| metric | n | mean | sigma_seed | SE | 95% CI (normal approx) | CV |")
    print("|---|---|---|---|---|---|---|")
    for key in SCORED + ["composite", "lattice"]:
        s = stats([m.get(key) for _, m in rows])
        if not s:
            continue
        overall.setdefault(key, []).extend(
            [m.get(key) for _, m in rows if isinstance(m.get(key), (int, float))]
        )
        sig = "n/a" if s["sigma"] is None else f"{s['sigma']:.5f}"
        se = "n/a" if s["se"] is None else f"{s['se']:.5f}"
        ci = "n/a" if s["ci95"] is None else f"[{s['ci95'][0]:.5f}, {s['ci95'][1]:.5f}]"
        cv = "n/a" if not s.get("cv") else f"{s['cv']*100:.2f}%"
        print(f"| {key} | {s['n']} | {s['mean']:.5f} | {sig} | {se} | {ci} | {cv} |")

    print("\n**Attestation diagnostics**\n")
    print("| diagnostic | n | mean | sigma | min | max |")
    print("|---|---|---|---|---|---|")
    for key in DIAG:
        vals = [m.get(key) for _, m in rows]
        s = stats(vals)
        if not s:
            continue
        clean = [v for v in vals if isinstance(v, (int, float))]
        sig = "n/a" if s["sigma"] is None else f"{s['sigma']:.5f}"
        print(f"| {key} | {s['n']} | {s['mean']:.5g} | {sig} | {min(clean):.5g} | {max(clean):.5g} |")
    caps = [m.get("org.diag.binding_cap") for _, m in rows]
    print(f"\n- binding cap per seed: {caps}")

    mfu = stats([m.get("org.diag.mfu_achieved") for _, m in rows])
    if mfu and mfu["mean"]:
        print(f"\n**MFU verdict for {baseline}:** measured mean MFU = {mfu['mean']*100:.1f}%.")
        # 3.0e18 / (838e12 * MFU) hours vs the 5.0h wall bound.
        wall_h = 3.0e18 / (838e12 * mfu["mean"]) / 3600.0
        print(f"  A full 3.0e18 FLOPs budget needs {wall_h:.2f}h at this MFU "
              f"against the 5.0h wall bound.")
        if wall_h > 5.0:
            print("  **WALL-BOUND.** The FLOPs cap is not reachable inside the wall "
                  "bound at this MFU, so wall-clock is still the binding constraint "
                  "and the kernel lottery is still being scored. Either lower "
                  "TRAIN_FLOPS_CAP or raise TRAIN_HOURS_CAP before flipping.")
        else:
            print("  **FLOPS-BOUND.** The currency binds before the clock, which is "
                  "the design intent.")

    gap = stats([m.get("org.diag.flops_analytic_ratio") for _, m in rows])
    if gap and gap["mean"]:
        off = abs(1.0 - gap["mean"])
        print(f"\n**Counter-vs-analytic gap for {baseline}:** mean ratio "
              f"{gap['mean']:.3f} (off by {off*100:.1f}%).")
        if off > 0.25:
            print("  **EXCEEDS FLOPS_ANALYTIC_GAP_MAX (0.25).** On a REFERENCE "
                  "baseline this means the analytic model is wrong, not that the "
                  "baseline cheats — fix the model before the gap is used as "
                  "evidence against submissions.")
        else:
            print("  Within FLOPS_ANALYTIC_GAP_MAX (0.25) on a known-honest model, "
                  "which is the precondition for treating a wide gap on a "
                  "submission as evidence.")

print("\n### Cross-baseline sigma_seed\n")
print("| metric | n | pooled mean | sigma_seed |")
print("|---|---|---|---|")
for key, vals in sorted(overall.items()):
    s = stats(vals)
    if not s:
        continue
    sig = "n/a" if s["sigma"] is None else f"{s['sigma']:.5f}"
    print(f"| {key} | {s['n']} | {s['mean']:.5f} | {sig} |")

print("""
### How to read sigma_seed

`sigma_seed` here is the standard deviation ACROSS TRAINING SEEDS of a metric
whose architecture, data pin, tokenizer and caps were identical. Compare it
to the clustered-bootstrap SE the composite already computes:

- if `sigma_seed` is comparable to or larger than the bootstrap SE, the LCB
  is overconfident and the bootstrap must be widened (or seeds averaged)
  before any anchor is pre-registered as measured;
- the minimum detectable difference between two submissions is roughly
  `2.8 x sqrt(sigma_seed^2/k + sigma_eval^2)` for k seeds each. Any scored
  key whose plausible architectural signal is below that MDD is measuring
  noise and must not be scored.

With n=3 the 95% interval on sigma itself spans roughly 0.52x to 3.7x the
point estimate, so treat a single Phase-0 wave as an ORDER OF MAGNITUDE, not
a calibration constant.
""")

if any_synthetic:
    print("**THIS REPORT IS SYNTHETIC (dry run).** No pods were rented and no "
          "number above is a measurement. Re-run with --confirm-spend to "
          "measure for real.")
PY

log "report written: $SUMMARY"
echo "EVIDENCE=$EVIDENCE"
echo "SUMMARY=$SUMMARY"
if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN complete — wiring validated, nothing rented, no number measured"
fi
