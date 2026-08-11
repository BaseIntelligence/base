# Prism overnight battery (100% plugged, public pack)

Operator recipe for a full-train + full-eval overnight Lium run with the
**public HF held-out** eval pack (`eval_tier=public`).

## Goals

- Full train budget (recipe default **6h** / `PRISM_TRAIN_HOURS_CAP`) — **no**
  `PRISM_TEST_TRAIN_MINUTES` unless you also keep `PRISM_TEST_EVAL_CAPS=0`
  and accept a short train.
- Full eval grids: G3 n16/g64, G4 all tiers, G5 4k–64k (RULER), G7 32k,
  G8 real µP sweep — `PRISM_TEST_EVAL_CAPS=0` and **unset** reduced
  `PRISM_EVAL_*_CAP` / `PRISM_EVAL_N_ITEMS` overrides.
- Staged pack from public HF (not embedded `public_dev` stubs).
- Real Lium: `PRISM_FORCE_SIM=false`, `PRISM_FLOW=v3`,
  `PRISM_MAX_CONCURRENT_EVALS=1`.

## 1. Build the public pack

```bash
export PRISM_EVAL_ASSETS_DIR=/tmp/prism-eval-assets
# Prefer a prebuilt LongBench+HELMET natural pool when available:
export G5_NATURAL_SRC=/tmp/natural-packs/g5/natural
python3 crates/prism-recipe/harness/eval/build_public_pack.py
# Expect: g1/domains/{prose,math,code,news}.jsonl, g1/fresh.jsonl,
#         g2/*.jsonl, g5/natural/natural_mcq.jsonl (≥16 rows), tier.json
cat "$PRISM_EVAL_ASSETS_DIR/tier.json"   # {"tier":"public",...}
```

Held-out rule: G1 fresh must come from a FineWeb dump **other than**
`HuggingFaceFW/fineweb-edu@sample/10BT` (train pin). The builder enforces that.

## 2. Overnight compose override

Write `/tmp/prism-overnight-compose.override.yml` (example also produced by
`deploy/scripts/prism-overnight-battery.sh`):

```yaml
services:
  prism-challenge:
    volumes:
      - /tmp/prism-eval-assets:/tmp/prism-eval-assets:ro
      - /tmp/prism-artifacts:/tmp/prism-artifacts
      - ./crates/prism-recipe/harness:/opt/prism/harness:ro
    environment:
      PRISM_FORCE_SIM: "false"
      PRISM_FLOW: "v3"
      PRISM_EVAL_ASSETS_DIR: "/tmp/prism-eval-assets"
      PRISM_ARTIFACT_DIR: "/tmp/prism-artifacts"
      PRISM_TEST_EVAL_CAPS: "0"
      PRISM_MAX_CONCURRENT_EVALS: "1"
      PRISM_TRAIN_HOURS_CAP: "6"
      # Intentionally unset / absent:
      # PRISM_TEST_TRAIN_MINUTES, PRISM_TEST_MAX_PARAMS,
      # PRISM_EVAL_N_ITEMS, PRISM_EVAL_G*_CAP (use harness production defaults)
```

Stack on top of the usual local/master compose files and recreate
`prism-challenge`. Confirm with:

```bash
docker exec base-prism-challenge-1 env | grep -E 'PRISM_(FORCE|FLOW|TEST_|EVAL_|TRAIN_|MAX_)' | sort
# Must NOT show PRISM_TEST_TRAIN_MINUTES; must show PRISM_TEST_EVAL_CAPS=0
```

## 3. Submit baseline (or miner tree) and poll

```bash
EVIDENCE=/tmp/prism-overnight-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$EVIDENCE"
# Use deploy/scripts/prism-overnight-battery.sh — submits transformer_pp
# baseline, writes SUMMARY.md, polls until terminal or agent wall.
```

## 4. Success checklist

| Check | Expect |
|-------|--------|
| `eval_tier` | `public` |
| `org.g1.bits_per_byte_{prose,math,code,fresh_crawl}` | present |
| tiny_caps / g8.mup.stub_reason_tiny_caps | absent |
| `org.g8.mup_lr_stability` | present (0.0 only if sweep diverged after full attempt) |
| `org.g6.*` | real curve (≥2 probes) |
| `inference_traces` | populated |
| artifacts | RECEIPT if checkpoint harvested |

## 5. Watch

```bash
tail -f /tmp/prism-overnight-*/SUMMARY.md
docker logs -f base-prism-challenge-1
curl -sS http://127.0.0.1:28092/v1/submissions/<id>
```
