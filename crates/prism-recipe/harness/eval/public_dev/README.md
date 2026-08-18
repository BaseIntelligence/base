# PRISM eval — public_dev fixtures (tiny, regenerable)

This directory is the **embedded** `public_dev` tier of the G1–G8 battery:
published to miners, embedded in the harness, and used on-pod whenever
`$PRISM_EVAL_ASSETS_DIR` is absent (the run is then reported with
`eval_tier: "public_dev"`).

For overnight / production scoring, stage a **public HF held-out pack**
(see `build_public_pack.py`) so the run reports `eval_tier: "public"` with
full G1 domains + fresh FineWeb dump + G2 + G5 natural. That pack is
**not secret** — it is built from public datasets; post-train staging only
blocks in-process train contamination.

## Layout

| Path | Role |
|------|------|
| `seeds.json` | Published dev seed (`public_seed`) for the procedural generators. |
| `g2/<task>.jsonl` | Tiny public 0-shot anchors for G2. |
| `g1/domains/<name>.jsonl` | Tiny multi-domain held-out anchors for G1 (`{"text"}` rows). |
| `g1/fresh.jsonl` | Tiny synthetic smoke rows so the canonical fresh-crawl key is contract-testable; production uses the staged FineWeb dump from `build_public_pack.py`. |
| `g5/natural/*` | Tiny LongBench/HELMET-shaped smoke fixtures — not production natural pools. |

## Staged public pack (operator-side, assets never committed)

```
$PRISM_EVAL_ASSETS_DIR/
  tier.json             # {"tier":"public"} — default
  g1/domains/*.jsonl    # prose/math/code/news from public HF
  g1/fresh.jsonl        # FineWeb CC-MAIN-2025-* (not fineweb-edu train pin)
  g2/<task>.jsonl       # official val splits
  g5/...                # filler + ruler_qa + natural (LongBench-v2 + HELMET)
```

Build:

```bash
export PRISM_EVAL_ASSETS_DIR=/tmp/prism-eval-assets
export G5_NATURAL_SRC=/tmp/natural-packs/g5/natural   # optional prebuilt
python3 crates/prism-recipe/harness/eval/build_public_pack.py
```

Optional secret contamination mirrors: `PACK_TIER=private` or
`PRISM_EVAL_TIER=private`. Gold answers for procedural tasks still come from
the generator seed (`SECRET_SEED` staged post-train), never from miner-visible
files.
