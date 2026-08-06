# PRISM eval — public dev family (regenerable)

This directory is the **public** tier of the G1–G8 evaluation battery:
published to miners, embedded in the harness, and used on-pod whenever
`$PRISM_EVAL_ASSETS_DIR` is absent (the run is then reported with
`eval_tier: "public_dev"`).

## Layout

| Path | Role |
|------|------|
| `seeds.json` | Published dev seed (`public_seed`) for the procedural generators. The private family is the *same code* (`eval/generators.py`, `eval/gen_reasoning.py`, `eval/gen_longctx.py`) with operator-held seeds: `seed = cantor(RECIPE_SEED, secret_seed, task_id)`. |
| `g2/<task>.jsonl` | Public 0-shot anchors for G2 (lambada, hellaswag, piqa, arc_easy, arc_challenge, winogrande, boolq, openbookqa). Rows: `{"prompt", "choices", "gold"}` — prompts are fully formed and frozen (OLMES-style, 0-shot); the module is generic. |
| `g1/domains/<name>.jsonl` | Public multi-domain held-out anchors for G1 (`{"text"}` rows). |
| `g1/fresh.jsonl` | *Private tier only* (fresh-crawl stream) — intentionally absent here. |

## Regenerating / rotating the public family

1. **Procedural items (G3/G4/G5)** — nothing to regenerate: pick a new
   `public_seed`, write it into `seeds.json`, and the items change while
   the templates stay identical to the private family.
2. **G2 anchors** — hand-authored tiny dev anchors. To rebuild at scale
   for local dev, sample from the official validation splits
   (lm-evaluation-harness task configs, `acc_norm`, 0-shot) and render
   each item to the frozen prompt format above. Keep this directory
   small: it ships to every pod.
3. **G1 domain anchors** — any held-out English/code text not present in
   the pinned training shard works; keep ≤ ~1 KiB rows.

## Private mirror ceremony (operator-side, assets never committed)

The operator stages `$PRISM_EVAL_ASSETS_DIR` into the container **after
the miner train subprocess is hard-killed**, with:

```
$PRISM_EVAL_ASSETS_DIR/
  seeds.json            # optional; the secret seed normally arrives via
                        # env PRISM_EVAL_SECRET_SEED instead (never on disk)
  g1/domains/*.jsonl    # difficulty-matched private mirrors
  g1/fresh.jsonl        # post-cutoff crawl sample
  g2/<task>.jsonl       # private mirrors of the public anchors
```

`PRISM_EVAL_SECRET_SEED` is delivered by env only, is combined with
`RECIPE_SEED` inside the eval code (Cantor pairing), and is unset by the
parent immediately after the eval subprocess is spawned. Gold answers
are always computed by the harness from the seed — never stored in any
miner-visible file.
