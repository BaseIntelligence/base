"""prismlib — library modules of the PRISM pod harness (recipe 2.1.0).

The harness is a multi-file package uploaded to the Lium pod by `prism-lium`:

- `main.py` (package root) — parent orchestrator: dataset pin verify, pod
  manifest, miner subprocess spawn, METRICS_JSON v2 emission.
- `prismlib/` — these library modules (imported by both the parent and the
  miner subprocess; the parent never imports miner code).
- `eval/` — G1–G8 battery registry (battery modules land in a later step).

Recipe 2.0 AutoModel staging + train adapter: `prismlib.automodel`
(operator-owned; miners submit pin id + unified diff only).

Pure stdlib + torch + transformers + pyarrow (preinstalled pod image).
"""

RECIPE_SEED = 0x00505249534D
RECIPE_VERSION = "2.1.0"
# Pinned FALLBACK tokenizer, used only when a submission declares none (see
# `prismlib.tokenizer`). The tokenizer is part of the submission; this pin is
# a default for legacy submissions and the reference baselines, never a rule.
DEFAULT_TOKENIZER = "gpt2"
VAL_ROWS = 256
TRAIN_ROWS = 2048
MAX_TELEMETRY_POINTS = 2048
MAX_LAYER_STAT_KEYS = 64
LN2 = 0.6931471805599453
