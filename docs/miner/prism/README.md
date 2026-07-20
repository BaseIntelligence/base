# Miner hub (Prism)

> **Monorepo hub.** Product source of truth is
> [`packages/challenges/prism`](../../../packages/challenges/prism/) inside
> **BaseIntelligence/base**. Public slug `/challenges/prism` is unchanged.
> Pack scripts and examples live under that package path from the monorepo root.

Mine Prism on **[joinbase.ai](https://joinbase.ai)** in minutes. Prism is the BASE
**research lab** challenge: try **new architectures**, earn when held-out generalization
wins under fair challenge-owned re-exec.

**Day-1 target:** hotkey ready → pack example seed → signed bridge submit → leaderboard,
in under 15 minutes. Science grade, promote ladder depth, and worker-plane GPU ops stay in
Concepts / linked deep docs — not on the first page.

| Page | What it covers |
|------|----------------|
| [Getting started](getting-started.md) | Hotkey, pack seed, sign headers, bridge POST, leaderboard, checklist |
| [Concepts](concepts.md) | Emission vs science, 50% share, NO-TEE honesty, ladder |
| [Troubleshooting](troubleshooting.md) | 401 / 409 / 429 / 502 and common rejects |

## Canonical public URLs

| Surface | URL |
|---------|-----|
| Product / dashboard | https://joinbase.ai |
| Base master API | https://chain.joinbase.ai |
| Submit bridge | `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` |
| Leaderboard | `GET https://chain.joinbase.ai/challenges/prism/leaderboard` |
| OpenAPI | `GET https://chain.joinbase.ai/challenges/prism/openapi.json` |

## Quick path

1. Link wallet hotkey on https://joinbase.ai.
2. Pack (from monorepo root / package checkout):
   `uv run --package prism-challenge python packages/challenges/prism/scripts/pack_seed_family.py --family transformer-tiny-1m --output-dir dist/seed-packages`
3. Sign canonical `prism:{hotkey}:{nonce}:{timestamp}:{sha256(zip)}` and POST the zip to the
   bridge URL above with `X-Hotkey` / `X-Nonce` / `X-Timestamp` / `X-Signature`.
4. Watch https://chain.joinbase.ai/challenges/prism/leaderboard.

Full walkthrough: [Getting started](getting-started.md).

## Purpose (lab identity)

PRISM is a **research lab**. The **norm** is to try **new architectures**. The **goal** is to find
**more performant** ones for our LLM target under fair challenge-owned re-exec. You submit two
scripts, a model `architecture.py` and a custom `training.py` loop; the challenge re-executes your
loop under a forced random init on locked FineWeb-Edu data. **Emission** ranks **held-out /
generalization primary** with prequential bits-per-byte **secondary**. You bring the model and the
loop, not the data or the metric.

For offline architecture-agnostic **Official Comparison Protocol v1** / multimetric scorecard
`multimetric.v1.1` (scientific multi-axis grade; held-out primary, prequential bpb secondary,
honest hooks, wall-clock never ranks; prior K=1 wins provisional only; multimetric is **not**
the emission scalar), see
[Official Comparison](../../../packages/challenges/prism/docs/official-comparison.md).
The live emission path is described in
[Scoring](../../../packages/challenges/prism/docs/scoring.md).
Brief miner-facing summary: [Concepts](concepts.md).

## Dual param ladder (start small)

| Stage | Cap | Role |
| --- | ---: | --- |
| Explore / provisional | **124M** | Default continuous thrash; may provisional-crown emission |
| Promote / final | **350M** | Confirm or revoke provisional crown on same package/family pin |

## Miner Flow

1. Build a two-script bundle that follows the PRISM contract. Novel `nn.Module` architectures
   under AST + dual ladder are **expected**, not second-class (Transformer, pure-torch SSM,
   LightDeepLoop-class looped depth when the forward contract holds, hybrids, …). Seeds
   `tiny-1m` / `mamba-tiny` are starting shapes only.
2. Sign and submit it with your miner hotkey via the joinbase bridge
   (`POST https://chain.joinbase.ai/v1/challenges/prism/submissions`).
3. PRISM runs the static sandbox and **deterministic admission** (similarity / anti-cheat). There is no
   LLM hard gate.
4. The challenge re-executes your `training.py` under a forced random init on the locked train split.
5. The challenge computes emission rank: **held-out primary**, prequential bits-per-byte **secondary**.
6. Better generalizing learners earn more weight after master aggregation of PRISM raw weights
   (architecture/training pools just **0.50 / 0.50**; validators submit on-chain; the challenge does not).
   Prism’s absolute emission share on BASE defaults to **50%** (paired with agent-challenge).

## Lab seed families (default exploration under 124M)

Tracked lab seeds under `examples/` package with the same outer two-script zip contract via
`packages/challenges/prism/scripts/pack_seed_family.py` / `prism_challenge.seed_packaging`. **Start here** under the explore cap
before any 350M promote:

| Family id | Path | Notes |
| --- | --- | --- |
| `transformer-tiny-1m` | `examples/tiny-1m` | Imp baseline: weight-tied ~1M decoder transformer; default explore shape under 124M; multi-GPU single-node ≤8 |
| `mamba-tiny-1m` | `examples/mamba-tiny` | Imp baseline: pure-PyTorch selective SSM (Mamba-style); **no** `mamba_ssm` C++/CUDA dep; same dual ladder + multi-GPU contract |
| `deeploop-tiny-1m` | `examples/deeploop-tiny` | Novel DeepLoop-class shared-weight looped residual (arXiv 2607.13491 class); ~1–1.5M; pure torch |
| `gated-delta-tiny-1m` | `examples/gated-delta-tiny` | Novel gated delta-rule linear recurrence (DeltaNet 2406.06484 class); ~1.5–3M; sequential pure torch |
| `hybrid-attn-ssm-tiny-1m` | `examples/hybrid-attn-ssm-tiny` | Novel hybrid causal attn × pure-torch SSM (Hymba/Jamba-mini spirit); ~2–4M; no `mamba_ssm`/`flash_attn` |

Family knobs that matter for lab interpretation (not product baseline tables):

- **Param counting** — architecture-agnostic, counts tensors from `build_model(ctx)`; both seeds weight-tie emb/lm_head. Mamba counts include `A_log`/`D`/conv/dt projections rather than MHA/MLP tensors.
- **Step throughput** — `LOCAL_BATCH`, optimizer LR, and token budget dominate step flu; score is compute-normalized. Pure-torch Mamba sequential scan is slower/token than fused CUDA kernels (use modest LR; default seed uses `0.003` vs transformer `0.005`).
- **Stability** — multi-GPU static contract requires distributed primitives + rank-0 writes; works at `world_size=1` for both families. Mamba pure-torch caveat: do not introduce blocked `mamba_ssm` / `cpp_extension` imports if you still need AST sandbox static pass.

## The Two-Script Contract

A bundle is a `.zip` (or directory) with two distinct scripts. An optional `prism.yaml` declares the
entrypoints and tokenizer:

```yaml
architecture:
  entrypoint: architecture.py
training:
  entrypoint: training.py
tokenizer: gpt2
```

`architecture.py` exposes the model factory; `training.py` exposes the loop you own:

```python
# architecture.py
def build_model(ctx):
    return MyModel(ctx.vocab_size)
```

```python
# training.py
from architecture import build_model

def train(ctx):
    model = build_model(ctx)
    # optimizer/schedule, read the locked train split from ctx.data_dir, tokenize,
    # run the loop, handle multi-GPU, write only under ctx.artifacts_dir.
    ...
```

`build_model(ctx)` returns any `torch.nn.Module` under the AST sandbox, the dual param ladder
(explore ≤ **124M**, promote ≤ **350M**), and the resource limits; it must not read data, open files,
touch the network, or reference the dataset. Novel architectures are expected.
`train(ctx)` owns the optimizer, schedule, dataloading, tokenization, multi-GPU strategy, and loop. The
single-module re-export idiom no longer satisfies the contract: the two roles must be distinct files.

## Context And Limits

`ctx` is a `PrismContext` supplying the metadata and limits you need:

- `vocab_size`, `max_seq_len` — token-id geometry;
- `max_params` — stage cap (**124M explore** / **350M promote**);
- `seed` — the forced seed you cannot change;
- `data_dir` — read-only path to the locked FineWeb-Edu **train** split;
- `artifacts_dir` — the only writable path;
- `world_size`, `rank`, `local_rank`, `device` — the distributed launch;
- `token_budget` / `step_budget` — the compute budget;
- `ctx.build_model()` and `ctx.reference_tokenizer("gpt2" | "llama")` — offline, no network.

Read raw text from `ctx.data_dir` and tokenize with your own tokenizer or a pre-staged reference; fail
closed if the locked data is missing.

## Locked Data, No Network

The train split is read-only at `ctx.data_dir`; the `val`/`test` splits are secret and never exposed to
your script. The eval container runs with `network=none`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, so there is no network during training: do not download data, tokenizers, or
weights at runtime.

## Multi-GPU

Your `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>`; PRISM is single-node (1-8 GPUs) and the
official scored run uses `torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path). A
correct loop calls `init_process_group`, wraps the model with DDP or FSDP, shards data per-rank, does
rank-0-only writes, all-reduces metrics, tears down the process group, and also works at `world_size=1`.
It is checked with a static contract and a gloo multi-rank test. See [Scaling](../../../packages/challenges/prism/docs/scaling.md).

## The Challenge Computes The Score

PRISM re-executes your loop under a forced random init, captures the single-pass online loss itself, and
writes a challenge-authored `prism_run_manifest.v2.json`; any value or manifest you write is ignored.
**Emission** ranks **held-out / generalization primary** (preferred: held-out delta-over-random-init)
with prequential bits-per-byte **secondary** (area under the from-scratch loss curve, normalized by
raw UTF-8 bytes). Multimetric / Complete View publish scientific multi-axis research grade and do
**not** silently replace emission. A smuggled pretrained model shows an anomalous step-0 loss and is
zeroed; an excessive train-vs-held-out gap is penalized as memorization.

## Submitting Work

**Day-1 production path** is the BASE bridge (preferred):

```http
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
Content-Type: application/zip
X-Hotkey: <ss58>
X-Nonce: <unique>
X-Timestamp: <unix-seconds>
X-Signature: <sr25519 over canonical payload>
```

Canonical payload:

```text
prism:{hotkey}:{nonce}:{timestamp}:{sha256_hex(body)}
```

Full pack/sign/checklist: [Getting started](getting-started.md).

When public direct submit is enabled on a local challenge, JSON shape is still supported:

```http
POST /v1/submissions
Content-Type: application/json
```

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

The hotkey must match the signature (timestamps and nonces block replay), and stay within the size
limit. Unsafe imports, network access, arbitrary filesystem access, deserialization escapes, and the
single-module idiom are rejected at static review before any GPU work. Close source copies of prior
work can be rejected by deterministic similarity (including borderline bands) with no operator review
queue.

## What Improves Your Score

- Grow generalization on the secret val split (**held-out primary** for emission).
- Drive the from-scratch loss down fast (lower bits-per-byte is **secondary** emission signal).
- Use the compute budget efficiently (scoring is compute-normalized, never wall-clock).
- Keep the train-vs-held-out gap small (a large gap is penalized as memorization).
- Ship correct, DDP-safe, rank-aware distributed behavior.
- Explore under **124M** first; promote to **350M** only to confirm durable claims.

## Miner Checklist

- Ship two distinct scripts: `architecture.py` with `build_model(ctx)` and `training.py` with `train(ctx)`.
- Keep `build_model` pure (no data, files, or network); read only `ctx.data_dir`, write only `ctx.artifacts_dir`.
- Stay under the stage param cap (**124M explore** / **350M promote**) and inside the AST sandbox.
- Prefer starting from `examples/tiny-1m` or `examples/mamba-tiny` under 124M.
- Make the loop deterministic under the forced seed and correct at `world_size=1`.
- Remove secrets, private endpoints, generated caches, and unrelated files.
- Submit via `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` with signed headers.
- When stuck, use [Troubleshooting](troubleshooting.md) before opening issues.
