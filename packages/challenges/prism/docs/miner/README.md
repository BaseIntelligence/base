> **API truth is OpenAPI** (`https://chain.joinbase.ai/challenges/prism/openapi.json`, `/docs`).
> Day-1 miners: repo-root [`docs/miner/getting-started.md`](../../../../docs/miner/getting-started.md).
> This page is a short product pin note, not a route dump.

# Miner hub (Prism)

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
2. Pack: `uv run python scripts/pack_seed_family.py --family transformer-tiny-1m --output-dir dist/seed-packages`

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

| Surface | URL |
|---------|-----|
| Product / dashboard | https://joinbase.ai |
| Base master API | https://chain.joinbase.ai |
| Submit bridge | `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` |
| Leaderboard | `GET https://chain.joinbase.ai/challenges/prism/leaderboard` |
| OpenAPI | `GET https://chain.joinbase.ai/challenges/prism/openapi.json` |

1. Link wallet hotkey on https://joinbase.ai.
2. Pack: `uv run python scripts/pack_seed_family.py --family transformer-tiny-1m --output-dir dist/seed-packages`
3. Sign canonical `prism:{hotkey}:{nonce}:{timestamp}:{sha256(zip)}` and POST the zip to the
   bridge URL above with `X-Hotkey` / `X-Nonce` / `X-Timestamp` / `X-Signature`.
4. Watch https://chain.joinbase.ai/challenges/prism/leaderboard.
**more performant** ones for our LLM target under fair challenge-owned re-exec. You submit two
scripts, a model `architecture.py` and a custom `training.py` loop; the challenge re-executes your
loop under a forced random init on locked FineWeb-Edu data. **Emission** ranks **held-out /
For offline architecture-agnostic **Official Comparison Protocol v1** / multimetric scorecard
`multimetric.v1.1` (scientific multi-axis grade; held-out primary, prequential bpb secondary,
honest hooks, wall-clock never ranks; prior K=1 wins provisional only; multimetric is **not**
the emission scalar), see [Official Comparison](../official-comparison.md). The live emission path
| Stage | Cap | Role |
| --- | ---: | --- |
| Explore / provisional | **124M** | Default continuous thrash; may provisional-crown emission |
| Promote / final | **350M** | Confirm or revoke provisional crown on same package/family pin |
1. Build a two-script bundle that follows the PRISM contract. Novel `nn.Module` architectures
   `tiny-1m` / `mamba-tiny` are starting shapes only.
2. Sign and submit it with your miner hotkey via the joinbase bridge
   (`POST https://chain.joinbase.ai/v1/challenges/prism/submissions`).
4. The challenge re-executes your `training.py` under a forced random init on the locked train split.
5. The challenge computes emission rank: **held-out primary**, prequential bits-per-byte **secondary**.
   Prism’s absolute emission share on BASE defaults to **50%** (paired with agent-challenge).
Tracked lab seeds under `examples/` package with the same outer two-script zip contract via
`scripts/pack_seed_family.py` / `prism_challenge.seed_packaging`. **Start here** under the explore cap
| Family id | Path | Notes |
| --- | --- | --- |
| `transformer-tiny-1m` | `examples/tiny-1m` | Imp baseline: weight-tied ~1M decoder transformer; default explore shape under 124M; multi-GPU single-node ≤8 |
| `mamba-tiny-1m` | `examples/mamba-tiny` | Imp baseline: pure-PyTorch selective SSM (Mamba-style); **no** `mamba_ssm` C++/CUDA dep; same dual ladder + multi-GPU contract |
| `deeploop-tiny-1m` | `examples/deeploop-tiny` | Novel DeepLoop-class shared-weight looped residual (arXiv 2607.13491 class); ~1–1.5M; pure torch |
| `gated-delta-tiny-1m` | `examples/gated-delta-tiny` | Novel gated delta-rule linear recurrence (DeltaNet 2406.06484 class); ~1.5–3M; sequential pure torch |
| `hybrid-attn-ssm-tiny-1m` | `examples/hybrid-attn-ssm-tiny` | Novel hybrid causal attn × pure-torch SSM (Hymba/Jamba-mini spirit); ~2–4M; no `mamba_ssm`/`flash_attn` |
- **Param counting** — architecture-agnostic, counts tensors from `build_model(ctx)`; both seeds weight-tie emb/lm_head. Mamba counts include `A_log`/`D`/conv/dt projections rather than MHA/MLP tensors.
- **Step throughput** — `LOCAL_BATCH`, optimizer LR, and token budget dominate step flu; score is compute-normalized. Pure-torch Mamba sequential scan is slower/token than fused CUDA kernels (use modest LR; default seed uses `0.003` vs transformer `0.005`).
- **Stability** — multi-GPU static contract requires distributed primitives + rank-0 writes; works at `world_size=1` for both families. Mamba pure-torch caveat: do not introduce blocked `mamba_ssm` / `cpp_extension` imports if you still need AST sandbox static pass.
A bundle is a `.zip` (or directory) with two distinct scripts. An optional `prism.yaml` declares the
```yaml
  entrypoint: architecture.py
```
`architecture.py` exposes the model factory; `training.py` exposes the loop you own:
```python
# architecture.py
def build_model(ctx):
```
```python
# training.py
from architecture import build_model
    model = build_model(ctx)
    # run the loop, handle multi-GPU, write only under ctx.artifacts_dir.
```
`build_model(ctx)` returns any `torch.nn.Module` under the AST sandbox, the dual param ladder
(explore ≤ **124M**, promote ≤ **350M**), and the resource limits; it must not read data, open files,
`train(ctx)` owns the optimizer, schedule, dataloading, tokenization, multi-GPU strategy, and loop. The
single-module re-export idiom no longer satisfies the contract: the two roles must be distinct files.
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
The train split is read-only at `ctx.data_dir`; the `val`/`test` splits are secret and never exposed to
your script. The eval container runs with `network=none`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, so there is no network during training: do not download data, tokenizers, or
Your `training.py` owns multi-GPU scaling. The harness launches
`torchrun --standalone --nnodes=1 --nproc-per-node=<gpu_count>`; PRISM is single-node (1-8 GPUs) and the
official scored run uses `torchrun --standalone --nnodes=1 --nproc-per-node=1` (the `nproc=1` path). A
correct loop calls `init_process_group`, wraps the model with DDP or FSDP, shards data per-rank, does
rank-0-only writes, all-reduces metrics, tears down the process group, and also works at `world_size=1`.
writes a challenge-authored `prism_run_manifest.v2.json`; any value or manifest you write is ignored.
**Emission** ranks **held-out / generalization primary** (preferred: held-out delta-over-random-init)
raw UTF-8 bytes). Multimetric / Complete View publish scientific multi-axis research grade and do
**not** silently replace emission. A smuggled pretrained model shows an anomalous step-0 loss and is
zeroed; an excessive train-vs-held-out gap is penalized as memorization.
```http
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
X-Hotkey: <ss58>
```
```text
```
```http
POST /v1/submissions
```
```json
```
The hotkey must match the signature (timestamps and nonces block replay), and stay within the size
single-module idiom are rejected at static review before any GPU work. Close source copies of prior
work can be rejected by deterministic similarity (including borderline bands) with no operator review
- Grow generalization on the secret val split (**held-out primary** for emission).
- Drive the from-scratch loss down fast (lower bits-per-byte is **secondary** emission signal).
- Use the compute budget efficiently (scoring is compute-normalized, never wall-clock).
- Keep the train-vs-held-out gap small (a large gap is penalized as memorization).
- Ship correct, DDP-safe, rank-aware distributed behavior.
- Explore under **124M** first; promote to **350M** only to confirm durable claims.
- Ship two distinct scripts: `architecture.py` with `build_model(ctx)` and `training.py` with `train(ctx)`.
- Keep `build_model` pure (no data, files, or network); read only `ctx.data_dir`, write only `ctx.artifacts_dir`.
- Stay under the stage param cap (**124M explore** / **350M promote**) and inside the AST sandbox.
- Prefer starting from `examples/tiny-1m` or `examples/mamba-tiny` under 124M.
- Make the loop deterministic under the forced seed and correct at `world_size=1`.
- Remove secrets, private endpoints, generated caches, and unrelated files.
- Submit via `POST https://chain.joinbase.ai/v1/challenges/prism/submissions` with signed headers.
- When stuck, use [Troubleshooting](troubleshooting.md) before opening issues.
