> **API truth is OpenAPI** (`https://chain.joinbase.ai/challenges/prism/openapi.json`, `/docs`).
> Day-1 miners: repo-root [`docs/miner/getting-started.md`](../../../../docs/miner/getting-started.md).
> This page is a short product pin note, not a route dump.

# Security Model

PRISM evaluates untrusted miner code. It assumes submissions may be malicious and layers identity
verification, a static AST sandbox, **deterministic admission** (project shape, source similarity,
anti-cheat), a terminal duplicate policy, and a forced-init re-execution that makes the common cheats
inert rather than merely detected. The former LLM hard gate and master LLM gateway are **removed**.

## Identity and Authentication

BASE handles miner-facing uploads, verifying hotkey identity, signatures, timestamps, nonces, request
freshness, and challenge routing before forwarding the payload to `POST /internal/v1/bridge/submissions`;
PRISM trusts the verified hotkey header only on authenticated internal requests. Internal endpoints
require the shared BASE challenge token (`Authorization: Bearer <shared-token>`), read from
`PRISM_SHARED_TOKEN`, `CHALLENGE_SHARED_TOKEN`, or a secret file (prefer secret files in production).

## Static Sandbox

Before any GPU work, PRISM runs the static gates over both scripts, in order:

1. **AST hard-blocks** over `architecture.py` and `training.py`: no `os`, `sys`, `subprocess`,
   `socket`, network clients, `pickle`/`torch.load` of untrusted paths, `ctypes`, dynamic `importlib`,
   `eval`/`exec`/`compile`, attribute escapes (`__globals__`, `__reduce__`, `__class__` walking), or
   filesystem writes outside `artifacts_dir`.
2. **Forced-seed parameter cap**: `build_model(ctx)` is instantiated under the forced seed in a bounded
   child process and rejected if it exceeds the stage param cap (124M explore / 350M promote; realized first-forward shapes).
3. **Multi-GPU static contract**: the training script must use the distributed primitives and a rank-0
   write guard; a `gpu_count > 8` or multi-node request is rejected.

A rejection at any static gate is terminal before similarity admission and before any GPU work.

BASE handles miner-facing uploads, verifying hotkey identity, signatures, timestamps, nonces, request
freshness, and challenge routing before forwarding the payload to `POST /internal/v1/bridge/submissions`;
PRISM trusts the verified hotkey header only on authenticated internal requests. Internal endpoints
require the shared BASE challenge token (`Authorization: Bearer <shared-token>`), read from
`PRISM_SHARED_TOKEN`, `CHALLENGE_SHARED_TOKEN`, or a secret file (prefer secret files in production).

1. **AST hard-blocks** over `architecture.py` and `training.py`: no `os`, `sys`, `subprocess`,
   `socket`, network clients, `pickle`/`torch.load` of untrusted paths, `ctypes`, dynamic `importlib`,
   `eval`/`exec`/`compile`, attribute escapes (`__globals__`, `__reduce__`, `__class__` walking), or
   filesystem writes outside `artifacts_dir`.
2. **Forced-seed parameter cap**: `build_model(ctx)` is instantiated under the forced seed in a bounded
   child process and rejected if it exceeds the stage param cap (124M explore / 350M promote; realized first-forward shapes).
3. **Multi-GPU static contract**: the training script must use the distributed primitives and a rank-0
   write guard; a `gpu_count > 8` or multi-node request is rejected.

A rejection at any static gate is terminal before similarity admission and before any GPU work.

After static gates pass, PRISM applies challenge-owned deterministic checks only:

- **Project shape** — two-script contract resolution and fingerprints (no single-module re-export).
- **Source similarity** — exact-source-hash duplicates are rejected; borderline (quarantine-band)
  similarity is also a **terminal reject**. There is no held-for-review or LLM quarantine path.
- **Anti-cheat / scoring gates** — forced-init invariants and score anomaly multipliers (see below).

Legacy env keys or settings related to LLM review, gateway URL/token, architecture auto-report, or
component-agent hold policies fail closed at configuration load. Unknown residual knobs are rejected
rather than silently ignored when they map to removed surfaces.

The challenge re-executes the miner's `training.py` under a **forced random init** with a fixed,
challenge-controlled seed and deterministic flags set **before** any miner code runs, feeds it fresh
single-pass batches from the locked train split, and records the online loss itself. This neutralizes
the three cheat classes:

- **No pretrained weights** — forced random init makes smuggled weights inert; an impossibly low step-0
  loss is flagged and zeroes the score; `network=none` and the sandbox block IO/network/deserialization
  escapes.
- **No metric manipulation** — the metric comes from the captured loss stream, so any miner-reported
  number and any miner-written manifest are ignored. The fixed seed and data order make runs
  reproducible within tolerance.
- **No memorization** — the `val`/`test` splits are secret and **never exposed** to the miner script; an
  excessive train-vs-held-out gap penalizes the score.

When the worker plane is enabled, Prism ingests reconciled external evaluation results **only** as the
Base SDK `ExternalResultEnvelope` (api_version, assignment/challenge bindings, execution proof). Dual
or legacy reduced bodies fail closed with a 422 before scoring or persistence. Proof verification and
plausibility gates run before finalization; duplicate deliveries are idempotent and conflicts refuse
mutation of sealed scores.

## Provider Trust And IMAGE_PIN

Prism does **not** ship a TEE-attestation verifier and does **not** require TEE evidence to finalize
production scores. Integrity for miner-funded GPU work rests on:

- **PROVIDER_TRUST** — operators trust **Lium/Targon** as compute providers (no Prism crypto TEE path).
- **IMAGE_PIN** — `worker_plane.pinned_image_digest` match grants audit effective tier **1** (maximum);
  pin mismatch yields an honest non-elevated downgrade (not silent ignore).
- **DEPLOY SMOKE** — paid provider lifecycle proofs are reachability/infra only; always terminate pods.
- **LAB-GPU** — remote CUDA lab scores under Official Comparison; scientific only.
- Ordinary ExecutionProof envelope + worker signature checks on the worker-plane path.

**REAL-PROVIDER TEE** (cryptographic provider attestation PASS as a Prism product goal) is
**retired**. Historical lab tables and reports may still show `real_provider_tee=BLOCKED` as honesty
history; do not implement or document a TEE production scoring gate as the live path. Opaque
non-empty `tdx_quote_b64` / `gpu_eat_jwt` fields never imply tier 2 by presence alone (max effective
tier is 1 via image pin).

The train split is mounted **read-only** at `ctx.data_dir` and is the only data the miner script sees;
the `val`/`test` splits are secret. The eval container runs with `network=none`, `HF_HUB_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1`, so there is **no network** during training and the miner cannot download data,
tokenizers, or weights at runtime.

An exact-source-hash duplicate is rejected, and a borderline-similarity quarantine is folded into a
terminal rejection at ingress (there is no operator hold-resolution surface; the v1-NAS component-review,
LLM review, and ownership machinery were decommissioned).

PRISM never executes submitted code inside the API process without isolation. The scored run happens in
a broker-backed container that is non-root, has a read-only rootfs except `artifacts_dir`, uses
`network=none` and `no-new-privileges`, and is bounded by CPU, memory, PID, and wall-clock caps.
Host-side static instantiation and held-out scoring run in bounded child processes with
`weights_only=True` for any deserialization. Application code does **not** create ephemeral evaluator
containers; evaluation is the long-lived challenge runtime (or external worker-plane GPUs on trusted
providers when enabled).

ZIP extraction rejects symlinks, path traversal, unsafe paths, unsupported file types, and excessive
file counts or bytes before code review begins.

PRISM exposes `get_weights` for inventory/compatibility and pushes authenticated **raw** hotkey weights
to the BASE master. The master aggregates the final vector; **validators** call `set_weights` with
their own wallets. The challenge and master never write weights on-chain.

- **Supply-chain attacks** — Gu, Dolan-Gavitt, and Garg, 2017/2019 (*BadNets*): treat submitted code
  and artifacts as adversarial even when metrics look normal.
- **Untrusted deserialization** — pickle/`torch.load` RCE guidance: load host-side artifacts with
  `weights_only=True` from the challenge-recorded path only.
- **Dataset provenance** — Penedo et al., 2024 (*The FineWeb Datasets*): pin the revision and shard
  hashes; keep held-out splits secret.

- Use real secret files in production, not inline tokens.
- Keep public submissions disabled when PRISM is deployed only behind BASE.
- Keep the eval container on `network=none` and the rootfs read-only except `artifacts_dir`.
- Do **not** configure LLM gateway URL/token fields; those surfaces are gone and residual knobs fail closed.
- Prefer `worker_plane.pinned_image_digest` for IMAGE_PIN tier-1; do not enable any removed TEE production scoring path.
- Monitor rejected, failed, and completed submissions separately (legacy held is not a live path).
