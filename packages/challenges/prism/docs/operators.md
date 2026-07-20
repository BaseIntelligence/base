# Operator Guide

Local validation and production configuration for running PRISM as a BASE challenge under
**Docker Compose** (long-lived combined challenge service). Swarm is not a supported install path.

## Installation

```bash
git clone https://github.com/BaseIntelligence/prism.git
cd prism
uv sync --frozen --extra dev
```

PRISM depends on the immutable Base public wheel **v3.1.2**:

```text
https://github.com/BaseIntelligence/base/releases/download/v3.1.2/base-3.1.2-py3-none-any.whl
#sha256=3a61c2d3a343ed6de55e80215486e3de0c9639276443d08f2ed316bc807f2ff0
```

(see `pyproject.toml`). There is no LLM gateway dependency in this pin.

## Local Validation

```bash
.venv/bin/ruff check src
.venv/bin/mypy src/prism_challenge
.venv/bin/python -m pytest tests -q
```

GPU re-execution, HuggingFace publication, and external provider calls are mocked in tests. The LLM
gateway is not part of the test or deploy path.

## Required Runtime Configuration

```bash
PRISM_DATABASE_URL=sqlite+aiosqlite:////data/prism.sqlite3
PRISM_SHARED_TOKEN_FILE=/run/secrets/base/challenge_token
PRISM_EXECUTION_BACKEND=base_gpu
```

The shared token must match the token configured in the BASE master for this challenge. Prefer secret
files over inline tokens.

## Combined Mode (Compose)

Preferred production layout is the BASE master Compose project with one long-lived PRISM service in
**combined mode**: the API process drains the eval queue in-process (same worker/DB as the HTTP app).
Do not deploy Swarm stacks or launch ephemeral evaluator jobs from application code.

Typical challenge container settings for combined mode:

```bash
# Enable API + in-process queue drain in one long-lived service
# (exact env name follows PrismSettings; set in the Compose challenge service)
PRISM_COMBINED_MODE=true
```

Digest-pin the PRISM image via the master Compose install / watcher path (`PRISM_IMAGE_REPOSITORY` +
`PRISM_IMAGE_DIGEST`). Mutable `latest` Swarm service mutation is unsupported.

## Docker Broker Configuration

When evaluation uses the BASE Docker broker with the CI-published evaluator image:

```bash
PRISM_DOCKER_ENABLED=true
PRISM_DOCKER_BACKEND=broker
PRISM_DOCKER_BROKER_URL=http://base-docker-broker:8082
PRISM_DOCKER_BROKER_TOKEN_FILE=/run/secrets/base/challenge_token
PRISM_BASE_EVAL_IMAGE=ghcr.io/baseintelligence/prism-evaluator@sha256:<pinned>
PRISM_BASE_EVAL_GPU_COUNT=1
PRISM_DOCKER_NETWORK=none
```

The scored run is single-node (`torchrun --standalone --nnodes=1 --nproc-per-node=1`). The evaluator
image must ship `sentencepiece` and an offline tiktoken gpt2 cache so reference tokenizers load with no
network. Prefer digest-pinned references over floating tags.

## Locked FineWeb-Edu Data Plane

The broker bind-mounts the locked FineWeb-Edu data read-only into the eval container (`network=none`):

```bash
PRISM_BASE_EVAL_DATA_DIR=/data/fineweb-edu/train       # miner-visible, read-only
PRISM_BASE_EVAL_VAL_DATA_DIR=/data/fineweb-edu/val     # secret; scorer-only, never mounted into eval
PRISM_BASE_EVAL_REFERENCE_TOKENIZER_DIR=/opt/reference-tokenizers
```

`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` are set inside the eval container. The `val`/`test`
splits are secret and must never be exposed to a miner script.

## Compute Budget

Compute-normalized scoring; wall-clock is only a layered safety cap:

```bash
PRISM_BASE_EVAL_BUDGET_SECONDS=1200            # graceful stop; score the partial stream
PRISM_BASE_EVAL_WATCHDOG_GRACE_SECONDS=120     # hard watchdog above the graceful budget
PRISM_BASE_EVAL_TIMEOUT_SECONDS=1800           # outer docker/broker backstop
PRISM_BASE_EVAL_ARTIFACTS_QUOTA_BYTES=2147483648
```

## Deterministic Admission (No LLM Gateway)

Admission is challenge-owned and deterministic. **Do not** set gateway URL/token or LLM review
settings; residual keys fail closed at load:

```text
# Removed — do not configure
# PRISM_LLM_REVIEW_ENABLED
# PRISM_LLM_GATEWAY_URL
# PRISM_GATEWAY_TOKEN_FILE
# BASE_LLM_GATEWAY_URL
# BASE_GATEWAY_TOKEN
```

Pipeline order before GPU:

1. AST sandbox hard-blocks over both scripts.
2. Forced-seed dual param ladder (124M explore / 350M promote).
3. Multi-GPU static contract and single-node bound.
4. Source similarity + duplicate policy (quarantine band → terminal **reject**, never held).

## Multi-GPU Static Contract

```bash
PRISM_DISTRIBUTED_CONTRACT_POLICY=reject     # reject | flag | off
PRISM_BASE_EVAL_MAX_GPU_COUNT=8
```

`reject` (default) hard-rejects a non-distributed `training.py`; `flag` advances but logs; `off` skips
the check.

## Duplicate / Similarity Policy

```bash
# thresholds also live in SQL runtime config under duplicate_thresholds
PRISM_PLAGIARISM_ENABLED=true
```

An exact-source-hash duplicate is rejected; a borderline-similarity quarantine is folded into a terminal
rejection at ingress. There is no operator hold-resolution surface.

## Raw-Weight Push

When master coordination is configured, PRISM periodically pushes authenticated raw hotkey weights to
the master private ingress (versioned epoch/revision, digest, idempotent). Configure master base URL
and challenge-scoped credentials via settings/secret files; never log tokens. Validators submit the
master-aggregated vector on-chain; PRISM does not call `set_weights`.

## External Results, Provider Trust, And IMAGE_PIN

Worker-plane result ingest accepts **only** `ExternalResultEnvelope` from the Base SDK. Legacy bodies
without the full binding/proof schema fail closed.

Provider trust (no TEE production scoring gate):

- Operators trust **Lium/Targon** as GPU providers (**PROVIDER_TRUST**). Prism does **not** ship a
  TEE verifier and does **not** fail closed on missing TEE evidence at score finalize.
- **IMAGE_PIN** — set `worker_plane.pinned_image_digest` (`sha256:<64hex>`) so proof claims that match
  the pin receive audit effective tier **1** (maximum). Mismatch is an honest downgrade.
- Paid Lium/Targon provision smoke is **`DEPLOY SMOKE PASS|FAIL` only** (always-terminate pods) and
  independent of emission ranking.
- Safe provider probes (inventory/health) prove reachability only; they do not unlock cryptographic
  provider attestation as a Prism product surface.
- **REAL-PROVIDER TEE** is **retired** for Prism product language. Historical lab tables may still
  show `real_provider_tee=BLOCKED` / non-claims; do not document or enable a TEE production scoring
  gate.
- Residual removed LLM env/keys (`PRISM_LLM_*`, gateway tokens, component-agent knobs) fail closed
  at settings load and never fall through to scored operation.

## Running Locally

```bash
PRISM_SHARED_TOKEN=dev-secret \
PRISM_DATABASE_URL=sqlite+aiosqlite:///./prism.sqlite3 \
.venv/bin/uvicorn prism_challenge.app:app --host 0.0.0.0 --port 8000
```

## BASE Deployment

PRISM registers as a challenge image reached by the master over the internal challenge network on
Compose. Public miner traffic goes through the BASE proxy. Use the master Compose install path
(`deploy/compose/install-master.sh` and related manifests), not Swarm.

> **Atomic rebrand deploy:** internal auth headers use `X-Base-*`
> (`X-Base-Challenge-Slug`, `X-Base-Verified-Hotkey`). PRISM accepts only `X-Base-*`, so cut over all
> services together; any sender left on legacy `X-Platform-*` fails internal auth until the cutover
> completes.

## Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/version
curl -H "Authorization: Bearer dev-secret" -H "X-Base-Challenge-Slug: prism" \
  http://localhost:8000/internal/v1/get_weights
```

## Offline Official Comparison (CPU / fixture)

Operators can rank two unknown-style seed packages under **Prism Official Comparison Protocol v1** without NVIDIA. The dual-family harness packages Transformer `tiny-1m` and pure-torch Mamba, builds comparable official score records from challenge-owned metrics, and prints a clear A-vs-B `compare_official` outcome. Long multi-step GPU pair trains remain **DEFERRED** when the host has no NVIDIA. Honesty labels use PROVIDER_TRUST / LAB-GPU / IMAGE_PIN; **REAL-PROVIDER TEE** is retired product language (historical tables may still say BLOCKED).

Full ranking axioms: [Official Comparison](official-comparison.md).

```bash
# From a Prism checkout (dev deps installed)
export UV_CACHE_DIR=/var/tmp/uv-cache

# 1) Package both families as two-script submit zips (shared outer contract)
uv run python -m prism_challenge.seed_packaging --output-dir dist/seed-packages

# 2) Run the dual-family official compare harness (CPU/fixture, no GPU train)
uv run python -m prism_challenge.evaluator.official_compare_harness \
  --output-dir dist/official-compare \
  --device-class fixture

# 3) Inspect the report
cat dist/official-compare/prism_compare_report.v1.json
```

Targeted unit proof (no NVIDIA):

```bash
uv run pytest tests/test_official_compare_harness.py tests/test_official_comparison_scoring.py -q
```

Report fields of interest:

| Field | Meaning |
| --- | --- |
| `ranking.winner` | `a`, `b`, or `tie` under held-out primary then bpb secondary |
| `ranking.outcome_label` | Human-readable winner label (package family id) |
| `side_a` / `side_b` | Bundle hashes, mean held-out + recomputed bpb, validity |
| `gpu_verification.status` | `DEFERRED` when nvidia-smi/`/dev/nvidia*`/nvidia runtime absent; never claimed PASS by the harness |
| `tee_note` / provider labels | Orthogonal to rank; PROVIDER_TRUST / IMAGE_PIN; REAL-PROVIDER TEE retired (historical BLOCKED OK) |

Do not treat the offline fixture winner as an automatic emission weight crown unless production emission scoring independently agrees (emission is held-out primary + bpb secondary; multimetric / Complete View remain published scientific research grade and do not silently replace emission).

### Multi-metric scorecard v1.1 (operators)

Additive annex on Protocol v1: **`scorecard_id=multimetric.v1.1`**. Full catalogue and polar rules: [Official Comparison §14](official-comparison.md#14-multi-metric-scorecard-annex-v11-scorecard_idmultimetricv11).

| Operator note | Detail |
| --- | --- |
| Default winner | When axes do **not** polar-conflict, keep v1 `compare_official` held-out primary then bpb secondary |
| Polar conflict | If short-gen winner ≠ long-ctx winner beyond ε (or floor veto on one side only), authoritative claim is **`TIE_POLAR`**, `crown_allowed=false` — publish the scorecard vector; no solitary arch crown |
| Multi-seed | Public non-provisional scorecard requires clean **K≥3**; K=1 is provisional only |
| Prior LAB-GPU | Prior short-ctx K=1 mamba heldout lead is **provisional only**; insufficient for architecture superiority |
| Suites | Validity V, short-gen, long-ctx (needle / MQAR / induction-copy / lag-NLL), sample-efficiency, memo, efficiency (VRAM / tok/s / params), stability — mark not-run honestly if a suite is missing |
| TEE / ops (retired product goal) | REAL-PROVIDER TEE stays **BLOCKED** in historical honesty fields only; production path is PROVIDER_TRUST + IMAGE_PIN; always-terminate paid pods; no live Swarm mutate; no `set_weights` from compare |

Inspect scorecard fields when present:

```bash
jq '{protocol_id, scorecard_id, scorecard, ranking, real_provider_tee, honesty_note}' \
  dist/official-compare/prism_compare_report.v1.json
```

### Complete View v1.3 (operators pointer)

MAX A→Z machine dashboard on Protocol v1: **`schema=complete_view.v1.3`**, **`scorecard_id=multimetric.complete.v1.3`**. Historical **`multimetric.complete.v1.2`** and multimetric.v1.1 annex stay valid. Full matrix, multi-axis comparison (per-axis leads including **reasoning**, disagreement matrix, expanded **TIE_POLAR**, **no opaque weighted sole crown**), **P10_reasoning_logic** synthetic probes (not GSM8K/MMLU primary; seed-scale lab comparison only, not human AGI), and non-claims: [Official Comparison §16](official-comparison.md) (plus §15 for v1.2 history).

| Operator note | Detail |
| --- | --- |
| Document | Prefer single machine file `complete_view.v1.3.json` reconciling panels P0–P10 + `comparison` |
| Rank | Multi-axis object only; scientific axis disagreements (including short_gen vs reasoning) → `TIE_POLAR` / `crown_allowed=false` |
| Reasoning panel | `P10_reasoning_logic` closed-acc + forced CE + chance baselines; suite_mean shell until probe suite fills |
| Honesty | Seed-scale synthetic logic is **lab/architecture comparison only**; not human AGI; not emission crown |
| Suites not-run | null + reason; never invent metrics |
| TEE / ops (retired product goal) | REAL-PROVIDER TEE **BLOCKED** in historical honesty only; use PROVIDER_TRUST + IMAGE_PIN; no live Swarm mutate; no `set_weights`; always-terminate paid remesure pods |
| Product module | `prism_challenge.evaluator.complete_view` |

```bash
jq '{schema, scorecard_id, historical_scorecard_id, comparison, real_provider_tee, non_claims, p10: .panels.P10_reasoning_logic.status}' \
  complete_view.v1.3.json
```

## Lab GPU short/long Official Comparison (host rank of remote CUDA)
When real dual-family CUDA trains already completed on a remote GPU host (for example paid Lium under a matched Protocol v1 pin), **rank on any CPU mission host** from the challenge-owned `prism_run_manifest.v2.json` artifacts. This path does **not** require local NVIDIA and is **not** fixture-only synthetic ranking.

| Class | When |
| --- | --- |
| **fixture / CPU** | Synth metrics + seed packaging only; host `gpu_verification.status=DEFERRED` with no local NVIDIA |
| **LAB-GPU** (short or long) | Real CUDA trains produced manifests; host recompute via `official_record_from_manifest` + `compare_official`; report `score_class=LAB-GPU` |
| **PROVIDER_TRUST / IMAGE_PIN** | Production integrity: trusted Lium/Targon + pinned image digest (max tier 1) |
| **REAL-PROVIDER TEE** (retired) | Historical honesty only: **BLOCKED / NOT_CLAIMED**. Lab score success never implements REAL-PROVIDER TEE PASS |

Layout expected under `--lab-gpu-artifacts`:

```text
{artifacts_root}/
  transformer-tiny-1m/seed-1337/prism_run_manifest.v2.json
  mamba-tiny-1m/seed-1337/prism_run_manifest.v2.json
```

```bash
export UV_CACHE_DIR=/var/tmp/uv-cache

# Host rank of real LAB-GPU long-train artifacts (example mission evidence layout)
uv run python -m prism_challenge.evaluator.official_compare_harness \
  --lab-gpu-artifacts /path/to/lium-train/artifacts/out \
  --output-dir dist/official-compare-lab-gpu \
  --seed 1337

# Inspect
jq '{score_class, ranking, real_provider_tee, gpu_verification}' \
  dist/official-compare-lab-gpu/prism_compare_report.v1.json
```

Protocol axioms still hold: held-out **primary**, prequential bpb **secondary**, wall-clock may be recorded but **never ranks**, miner self-report is non-authoritative. Exit code `2` with `BLOCKED:` means dual-family manifests were missing — do not invent scores. No HA-live Swarm mutate and no `set_weights` are required for ranking.

**Provisional honesty for prior K=1 short-ctx LAB-GPU wins:** a single-seed mamba heldout_delta lead under short context is a valid **provisional** lab observation under Protocol v1, **not** multi-seed public architecture superiority. Apply the multimetric.v1.1 scorecard (long-ctx, sample-eff, K≥3, efficiency, polar rules) before any crown language. Label integrity with PROVIDER_TRUST / IMAGE_PIN / LAB-GPU; REAL-PROVIDER TEE remains a retired / historical BLOCKED label only.

### Challenge-owned train series telemetry (operators)

Machine identity: **`schema=prism_train_series.v1`**. Full protocol: [Official Comparison §17](official-comparison.md#17-challenge-owned-train-series-telemetry-prism_train_seriesv1).

| Operator note | Detail |
| --- | --- |
| Authority | **Challenge-owned** series only (online CE/bpb, tokens_seen, wall, **mandatory grad_norm + clip_events** under the telemetry pin). Miner dashboards / self-logs are **non-authoritative** and never certify grade |
| Scientific miner grade | Multi-axis Official Comparison / Complete View (held-out primary + bpb secondary + polar honesty) |
| Emission leaderboard | Held-out primary + bpb secondary — series are **not** emission substitution |
| Rank role of series | Visibility + densify **sample-eff / stability residual** only — **never sole primary rank** over held-out/bpb |
| Fail-closed Official pin | If grading pin sets `require_train_series` and series is missing/empty/corrupt → Official scientific grade **fail-closes** (not silent PASS) |
| APIs | **`GET /v1/submissions/{id}/curve`** (public challenge routes under existing Base proxy / internal auth) returns legacy `loss_curve` **and** optional challenge-owned **`train_series`** (`prism_train_series.v1`) with downsample-safe multichannel points: train CE / running bpb, tokens_seen, wall_s, **grad_norm**, **clip_event**. Miner authority payloads are **never** returned. Chart-safe key projection strips unknown/secret fields. Frontend Architecture Lab already plots loss vs covered bytes from this route; operators may also plot cosine-style time-flow via CLI `jq` on `train_series.points` until UI surfaces `grad_norm` series natively. |
| TEE / ops (retired product goal) | Series realize lab observability only; REAL-PROVIDER TEE historical **BLOCKED**; production uses PROVIDER_TRUST + IMAGE_PIN; no live Swarm mutate; no `set_weights` from telemetry |

```bash
# Operator time-flow: loss + grad_norm vs tokens / wall (same auth as other curve reads)
curl -sS -H "Authorization: Bearer $TOKEN" -H "X-Base-Challenge-Slug: prism" \
  "http://localhost:8000/internal/v1/submissions/${SUBMISSION_ID}/curve" \
  | jq '{
      loss_points: .loss_curve.points,
      schema: .train_series.schema,
      authority: .train_series.authority,
      downsampled: .train_series.downsampled,
      n_total: .train_series.points_total,
      sample: [.train_series.points[] | {i, tokens_seen, wall_s, train_ce_nats, running_bpb, grad_norm, clip_event}][0:8],
      clip_events: .train_series.aggregates.clip_events
    }'

# Artifact side-car (challenge-owned only; miner dashboards non-authoritative)
jq '{schema, authority, miner_reported_ignored, n: (.points|length), grads: [.points[].grad_norm][0:5]}' \
  prism_train_series.v1.json
```

**UI note (VAL-TELE-008):** the public Architecture Lab already wires `GET .../curve` into the loss/bpb chart (`getSubmissionCurve` → loss vs covered bytes). The API now also returns `train_series` for grad_norm / clip time-flow under the same trust path. Operators verify multichannel series via the curl/jq path above when frontend render of grad has not yet recaptured the optional field; **do not** invent a parallel miner-trust chart.

Do not rank packages solely on prettier grad_norm aesthetics. Use multi-axis Official / Complete View for scientific claims; keep emission on the held-out-primary leaderboard path (bpb secondary).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `invalid internal token` | Shared token mismatch between BASE and PRISM |
| submission rejected before container | Static sandbox, two-script contract, param cap, distributed contract, or similarity/anti-cheat |
| residual LLM/gateway env rejected | Removed gateway fields still set; remove them |
| evaluation failed | Broker, image, GPU, timeout, missing locked data, or container error |
| empty weights / no push | No completed scored submissions yet; raw-weight push disabled or master URL missing |
| `result_envelope_invalid` | Body is not a full `ExternalResultEnvelope` |
| TEE elevated tier never grants | Expected: Prism has no TEE verifier; max tier is IMAGE_PIN tier-1 |
| `missing_locked_data` | The read-only FineWeb-Edu train mount is absent or empty on the GPU node |
| Offline compare says GPU DEFERRED | Expected without `nvidia-smi` on the **fixture/CPU** path; use fixture metrics or list CUDA as metadata only |
| LAB-GPU report still fixture | Forgot `--lab-gpu-artifacts`; fixture path is the default |
| `BLOCKED: LAB-GPU host compare` | Missing `{family}/seed-*/prism_run_manifest.v2.json` for one or both families |
| LAB-GPU vs REAL-PROVIDER | Lab GPU score class is scientific only; REAL-PROVIDER TEE is retired product language (historical BLOCKED OK) |
| Prior mamba short-ctx win treated as “Mamba better architecture” | K=1 short-ctx LAB-GPU is provisional; require multimetric.v1.1 scorecard + K≥3; TIE_POLAR if short vs long axes disagree |
| Scorecard long-ctx / sample-eff fields missing | Suite not run yet — mark not-run/BLOCKED; do not invent metrics |
| Official grade missing train series | Pin requires `prism_train_series.v1` and capture empty/miner-only — fail-closed Official grade; do not PASS on miner dashboard |
| Ranking only on grad aesthetics / wall_s | Forbidden — series residual never sole-primary; emission stays held-out primary + bpb secondary; Official multi-axis scientific grade still rule |
| Need K≥3 scale-eval pin + densify APIs | Use `prism_challenge.evaluator.scale_eval` (`scale_p0_protocol_pin`, `densify_complete_view_pair`, `run_scale_multi_family_host_compare`); see official-comparison §11.3 |
| Public multi-seed claim with K=1 | Provisional only — `assert_public_multi_seed_pin` / `OFFICIAL_MIN_PUBLIC_SEEDS=3` rejects public K |
