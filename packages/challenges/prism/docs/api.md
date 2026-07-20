# API Reference

PRISM exposes public challenge routes and internal BASE routes. Wire contracts for external results
use the Base SDK **v3.1.2** `ExternalResultEnvelope` only on the worker-plane ingest path.

## Public Routes

- `GET /health` — challenge health metadata.
- `GET /version` — version, API version, SDK version, and capabilities.
- `GET /v1/submissions/history` — daily submission counts over a window.
- `GET /v1/leaderboard` — submissions ranked by `final_score` for the current epoch
  (earliest-commit-wins on a tie, one entry per hotkey).
- `GET /v1/architectures/{architecture_id}` — one architecture's lab detail (name, owner, best
  score/submission, variant and submission counts, `first_seen_at`, `updated_at`); `404` if absent.
- `GET /v1/architectures/{architecture_id}/variants` — the architecture's training-script variants
  (best first); `404` if the architecture is absent, empty `variants` is valid.
- `GET /v1/submissions/{submission_id}/curve` — the persisted loss curve (`online_loss` +
  `covered_bytes_cumulative`, downsampled to at most 500 points, first and last preserved), the
  prequential bits-per-byte scalars, the reconciled compute profile (`estimated_flops`,
  `gpu_hours`, peak VRAM/RSS, wall-clock), and when challenge capture produced a series,
  a downsample-safe **`train_series`** payload with schema **`prism_train_series.v1`**
  (train CE / running bpb, tokens_seen, wall_s, **grad_norm**, **clip_event** aggregates);
  `train_series` is `null` for legacy rows without instrumented capture; `404` if no curve row.
- `GET /v1/epochs/current` — current epoch id and length.
- `GET /v1/epochs` — recent epochs.
- `GET /v1/health/eval-jobs` — recent eval-job health entries (id, submission id, level, status, attempts).
- `GET /v1/gpu/status` — GPU-lease summary (total GPUs, active leases, by status, by tier).

The former public architecture auto-report route (`GET /v1/architectures/{architecture_id}/report`)
is **removed** with the LLM gateway. Callers receive a normal 404; no gateway/provider call is made.

### `POST /v1/submissions`

Submit a two-script bundle directly to PRISM (miner authentication headers). In production, submissions
usually enter through the BASE proxy and the internal bridge route instead.

```json
{
  "filename": "project.zip",
  "code": "<base64 zip payload>",
  "metadata": {}
}
```

### `GET /v1/submissions/{submission_id}`

```json
{
  "id": "...",
  "hotkey": "...",
  "epoch_id": 123,
  "status": "completed",
  "final_score": 0.72,
  "anti_cheat_multiplier": 1.0
}
```

`final_score` is the challenge-computed emission rank (held-out / generalization primary; a larger
honest held-out delta yields a higher `final_score`, with prequential bpb secondary); `q_arch`,
`q_recipe`, `diversity_bonus`, and `penalty` are legacy fields retained for
response-schema stability. `status` is `pending`, `running`, `completed`, `failed`, or `rejected`
(rejected = failed a static gate, the two-script contract, deterministic similarity/anti-cheat, or
other admission gates). There is **no** live `held` status after gateway removal.

### `GET /v1/architectures`

Architecture-lab leaderboard grouped by architecture family, ranked by `best_final_score` descending.
Optional `epoch_id` scopes to architectures with a completed submission in that epoch; omitting it
resolves to the most-recent non-empty epoch. `name` is the miner-declared, deterministically moderated
architecture name (may be `null`).

```json
{
  "epoch_id": 42,
  "architectures": [
    {
      "rank": 1,
      "architecture_id": "...",
      "arch_hash": "...",
      "name": "Rotary MoE v3",
      "owner_hotkey": "...",
      "best_final_score": 1.2345,
      "best_submission_id": "...",
      "variant_count": 3,
      "submission_count": 7,
      "updated_at": "..."
    }
  ]
}
```

## Internal BASE Routes

All internal routes require `Authorization: Bearer <shared-token>` unless noted.

### `GET /internal/v1/get_weights`

Standard BASE challenge contract. Returns normalized hotkey weights (one per hotkey, from that
hotkey's best `final_score`) for inventory/compatibility. Live emissions use authenticated
**raw-weight push** to the BASE master; validators submit the master-aggregated vector on-chain.
PRISM never calls `set_weights`.

### `POST /internal/v1/bridge/submissions`

Receives BASE-verified submissions.

```text
Authorization: Bearer <shared-token>
X-Base-Verified-Hotkey: <hotkey>
X-Submission-Filename: project.zip
Content-Type: application/zip
```

The body can be raw ZIP bytes or JSON matching `SubmissionCreate`.

### `POST /internal/v1/worker/process-next`

Claims and processes one pending submission through the full pipeline: static gates, deterministic
admission (similarity/anti-cheat), forced-init re-execution (when configured), and prequential
bits-per-byte scoring. In **combined mode**, the API process drains this queue in-process via the
background worker loop.

### `GET /internal/v1/work_units`

Lists pending challenge work units for the master coordination plane (execution-free enumeration).
When the worker plane is enabled, also lists pending audit units.

### `POST /internal/v1/work_units/result`

Accepts a base-reconciled worker result as **`ExternalResultEnvelope` only** (Base SDK schema). The
body must include `api_version`, assignment/challenge bindings, and an execution proof. Dual/legacy
reduced bodies without those fields are rejected with **422** before scoring or persistence. Enabled
only when the worker plane is on; otherwise **404**.

Proof envelopes may carry optional claim fields; Prism does **not** run a TEE verifier on them.
**IMAGE_PIN** (`worker_plane.pinned_image_digest`) is the product elevation path (max effective tier
**1**). Score finalization is never blocked for missing TEE evidence. Provider trust is Lium/Targon
plus ordinary signature/plausibility gates on the worker plane.

### Other internal routes

- `POST /internal/v1/checkpoints` — validator-signed checkpoint publish (permit-gated).
- `POST /internal/v1/audit_units/{audit_unit_id}/result` — audit-replay resolution when the worker
  plane is enabled.
