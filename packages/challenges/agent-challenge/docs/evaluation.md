> **API truth is OpenAPI** (`https://chain.joinbase.ai/challenges/agent-challenge/openapi.json`, `/docs`).
> Day-1 miners: repo-root [`docs/miner/getting-started.md`](../../../../docs/miner/getting-started.md).
> This page is a short product pin note, not a route dump.

# Evaluation

This page covers the submission lifecycle, public status vocabulary, scoring, and how production
self-deploy relates to acceptance and weights.

## Production path (mandatory)

Production scoring requires:

1. `phala_attestation_enabled` (`CHALLENGE_PHALA_ATTESTATION_ENABLED`) **ON**
2. `attested_review_enabled` **ON**
3. Miner-driven Phala Cloud CPU TDX **review** CVM, then (only after fresh re-verified allow) **eval** CVM
4. Attestation-only grading: measured OpenRouter under review `.rules`; **no** Base LLM gateway on the scored path
5. Direct `POST /evaluation/v1/runs/{eval_run_id}/result` with score-domain attestation and durable key-grant

### Agent-driven order (package verify → tree SHA → TEE → eval)

Evaluation is **agent-driven**. Before any TEE-authorized eval or score attestation:

1. **Verify the package** with measured **agent LLM rules** residual under harness / `.rules`.
2. **Prove the folder** with canonical **`package_tree_sha`** (content-addressed tree SHA of the
   extracted package), durable next to the ZIP digest and bound into plan / review materials.
3. **TEE authorization** only after residual **allow** + matching `package_tree_sha` are bound into
   fresh review / quote materials. Host-static analyzer alone is **insufficient**.
4. **Only then** eval prepare / deploy / key-release / trials / score attestation.

Without residual + tree SHA proof chain: **no eval prepare, no KR grant, no score attestation**.
Agent models: **no closed catalog**; ban personal / custom finetunes only (review judge pin stays
separate). Guest recomputes `package_tree_sha` before scored trials and refuse on mismatch.

Validators do **not** deploy scored jobs for miners in production. Work-unit pull /
`list_pending_work_units` style execution is legacy relative to the attested self-deploy path.

See [architecture](architecture.md), [miner self-deploy](miner/self-deploy.md), and
[Attestation TEE](miner/attestation-tee.md).

## High-level lifecycle

1. Miner signs and uploads an immutable ZIP (`POST /submissions`).
2. Digest becomes the stable agent hash; service also records **`package_tree_sha`** (canonical
   folder-tree SHA of the extracted package) as durable package proof next to the ZIP digest.
   AST and similarity analysis may run as service gates (host-static alone never TEE-authorizes eval).
3. Attested **review** session: miner prepare/deploy measured review image; CVM produces a
   domain-separated review report (OpenRouter under harness / `.rules`) and a measured
   **LLM rules residual** for the package.
4. Host binds residual verdict + rules digests + `package_tree_sha` into durable review outcome
   materials. Validator re-verifies quote + review allowlist + review-domain `report_data`
   (bound times ≤24h) **with residual + tree SHA present**.
5. Verdict outcomes:
   - `allow` + residual allow + matching tree SHA unlocks eval prepare only while fresh
     re-verify materials still admit
   - residual fail / tree SHA mismatch ends eval eligibility (fail-closed)
   - `reject` ends eval eligibility
   - `escalate` pauses for signed owner review
6. Miner **eval** prepare/deploy on the separate measured **canonical** image. Prepare admission
   refuses cache-only DB `review_allowed` bits, missing residual, or missing/mismatched
   `package_tree_sha` without re-running bound-outcome checks.
7. Eval guest obtains dstack **GetTlsKey** client certs, then obtains golden AES-256 key only over
   raw RA-TLS key-release (domain-separated, allowlist + mTLS). KR also requires the package
   proof chain.
8. Guest **rechecks `package_tree_sha`** against the plan before trials; trials run from baked
   live-task-cache; CVM emits attested result; challenge receipts body then verifies.
9. Accepted results may contribute to leaderboard and BASE raw weights.

Public clients poll `GET /submissions/{id}/status` or SSE `GET /submissions/{id}/events`.

## Prepare / deploy / key-release / score gate

| Stage | What must hold | Fail closed when |
| --- | --- | --- |
| Review prepare | Signed assignment; immutable ZIP digests + durable `package_tree_sha` | Signature / rate / capability failures |
| Review deploy | Review allowlist compose_hash; Phala `encrypted_env` for OpenRouter + session | Missing encrypted_env, GPU shape, money cap |
| Review result | Review-domain quote + allowlist + bound times + **LLM rules residual** + tree SHA bind | Stale >24h, wrong domain, measurement mismatch, residual missing/reject |
| Eval prepare | Fresh re-verified allow + residual allow + plan `package_tree_sha` (not DB phase alone) | `review_allow_required`, stale allow, `package_residual_missing`, tree SHA absent/mismatch, cached-allow-only refuse |
| Eval deploy | Canonical compose_hash + measurement; plan nonces | Plan/compose mismatch, OS pin drift |
| Key release | Raw TCP TLS 1.3 + client cert + keyrelease-domain quote + allowlist + package proof chain | Deny returns no key; no L7 `/release` production path |
| Guest trials | Guest recomputes `package_tree_sha` matches plan | Tree SHA mismatch → no scored trials |
| Score admit | Score-domain quote + event log + allowlist + durable key-grant + nonces + residual/tree proof | Missing key-grant, residual, tree SHA, or attestation materials write **no** score |

## Public phases (attested mode)

Exact public strings can evolve with the service, but the conceptual map is:

| Concern | Phases / outcomes (illustrative) |
| --- | --- |
| Review | `review_queued`, `review_cvm_running`, `review_provider_standby`, `review_verifying`, `review_allowed`, `review_rejected`, `review_escalated`, `review_expired`, `review_cancelled`, `review_error` |
| Eval | `eval_prepared`, `eval_running`, `eval_verifying`, `eval_accepted`, `eval_rejected`, `eval_expired`, `eval_cancelled`, `eval_error` |
| Pre-receipt failures | `eval_deploy_failed`, `eval_tunnel_failed`, `eval_key_release_unavailable`, `eval_no_result` |
| Terminal public labels | `valid`, `invalid`, `suspicious`, `error` (and owner override forms where configured) |

Review and eval history routes use stable cursor pagination (default 10, max 16) and retain cancelled,
expired, failed, superseding attempts. Safe fields only: digests, phases, reason codes, timestamps.

## Scoring

- Each selected task contributes a task score. Defaults and caps are configured by the challenge
  (`evaluation_task_count`, concurrency helpers; task selection is deterministic for a given agent
  hash when plans are fixed by eval prepare).
- Aggregate score is typically the average across selected tasks for a completed valid submission.
- Defaults for raw BASE weights are **winner-take-all** among valid submissions when
  `weights_winner_take_all` is on (default); off falls back to best score per miner hotkey.
- Only effective status `valid` or `overridden_valid` can produce weight entries and leaderboard rows.
- Timed-out tasks are terminal, non-passing, score 0, counted once.
- On the attested path, weight eligibility requires **verified** attestation acceptance including
  durable key-grant, not merely an unauthenticated number claim.

`/internal/v1/get_weights` is the challenge weight contract; BASE normalizes to UIDs (cross-repo).

## Acceptance checks (eval result)

Before writing an accepted score the challenge verifies, in conjunction:

- TDX quote integrity and acceptable TCB
- Event log replay / compose identity
- Measurement present on the **eval** validator allowlist
- Score-domain `report_data` binds measurement, agent hash, task ids, scores digest, score nonce
  and eval_run_id as specified in the wire schema
- **Key-grant consistency** for that eval run (no grant, no score)
- Nonces single-use / fresh; conflicting body digests conflict; verifier unavailable is retryable
  without treating score nonce as successfully consumed
- Production path also required a fresh re-verified review allow at eval admit time

Invalid or rejected results write no accepted score.

## Signed miner requests

Miner writes use:

```http
X-Hotkey: <ss58-hotkey>
X-Signature: <signature>
X-Nonce: <unique-nonce>
X-Timestamp: <timestamp>
```

Canonical string:

```text
{METHOD}
{PATH_WITH_SORTED_QUERY}
{X-TIMESTAMP}
{X-NONCE}
{SHA256_HEX_OF_RAW_BODY}
```

Timestamp skew tolerance defaults to 300 seconds. Reused `(hotkey, nonce)` returns HTTP 409.
ZIP compressed size limit is 1048576 bytes (1 MiB); larger returns `413` `zip_too_large`.

## Offline and flag-off

With attestation flags **OFF**, the service may still run offline AST helpers and historical
evaluation helpers for local CI and compatibility. That mode:

- must not be described as production scoring
- does not require miners to spend Phala credits
- keeps validators as operators of the challenge service, not as substitutes for miner TEE self-deploy

Details for flags: [operator self-deploy](validator/self-deploy.md).

## Guest task-cache bake (terminal-bench prepare)

Eval CVMs resolve Terminal-Bench task trees from the **measured** guest path
`/opt/agent-challenge/task-cache` (no network at eval time). The canonical image
bakes that root via:

```dockerfile
COPY docker/canonical/live-task-cache/ /opt/agent-challenge/task-cache/
```

Populate from the pinned local harbor package cache (Task-1 acquisition;
digest-gated against `golden/dataset-digest.json`):

```bash
uv run python scripts/populate_live_task_cache.py           # full digest set
uv run python scripts/populate_live_task_cache.py --fallback-only  # FALLBACK subset only
```

Prepare/select draws from `TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS` when residual count paths use that
fallback list, so the baked cache **must** include at least those bare dirs. Incomplete bake surfaces
as guest `TaskDefNotFoundError` at `preflight_tasks` (before key-release). Fallback IDs are a subset
of the frozen 89-task digest set. Custom local cache paths remain digest-gated; there is no miner
`task_url` / git field. Plan `selected_tasks` is produced only by validator `eval/prepare`.

Contract tests: `tests/test_live_task_cache_prepare_complete.py`,
`tests/test_tbench_integrity_policy.py`.

See also [Security](security.md) for `allow_internet` product policy (retain task-authored default;
answer hardcoding remains cheat; harness pins required).

## Related

- [Architecture](architecture.md)
- [Miner self-deploy](miner/self-deploy.md)
- [Attestation TEE](miner/attestation-tee.md)
- [Frontend API contract](frontend-api-contract.md)
