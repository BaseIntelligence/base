# Validator / Operator Guide

## Purpose

Agent Challenge lets validators and operators run the challenge trust root for a software engineering
agent benchmark. In production, **miners self-deploy Phala Intel TDX CVMs** for review and eval.
Validators publish measurements, operate RA-TLS golden key release, verify quotes, keep production
attestation flags ON, and expose accepted scores as BASE weights. They are **not** production
scored-job deployers for miners.

See [Operator self-deploy](self-deploy.md) for allowlist, KR 8701, and flags.

## Production scored path (attestation)

When `phala_attestation_enabled` and `attested_review_enabled` are **both ON**, the scored path is
**miner self-deploy** on Phala. Evaluation is **agent-driven** in a fixed order:

1. **Verify the package** with measured **agent LLM rules** residual under harness / `.rules`.
2. **Prove the folder** with canonical **`package_tree_sha`** (tree content-addressed SHA).
3. **TEE authorization** only after residual allow + tree SHA are bound into fresh review materials
   (host-static analyzer alone is **not** enough).
4. **Only then** attested eval with GetTlsKey + raw RA-TLS golden key release, and direct RESULT
   admission with durable key-grant (guest rechecks `package_tree_sha` before trials).

Without residual + tree SHA proof: **no eval prepare, no KR grant, no score attestation**.
Agent models: **no closed catalog**; ban personal finetunes only. Dual measurement allowlists pin
review vs canonical images. Details: [Operator self-deploy](self-deploy.md),
[Architecture](../architecture.md), [Evaluation](../evaluation.md),
[Attestation TEE (agent-driven order)](../miner/attestation-tee.md#agent-driven-order-package-verify--tree-sha--tee--eval).

Historical sections below document shared surfaces (signing, public status, owner routes,
BASE weight contract) and offline / compatibility Terminal-Bench helpers. **Do not** treat
validator broker `own_runner` job deployment or Base master LLM gateway review as the
production scored path under dual-flag attestation.

## Responsibilities

Validators and operators are responsible for:

- publishing the active benchmark configuration and measurement allowlist;
- accepting valid miner artifacts and verifying signed surfaces;
- operating RA-TLS key release (port 8701) and quote verification in production;
- keeping `phala_attestation_enabled` and `attested_review_enabled` **ON** in production;
- protecting the shared BASE token and golden key materials;
- monitoring task failures, timeouts, and queue health;
- keeping persisted results available for audit;
- exposing completed and verified scores as BASE weights;
- **not** funding or operating miner production CVMs as the scored path.

## Submitted Agent Runtime Policy

Miner artifacts must be based on [`BaseIntelligence/baseagent`](https://github.com/BaseIntelligence/baseagent).
**Base LLM gateway is forbidden** (`BASE_LLM_GATEWAY_URL`, `BASE_GATEWAY_TOKEN`, `/llm/v1`).
Legal LLM paths: measured OpenRouter inside review/eval CVMs with digests under `.rules`, and/or
tools-only agents. Submitted agents MUST NOT restore Base gateway clients or embed non-measured
provider secrets / emission model pins.

Continuous static analysis flags residual Base gateway clients (`base_gateway_forbidden`),
non-measured provider embeds (`unauthorized_llm_provider`), or hardcoded emission models
(`hardcoded_llm_model`); flagged artifacts should be rejected or escalated before scoring.

`validator_role` is a legacy, inert setting. It is still accepted (default `normal`) for backward
compatibility, but it no longer gates any behavior: setting `CHALLENGE_VALIDATOR_ROLE=master`,
`normal`, or leaving it unset all behave identically. Every eligible online validator accepts and
persists signed submissions, creates queued jobs, claims work, runs analyzers, and publishes
effective scores. Execution is driven by the validator worker pull loop that claims queued jobs, not
by a master/normal role or a centralized launch bridge.

## Evaluation Lifecycle

1. A miner submits a signed immutable ZIP artifact and hotkey.
2. The API verifies the signature, ZIP safety, ZIP digest, and 1-per-3h hotkey rate limit.
3. The challenge stores the artifact digest as the stable agent hash and records durable status events.
4. A validator moves through explicit raw statuses: `analysis_queued -> ast_running -> llm_running -> analysis_allowed -> waiting_miner_env -> tb_queued -> tb_running`.
5. AST review extracts Python features and same-challenge similarity. LLM review asks the central-gate reviewer, which routes through the master LLM gateway (the gateway selects the model), for a final verdict when configured. Public copy distinguishes `AST review`, `LLM review`, `LLM standby`, `Waiting environments`, `evaluation queued`, and `evaluating`.
6. A missing LLM gateway token, provider unavailable, rate limit, and timeout move to `llm_standby` with sanitized reason codes. Standby retries through `llm_standby -> analysis_queued` when gateway config becomes available and does not create `LlmVerdict`, `EvaluationJob`, `AdminReviewDecision`, or weights.
7. LLM `allow` records `analysis_allowed`, then moves to `waiting_miner_env`. Env-ready submissions lock and enqueue exactly once; env-missing submissions show public `Waiting environments`. `reject` ends as public invalid, and `escalate` pauses for signed owner review.
8. Terminal-Bench attempts run through `own_runner`, the only supported execution backend, which
   runs the runner image's native Docker environment inside a privileged Docker-in-Docker runner
   and persists stable job directories plus provider-neutral trial refs.
9. The recovery reconciler restores progress after process restarts, finalizes completed job dirs, and
   applies retry/final policy for missing execution state.
10. BASE reads the best completed score per miner hotkey after effective-status filtering.

## Benchmark Backends

### SWE-Forge Style Tasks

SWE-Forge tasks evaluate whether an agent can repair repositories. Each task provides a prepared
workspace, a task-specific evaluator, and a pass or fail outcome.

Key settings:

| Setting | Purpose |
| --- | --- |
| `CHALLENGE_BENCHMARK_BACKEND=swe_forge` | Selects repository-repair evaluation. |
| `CHALLENGE_SWE_FORGE_TREE_URL` | Dataset tree used to discover available tasks. |
| `CHALLENGE_SWE_FORGE_IMAGE_PREFIX` | Image prefix for task environments. |
| `CHALLENGE_EVALUATION_TASK_COUNT` | Number of tasks selected per agent, default 4 and maximum 4. |

### Terminal-Bench Style Tasks

Terminal-Bench tasks evaluate agents through Harbor-compatible terminal environments. This mode is
useful for broader command-line and environment-interaction benchmarks. The production dataset is
`terminal-bench/terminal-bench-2-1`; `terminal-bench@2.1` is the mandatory display and legacy label
for operator and public metadata. Do not use earlier Terminal-Bench 2.x labels.

Key settings:

| Setting | Purpose |
| --- | --- |
| `CHALLENGE_BENCHMARK_BACKEND=terminal_bench` | Selects terminal benchmark evaluation. |
| `CHALLENGE_TERMINAL_BENCH_DATASET` | Harbor dataset identifier, `terminal-bench/terminal-bench-2-1` in production. |
| `CHALLENGE_TERMINAL_BENCH_LABEL` | Mandatory display and legacy label, `terminal-bench@2.1`. |
| `CHALLENGE_TERMINAL_BENCH_TASK_IDS` | Optional explicit task IDs. |
| `CHALLENGE_TERMINAL_BENCH_SHARDS` | Number of generated shards when explicit IDs are not used. |
| `CHALLENGE_TERMINAL_BENCH_TASKS_PER_SHARD` | Number of tasks per generated shard. |
| `CHALLENGE_HARBOR_AGENT_IMPORT_PATH` | Import path for submitted agents. Production default is `agent:Agent`; submitted ZIPs must include root `agent.py` with top-level `class Agent`. |
| `CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND=own_runner` | Selects the only supported execution backend, `own_runner`. |
| `CHALLENGE_HARBOR_RUNNER_IMAGE` | Prebuilt runner image used by own_runner, `ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1`. |
| `CHALLENGE_HARBOR_FORWARD_ENV_VARS` | Empty by default; explicit opt-in list for provider credentials when a benchmark requires them. |
| `CHALLENGE_HARBOR_N_CONCURRENT` | Harbor per-task concurrency inside a run, separate from selected task count and validator runtime concurrency. |

Production Terminal-Bench mode is `own_runner`. Use `CHALLENGE_DOCKER_ENABLED=true`,
`CHALLENGE_DOCKER_BACKEND=broker`, `CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND=own_runner`,
`CHALLENGE_DOCKER_BROKER_URL`, `CHALLENGE_DOCKER_BROKER_TOKEN_FILE=/run/secrets/base/docker_broker_token`,
`CHALLENGE_HARBOR_RUNNER_IMAGE=ghcr.io/baseintelligence/terminal-bench-harbor-runner:2.1`, and
`CHALLENGE_DOCKER_NETWORK=default`. The privileged Docker-in-Docker runner requires a writable root
filesystem, so leave `CHALLENGE_DOCKER_READ_ONLY` unset for the own_runner path. Prefer the token file over
`CHALLENGE_DOCKER_BROKER_TOKEN`; do not
paste raw broker tokens into shell commands, docs, screenshots, or support logs. The production
allowlist should scope Terminal-Bench to `ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner:latest`
rather than a broad `ghcr.io/`, `baseintelligence/`, or `python:` pattern.

The own_runner backend runs the runner image's prebuilt Harbor tooling against its native Docker
environment inside a privileged Docker-in-Docker container, and it does not require a Harbor fork or
any runtime Harbor install. Harbor provider credentials are not forwarded by default; only
set `CHALLENGE_HARBOR_FORWARD_ENV_VARS` after accepting the risk for a specific benchmark provider.

In BASE registry metadata, set `required_capabilities=["get_weights",
"proxy_routes", "docker_executor"]` so BASE injects the broker URL and the broker token file at
`/run/secrets/base/docker_broker_token`. The broker uses the controlled runner image and token file;
production does not run `pip install harbor` or any other runtime Harbor install path.

There is no runtime Harbor install path. own_runner always uses the prebuilt runner image, and the
former local Docker CLI runtime-install override is no longer accepted by the production broker path.

## Runtime Configuration

All runtime settings use the `CHALLENGE_` environment prefix.

| Setting | Purpose |
| --- | --- |
| `CHALLENGE_SLUG` | Challenge identifier; defaults to `agent-challenge`. |
| `CHALLENGE_NAME` | Human-readable challenge name. |
| `CHALLENGE_DATABASE_URL` | Persistent result storage. |
| `CHALLENGE_DATA_DIR` | Base data directory. |
| `CHALLENGE_ARTIFACT_ROOT` | Trusted root for mounted agent artifacts. |
| `CHALLENGE_SHARED_TOKEN` | Shared token for BASE internal calls. |
| `CHALLENGE_SHARED_TOKEN_FILE` | File containing the BASE shared token. |
| `CHALLENGE_LLM_GATEWAY_BASE_URL` | Master LLM gateway base URL for the central-gate reviewer. |
| `CHALLENGE_LLM_GATEWAY_TOKEN` | Scoped central-gate gateway token; when missing, LLM review enters retryable standby. |
| `CHALLENGE_LLM_GATEWAY_TOKEN_FILE` | File containing the gateway token, e.g. a mounted Docker secret at `/run/secrets/base_gateway_token`. |
| `CHALLENGE_DOCKER_ENABLED` | Allows the configured master-validator execution path to run Docker-backed task environments. |
| `CHALLENGE_DOCKER_BACKEND` | Local executor or BASE broker mode. |
| `CHALLENGE_DOCKER_BROKER_URL` | BASE broker URL when broker mode is used. |
| `CHALLENGE_DOCKER_BROKER_TOKEN` | Broker token. |
| `CHALLENGE_DOCKER_BROKER_TOKEN_FILE` | File containing the broker token. BASE mounts it at `/run/secrets/base/docker_broker_token`. |
| `CHALLENGE_DOCKER_ALLOWED_IMAGES` | Allowed task environment images; production must allow the own_runner runner image and avoid broad prefixes. |
| `CHALLENGE_EVALUATION_TIMEOUT_SECONDS` | Per-task timeout. |
| `CHALLENGE_EVALUATION_CONCURRENCY` | Number of tasks evaluated in parallel per submitted agent, default 4 and maximum 4. |
| `CHALLENGE_EVALUATION_LOG_LIMIT_BYTES` | Stored log size cap per task. |

Default security and execution limits:

| Setting | Default |
| --- | --- |
| `CHALLENGE_OWNER_HOTKEY` | `5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At` |
| `CHALLENGE_SIGNING_TTL_SECONDS` | `300` |
| `CHALLENGE_ZIP_MAX_BYTES` | `1048576` |
| `CHALLENGE_DOCKER_CPUS` | `4.0` |
| `CHALLENGE_DOCKER_MEMORY` | `8g` |
| `CHALLENGE_EVALUATION_TIMEOUT_SECONDS` | `3600` |
| `CHALLENGE_DOCKER_NETWORK` | `none` |
| `CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS` | `10800` |
| `CHALLENGE_SSE_HEARTBEAT_SECONDS` | `15` |
| `CHALLENGE_LLM_REVIEWER_TIMEOUT_SECONDS` | `120` |
| `CHALLENGE_LLM_REVIEWER_MAX_ATTEMPTS` | `3` |
| `CHALLENGE_ANALYZER_SIMILARITY_HIGH_RISK_THRESHOLD` | `90.0` |
| `CHALLENGE_ANALYZER_SIMILARITY_MEDIUM_RISK_THRESHOLD` | `70.0` |

The ZIP limit is checked against compressed archive size. `1048576` bytes is treated as 1MB, and an
oversized archive returns HTTP `413` with `detail.code="zip_too_large"`. Analyzer runs use strict
container defaults of `cpus=4.0`, `memory=8g`, `timeout_seconds=3600`, and `network=none`.

LLM gateway token, broker, BASE shared-token, and database secrets must come from environment
variables or Docker secrets. Safe config rendering redacts those values, and operators must not put
actual API keys, bearer tokens, mnemonics, wallet material, or database credentials in config files,
logs, status events, or public documentation. A missing LLM gateway token, provider timeout,
provider rate-limit, and provider unavailable are visible as retryable `LLM standby` with sanitized
reason codes. They are not rejection, escalation, or evaluation. LLM reviewer retries also include
missing tool-call and malformed verdict failures; unsafe paths, disallowed tools, and non-final verdict
tool calls are excluded from retry policy.

Analyzer policy comes from the repository `.rules` directory. Missing `.rules` returns `error`.
Hardcoding detection is evidence-based, bounded, owner-auditable, and not proof that hardcoding is
absent. The static analyzer also flags unauthorized submitted-agent LLM access so continuous review
can reject artifacts that embed a provider API key or base URL (`unauthorized_llm_provider`) or
hardcode a model name (`hardcoded_llm_model`) early.


## Central Gateway LLM Reviewer

The central AST + LLM gate routes all LLM calls through the master LLM gateway using a scoped
central-gate token. The gateway injects the provider key + model server-side, so the challenge holds
no provider key and pins no model. When the gateway token is missing or the provider is unavailable,
rate-limited, or timed out, submissions enter visible retryable `LLM standby` with sanitized reason
codes.

Environment variable setup:

```bash
export CHALLENGE_LLM_GATEWAY_BASE_URL='https://<master-gateway-host>'
export CHALLENGE_LLM_GATEWAY_TOKEN='<gateway-token-write-only>'
```

Docker secret file setup (the installer mounts the central-gate token on the challenge service at
`/run/secrets/base_gateway_token`):

```bash
export CHALLENGE_LLM_GATEWAY_BASE_URL='https://<master-gateway-host>'
export CHALLENGE_LLM_GATEWAY_TOKEN_FILE='/run/secrets/base_gateway_token'
```

Redaction policy:

- `safe_model_dump()` redacts gateway tokens, broker tokens, shared tokens, and database URLs.
- Public status, SSE, logs, and docs must never contain real API keys, bearer tokens, mnemonics, wallet material, private endpoints, raw provider transcripts, or live database URLs.
- A missing gateway token, provider timeout, rate limit, and provider unavailable become retryable `LLM standby` with sanitized reason codes, not `LlmVerdict`, `EvaluationJob`, `AdminReviewDecision`, or weights.
- Standby retries through `llm_standby -> analysis_queued` when gateway config becomes available.
- Retry policy also covers missing tool call and malformed verdict failures.
- Unsafe paths, disallowed tools, and non-final verdict tool calls are not retried.

## Signed Request Contract

Public miner submissions and owner controls use the same signed request envelope. Clients must send
these exact headers:

```http
X-Hotkey: <ss58-hotkey>
X-Signature: <signature>
X-Nonce: <unique-nonce>
X-Timestamp: <timestamp>
```

The canonical string is exactly:

```text
{METHOD}
{PATH_WITH_SORTED_QUERY}
{X-TIMESTAMP}
{X-NONCE}
{SHA256_HEX_OF_RAW_BODY}
```

`PATH_WITH_SORTED_QUERY` includes the path and query string sorted by key and value. The body digest
is the SHA-256 hex digest of the raw request body bytes. Requests allow `300` seconds of timestamp
skew. Replay protection stores `(hotkey, nonce)` pairs, and a reused pair returns HTTP `409`.

Owner controls require the owner hotkey exactly:

```text
5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At
```

## Public Miner Surface

Miners and dashboards use:

```http
GET /benchmarks
GET /benchmarks/tasks
POST /submissions
GET /submissions
GET /submissions/count
GET /submissions/{submission_id}
GET /submissions/{submission_id}/versions
GET /submissions/{submission_id}/status
GET /submissions/{submission_id}/events
GET /submissions/{submission_id}/env
PUT /submissions/{submission_id}/env
POST /submissions/{submission_id}/env/confirm-empty
POST /submissions/{submission_id}/launch
GET /submissions/{submission_id}/task-events
GET /submissions/{submission_id}/task-events/stream
GET /agents/{agent_hash}/evaluation
GET /leaderboard
```


Through BASE, the canonical frontend read base is `/challenges/agent-challenge/...`. The BASE page can also read `/v1/registry` for hero metadata. Frontend examples include `GET /challenges/agent-challenge/benchmarks`, `GET /challenges/agent-challenge/submissions/{id}/status`, `GET /challenges/agent-challenge/submissions/{id}/events`, `GET/PUT /challenges/agent-challenge/submissions/{id}/env`, `POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty`, `POST /challenges/agent-challenge/submissions/{id}/launch`, and `GET /challenges/agent-challenge/leaderboard`. Raw ZIP uploads use `POST /v1/challenges/agent-challenge/submissions`; JSON base64 uploads use `POST /challenges/agent-challenge/submissions` and sign the challenge-local `/submissions` path. Env and launch routes sign the challenge-local env or launch path. `/challenges/agent-challenge/submissions` returns the latest 100 submissions newest-first, and `/challenges/agent-challenge/leaderboard` returns one best scoring row per hotkey. Pagination, filtering, and client-selected sorting are deferred to future v2. BASE blocks `/internal/*`, `/health`, and `/version` from the public proxy. BASE registry and proxy do not store per-submission env values.

`POST /submissions` stores the signed immutable artifact and metadata and, once analysis allows it,
creates one queued job for the immutable artifact. `validator_role` is inert and does not change this
behavior; every eligible online validator enqueues, claims, runs, and evaluates work.

Public status responses expose bounded latest evaluation summaries, effective status, ZIP SHA, and
timestamps. They do not expose logs, analyzer report JSON, signatures, raw status, reason-code
internals, own_runner provider refs, job directories, Swarm service or task names, broker tokens, or
raw execution refs.

Public version fields available to frontend reads where applicable are `family_id`, `display_name`,
`version_number`, `version_label`, `version_count`, `latest_submission_id`, and `is_latest_version`.
The public `family_id` is the family public identifier, not the raw `submission_family_id` database key.


## Submission Operations

### Signed Upload And Receipt Verification

`POST /submissions` is the only public upload path. The signed body should include exactly one artifact
source and a miner hotkey that matches the signed identity. A successful response is the operator and
miner receipt: compare the returned `submission_id`, `agent_hash`, `zip_sha256`, `zip_size_bytes`, and
`status` with the local ZIP digest before announcing the artifact accepted. The server ignores any
client naming attempt through `agent_hash`; it stores the artifact digest as `agent_hash` and canonical
artifact identity. Reused signed nonces return HTTP `409`; a second accepted submission for the same hotkey inside
`CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS=10800` returns HTTP `429` with
`detail.code="submission_rate_limited"` and a `next_allowed_at` timestamp.

Global submission name and version rules:

- The first successful submitter owns the normalized name globally within Agent Challenge.
- Later accepted submissions from the same owner and normalized name become the next family version.
- Version labels are exact integer labels: `v1`, `v2`, `v3`, and so on.
- Name ownership conflicts return HTTP `409` with `detail.code="name_taken"`.
- Duplicate artifact or code hash conflicts return HTTP `409` with `detail.code="duplicate_code_hash"`.
- Duplicate artifact or code hashes are rejected globally, regardless of name or miner.
- Duplicate hash checks take precedence over name ownership checks.

```bash
curl -sS -X POST "https://<challenge-host>/submissions" \
  -H "Content-Type: application/json" \
  -H "X-Hotkey: <miner-hotkey>" \
  -H "X-Signature: <signature-over-canonical-request>" \
  -H "X-Nonce: <unique-nonce>" \
  -H "X-Timestamp: <iso8601-or-unix-timestamp>" \
  --data '{"miner_hotkey":"<miner-hotkey>","name":"example-agent","artifact_zip_base64":"<base64-zip>"}'
```

### Analyzer Gate

Analyzer evidence is durable and source-safe. Python AST extraction reads only manifest-listed text files
from the immutable ZIP. Same-challenge AST similarity stores scores, risk bands, and source-free file
pair evidence. The central-gate reviewer routes through the master LLM gateway (the gateway selects
the model) and must end with one of three verdicts:

| Verdict | Effect |
| --- | --- |
| `allow` | Records `analysis_allowed`, moves to `waiting_miner_env`, and exposes `Waiting environments` until env rows exist or the miner confirms empty. Env-ready submissions enqueue exactly once. |
| `reject` | Records `analysis_rejected`; public status becomes `invalid`; no Terminal-Bench job is created. |
| `escalate` | Records `analysis_escalated` and `admin_paused`; owner review is required. |

If the LLM gateway is not configured or is temporarily unavailable, review enters visible retryable standby.
Operators should not treat missing LLM evidence as an allow, reject, escalation, or evaluation decision.
Configure `CHALLENGE_LLM_GATEWAY_BASE_URL` plus either `CHALLENGE_LLM_GATEWAY_TOKEN` or
`CHALLENGE_LLM_GATEWAY_TOKEN_FILE=/run/secrets/base_gateway_token`. Safe config dumps redact the
gateway token, broker token, shared token, and database URL. Never place real keys, bearer tokens, mnemonics,
wallet material, or DB URLs in docs, logs, status metadata, or example commands.

### Polling And SSE Status

Use `GET /submissions/{submission_id}/status` for polling and
`GET /submissions/{submission_id}/events` for SSE. Both are public proxy routes. Status and SSE expose
these exact non-terminal mappings: `analysis_queued` to `queued` and phase `queued`, `ast_running` to
`AST review` and phase `ast_review`, `llm_running` to `LLM review` and phase `llm_review`,
`llm_standby` to `LLM standby` and phase `llm_standby`, `analysis_allowed` to `queued` and phase
`evaluation_queued`, `waiting_miner_env` to `Waiting environments` and phase `waiting_environments`,
`tb_queued` to `evaluation queued` and phase `evaluation_queued`, and `tb_running` to `evaluating` and
phase `evaluation`. Terminal public states include `valid`, `invalid`, `suspicious`, `error`,
`admin_paused`, and owner override states. Responses omit raw LLM prompts/responses, provider errors,
private artifact paths, source snippets, worker leases, raw trial artifacts, broker refs, own_runner
provider refs, job directories, Swarm service or task names, tokens, raw refs, and free-form internal
reasons.

```bash
curl -sS "https://<challenge-host>/submissions/<submission-id>/status"

curl -N "https://<challenge-host>/submissions/<submission-id>/events"

curl -N "https://<challenge-host>/submissions/<submission-id>/events" \
  -H "Last-Event-ID: <last-seen-event-id>"
```

SSE emits `event: submission.status`, a durable integer `id`, and JSON data with public `status`,
`public_state`, `phase`, `sequence`, `submission_id`, `created_at`, and allowlisted machine
`reason_code` values. On reconnect, `Last-Event-ID` replays rows with larger DB event ids. If the id is
unknown, stale before this submission's first event, or belongs to another submission, the server
returns HTTP `409` with:

```json
{"detail": "unknown Last-Event-ID", "replay_from": "<first-event-id>"}
```

### Task Event Replay And SSE

Use task events for stored per-task progress, capped logs, and terminal task outcomes. This is a public
contract for existing payload fields only; it does not promise frontend UI behavior.

```http
GET /submissions/{submission_id}/task-events
GET /submissions/{submission_id}/task-events/stream
```

`GET /submissions/{submission_id}/task-events` replays persisted events after an integer cursor. A
missing `cursor` or `cursor=0` starts at the beginning. `cursor` is the last seen per-submission
`TaskLogEvent.sequence`; results include only larger sequences. `limit` bounds the response page,
`task_id` filters to one task, and `event_type` filters to one event name. The response includes
`cursor`, `next_cursor`, `has_more`, public version fields, and an `events` array. Malformed, negative,
or future cursors return HTTP `409` with `detail.code="task_event_cursor_invalid"`.

`GET /submissions/{submission_id}/task-events/stream` streams the same durable task events as SSE.
Each SSE frame uses `id` equal to `TaskLogEvent.sequence`, `event` equal to the task event type, and
redacted public JSON data. Resume with `cursor` or `Last-Event-ID`; when both are present, `cursor`
takes precedence. Malformed, negative, or future resume ids return HTTP `409` with
`detail.code="task_event_cursor_invalid"`.

Terminal task event types are exact. Use `task.completed` for success and `task.failed` for failed or
error terminal outcomes. `submission.completed` can close a submission-level stream, but it is not a
per-task success marker.

Task log storage has fixed caps: `64KB/event`, `10MB/task`, and `50MB/submission`. Cap marker events
are durable public events named `task_log_cap_reached` and `submission_log_cap_reached`, with
`cap_reached=true`. Log caps do not stop progress, status, terminal, or cap marker events from being
stored and serialized. Do not document or depend on unlimited logs, raw unbounded downloads, or
permanent unlimited retention.

Task replay and task SSE payloads are redacted before persistence and again kept within a public
serialization boundary. Public payloads must not include raw DB ids, normalized names, canonical
hashes, signatures, nonces, artifact paths, worker paths, stdout/stderr refs, log refs, private paths,
refs, tokens, raw artifact paths, worker internals, raw job directories, broker refs, external refs,
container ids, raw stdout or stderr beyond capped stored messages, or raw Terminal-Bench artifacts.

### Admin Escalation

Escalated submissions are resolved through the signed owner endpoint. Use placeholder signed owner
headers in runbooks and never paste live owner signatures.

```bash
curl -sS -X POST "https://<challenge-host>/owner/submissions/<submission-id>/admin-escalation" \
  -H "Content-Type: application/json" \
  -H "X-Hotkey: <owner-hotkey>" \
  -H "X-Signature: <owner-signature>" \
  -H "X-Nonce: <owner-unique-nonce>" \
  -H "X-Timestamp: <iso8601-or-unix-timestamp>" \
  --data '{"decision":"admin_allow","reason":"<operator-reviewed-reason>"}'
```

Decision options are:

| Decision | Effect |
| --- | --- |
| `admin_allow` | Preserves analyzer evidence, records `analysis_allowed`, then moves to `waiting_miner_env`; env-ready submissions enqueue exactly once and env-missing submissions show `Waiting environments`. |
| `admin_reject` | Preserves analyzer evidence, records `analysis_rejected`, and does not create evaluation work. |
| `admin_request_rerun` | Preserves prior evidence and requeues analyzer work for the same immutable artifact. |

### Miner Env Action Runbook

After analyzer allow or admin allow, the exact lifecycle is `analysis_allowed -> waiting_miner_env -> tb_queued -> tb_running`. Public status shows `Waiting environments` while env is missing. The miner must save env vars or call the explicit empty confirmation endpoint. If env rows already exist or empty env is confirmed, the validator locks env metadata and enqueues exactly once.

Local signed routes, including the exact shorthand `GET/PUT /submissions/{id}/env`:

```http
GET /submissions/{submission_id}/env
PUT /submissions/{submission_id}/env
POST /submissions/{submission_id}/env/confirm-empty
POST /submissions/{submission_id}/launch
```

BASE public paths, including the exact shorthand `GET/PUT /challenges/agent-challenge/submissions/{id}/env`:

```http
GET /challenges/agent-challenge/submissions/{id}/env
PUT /challenges/agent-challenge/submissions/{id}/env
POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty
POST /challenges/agent-challenge/submissions/{id}/launch
```

The centralized `POST /internal/v1/submissions/{submission_id}/launch` bridge route has been removed
and now returns 404. Execution is no longer triggered by a launch call; it is driven by the validator
worker pull loop that claims queued jobs. BASE must still keep generic benchmark execution routes
such as `/benchmark-executions` blocked by the public proxy.

Signed miner header examples must use fake placeholders only:

```http
X-Hotkey: <miner-hotkey>
X-Signature: <signature>
X-Nonce: <nonce>
X-Timestamp: <timestamp>
```

Env keys must match `^[A-Za-z_][A-Za-z0-9_]{0,127}$`. Limits are 64 keys, 16 KiB per value, and 128 KiB total payload. `PUT /env` replaces the complete stored set on a waiting submission, then locks/env-ready and enqueues exactly once. `POST /env/confirm-empty` is required for zero-env submissions and also locks/env-ready and enqueues exactly once. Repeat writes or repeated empty confirmation after lock return a conflict. `POST /launch` returns an existing queued or running job idempotently without duplicating it. Values are write-only and never appear in reads, status, SSE, task events, docs, evidence, or logs.

Env values are master-validator scoped, encrypted at rest in Agent Challenge storage, decrypted only for Harbor/Terminal-Bench runtime injection, and cannot be retrieved after submission. BASE registry and proxy do not store per-submission env values.

### Restart Recovery Runbook

Run recovery through the validator worker path or by invoking `run_reconciler_once` from an
operator shell that has the same database and artifact root. The reconciler is idempotent: it reclaims
expired analyzer leases, finalizes completed Terminal-Bench job directories by reading persisted trial
results, marks missing job directories or missing Harbor broker refs retryable until the configured
attempt cap, and then records final failure. Polling and SSE rebuild from DB rows after API restarts.

Do not start duplicate Terminal-Bench jobs when a stable job dir such as
`tb21-<submission-id>-<attempt>` or an external ref already exists. Harbor `harbor jobs resume -p <job_dir>` is policy context for
operators who have confirmed a resumable Harbor job directory; it is not a default duplicate-start
instruction. First check the submission status endpoint, durable attempt row, external ref, and job dir;
then let the reconciler finalize, retry, or fail according to policy.

Known production caveats:

- Normal validators accept signed artifacts but do not evaluate; recovery and analyzer work require the
  master role.
- Terminal-Bench production uses `own_runner` with BASE broker policy and Docker secrets; there
  is no Docker Compose path and no Daytona or `platform_sdk` backend.
- `own_runner` is the only `CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND` value; the runner uses its
  prebuilt Harbor tooling against its native Docker environment.
- Harbor provider credentials are not forwarded unless explicitly listed in
  `CHALLENGE_HARBOR_FORWARD_ENV_VARS`.
- Public status is intentionally summarized; raw analyzer, LLM, own_runner provider refs, job dirs,
  Swarm service or task names, broker tokens, raw refs, and Harbor artifacts stay operator-only.

## Owner Control Surface

Owner endpoints are signed with the owner hotkey and the signed request contract above:

```http
POST /owner/submissions/{submission_id}/revalidate
POST /owner/submissions/{submission_id}/override
POST /owner/submissions/{submission_id}/suspicious
POST /owner/submissions/{submission_id}/admin-escalation
GET /owner/audit
```

`revalidate` creates a new queued job for the same immutable artifact. `override` changes only
`effective_status`; it does not rewrite raw submission status or persisted job evidence.
`suspicious` marks or clears only the effective suspicious state. `admin-escalation` resolves an
LLM/analyzer escalation with `admin_allow`, `admin_reject`, or `admin_request_rerun` while preserving
prior evidence. `/owner/audit` returns append-only audit rows for owner actions.

Owner nonce and replay behavior is the same as miner signing: timestamps allow `300` seconds of skew,
and a reused `(hotkey, nonce)` pair returns HTTP `409`. Audit rows record the owner hotkey, action,
reason, nonce, signature, body hash/request hash, request timestamp, and before and after effective
status.

## BASE Contract

Health check:

```http
GET /health
```

Version and capability check:

```http
GET /version
```

Weight request:

```http
GET /internal/v1/get_weights
Authorization: Bearer <shared-token>
X-Base-Challenge-Slug: agent-challenge
```

Example weight response:

```json
{
  "challenge_slug": "agent-challenge",
  "epoch": 1760000000,
  "weights": {
    "5Abc...": 0.75
  }
}
```

## Scoring And Weights

For each completed job:

```text
aggregate_score = sum(task_scores) / selected_task_count
```

The exported weight map uses the best completed aggregate score from a valid submission for each
miner hotkey. Failed, pending, standby, or running jobs are not included in the weight map. Each
submitted agent or evaluation job selects at most 30 benchmark tasks and runs at most 30 task evaluations
concurrently. Defaults are `evaluation_task_count: 30` and `evaluation_concurrency: 4`; config values
above 30 are rejected by settings validation or capped by runtime helpers for patched tests and stale job
payloads. `harbor_n_concurrent` is separate per-task Harbor behavior inside Terminal-Bench.

Effective-status filtering is stricter than raw job completion. Job lifecycle status remains
`queued`, `running`, `completed`, or `failed`, but public submission status vocabulary includes `received`, `queued`, `AST review`, `LLM review`,
`LLM standby`, `Waiting environments`, `evaluation queued`, `evaluating`, `valid`, `invalid`,
`suspicious`, and `error`. Only completed jobs whose
submission `effective_status` is `valid` or `overridden_valid` can produce weights or leaderboard
rows. Older `completed` submission fixtures are translated for compatibility. Submissions with
`effective_status` of `suspicious`, `invalid`, `error`, or `overridden_invalid` are excluded even if
older job evidence exists.


## Operator Checklist

Before accepting submissions:

1. Configure the benchmark backend and task count.
2. Configure artifact storage and persistent result storage.
3. Configure shared BASE token delivery.
4. Configure `CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND=own_runner`, broker URL plus token file,
   controlled runner image, and allowed-image policy before enabling Terminal-Bench.
5. Enable evaluation only after task environments and broker settings are ready.
6. Verify benchmark metadata is visible.
7. Submit a small test artifact.
8. Confirm the evaluation reaches a terminal status.
9. Confirm the leaderboard reflects completed scores.
10. Confirm BASE can read the protected weight contract.

own_runner operator verification commands use placeholders only and must not print raw tokens. The
challenge runs as Docker Swarm services (`challenge-agent-challenge` API, `challenge-agent-challenge-worker`
eval loop, `base-master-broker` broker):

```bash
docker service ps challenge-agent-challenge
docker service logs challenge-agent-challenge-worker --since 30m | rg 'terminal_bench|own_runner|tb_running'
docker service logs base-master-broker --since 30m | rg 'run request|created job|terminal-bench-harbor-runner'
docker service logs challenge-agent-challenge-worker --since 30m | rg --fixed-strings -- 'agent-challenge-terminal-bench-runner'
curl -sS '<api-base-url>/submissions/<submission-id>/status' | rg '"status":"evaluating"|"phase":"evaluation"|"status":"valid"|"status":"error"'
```

There is no execution-backend rollback. `own_runner` is the only accepted
`CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND` value, and any other value is rejected by settings
validation; there is no Daytona or `platform_sdk` path to roll back to.

During operation:

- watch failed and timed-out task counts;
- keep benchmark settings stable during a scoring epoch;
- rotate tokens if they are exposed;
- back up persistent result storage;
- announce entrypoint and packaging expectations to miners;
- avoid changing task counts mid-round unless the round is intentionally reset.

## Safety Notes

- Run submitted artifacts only in isolated environments.
- Keep network and resource limits strict.
- Do not pass private credentials into untrusted agent code unless the benchmark explicitly requires
  them and the risk is accepted.
- Limit logs to prevent storage exhaustion.
- Treat mounted artifact paths as trusted operator inputs only.
- Keep broker tokens and BASE shared tokens out of public logs.
