# Miner hub (Agent Challenge)

> **Monorepo hub.** Product source of truth is
> [`packages/challenges/agent-challenge`](../../../packages/challenges/agent-challenge/) inside
> **BaseIntelligence/base**. Public slug `/challenges/agent-challenge` is unchanged.
> Submit helpers live under that package path from the monorepo root.


Mine Agent Challenge on **[joinbase.ai](https://joinbase.ai)** in minutes. Build software
engineering agents, submit a signed ZIP, then (for production score) self-deploy attested CVMs.

**Day-1 target:** hotkey ready → package agent → signed submit on joinbase → status / leaderboard,
in under 15 minutes. Phala TDX self-deploy depth stays in Concepts / advanced How-to — not on the
first page.

| Page | What it covers |
|------|----------------|
| [Getting started](getting-started.md) | joinbase URLs, wallet, dashboard and/or `submit_agent.py`, checklist, Troubleshooting |
| [Submit agent](submit-agent.md) | Packaging, signing, env gate, log streams (A→Z) |
| [Self-deploy (how-to advanced)](self-deploy.md) | Production Phala review/eval CLI after upload |
| [Attestation TEE (concepts)](attestation-tee.md) | Intel TDX measurements, report_data domains, RA-TLS |
| This hub (reference) | Full signing contract, status table, env, leaderboard, BASE routes |

## Canonical public URLs

| Surface | URL |
|---------|-----|
| Product / dashboard | https://joinbase.ai |
| Base master API | https://chain.joinbase.ai |
| Submit (proxy default for CLI) | `POST https://chain.joinbase.ai/challenges/agent-challenge/submissions` |
| ZIP bridge | `POST https://chain.joinbase.ai/v1/challenges/agent-challenge/submissions` |
| Leaderboard | `GET https://chain.joinbase.ai/challenges/agent-challenge/leaderboard` |
| OpenAPI | `GET https://chain.joinbase.ai/challenges/agent-challenge/openapi.json` |

Agent Challenge receives **50%** absolute emission share (paired with Prism at 50%). See BASE
miner [Concepts](https://github.com/BaseIntelligence/base/blob/main/docs/miner/concepts.md).

## Quick path

1. Link wallet hotkey on https://joinbase.ai.
2. Confirm network: `curl -fsS https://chain.joinbase.ai/health`.
3. Package + submit with [`packages/challenges/agent-challenge/scripts/submit_agent.py`](../../../packages/challenges/agent-challenge/scripts/submit_agent.py)
   (default `--api-base` is the joinbase proxy) **or** the joinbase dashboard.
4. Watch https://chain.joinbase.ai/challenges/agent-challenge/leaderboard.
5. When ready to score in production: [Self-deploy](self-deploy.md) (advanced).

Full walkthrough: [Getting started](getting-started.md). Common errors:
[Troubleshooting in Getting started](getting-started.md#troubleshooting).

---

## Purpose

Agent Challenge rewards miners for submitting software engineering agents that solve benchmark
tasks. Your score comes from completed task evaluations, and your best completed score becomes the
raw weight BASE uses for your hotkey.

**Production scoring is miner self-deploy on Phala Cloud Intel TDX CVMs (attestation mandatory; Base
LLM gateway forbidden).** After submit you fund and operate the attested review CVM (shipping
script + agent ZIP measured with real OpenRouter under the harness / `.rules`) and, after a
**fresh re-verified** allow (review-domain `issued_at` / `received_at` bound into `report_data`,
≤24h freshness), the attested eval CVM. See [Self-deploy (how-to advanced)](self-deploy.md).
Validators re-verify measurements, quotes, durable KR grant, and the full score chain; they do not
deploy your production scored jobs for you. Day-1 upload only: [Getting started](getting-started.md).

## Miner Flow

1. Build an agent that can operate inside benchmark workspaces.
2. Package the agent artifact.
3. Submit the artifact with your miner hotkey (joinbase dashboard and/or CLI).
4. Self-deploy review then eval CVMs when production attestation flags are ON ([self-deploy](self-deploy.md)).
5. Track evaluation progress and tear down CVMs to `total: 0`.
6. Review failed tasks and improve your agent.
7. Submit a new version when ready.

For a copy-paste, end-to-end walkthrough of every step — packaging, request
signing, the env gate, and the per-channel evaluation log streams — see the
[A→Z submit walkthrough](submit-agent.md). A ready-to-run implementation lives in
[`packages/challenges/agent-challenge/scripts/submit_agent.py`](../../../packages/challenges/agent-challenge/scripts/submit_agent.py). Day-1 front door:
[Getting started](getting-started.md).


## BASE Frontend API

A BASE-hosted Agent Challenge page should read through the BASE master/proxy base
(`https://chain.joinbase.ai`), not a direct challenge host:

```http
GET /v1/registry
GET /challenges/agent-challenge/benchmarks
GET /challenges/agent-challenge/submissions/{id}/status
GET /challenges/agent-challenge/submissions/{id}/events
GET /challenges/agent-challenge/submissions/{id}/env
PUT /challenges/agent-challenge/submissions/{id}/env
POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty
POST /challenges/agent-challenge/submissions/{id}/launch
GET /challenges/agent-challenge/leaderboard
```

There are two upload paths:

```http
POST /v1/challenges/agent-challenge/submissions
POST /challenges/agent-challenge/submissions
```

Use `POST /v1/challenges/agent-challenge/submissions` for raw ZIP bridge uploads through BASE. Use `POST /challenges/agent-challenge/submissions` for JSON base64 uploads through the generic proxy. The generic proxy request still signs the challenge-local path, `/submissions`.

For v1 lists, `/challenges/agent-challenge/submissions` returns the latest 100 submissions newest-first. `/challenges/agent-challenge/leaderboard` returns one best scoring row per hotkey. Pagination, filtering, and client-selected sorting are deferred to future v2. BASE blocks `/internal/*`, `/health`, and `/version` from the public proxy.

## Understanding The Benchmark

Validators publish the active benchmark configuration through:

```http
GET /benchmarks
```

Task metadata is available through:

```http
GET /benchmarks/tasks
```

Important benchmark fields:

| Field | Meaning |
| --- | --- |
| `backend` | Active benchmark family, such as repository-repair or terminal-task evaluation. |
| `dataset` | Dataset or benchmark collection currently selected. |
| `task_count` | Number of available benchmark tasks or configured task shards. |
| `evaluation_concurrency` | Number of task evaluations the validator can run at once for one submitted agent, capped at 30. |
| `task_id` | Stable task identifier. |
| `docker_image` | Isolated task environment used by the validator. |
| `prompt` | Human-readable task prompt or dataset reference. |

## Agent Expectations

A strong agent should be able to:

- read task instructions and repository context;
- inspect files and understand failing behavior;
- modify source code safely;
- run relevant checks when available;
- avoid destructive or unrelated changes;
- finish within the validator timeout;
- handle repeated runs consistently;
- keep secrets and external credentials out of outputs.

## Required Base Agent And Legal LLM Paths

Build your submission from [`BaseIntelligence/baseagent`](https://github.com/BaseIntelligence/baseagent).

**Base LLM gateway is removed and forbidden** for production:

- Do **not** set, hardcode, or require `BASE_LLM_GATEWAY_URL`, `BASE_GATEWAY_TOKEN`, `GATEWAY_TOKEN`,
  or `/llm/v1` client wiring (`base_gateway_forbidden`).
- Do **not** embed provider API keys, non-measured base URLs, or emission model pins on the host
  (`unauthorized_llm_provider`, `hardcoded_llm_model`).

Legal LLM paths under **attestation-only** policy (no Base master gateway on the scored path):

1. **Measured OpenRouter** inside the review harness (shipping script + agent ZIP under `.rules`,
   real OpenRouter on the measured guest) and, when product permits agent models, OpenRouter
   **only inside the measured eval CVM** with planned/observed digests bound into attestation.
2. **Tools-only** agents with no model egress (no LLM key required).

Continuous static analysis automatically flags residual Base gateway clients and non-measured
provider embeds before scoring.

For Terminal-Bench style tasks, the ZIP entrypoint is mandatory and fixed. Every submitted ZIP
must include `agent.py` at the archive root, and that file must define a top-level `class Agent`.
Production validators import `agent:Agent`; `submitted_agent.py` is not accepted as the entrypoint.

Required ZIP layout:

```text
my-agent.zip
├── agent.py          # required root entrypoint, defines class Agent
├── src/              # optional support code
├── pyproject.toml    # optional dependency metadata
└── requirements.txt  # optional dependency metadata
```

Minimal valid `agent.py` shape:

```python
class Agent:
    async def run(self, instruction, environment, context):
        return "Task completed"
```

Production validators use dataset `terminal-bench/terminal-bench-2-1` with display label
`terminal-bench@2.1` and import `agent:Agent` from the submitted artifact. Each submitted
agent selects at most 30 benchmark tasks, and at most 30 task evaluations run concurrently for that
agent. Defaults are `evaluation_task_count: 30` and `evaluation_concurrency: 4`;
`harbor_n_concurrent` is separate per-task Harbor behavior.

## Submitting An Agent

Submit either a base64-encoded zip archive or a trusted artifact path already mounted on the
challenge host.

```http
POST /submissions
Content-Type: application/json
```

Every public miner submission must include these exact signed request headers:

```http
X-Hotkey: <miner-hotkey>
X-Signature: <signature>
X-Nonce: <unique-nonce>
X-Timestamp: <timestamp>
```

Sign this canonical string exactly, preserving the newline order:

```text
{METHOD}
{PATH_WITH_SORTED_QUERY}
{X-TIMESTAMP}
{X-NONCE}
{SHA256_HEX_OF_RAW_BODY}
```

For `POST /submissions`, the method is `POST`, the path is `/submissions` with any query string
sorted, and the body hash is the SHA-256 hex digest of the raw request body bytes. The validator
accepts timestamps within `300` seconds. Each `(hotkey, nonce)` pair can be used once; replaying it
returns HTTP `409`. Accepted submissions are rate-limited to one per hotkey per active
`CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS` window (Settings: `submission_rate_limit_window_seconds`;
product default **10800** seconds / 3 hours when operators leave the default). Another accepted upload
inside the active window returns HTTP `429` with `detail.code="submission_rate_limited"`, a
window-aware human message (not a hard-coded “every 3 hours” string when the configured window
differs), and `next_allowed_at`. Always honor `next_allowed_at` rather than assuming a fixed 3-hour
clock.

Zip archive submission:

```json
{
  "miner_hotkey": "5Abc...",
  "name": "my-agent",
  "artifact_zip_base64": "<base64-encoded-agent-zip>"
}
```

Mounted artifact submission:

```json
{
  "miner_hotkey": "5Abc...",
  "name": "my-agent",
  "artifact_uri": "/data/agents/my-agent"
}
```

Example signed upload with placeholders only:

```bash
curl -X POST '<api-base-url>/submissions' \
  -H 'Content-Type: application/json' \
  -H 'X-Hotkey: <miner-hotkey>' \
  -H 'X-Signature: <signature>' \
  -H 'X-Nonce: <unique-nonce>' \
  -H 'X-Timestamp: <iso-8601-timestamp>' \
  --data '{"miner_hotkey":"<miner-hotkey>","name":"<agent-name>","artifact_zip_base64":"<base64-zip>"}'
```

After upload, verify the response receipt. `zip_sha256` should match the SHA-256 digest of your local
compressed ZIP bytes, `agent_hash` is the server-stored artifact digest, and `submission_id` is the id
used for polling, task replay, and SSE. Client-supplied `agent_hash` values are not a public naming
contract.

Rate limit and replay rules:

- one submission per hotkey is allowed per active `submission_rate_limit_window_seconds` window
  (env `CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS`; product default **10800** seconds / 3 hours).
  Live joinbase residual ops may set a shorter window (for example `1`); the 429 text matches that
  window via product `submission_rate_limit_message(window_seconds)` (Settings wire `cebc3ad`).
- Reusing the same `(hotkey, nonce)` pair returns HTTP `409`, even if the body changes.
- A rate-limited accepted upload returns HTTP `429` with `detail.code="submission_rate_limited"` and
  `next_allowed_at`. Wait until `next_allowed_at` (do not reinvent dual-submit storms on a fixed 3h belief).
- Use a fresh nonce and timestamp for every request.

Submission rules:

- `miner_hotkey` is the hotkey that receives score credit.
- `name` is a human-readable label. The first successful submitter owns the normalized name globally
  within Agent Challenge.
- Exactly one artifact source should be provided.
- Duplicate artifact or code hashes are rejected globally, regardless of name or miner.
- Duplicate artifact or code hash conflicts return HTTP `409` with `detail.code="duplicate_code_hash"`.
- A name already owned by another miner returns HTTP `409` with `detail.code="name_taken"`.
- Duplicate hash checks run before name ownership checks.
- Mounted artifact paths must be inside the validator-approved artifact root.
- ZIP submissions have a maximum compressed size of `1048576` bytes, also described as 1MB.
- Oversized ZIP submissions return HTTP `413` with `detail.code="zip_too_large"`.
- Submitted ZIPs are stored immutably by their SHA-256 digest.

Submission versions:

- Reusing your globally owned normalized `name` creates the next integer version of that agent family.
- Version labels are exact integer labels, such as `v1`, `v2`, and `v3`.
- Public read payloads can include `family_id`, `display_name`, `version_number`, `version_label`,
  `version_count`, `latest_submission_id`, and `is_latest_version` where the route returns versioned
  submission data.
- Public `family_id` is the stable public family identifier, not an internal database id.

## Tracking Evaluation

List recent submissions:

```http
GET /submissions
```

Read one submission:

```http
GET /submissions/{submission_id}
```

Read the versions in the same submission family:

```http
GET /submissions/{submission_id}/versions
```

Read public status:

```http
GET /submissions/{submission_id}/status
```

Stream status events:

```http
GET /submissions/{submission_id}/events
```

Replay task events:

```http
GET /submissions/{submission_id}/task-events
```

Stream task events:

```http
GET /submissions/{submission_id}/task-events/stream
```

Read the number of stored submissions:

```http
GET /submissions/count
```

Read evaluation details for an agent hash:

```http
GET /agents/{agent_hash}/evaluation
```


Status polling example:

```bash
curl '<api-base-url>/submissions/<submission-id>/status'
```

SSE example:

```bash
curl -N '<api-base-url>/submissions/<submission-id>/events'
```

Reconnect with the last durable event id:

```bash
curl -N \
  -H 'Last-Event-ID: <last-event-id>' \
  '<api-base-url>/submissions/<submission-id>/events'
```

If the reconnect id is stale, unknown, or from another submission, the validator returns HTTP `409`:

```json
{
  "detail": "unknown Last-Event-ID",
  "replay_from": "<first-event-id>"
}
```

Task event replay returns durable per-submission task events after an integer cursor:

```bash
curl '<api-base-url>/submissions/<submission-id>/task-events?cursor=0&limit=100'
```

For `GET /submissions/{submission_id}/task-events`, a missing `cursor` or `cursor=0` starts at the
beginning. `cursor` is the last seen `TaskLogEvent.sequence`, and the next page returns rows with a
larger sequence plus `next_cursor` and `has_more`. `limit` bounds the page size. `task_id` filters to
one task, and `event_type` filters to one event name. Malformed, negative, or future cursors return
HTTP `409` with `detail.code="task_event_cursor_invalid"`.

Task event SSE uses the same durable event rows:

```bash
curl -N '<api-base-url>/submissions/<submission-id>/task-events/stream?cursor=<last-sequence>'
```

For `GET /submissions/{submission_id}/task-events/stream`, the SSE `id` is the `TaskLogEvent.sequence`.
Reconnect with either `cursor` or `Last-Event-ID`; when both are present, `cursor` takes precedence.
Malformed, negative, or future cursors return HTTP `409` with `detail.code="task_event_cursor_invalid"`.
Terminal task outcomes use `task.completed` for success and `task.failed` for failed or error outcomes.

Both `GET /submissions/{submission_id}/task-events` and
`GET /submissions/{submission_id}/task-events/stream` accept an optional `stream` query parameter
that isolates a single log channel, so you can read the agent's own output separately from the
harness and the verifier:

| `stream` value | Contents |
| --- | --- |
| _(omitted)_ | Every channel plus progress, status, terminal, and cap marker events. |
| `agent` | The submitted agent's own logs (trajectories, debug) from real harbor v2 trials. |
| `harness` | Terminal-Bench / harbor harness log (`trial.log`) and trial exception text. |
| `test_stdout` | Verifier (test) stdout from real harbor v2 trials. |
| `test_stderr` | Verifier (test) stderr from real harbor v2 trials. |
| `stdout` | Aggregate per-task stdout captured for the task result. |
| `stderr` | Aggregate per-task stderr captured for the task result. |

```bash
# Replay only the agent's own logs:
curl '<api-base-url>/submissions/<submission-id>/task-events?stream=agent&cursor=0&limit=100'

# Live-stream only the verifier stderr:
curl -N '<api-base-url>/submissions/<submission-id>/task-events/stream?stream=test_stderr'
```

`stream` composes with `cursor`, `limit`, `task_id`, and `event_type`, and is part of the signed path
when present (sort the query string before signing). It is an exact-match filter: an unrecognized
`stream` value is not an error, it simply matches no log rows. The channel filter narrows `task.log`
lines only; progress, status, terminal, and cap marker events are not stream-tagged and are returned
regardless of the filter. The separated `agent`, `harness`, `test_stdout`, and `test_stderr` channels
populate for real (non-mock) harbor v2 trials that emit those artefacts; `stdout` and `stderr` carry
the aggregate per-task output.

Task log storage is capped, not unlimited. One task event message is capped at `64KB/event`, counted
task logs are capped at `10MB/task`, and counted submission logs are capped at `50MB/submission`.
When a cap is reached, replay and SSE can include cap marker events named `task_log_cap_reached` or
`submission_log_cap_reached` with `cap_reached=true`. Progress, status, terminal, and cap marker events
can continue after log caps.

Task event payloads are public and redacted before persistence and serialization. They must not expose
raw DB ids, normalized names, canonical hashes, signatures, nonces, artifact paths, worker paths,
stdout/stderr refs, log refs, private paths, refs, tokens, raw artifact paths, or worker internals. Do
not expect raw unbounded stdout, stderr, artifact paths, log downloads, or permanent unlimited
retention from public routes.

Public status meanings:

| Raw status | Public status copy | Public phase | Meaning |
| --- | --- | --- | --- |
| `received` | `received` | `received` | The validator accepted the signed upload. |
| `analysis_queued` | `queued` | `queued` | The submission is waiting for analysis work. |
| `ast_running` | `AST review` | `ast_review` | ZIP receipt, Python AST features, and similarity review are in progress. |
| `llm_running` | `LLM review` | `llm_review` | Attested measured OpenRouter review (review CVM / harness) is in progress. **Not** Base master gateway. |
| `llm_standby` | `LLM standby` | `llm_standby` | Measured review provider/session material is missing or temporarily unavailable; review retries when attested path config is available. Base gateway is **not** required and must not be restored. |
| `analysis_allowed` | `queued` | `evaluation_queued` | The analyzer allowed the artifact and evaluation can be queued once env is ready. |
| `waiting_miner_env` | `Waiting environments` | `waiting_environments` | The validator is waiting for you to provide env vars or confirm that no env vars are needed. |
| `tb_queued` | `evaluation queued` | `evaluation_queued` | Terminal-Bench work is queued. |
| `tb_running` | `evaluating` | `evaluation` | Terminal-Bench work is running. |
| terminal success | `valid` | `complete` | The submission completed and can count for scoring. |
| terminal rejection | `invalid` | `error` | The analyzer, admin review, or evaluation policy rejected the submission. |
| owner exclusion | `suspicious` | `error` | Owner policy has marked the submission for exclusion. |
| terminal error | `error` | `error` | The submission reached a terminal error. |

The raw happy path is `analysis_queued -> ast_running -> llm_running -> analysis_allowed -> waiting_miner_env -> tb_queued -> tb_running`. On the production dual-flag path the scored happy path is miner self-deploy (attested review allow then attested eval RESULT) rather than validator-brokered tb jobs. A missing measured-review provider token, provider unavailable, rate limit, and timeout produce sanitized standby reason codes and do not create `LlmVerdict`, `EvaluationJob`, `AdminReviewDecision`, or weights. When attested review config becomes available, standby retries through `llm_standby -> analysis_queued`. Do not treat Base LLM gateway as required.

Analyzer verdict meanings:

| Verdict | Meaning |
| --- | --- |
| `allow` | The submission can continue to Terminal-Bench. |
| `reject` | The submission stops as invalid. |
| `escalate` | The submission waits for owner review. |

When a submission is at raw `waiting_miner_env`, public list, detail, and status payloads include
redacted metadata fields: `env_action_required`, `env_keys`, `env_var_count`,
`env_confirmed_empty`, `env_locked`, and `env_updated_at`. `env_keys` are miner-provided variable
names so a frontend can show which identifiers were supplied; they are not secret values. Env
values, ciphertext, hashes, request bodies, signatures, nonces, key file paths, and runtime injected
values are never returned in public status, list, SSE, task event, docs, evidence, or notepad output.
The `waiting_miner_env` SSE transition is allowlisted only as a safe reason code; fetch status or
detail after that event to read the redacted env action fields.

The public responses intentionally omit raw internal metadata, source code, signatures, provider transcripts, private paths, tokens, and env values.

## Miner Env Vars Before Launch

When analysis allows your artifact, the exact raw lifecycle is `analysis_allowed -> waiting_miner_env -> tb_queued -> tb_running`. If env is missing, public state is `Waiting environments` with phase `waiting_environments`. Terminal-Bench will not launch until you save env vars or confirm that no env vars are needed. If env rows already exist or empty env was already confirmed, the validator locks env metadata and enqueues exactly once without waiting for a separate launch call.

Agent Challenge local signed routes, including the exact shorthand `GET/PUT /submissions/{id}/env`:

```http
GET /submissions/{id}/env
PUT /submissions/{id}/env
POST /submissions/{id}/env/confirm-empty
POST /submissions/{id}/launch
```

Exact local shorthand: `GET/PUT /submissions/{id}/env`, `POST /submissions/{id}/env/confirm-empty`, `POST /submissions/{id}/launch`.

BASE public paths for the same actions, including the exact shorthand `GET/PUT /challenges/agent-challenge/submissions/{id}/env`:

```http
GET /challenges/agent-challenge/submissions/{id}/env
PUT /challenges/agent-challenge/submissions/{id}/env
POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty
POST /challenges/agent-challenge/submissions/{id}/launch
```

Exact BASE shorthand: `GET/PUT /challenges/agent-challenge/submissions/{id}/env`, `POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty`, `POST /challenges/agent-challenge/submissions/{id}/launch`.

Use the same signed request header names with placeholders only:

```http
X-Hotkey: <miner-hotkey>
X-Signature: <signature>
X-Nonce: <nonce>
X-Timestamp: <timestamp>
```

Env keys must match `^[A-Za-z_][A-Za-z0-9_]{0,127}$`. You can submit at most 64 keys, each value can be at most 16 KiB, and the full request payload can be at most 128 KiB. Use a fresh nonce and timestamp for every env, confirm-empty, or launch request.

`PUT /submissions/{id}/env` replaces the complete env set for your waiting submission, then locks/env-ready and enqueues exactly once. Values are write-only. Read and write responses return metadata only: keys, count, timestamps, empty confirmation, and lock state. Example request with redacted values:

```json
{
  "env": {
    "EXAMPLE_API_TOKEN": "<env-value-write-only>"
  }
}
```

Example metadata-only response:

```json
{
  "submission_id": "<submission-id>",
  "env_keys": ["EXAMPLE_API_TOKEN"],
  "env_var_count": 1,
  "env_confirmed_empty": false,
  "env_locked": false,
  "env_updated_at": "<timestamp>"
}
```

If your agent needs no env vars, call `POST /submissions/{id}/env/confirm-empty`. This explicit zero-env confirmation prevents the submission from getting stuck in `Waiting environments`. `PUT /submissions/{id}/env` and `POST /submissions/{id}/env/confirm-empty` on a waiting submission lock/env-ready and enqueue Terminal-Bench exactly once. Repeat writes or repeated empty confirmation after lock return a conflict. `POST /submissions/{id}/launch` returns an existing queued or running job idempotently without duplicating it. After lock, env values cannot be retrieved, changed, or deleted through public APIs.

Env values are scoped to the validator, encrypted at rest in Agent Challenge storage, injected into the Harbor/Terminal-Bench runtime only for the submission launch, and cannot be retrieved after submission. BASE registry and BASE proxy do not store per-submission env values.

## BASE 502 Handling

If a BASE URL under `/challenges/agent-challenge/...` returns 502, treat it as temporary challenge unavailable state. Frontends should show safe copy such as `Agent Challenge is temporarily unavailable. Please try again shortly.` They must not display raw text such as `BASE request failed with status 502`.

Operator checklist:

1. Verify ingress sends `/challenges` traffic to the BASE proxy.
2. Verify BASE proxy routing for `agent-challenge` and confirm private paths stay blocked.
3. Verify the Agent Challenge service is healthy and reachable from BASE.
4. Verify the Agent Challenge Swarm service is healthy: `docker service ps challenge-agent-challenge`, that overlay DNS resolves (`tasks.challenge-agent-challenge` on `base_challenges`), and that the published port answers.
5. Separate transport failures from challenge-origin non-2xx responses. BASE should return safe 502 only for transport failures and should pass through safe challenge responses for validation errors, auth failures, replay conflicts, rate limits, and challenge-origin server errors.
6. For env actions, verify BASE forwards only `X-Hotkey`, `X-Signature`, `X-Nonce`, and `X-Timestamp` on the allowed env and launch paths, while keeping other sensitive headers stripped.

Evaluation response fields:

| Field | Meaning |
| --- | --- |
| `status` | Public state such as `received`, `queued`, `AST review`, `LLM review`, `LLM standby`, `Waiting environments`, `evaluation queued`, `evaluating`, `valid`, `invalid`, `suspicious`, `error`, or `admin_paused`. |
| `phase` | Public phase such as `queued`, `ast_review`, `llm_review`, `llm_standby`, `waiting_environments`, `evaluation_queued`, `evaluation`, `complete`, or `error`. |
| `effective_status` | Submission result used for leaderboard and weight eligibility. |
| `env_action_required` | `true` only while the submission is in the miner env action step and env input is not locked. |
| `env_keys` | Miner-provided env variable identifiers, never values. |
| `env_var_count` | Count of stored miner env variable identifiers. |
| `env_confirmed_empty` | Whether the miner confirmed that no env vars are needed. |
| `env_locked` | Whether env metadata is locked for launch. |
| `env_updated_at` | Last env metadata update or empty-env confirmation timestamp. |
| `last_event_id` | Durable SSE id to store for reconnect. |
| `analyzer` | Safe analyzer verdict summary. LLM verdict meanings are `allow`, `reject`, or `escalate`. |
| `similarity` | Safe AST similarity score/risk summary without raw source. |
| `terminal_bench` | Terminal-Bench trial counts for the current durable attempt. |
| `score` | Average score across selected tasks. |
| `passed_tasks` | Number of tasks scored as passed. |
| `total_tasks` | Number of selected tasks. |
| `tasks` | Per-task status, score, return code, and duration. |

Version fields available to frontend reads where applicable are `family_id`, `display_name`,
`version_number`, `version_label`, `version_count`, `latest_submission_id`, and `is_latest_version`.

## Leaderboard

Read the current leaderboard:

```http
GET /leaderboard
```

The leaderboard keeps the best completed score from a valid submission per miner hotkey. Each submitted agent selects at most 30 benchmark tasks and runs at most 30 task evaluations concurrently; defaults are `evaluation_task_count: 30` and `evaluation_concurrency: 4`, values above 30 are rejected or capped, and `harbor_n_concurrent` remains separate per-task Harbor behavior. If you
submit several agent versions, only your strongest valid completed score is used for weight
calculation.

Weights use effective status. Only completed jobs whose submission `effective_status` is
`valid` or `overridden_valid` can appear on the leaderboard or in BASE weights. Older
`completed` submission fixtures are translated for compatibility. Submissions marked `suspicious`,
`invalid`, `error`, or `overridden_invalid` are excluded.

## Scoring Model

Each submission selects tasks deterministically from the agent hash. Each submitted agent or evaluation job selects at most 30 benchmark tasks and runs at most 30 task evaluations concurrently. Defaults are `evaluation_task_count: 30` and `evaluation_concurrency: 4`; config values above 30 are rejected or capped, while `harbor_n_concurrent` stays separate per-task Harbor behavior. This prevents miners from
choosing only favorable tasks while keeping results reproducible.

The aggregate score is:

```text
sum(task_scores) / selected_task_count
```

For binary tasks, a passing task contributes `1.0` and a failing or timed-out task contributes
`0.0`. Terminal-task benchmarks can return fractional scores when the benchmark provides them.

Analyzer checks use the validator's `.rules` directory. If `.rules` is missing, the analyzer returns
`error`. Hardcoding detection is evidence-based, bounded, owner-auditable, and not proof that
hardcoding is absent.

## Packaging Checklist

Before submitting:

- Confirm your artifact contains all files required by the published agent contract.
- Confirm the artifact is based on `BaseIntelligence/baseagent`.
- Confirm the artifact embeds no Base LLM gateway client (`BASE_LLM_GATEWAY_URL` / `BASE_GATEWAY_TOKEN` / `/llm/v1`) and no non-measured provider secrets or emission model pins; legal LLM path is measured OpenRouter under the review harness (shipping script + ZIP / `.rules`) and measured eval CVM, or tools-only.
- Keep the archive small and focused.
- Keep the compressed ZIP at or below `1048576` bytes, 1MB.
- Remove local caches, logs, and secrets.
- Test the agent in a clean workspace.
- Ensure the expected entrypoint resolves from the artifact root.
- Make failures readable so you can improve the next version.
- Submit a new artifact under your owned name when you want the next `v1`, `v2`, `v3` style version.
