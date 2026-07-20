# Agent Challenge Frontend API Contract

## Scope

This contract describes the API surface a future BASE Agent Challenge page can use without calling challenge hosts directly. Frontend reads should use the canonical BASE proxy base:

```text
/challenges/agent-challenge/...
```

The only non-proxy frontend inputs are BASE registry metadata from `/v1/registry` and the raw ZIP upload bridge at `/v1/challenges/agent-challenge/submissions`. The bridge verifies the miner upload at BASE before forwarding it to Agent Challenge's internal bridge route.

Do not advertise private challenge routes as frontend-consumable. The public proxy blocks `/health`, `/version`, `/internal/*`, and generic benchmark execution-shaped routes such as `/benchmark-executions`. It strips sensitive request headers and adds `X-Base-Proxy: true` plus `X-Base-Challenge-Slug: agent-challenge` upstream.

## Auth Modes

| Auth | Meaning |
| --- | --- |
| None | Public read through BASE proxy. Sensitive caller headers are stripped before the challenge receives the request. |
| Signed miner JSON | Challenge-local signed request headers on a JSON body sent through generic proxy to `/challenges/agent-challenge/submissions`. |
| Signed miner env action | Challenge-local signed request headers on env, confirm-empty, and launch requests sent through generic proxy to `/challenges/agent-challenge/submissions/{id}/...`. |
| BASE bridge upload | Miner upload signature is verified by BASE at `/v1/challenges/agent-challenge/submissions`, then BASE forwards verified headers to the challenge internal bridge route. |

## Frontend Route Matrix

| Section | BASE-served route | Raw challenge route | Method | Auth | Purpose | Response fields | Empty/loading/error behavior | Cache/SSE key | Redaction | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Hero | `/v1/registry` entry where `slug="agent-challenge"` | Not a challenge route | GET | None | Render the challenge hero card and public proxy base. | Fields: `slug`, `name`, `image`, `version`, `emission_percent`, `status`, `public_proxy_base_path`, `description`, frontend-safe scalar `metadata`, `required_capabilities`, `resources`, `volumes`, `env`, and `secrets`. | Empty: no active entry means show challenge unavailable. Loading: load once on page entry. Error: retry 5xx with backoff; treat 404 as unavailable after registry refresh. | `registry:agent-challenge` | Never display or persist `internal_base_url`, tokens, token hints, broker tokens, secret env values, secret file paths, or unallowlisted metadata. | AVAILABLE. |
| Benchmark | `/challenges/agent-challenge/benchmarks` | `/benchmarks` | GET | None | Show active benchmark family, dataset, task count, and evaluation capacity. | `backend`, `dataset`, `task_count`, `evaluation_concurrency`. | Empty: `task_count: 0` means no published tasks. Loading: load with page. Error: retry transient 502/5xx; do not treat 404 as success. | `benchmark:agent-challenge` | Do not add private dataset paths or operator config beyond returned public fields. | AVAILABLE. |
| Evaluation task details | `/challenges/agent-challenge/benchmarks/tasks` | `/benchmarks/tasks` | GET | None | Show public task catalog or benchmark shards selected by the validator. | Array of `task_id`, `benchmark`, `docker_image`, `prompt`. | Empty: empty array means no task details are published. Loading: lazy-load below benchmark summary. Error: retry 502/5xx. | `benchmark-tasks:agent-challenge` | Do not add source archives, expected answers, private evaluator paths, or raw artifacts. | AVAILABLE. |
| Upload | `/v1/challenges/agent-challenge/submissions` | `POST /internal/v1/bridge/submissions` | POST | BASE bridge upload | Raw ZIP bridge upload. BASE verifies miner signature and forwards verified identity to the challenge. | Receipt: `submission_id`, `zip_sha256`, `status`, `effective_status`, `submitted_at`, `created_at`, `latest_evaluation`. | Empty: not applicable. Loading: disable submit while uploading. Error: 401 auth failure, 409 nonce replay/duplicate, 413 ZIP too large, 429 rate limit; retry only with a new nonce/signature. | `submission-upload:raw-zip:{nonce}` then `submission:{submission_id}` | Do not send or show bearer tokens. Do not trust client hotkey after BASE verifies it. Do not expose signatures, request hashes, nonce storage, uploaded ZIP contents, or source files. | AVAILABLE. BASE route forwards to Agent Challenge `POST /internal/v1/bridge/submissions`. |
| Upload | `/challenges/agent-challenge/submissions` | `/submissions` | POST | Signed miner JSON | JSON base64 generic proxy upload for clients that intentionally sign the challenge-local path. | `submission_id`, `zip_sha256`, `status`, `effective_status`, `submitted_at`, `created_at`, `latest_evaluation`. | Empty: not applicable. Loading: disable submit while uploading. Error: 400/422 validation, 401 signature failure, 409 duplicate/replay, 413 ZIP too large, 429 rate limit; retry only after changing and re-signing the request. | `submission-upload:json:{agent_hash_or_nonce}` then `submission:{submission_id}` | Do not persist raw base64 ZIP in frontend state. Do not expose signatures, canonical strings, artifact paths, source files, or raw ZIP contents. | AVAILABLE. |
| Submission list/count/detail | `/challenges/agent-challenge/submissions` | `/submissions` | GET | None | Show latest public submissions. v1 returns latest 100 newest-first. Pagination/filter/sort are deferred to future v2. | Array of `id`, `miner_hotkey`, `name`, `agent_hash`, `zip_sha256`, `status`, `effective_status`, `score`, `submitted_at`, `created_at`, `latest_evaluation`, and redacted env action fields `env_action_required`, `env_keys`, `env_var_count`, `env_confirmed_empty`, `env_locked`, `env_updated_at`. | Empty: no submissions yet. Loading: show skeleton rows. Error: retry 502/5xx and refresh after upload or terminal status. | `submissions:list:latest100` | Do not add signatures, nonces, request hashes, artifact paths, source snippets, raw status internals, worker lease owners, env values, env ciphertext, env hashes, or env key file paths. `env_keys` are miner-provided variable names only. | AVAILABLE. |
| Submission list/count/detail | `/challenges/agent-challenge/submissions/count` | `/submissions/count` | GET | None | Show aggregate submission count for page stats. | `count`. | Empty: `count: 0` means no submissions yet. Loading: load with list. Error: retry 502/5xx. | `submissions:count` | Count only; do not infer or expose hidden records. | AVAILABLE. |
| Submission list/count/detail | `/challenges/agent-challenge/submissions/{id}` | `/submissions/{id}` | GET | None | Show one public submission receipt or selected table row. | `id`, `miner_hotkey`, `name`, `agent_hash`, `zip_sha256`, `status`, `effective_status`, `score`, `submitted_at`, `created_at`, `latest_evaluation`, and redacted env action fields `env_action_required`, `env_keys`, `env_var_count`, `env_confirmed_empty`, `env_locked`, `env_updated_at`. | Empty: 404 means not known or not visible. Loading: load on detail open. Error: retry transient 502/5xx. | `submission:{id}` | Do not add signatures, private paths, raw analyzer reports, source snippets, worker lease state, env values, env ciphertext, env hashes, or env key file paths. `env_keys` are identifiers, not values. | AVAILABLE. |
| Submission list/count/detail | `/v1/challenges/agent-challenge/submissions/{id}` | `GET /v1/submissions/{id}` | GET | None through BASE bridge helper | Compatibility status lookup used by existing BASE bridge helper. Prefer canonical read base for new UI reads. | Same safe detail fields as `/submissions/{id}`. | Empty: 404 means missing submission. Loading: use only when client is tied to bridge helper. Error: retry transient 502/5xx. | `submission-bridge:{id}` | Do not expose bridge auth tokens, internal lookup details, signatures, private paths, raw analyzer reports, or source snippets. | AVAILABLE. BASE bridge helper can read Agent Challenge `GET /v1/submissions/{id}`. |
| Submission status | `/challenges/agent-challenge/submissions/{id}/status` | `/submissions/{id}/status` | GET | None | Poll public lifecycle snapshot for one submission. | `submission_id`, `agent_hash`, `status`, `public_state`, `phase`, `effective_status`, `env_action_required`, `env_keys`, `env_var_count`, `env_confirmed_empty`, `env_locked`, `env_updated_at`, `last_event_id`, `last_event_sequence`, `current_attempt`, `analyzer`, `similarity`, `evaluation`, `terminal_bench`, `progress`, `submitted_at`, `updated_at`. Under dual-flag attestation mode (`phala_attestation_enabled` + `attested_review_enabled`) also includes `review`: `session_id`, `assignment_id`, `attempt`, `phase`, `terminal`, `verdict`, `verified`, `retryable`, `reason_code`, `report_available`, `issued_at`, `finished_at`. Dual-flag `review.*` is independent of list/public lifecycle status (for example allow+verified may coexist with public `queued`). Prefer dual-flag fields for STATUS badges; do not remap product `PUBLIC_STATUS` as the sole badge source. **Dual-flag `evaluation.task_rows`:** when a latest `EvalRun` exists, `evaluation.task_rows` is projected from `EvalRun.plan_json.selected_tasks` (not from validator `EvaluationJob`, which stays `None` under dual flags). `eval_prepared` / `eval_expired` still return planned rows. When `canonical_score_record_json` is present, matching planned rows set `has_result` and phase/outcome; `score` / `passed_tasks` / `total_tasks` come from the EvalRun ledger (not hard-coded zeros). Empty `task-events` remains honest empty (no invented guest logs). | Empty: 404 means not known; nested nulls/zero counts mean pending. For `Waiting environments`, use the env action fields to render the miner env confirmation state. Under dual flags, empty `task_rows` only means no EvalRun plan yet (or empty plan), not "evaluation has not started" solely from list `queued`. Loading: poll only when SSE is unavailable or reconnecting. Error: back off on 502/5xx; stop on terminal states. | `submission-status:{id}` | Do not expose raw analyzer JSON, raw LLM prompts/responses, raw source, provider errors, signatures, private job dirs, broker refs, worker lease owners, env values, env ciphertext, env hashes, or env key **** paths. `env_keys` are miner-provided names only. Never expose review nonce plaintext, session tokens, capabilities, evidence bodies, or model IO on status. Do not invent Terminal-Bench task rows or guest log bodies when plan/score/events are absent. | AVAILABLE. |
| Public TEE math | `/challenges/agent-challenge/submissions/{id}/review/tee` | `/submissions/{id}/review/tee` (v1 alias `/v1/submissions/{id}/review/tee`) | GET | None | Independently inspectable TEE math after a verified review report exists. Same public trust class as status/events (no miner signature). | When no report: HTTP **200** body exactly `{"available": false}`. When available: `available: true`, `submission_id`, `domain`, `review_digest`, `report_data_hex`, `report_data_preimage` (**without** raw `review_nonce`; may include `review_nonce_sha256`), `measurement` (`mrtd`, `rtmr0`–`rtmr3`, `compose_hash`, `os_image_hash`, `key_provider`, `vm_shape`), size-capped `tdx_quote_hex`, size-capped `event_log`, `verification_outcome` public subset (`status`, `measurement_allowlisted`, `report_data_matched`, `verified_at_ms`, `reason_code`), `quote_fingerprint_sha256`, and optional cross-check digests (`agent_hash`, `zip_sha256`, `verdict`, `assignment_digest`, `session_id`, `assignment_id`). | Empty: `available: false` means no durable report yet (not an error). Loading: fetch when status `review.report_available` is true, or always fetch and honor closed form. Error: 404 only for unknown submission; retry 502/5xx. | `submission-tee:{id}` | **Never** expose nonce plaintext, session/bearer tokens, capabilities, evidence bodies, model request/response IO, encryption KEY material, wallets, or internal `/internal/...` envelope fields. Miner signed `GET .../review/report` remains a separate redacted surface. | AVAILABLE. Requires BASE proxy allowlist for the public GET path. |
| Submission status | No BASE bridge status suffix yet; use canonical read route above | `GET /v1/submissions/{id}/status` | GET | None through future bridge helper if added | Symmetric Agent Challenge v1 alias for bridge status consumers. | Same safe status fields as `/submissions/{id}/status`. | Empty: 404 means missing submission. Loading: prefer canonical read route until bridge helper exists. Error: retry transient 502/5xx. | `submission-status-bridge:{id}` | Same as polling status route; keep summarized fields only. | AVAILABLE in Agent Challenge as a v1 alias. Prefer the canonical read route for new frontend code. |
| SSE | `/challenges/agent-challenge/submissions/{id}/events` | `/submissions/{id}/events` | GET | None | Stream public status events and reconnect from `Last-Event-ID`. | Event frames: durable `id`, `event: submission.status`, JSON `id`, `sequence`, `submission_id`, `status`, `public_state`, `phase`, `created_at`, optional allowlisted `reason_code` such as `waiting_miner_env`, optional safe `actor`; 409 replay conflict returns `detail` and `replay_from`. | Empty: stream may close after terminal event; 404 means missing submission. Loading: use EventSource or fetch-stream and persist last id. On `waiting_miner_env`, invalidate/refetch status or detail to read redacted env action metadata. Error: reconnect with `Last-Event-ID`; on 409 fetch status and restart from `replay_from` when available. | `submission-events:{id}:last-event:{last_event_id}` | Do not expose tokens, signatures, source, raw analyzer evidence, raw LLM transcripts, raw Terminal-Bench artifacts, private paths, broker refs, lease owners, env values, env ciphertext, env hashes, or env key file paths. SSE events remain allowlisted and do not carry env metadata. | AVAILABLE through Agent Challenge and the BASE proxy. |
| Miner env read | `/challenges/agent-challenge/submissions/{id}/env` | `/submissions/{id}/env` | GET | Signed miner env action | Read redacted env metadata for a waiting or locked submission owned by the signed miner. | `submission_id`, `env_keys`, `env_var_count`, `env_confirmed_empty`, `env_locked`, `env_updated_at`, and optional safe status fields. | Empty: no keys plus `env_confirmed_empty: false` means action is still needed. Error: 401 signature failure, 403 owner mismatch, 404 missing submission, 409 wrong lifecycle. | `submission-env:{id}` | Metadata only. Never expose env values, ciphertext, hashes, request body, key file paths, runtime injected values, signatures, or nonces. | AVAILABLE through Agent Challenge and the BASE proxy. |
| Miner env save | `/challenges/agent-challenge/submissions/{id}/env` | `/submissions/{id}/env` | PUT | Signed miner env action | Replace the full env set while the submission is in `Waiting environments`. | Metadata only: `submission_id`, `env_keys`, `env_var_count`, `env_confirmed_empty`, `env_locked`, `env_updated_at`. | Empty: `{}` is not a launch confirmation; use confirm-empty for zero-env submissions. Error: 400 or 422 validation, 401 signature failure, 403 owner mismatch, 409 locked or wrong lifecycle, 413 payload too large. | `submission-env-save:{id}:{nonce}` then `submission-env:{id}` | Body values are write-only and must be cleared from frontend state after submit. Never persist or render submitted values. | AVAILABLE through Agent Challenge and the BASE proxy. |
| Miner env confirm empty | `/challenges/agent-challenge/submissions/{id}/env/confirm-empty` | `/submissions/{id}/env/confirm-empty` | POST | Signed miner env action | Explicitly confirm that no env vars are needed so zero-env submissions do not get stuck. | Metadata only with `env_confirmed_empty: true`, count, keys, lock state, and timestamp. | Empty: no body required except the signed empty JSON or raw empty body chosen by the client signer. Error: 401 signature failure, 403 owner mismatch, 409 locked or wrong lifecycle. | `submission-env-confirm-empty:{id}:{nonce}` then `submission-env:{id}` | Do not send or display env values on this route. | AVAILABLE through Agent Challenge and the BASE proxy. |
| Miner launch | `/challenges/agent-challenge/submissions/{id}/launch` | `/submissions/{id}/launch` | POST | Signed miner env action | Return launch state for the lifecycle `analysis_allowed -> waiting_miner_env -> tb_queued -> tb_running`; when a queued or running job already exists, return it idempotently without duplicating it. | Metadata-only launch receipt with submission id, safe status, keys/count, empty confirmation, lock state, and updated timestamp. | Empty: requires saved env vars or prior confirm-empty unless an existing queued or running job is present. Error: 401 signature failure, 403 owner mismatch, 409 missing env action, locked without a queued or running job, or wrong lifecycle. | `submission-launch:{id}:{nonce}` then `submission-status:{id}` | Do not expose env values, ciphertext, hashes, injected runtime env, Harbor job paths, broker refs, own_runner provider refs, Swarm service or task names, raw refs, or signatures. | AVAILABLE through Agent Challenge and the BASE proxy. |
| Analyzer | `/challenges/agent-challenge/submissions/{id}/status` | `/submissions/{id}/status` | GET | None | Render AST review, LLM review, LLM standby, and analyzer verdict summary from the status snapshot. | `analyzer.phase`, `analyzer.status`, `analyzer.verdict`, `analyzer.reason_codes`, `analyzer.llm_verdict`, `analyzer.llm_confidence`, `analyzer.llm_reason_codes`, `analyzer.started_at`, `analyzer.finished_at`. | Empty: `pending` or null fields mean waiting for analyzer. `LLM standby` means the provider path is retryable when configuration becomes available. Loading: reuse status cache and SSE invalidation. Error: same as status route. | `submission-status:{id}:analyzer` | Do not expose raw AnalyzerRun rows, raw analyzer reports, raw AST features, source snippets, prompt text, provider request/response bodies, or provider errors. | AVAILABLE through status route. |
| Similarity | `/challenges/agent-challenge/submissions/{id}/status` | `/submissions/{id}/status` | GET | None | Render AST similarity summary without source evidence. | `similarity.max_score_percent`, `similarity.match_count`, `similarity.top_matches` with `matched_submission_id`, `match_kind`, `score_percent`, `risk_band`. | Empty: `match_count: 0` and null max score means no matches. Loading: reuse status cache and SSE invalidation. Error: same as status route. | `submission-status:{id}:similarity` | Do not expose raw similarity evidence, file paths, source text, AST dumps, or matched snippets. | AVAILABLE through status route. |
| Terminal-Bench | `/challenges/agent-challenge/submissions/{id}/status` | `/submissions/{id}/status` | GET | None | Render Terminal-Bench progress counts for the current durable attempt. | `terminal_bench.total_trials`, `completed_trials`, `failed_trials`, `errored_trials`, `final_trials`, plus `evaluation.current_attempt` and `evaluation.attempt_status`. | Empty: all counts 0 means evaluation has not started or has no trials. Loading: refresh via SSE; polling fallback backs off during long runs. Error: same as status route. | `submission-status:{id}:terminal-bench` | Do not expose raw logs, job directories, container IDs, broker refs, external refs, private artifact paths, or raw Terminal-Bench artifacts. | AVAILABLE through status route. |
| Evaluation task details | `/challenges/agent-challenge/agents/{agent_hash}/evaluation` | `/agents/{agent_hash}/evaluation` | GET | None | Show latest evaluation details for an agent hash, including public per-task results. | `job_id`, `agent_hash`, `zip_sha256`, `status`, `effective_status`, `score`, `passed_tasks`, `total_tasks`, `verdict`, `rules_version`, `created_at`, `started_at`, `finished_at`, `tasks` with `task_id`, `docker_image`, `status`, `score`, `returncode`, `duration_seconds`. | Empty: 404 means no evaluation exists yet. Loading: load after `agent_hash` is known; refresh on status/SSE progress. Error: retry 502/5xx. | `agent-evaluation:{agent_hash}` | Do not expose task logs, raw stdout/stderr, workspace paths, patch contents, source snippets, provider credentials, or raw artifacts. | AVAILABLE. |
| Leaderboard | `/challenges/agent-challenge/leaderboard` | `/leaderboard` | GET | None | Show best scoring public row per miner hotkey. v1 returns one best scoring row per hotkey. Pagination/filter/sort are deferred to future v2. | Array of `miner_hotkey`, `agent_hash`, `score`, `passed_tasks`, `total_tasks`. | Empty: no valid completed scoring submissions yet. Loading: load on page entry and refresh after terminal valid statuses. Error: retry 502/5xx. | `leaderboard:agent-challenge:v1` | Do not expose excluded submissions, internal weight maps, private validator notes, raw scoring evidence, worker leases, or broker refs. | AVAILABLE. |
| Scoring | Static docs linked from hero metadata, plus `/challenges/agent-challenge/benchmarks` and `/challenges/agent-challenge/benchmarks/tasks` for live config | `/benchmarks`, `/benchmarks/tasks` | GET for live API data | None | Explain upload rules, analyzer gate, Terminal-Bench scoring, retry policy, and leaderboard eligibility. | Live benchmark fields plus static docs text for score formula, effective-status eligibility, ZIP limit, rate limit, SSE reconnect, and redaction policy. | Empty: if docs link is absent, show inline rules text and live benchmark data. Loading: cache static docs and refresh live benchmark metadata with page data. Error: retry live data 502/5xx; keep static copy. | `docs:agent-challenge:scoring-rules` | Docs and UI must not include real hostnames, bearer tokens, gateway tokens, mnemonics, DB URLs, signatures, private paths, raw LLM transcripts, raw analyzer reports, source snippets, worker lease owners, broker refs, or Terminal-Bench raw artifacts. | PARTIAL. Existing miner and validator docs cover challenge-local behavior. Task 7 updates canonical frontend wording. |

## Internal Launch Boundary

The centralized `POST /internal/v1/submissions/{submission_id}/launch` bridge route has been removed and returns 404; it never drives execution. Execution is driven by the validator worker pull loop that claims queued jobs. Frontend and miner clients keep using signed public env routes and `POST /submissions/{id}/launch` when they need idempotent launch state. BASE must also avoid exposing generic benchmark execution routes; the generic broker remains an internal execution substrate, not a public challenge API.

## Public Lifecycle Status Contract

Status and SSE surfaces use exact public copy and phases for the repaired lifecycle:

| Raw status | Public status copy | Public phase |
| --- | --- | --- |
| `analysis_queued` | `queued` | `queued` |
| `ast_running` | `AST review` | `ast_review` |
| `llm_running` | `LLM review` | `llm_review` |
| `llm_standby` | `LLM standby` | `llm_standby` |
| `analysis_allowed` | `queued` | `evaluation_queued` |
| `waiting_miner_env` | `Waiting environments` | `waiting_environments` |
| `tb_queued` | `evaluation queued` | `evaluation_queued` |
| `tb_running` | `evaluating` | `evaluation` |

The raw happy path is `analysis_queued -> ast_running -> llm_running -> analysis_allowed -> waiting_miner_env -> tb_queued -> tb_running`. A missing LLM gateway token, provider unavailable, rate limit, and timeout move to `llm_standby` with sanitized reason codes such as `missing_llm_gateway_token`, `llm_provider_unavailable`, `llm_provider_rate_limited`, and `llm_provider_timeout`. This does not create `LlmVerdict`, `EvaluationJob`, `AdminReviewDecision`, or weights. When gateway config becomes available, standby retries through `llm_standby -> analysis_queued`.

## Current Route Availability Summary

| Contract item | Status |
| --- | --- |
| `GET /challenges/agent-challenge/benchmarks` | AVAILABLE via generic proxy to `/benchmarks`. |
| `GET /challenges/agent-challenge/benchmarks/tasks` | AVAILABLE via generic proxy to `/benchmarks/tasks`. |
| `POST /challenges/agent-challenge/submissions` | AVAILABLE via generic proxy to `/submissions` for JSON base64 signed upload. |
| `GET /challenges/agent-challenge/submissions` | AVAILABLE via generic proxy to `/submissions`. Latest 100 newest-first. |
| `GET /challenges/agent-challenge/submissions/count` | AVAILABLE via generic proxy to `/submissions/count`. |
| `GET /challenges/agent-challenge/submissions/{id}` | AVAILABLE via generic proxy to `/submissions/{id}`. |
| `GET /challenges/agent-challenge/submissions/{id}/status` | AVAILABLE via generic proxy to `/submissions/{id}/status`. Dual-flag mode adds safe `review.*` fields. |
| `GET /challenges/agent-challenge/submissions/{id}/review/tee` | AVAILABLE via generic proxy to `/submissions/{id}/review/tee` once the BASE proxy allowlists the path. Locked closed form `{"available": false}` when no report. |
| `GET /challenges/agent-challenge/submissions/{id}/events` | AVAILABLE via generic proxy to `/submissions/{id}/events` and streams through BASE as `text/event-stream`. |
| `GET /challenges/agent-challenge/submissions/{id}/env` | AVAILABLE via generic proxy to `/submissions/{id}/env`; preserves only signed miner headers `X-Hotkey`, `X-Signature`, `X-Nonce`, and `X-Timestamp`. |
| `PUT /challenges/agent-challenge/submissions/{id}/env` | AVAILABLE via generic proxy to `/submissions/{id}/env`; request values are write-only and never stored by BASE. |
| `POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty` | AVAILABLE via generic proxy to `/submissions/{id}/env/confirm-empty`. |
| `POST /challenges/agent-challenge/submissions/{id}/launch` | AVAILABLE via generic proxy to `/submissions/{id}/launch`. |
| `GET /challenges/agent-challenge/agents/{agent_hash}/evaluation` | AVAILABLE via generic proxy to `/agents/{agent_hash}/evaluation`. |
| `GET /challenges/agent-challenge/leaderboard` | AVAILABLE via generic proxy to `/leaderboard`. Best scoring row per hotkey. |
| `POST /v1/challenges/agent-challenge/submissions` | AVAILABLE as the raw ZIP bridge to Agent Challenge `POST /internal/v1/bridge/submissions`. |
| `GET /v1/challenges/agent-challenge/submissions/{id}` | AVAILABLE as a BASE bridge helper backed by Agent Challenge `GET /v1/submissions/{id}`. |
| `GET /v1/submissions/{id}/status` | AVAILABLE in Agent Challenge as a v1 status alias. New frontend code should use `/challenges/agent-challenge/submissions/{id}/status`. |

## Benchmark Task Limit Contract

Each submitted agent or evaluation job can select at most 30 benchmark tasks, and the validator runs at most 30 task evaluations concurrently for that job. Defaults are `evaluation_task_count: 30` and `evaluation_concurrency: 4`; config values above 30 are rejected by settings validation or capped by runtime helpers for patched tests and stale job payloads. `harbor_n_concurrent` is separate and controls per-task Harbor behavior inside a Terminal-Bench run.

## Redaction Baseline

All public frontend responses and docs must stay source-safe and secret-safe. The frontend contract bans these values from public fields:

| Forbidden value | Rule |
| --- | --- |
| Tokens and keys | Never expose challenge tokens, broker tokens, bearer tokens, shared BASE tokens, gateway tokens, private keys, mnemonics, or database URLs. |
| Internal routes and hosts | Do not expose challenge internal hostnames, private proxy paths, secret file paths, private job dirs, or operator-only service URLs. |
| Submitted code | Do not expose raw ZIP contents, source snippets, AST dumps, patch files, or manifest-listed source text. |
| Analyzer and LLM internals | Do not expose raw analyzer reports, raw AnalyzerRun rows, prompts, provider request bodies, provider response bodies, raw transcripts, provider errors, or free-form internal reason text. |
| Similarity internals | Do not expose raw evidence JSON, source pair paths, AST match details, or matched snippets. |
| Terminal-Bench internals | Do not expose raw logs, stdout, stderr, workspace paths, job dirs, broker refs, external refs, own_runner provider refs, Swarm service or task names, raw refs, container IDs, tokens, or artifacts. |
| Worker and audit internals | Do not expose worker lease owners, nonce stores, request hashes, signatures, owner audit signatures, or admin-only review metadata. |
| Miner env action data | Public list, detail, status, and env metadata responses may expose only `env_action_required`, `env_keys`, `env_var_count`, `env_confirmed_empty`, `env_locked`, and `env_updated_at`. `env_keys` are miner-provided identifiers such as variable names; env values, ciphertext, hashes, request bodies, key file paths, and runtime injected values are never public. Env values are master-validator scoped, encrypted at rest by Agent Challenge, injected only into Harbor/Terminal-Bench runtime, and cannot be retrieved after submission. BASE registry and BASE proxy do not store per-submission env values. |

## Error Shape Guidance

Frontend copy should be based on HTTP class and safe `detail` fields only:

| Status | Frontend handling |
| --- | --- |
| 400 or 422 | Show validation problem for the current input. |
| 401 | Show signed request or bridge auth failure. |
| 403 | Show route not allowed or challenge unavailable to this caller. |
| 404 | Show missing submission, missing evaluation, or inactive challenge. |
| 409 | Show nonce replay, duplicate agent hash, or SSE replay conflict. |
| 413 | Show ZIP too large. |
| 429 | Show submission rate limit and `next_allowed_at` when provided. |
| 502 or 5xx | Show temporary challenge unavailable state and retry with backoff. Use safe copy, not raw text such as `BASE request failed with status 502`. |

## v1 List Semantics

`/challenges/agent-challenge/submissions` maps to challenge `/submissions` and returns the latest 100 submissions newest-first, ordered by `created_at` descending.

`/challenges/agent-challenge/leaderboard` maps to challenge `/leaderboard` and returns one best scoring row per hotkey. Rows are selected from scoring-eligible submissions only.

Pagination, filtering, and client-selected sorting are deferred to future v2.


## Miner Env Contract

The exact env path shorthand is `GET/PUT /challenges/agent-challenge/submissions/{id}/env` through BASE and `GET/PUT /submissions/{id}/env` locally. The explicit zero-env and launch paths are `POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty`, `POST /challenges/agent-challenge/submissions/{id}/launch`, `POST /submissions/{id}/env/confirm-empty`, and `POST /submissions/{id}/launch`.

Signed miner env requests use fake placeholders only in docs and examples:

```http
X-Hotkey: <miner-hotkey>
X-Signature: <signature>
X-Nonce: <nonce>
X-Timestamp: <timestamp>
```

Env keys must match `^[A-Za-z_][A-Za-z0-9_]{0,127}$`. A request can contain at most 64 keys, each value is at most 16 KiB, and the total payload is at most 128 KiB. Values are write-only. `GET /submissions/{id}/env` and all write responses return metadata only. The frontend must clear value inputs after save, show saved keys only, and treat `env_locked: true` as final.

`POST /submissions/{id}/env/confirm-empty` is required for zero-env submissions. Saving an empty map is not the same as confirming empty. `PUT /submissions/{id}/env` and `POST /submissions/{id}/env/confirm-empty` on `Waiting environments` lock/env-ready and enqueue exactly once. Repeat writes or repeated empty confirmation after lock return a conflict. `POST /submissions/{id}/launch` returns an existing queued or running job idempotently without duplicating it.

## 502 Runbook For Frontend And Operators

A 502 from `/challenges/agent-challenge/...` is a BASE proxy unavailable state. Frontend copy should say the challenge is temporarily unavailable and should retry with backoff. It must not render raw text such as `BASE request failed with status 502`.

Troubleshooting checklist:

1. Check ingress includes `/challenges`, not only `/v1/challenges`, and routes it to the BASE proxy.
2. Check BASE proxy path handling for the challenge slug and confirm `/internal/*`, `/health`, and `/version` remain blocked.
3. Check Agent Challenge Swarm service health (`docker service ps challenge-agent-challenge`), overlay DNS (`tasks.challenge-agent-challenge`), and the published port from inside the `base_challenges` overlay.
4. Check the Swarm task placement and health: confirm replicas are running, not restarting or unscheduled for missing capacity, with `docker service ps challenge-agent-challenge`.
5. Split transport failures from challenge-origin non-2xx responses. BASE should convert transport failures to safe 502 responses, but pass through challenge-origin non-2xx responses with safe fields.
6. For env routes, confirm the proxy forwards `X-Hotkey`, `X-Signature`, `X-Nonce`, and `X-Timestamp` only on the allowed Agent Challenge env and launch paths. BASE must not parse, persist, log, or registry-serialize submitted env values.
