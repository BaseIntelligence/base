> OpenAPI: https://chain.joinbase.ai/challenges/agent-challenge/openapi.json · Day-1: root docs/miner/getting-started.md

# Miner self-deploy (How-to advanced)

> **Day-1 front door is not this page.** Upload first via the joinbase dashboard
> and/or [`packages/challenges/agent-challenge/scripts/submit_agent.py`](../../../scripts/submit_agent.py): see
> [Getting started](getting-started.md). Concepts for TEE trust:
> [Attestation TEE](attestation-tee.md).

**Production scoring is miner self-deploy on Phala Intel TDX CVMs.** You fund and
operate the attested review CVM and, after a verified allow, the attested eval CVM.
The validator/subnet keeps the trust root: measurement allowlist, golden
key-release endpoint, and quote verification. Validators do **not** deploy your
production scored jobs for you.

### Agent-driven gate (required before eval)

Eval is **agent-driven**. The product order is hard:

1. **Package + LLM rules residual** — measured review under harness / `.rules` must **allow**.
2. **`package_tree_sha` proof** — canonical folder-tree SHA of the extracted agent package,
   stored with the submission and bound into the eval plan and review materials.
3. **TEE auth** — fresh re-verified review allow whose authorizing materials bind residual +
   tree SHA (dual flags ON). Host-static analyzer alone is **not** enough.
4. **Only then** — `eval prepare` / deploy / key-release / trials / score attestation.

If residual fails or tree SHA is missing/mismatched: prepare, KR, and score **refuse**.
There is no closed agent-model catalog; personal finetunes are banned. Concepts:
[Attestation TEE — agent-driven order](attestation-tee.md#agent-driven-order-package-verify--tree-sha--tee--eval).

The mission is **CPU Intel TDX only** (no GPU) with a hard **$20** spend cap and a
preference for the smallest CPU shape that works (`tdx.small`/`tdx.medium`). GPU
targets, over-cap shapes, and missing Phala credentials are refused **before** any
Phala call.

Invoke the CLI with:

```
python -m agent_challenge.selfdeploy <subcommand> [options]
```

Run `python -m agent_challenge.selfdeploy --help` (or `<subcommand> --help`) for
the full option list.

## Production flags and credentials

Production requires both feature flags ON on the challenge service:

- `phala_attestation_enabled` / `CHALLENGE_PHALA_ATTESTATION_ENABLED`
- `attested_review_enabled`

Only an explicit `deploy` (without `--dry-run`) or `teardown` reaches Phala for
spend. Dry-runs, unit tests, and offline tooling never create CVMs.

### Offline compatibility

Flag-off / mixed production settings are closed for production scoring. Running
both flags **false** remains a local test and offline compatibility path only
(no Phala spend, no production scores). That offline path is **not** a supported
production scored deployment and is not the default product narrative.

Provide your Phala credential through the `PHALA_CLOUD_API_KEY` environment
variable only. Never write the key into a file you commit, and never paste it into
these docs or a compose file — the CLI reads it from the environment and never
prints it.

## Ordered attested lifecycle

Full attested production mode is a miner-driven two-stage lifecycle. A signed
submission creates at most the durable review session and assignment; it does not
create a CVM or spend Phala credits. The validator must return a verified
`allow` before `eval prepare` can authorize an eval run. A review `reject`,
`escalate`, expiry, cancellation, provider failure, or attestation failure never
creates benchmark work, a score, or a weight.

The ordered top-level commands are `review` and `eval`. Their exact stages are:

```text
review prepare, deploy, deployed, result, history, cancel, retry, teardown
eval prepare, deploy, result, status, cancel, retry, failure, teardown
```

The CLI's `review deploy` and `eval deploy` commands fetch the current signed
prepare response and immediately consume the one-time capability. Use those
commands directly for deployment. A standalone `review prepare` or `eval
prepare` is for a custom client that receives and immediately encrypts the
one-time capability; the CLI redacts it from output and persisted JSON, so a
redacted prepare file cannot be used as a deployment credential. Repeated prepare
requests return the existing assignment or plan without delivering the capability
again.

### `review prepare`

Request the immutable review assignment over the signed miner route. The response
contains the assignment and delivers `REVIEW_SESSION_TOKEN` at most once. The CLI
does not print or persist the token.

```bash
python -m agent_challenge.selfdeploy review prepare \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --signature <signature> \
    --nonce <unique-nonce> \
    --timestamp <timestamp>
```

### `review deploy`

Fetch the current assignment, encrypt exactly `OPENROUTER_API_KEY` and
`REVIEW_SESSION_TOKEN` to the validator-pinned review app KMS key, and send the
ciphertext plus `env_keys` in the Phala create request. The plaintext values are
never put in compose bytes, ordinary environment, arguments, logs, reports, or
the eval CVM. Signing accepts either `--auto-sign` (uses the configured miner
signing key from `MINER_HOTKEY_MNEMONIC` / `MINER_HOTKEY_URI`) or the explicit
header path (`--signature` and `--nonce`, with optional `--timestamp`).

Auto-sign form:

```bash
python -m agent_challenge.selfdeploy review deploy \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --openrouter-key-env OPENROUTER_API_KEY \
    --phala-api https://cloud-api.phala.com/api/v1 \
    --review-instance-type tdx.small \
    --eval-instance-type tdx.small \
    --review-runtime-hours 6 \
    --eval-runtime-hours 6 \
    --money-cap-usd 20
```

Explicit-signature form (every required argument shown; no `--auto-sign`):

```bash
python -m agent_challenge.selfdeploy review deploy \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --signature <signature> \
    --nonce <unique-nonce> \
    --timestamp <timestamp> \
    --openrouter-key-env OPENROUTER_API_KEY \
    --phala-api https://cloud-api.phala.com/api/v1 \
    --review-instance-type tdx.small \
    --eval-instance-type tdx.small \
    --review-runtime-hours 6 \
    --eval-runtime-hours 6 \
    --money-cap-usd 20
```

Provisioning is a two-request sequence, `POST /cvms/provision` followed by
`POST /cvms`. The create body contains the validator-returned `compose_hash`,
non-empty `encrypted_env`, and the exact `env_keys`. A dry run validates the
assignment and prints names, digests, measurement metadata, and projected cost,
but sends no Phala create request. Dry-run verification of the assignment's
bound allowlist reports a real `IN-LIST`/`NOT-IN-LIST` result or `UNKNOWN` when
no allowlist is present; it never fabricates allowlist membership:

```bash
python -m agent_challenge.selfdeploy review deploy \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --dry-run
```

A post-create failure (ack rejection, interrupted create, or budget/identical
failure after provision) deletes the attributable review CVM before returning.
The CLI does not leave residual funded CVMs after a failed stage.

### `review deployed`

Record the Phala create receipt and CVM identity on the signed route. This is
deployment bookkeeping only, not attestation evidence or an authorization
override.

```bash
python -m agent_challenge.selfdeploy review deployed \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --acknowledgement ./review-acknowledgement.json
```

### `review result`

Read the safe, redacted public review audit bundle. It includes attempt identity,
phase, terminal and retryable state, bounded reason code, timestamps, and safe
report projection digests. It does not include the session token, nonce value,
raw model request or response, unrestricted source, or internal evidence.
Only digests and bounded status metadata are exposed.
Before a report projection exists, the route returns
`review_report_not_available` (HTTP 404). With `--auto-sign`, the CLI signs the
exact request including any query string (`canonical_request_string`); use
`--cursor` when continuing a paged report read.

For **unauthenticated** independent inspection of measurements, quote hex,
`report_data_hex`, and verification outcome (joinbase math panel / operator
probes), use public `GET /submissions/{id}/review/tee` instead. That route does
not require miner signatures; when no report exists it returns HTTP 200
`{"available": false}`. See [Attestation TEE](attestation-tee.md#public-tee-math-joinbase--self-deploy-inspection).

```bash
python -m agent_challenge.selfdeploy review result \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --cursor <next-cursor>
```

### `review history`

Read all retained review attempts in stable cursor order. Cancelled, expired,
failed, rejected, and superseded attempts remain visible; retries do not hide
prior history. The default page is 10 items and the maximum is 16. Responses
include `next_cursor` and `total_count` when more attempts exist. Auto-sign binds
the exact query string (`?cursor=...`) so the signature matches the server
`canonical_request_string`.

```bash
python -m agent_challenge.selfdeploy review history \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --cursor <next-cursor>
```

### `review cancel`

Cancel only the expected active assignment. Cancellation revokes its capability
and nonce and must be followed by teardown of the attributable review CVM.

```bash
python -m agent_challenge.selfdeploy review cancel \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --assignment-id assignment-1
```

### `review retry`

Retry only the expected terminal retryable assignment. Ordinary cancel/expiry
retries omit the optional approval field. A rejected or escalated review also
requires the validator's one-use operator `approval_id`, which the server
atomically consumes. Retry creates a fresh assignment, nonce, TTL, and capability
while retaining prior attempts.

Ordinary (no approval) retry:

```bash
python -m agent_challenge.selfdeploy review retry \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --assignment-id assignment-1
```

Approval-backed retry after reject/escalate (every required argument shown):

```bash
python -m agent_challenge.selfdeploy review retry \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --assignment-id assignment-1 \
    --approval-id <one-use-operator-approval-id>
```

### `review teardown`

Delete the review CVM after allow, reject, expiry, cancellation, provider
failure, verification failure, or interruption. The CLI uses the exact
`phala cvms delete <id> -f` operation. If deletion fails, the command exits
non-zero and prints only size-bounded diagnostics (never secrets or full remote
logs).

```bash
python -m agent_challenge.selfdeploy review teardown --cvm-id review-cvm-1
```

### `eval prepare`

After a verified review `allow`, request the immutable eval plan. The response
contains the selected task plan, exact image and compose identity, separate
key-release and score nonce identities, six-hour expiry, and delivers
`EVAL_RUN_TOKEN` at most once. It is forbidden before allow and returns
`review_allow_required` (HTTP 403); no task plan or eval CVM exists on rejection.

```bash
python -m agent_challenge.selfdeploy eval prepare \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --signature <signature> \
    --nonce <unique-nonce> \
    --timestamp <timestamp>
```

### `eval deploy`

Fetch the current authorized plan and encrypt the scoped eval capabilities:
`EVAL_RUN_TOKEN`, `LLM_COST_LIMIT`, and the attestation plan fields.
**Base LLM gateway secrets are not required and must not be injected**
(`BASE_GATEWAY_TOKEN` / `BASE_LLM_GATEWAY_URL` are removed from
`EVAL_REQUIRED_SECRET_ENVS`). Measured OpenRouter keys for review stay on the
review CVM encrypted_env path only. The encrypted ciphertext and `env_keys` are
transmitted in the same `POST /cvms/provision` then `POST /cvms` sequence. The
eval app receives no review evidence or unrestricted report data. Pre-create
spend projection counts both the review and eval stage shapes against the shared
money cap. Signing again accepts either `--auto-sign` or explicit `--signature`
+ `--nonce` (+ optional `--timestamp`).

```bash
python -m agent_challenge.selfdeploy eval deploy \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --llm-cost-limit-env LLM_COST_LIMIT \
    --phala-api https://cloud-api.phala.com/api/v1 \
    --eval-instance-type tdx.small \
    --money-cap-usd 20
```

Use `--dry-run` to validate the signed plan and show only safe names, digests,
measurement metadata, and cost. Without a validator eval allowlist the dry-run
allowlist field is `UNKNOWN` (never fabricated `IN-LIST`):

```bash
python -m agent_challenge.selfdeploy eval deploy \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --dry-run
```

A post-create failure deletes the attributable eval CVM before the command
returns.

### `eval result`

Post the exact CVM-emitted result bytes directly to
`POST /evaluation/v1/runs/{eval_run_id}/result`. The CLI reads the
`EVAL_RUN_TOKEN` from the named environment variable and sends it as a bearer
credential. The validator durably receipts the body before verification and
returns the safe receipt phase, body digest, terminal/retryable state, reason
code, and finalization time. It never accepts authentication without quote,
measurement, event-log, key-grant, nonce, and score binding checks.

```bash
python -m agent_challenge.selfdeploy eval result \
    --base-url https://<challenge-host> \
    --run-id eval-run-1 \
    --result ./eval-result.json \
    --token-env EVAL_RUN_TOKEN
```

An invalid, rejected, expired, or verifier-unavailable result produces no
accepted score. A conflicting receipt is a conflict, and a verifier outage is
retryable; neither consumes the score nonce as a successful result.

### `eval status`

Read the attempt-ordered eval history. Each item exposes only
`eval_run_id`, attempt and predecessor identity, receipt and body digest,
phase, terminal/verified/retryable flags, bounded `reason_code`, key-grant
state, key-release and score nonce states, issue/expiry/receipt/finalization
times, and `result_available`. It does not expose the run token, nonce values,
selected task contents, quote, or raw evidence. Pass `--cursor` to continue a
page; `--auto-sign` signs the exact query string so the signature matches the
server `canonical_request_string`.

```bash
python -m agent_challenge.selfdeploy eval status \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --cursor <next-cursor>
```

The default page is 10 items and the maximum is 16; responses include
`next_cursor` and `total_count`. `eval_prepared`,
`eval_running`, `eval_verifying`, `eval_expired`, `eval_cancelled`,
`eval_error`, `eval_rejected`, and `eval_accepted` are observable phases.
`eval_deploy_failed`, `eval_tunnel_failed`, `eval_key_release_unavailable`,
and `eval_no_result` are the only accepted pre-receipt failure reasons.

### `eval cancel`

Cancel only the expected active, pre-receipt, never-key-granted run. A delayed
or wrong run id is a conflict. Revoke the run and then delete its attributable
eval CVM.

```bash
python -m agent_challenge.selfdeploy eval cancel \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --run-id eval-run-1
```

### `eval retry`

Retry only a pre-receipt, pre-key-grant no-receipt, cancellation, or expiry
state. A key-granted, receipted, accepted, or permanently failed run cannot
retry. The new attempt receives fresh run, capability, TTL, key-release nonce,
and score nonce identities while the old history remains.

```bash
python -m agent_challenge.selfdeploy eval retry \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --run-id eval-run-1
```

### `eval failure`

Record one bounded pre-receipt failure when deployment, tunnel, key release, or
result delivery cannot complete. This cannot mutate a key-granted or receipted
run and returns the safe eval history item.

```bash
python -m agent_challenge.selfdeploy eval failure \
    --base-url https://<challenge-host> \
    --submission-id 1 \
    --hotkey <miner-hotkey> \
    --auto-sign \
    --run-id eval-run-1 \
    --reason-code eval_key_release_unavailable
```

### `eval teardown`

Delete the eval CVM after acceptance, rejection, expiry, cancellation, result
failure, or interruption. Deletion failure exits non-zero and returns
size-bounded diagnostics only.

```bash
python -m agent_challenge.selfdeploy eval teardown --cvm-id eval-cvm-1
```

## Raw-TCP RA-TLS key release

Production eval guests run on Phala Cloud Intel TDX. Inside the measured guest, dstack **GetTlsKey** materializes mTLS client cert material. The guest then dials the validator **raw TCP** RA-TLS listener (not public L7 HTTP `/release`) to obtain the validator AES-256 golden key used to decrypt oracle material. Server CA inject for verifying the KR listener is separate from the host client-trust CA that verifies guest certificates.

The eval plan's `key_release_endpoint` points to the validator's external raw
TCP tunnel, normally an authority ending in port `8701`. The production key
release listener requires TLS 1.3 and a dstack RA-TLS client certificate. It
validates the certificate's quote and event log and derives the peer SPKI
digest from the negotiated certificate; a caller-supplied peer header is not
trusted. Production eval compose provisions the listener target through
`KEY_RELEASE_RA_TLS_HOST` and `KEY_RELEASE_RA_TLS_PORT`. There is no HTTP
key-release fallback on the measured production path.

The client reads its mTLS files from
`CHALLENGE_PHALA_RA_TLS_CERT_FILE`,
`CHALLENGE_PHALA_RA_TLS_KEY_FILE`, and
`CHALLENGE_PHALA_RA_TLS_CA_FILE`. The raw exchange is one 4-byte big-endian
length followed by canonical JSON:

```json
{
  "schema_version": 1,
  "eval_run_id": "<eval-run-id>",
  "nonce": "<key-release-nonce>",
  "quote_hex": "<quote-hex>",
  "event_log": []
}
```

The response is a bounded canonical JSON frame with `released` and either
`key_b64` on success or a bounded `reason_code` on denial. The frame cap is
3 MiB, the TLS handshake deadline is 10 seconds, the total exchange deadline
is 30 seconds, and there is no HTTP status framing on this socket. HTTP
`/release` is not the production transport.

## Subcommands

The legacy top-level commands below remain useful for compatibility and offline
artifact/measurement checks. They do not replace the ordered review-before-eval
flow when attested mode is enabled.

### `prepare`

Fetch/prepare the canonical image + generated compose. Resolves the canonical
image to an immutable `repo@sha256:<digest>` reference (a floating tag such as
`:latest` is refused) and writes the deployable `app-compose.json`, which mounts
the dstack socket (`/var/run/dstack.sock`) and the guest Docker socket
(`/var/run/docker.sock`) and carries the operator-supplied validator key-release
endpoint.

```
python -m agent_challenge.selfdeploy prepare \
    --image ghcr.io/baseintelligence/agent-challenge-canonical@sha256:<digest> \
    --key-release-url https://validator.example/keyrelease \
    --out ./deploy
```

### `measurements`

Publish/reproduce the canonical measurement record
`{mrtd, rtmr0, rtmr1, rtmr2, compose_hash, os_image_hash}` deterministically, so
the miner and validator agree on the same allowlist entry. Requires the pinned
dstack image `metadata.json`, the VM shape (`--cpu`/`--memory`), and the compose.

```
python -m agent_challenge.selfdeploy measurements \
    --metadata ./metadata.json --cpu 1 --memory 2G --compose ./deploy/app-compose.json
```

### `verdict`

Report a measurement's canonical fields and whether it is **IN-LIST** or
**NOT-IN-LIST** against a validator-owned allowlist. The measurement can be given
directly (`--measurement`) or read from a captured run output (`--from-result`).

```
python -m agent_challenge.selfdeploy verdict \
    --measurement ./measurement.json --allowlist ./allowlist.json
```

### `deploy`

Deploy a CPU-only, miner-funded CVM. Absent `--instance-type`, the smallest CPU
shape (`tdx.small`) is chosen. A GPU instance type or GPU OS image is refused, and
a shape whose projected cost would breach the money cap is refused, both before
any provisioning. Use `--dry-run` to print the full plan (compose, image digest,
instance type, region, key-release endpoint, projected cost) and make zero
CVM-creating calls.

```
python -m agent_challenge.selfdeploy deploy \
    --image ghcr.io/baseintelligence/agent-challenge-canonical@sha256:<digest> \
    --key-release-url https://validator.example/keyrelease \
    --dry-run
```

Set `PHALA_CLOUD_API_KEY` before a real (non-dry-run) deploy; if it is unset the
command errors clearly without any Phala call and never prints the key.

### `run`

Run the eval against the validator key-release endpoint. The in-CVM backend
obtains the golden key from exactly that endpoint before scoring; if the endpoint
is unreachable or denies the quote, the run fails closed with a clear error and
produces **no** attested result or score.

```
python -m agent_challenge.selfdeploy run \
    --job-dir ./job --task <task-id> \
    --key-release-url https://validator.example/keyrelease
```

### `result`

Surface + verify the attested-result envelope from a captured run output: the TDX
quote, event log, `report_data`, the measurement block, and the per-task scores.
It recomputes `report_data` from the reported binding and confirms it equals the
quote's value (a tampered score/measurement/nonce fails the check). Pass
`--allowlist` to also report the measurement's allowlist verdict.

```
python -m agent_challenge.selfdeploy result --from ./run-output.txt
```

The command also reports a coarse, non-sensitive **acceptance verdict** so a
result the validator does not accept is surfaced to you, never silently dropped.
Acceptance is a conjunction of binding, quote, measurement, nonce, and key-grant
signals; a single failing check rejects the result. Fold in the validator's
checks with `--allowlist` (measurement), `--quote-verified true|false` (the
Phala verify / `dcap-qvl` verdict), and `--nonce-state
ok|stale|consumed|unknown` (the validator nonce-ledger verdict):

```
python -m agent_challenge.selfdeploy result --from ./run-output.txt \
    --allowlist ./allowlist.json --quote-verified true --nonce-state ok
```

When a result is not accepted the command exits non-zero and prints only
`{accepted: false, reason: <coarse>}` (never a score, quote, or secret). The
coarse reasons are: `attestation absent`, `attestation not verified`, `measurement
not allowlisted`, `nonce stale` (or `nonce already used` / `nonce not
recognized`), and `attestation binding mismatch`.

### `teardown`

Delete a deployed CVM so no resource is left running. Successful deletion of an
already-gone CVM still exits zero (idempotent). A real deletion failure exits
non-zero and returns size-bounded diagnostics only (returncode/stderr truncated,
never secrets or full remote logs).

```
python -m agent_challenge.selfdeploy teardown --cvm-id cvm-1
```

## Mandatory teardown and the money cap

Every CVM you deploy is **miner-funded** and must be deleted when you are done.
The total mission spend cap is **$20**; always use the smallest CPU shape that
works (`tdx.small`/`tdx.medium`) and never deploy a GPU CVM. Review and eval
spend are projected together before either create.

The `teardown` subcommand runs `phala cvms delete <id> -f` for you, but you can
also delete and confirm directly with the `phala` CLI:

```
phala cvms delete <id> -f
phala cvms list
```

After teardown, `phala cvms list` must report `total: 0` — the CVM is **deleted**,
not merely stopped. If you cannot confirm `total: 0`, delete the residual CVM
before ending the session.

Live deploy, run, and teardown against a real Phala CVM remain under the money
guardrails (smallest CPU shape, mandatory teardown to `total: 0`).

## Validator operations

The validator-operated trust root (measurement allowlist, golden key-release
endpoint, and quote verification) is documented in
[`docs/validator/self-deploy.md`](../../../packages/challenges/agent-challenge/docs/validator/self-deploy.md). The validator/master
integration lives in the separate base repository
([`BaseIntelligence/base`](https://github.com/BaseIntelligence/base) (available after PR merge)).
