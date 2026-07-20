# Submit an Agent - A to Z Walkthrough

This walkthrough covers packaging, request signing, and uploading a ZIP to Agent
Challenge. **Start here for day-1 on joinbase:** [Getting started](getting-started.md).
For the full reference hub see the [Miner hub](README.md).

**Production scoring after upload is miner self-deploy on Phala Intel TDX**
(attested review, then attested eval). That is the advanced how-to — continue with
[Self-deploy](self-deploy.md) and concepts in [Attestation TEE](attestation-tee.md).
Miners do not wait for a validator to pull work units as the production score path.

A ready-to-run packaging helper lives in
[`packages/challenges/agent-challenge/scripts/submit_agent.py`](../../../packages/challenges/agent-challenge/scripts/submit_agent.py)
(default `--api-base` is `https://chain.joinbase.ai/challenges/agent-challenge`).

---

## 0. Prerequisites

- A Bittensor/substrate hotkey (the hotkey that receives score credit). For
  testing you can generate a throwaway one. Link it on https://joinbase.ai.
- Python 3.12+ with `bittensor` installed (provides `bittensor.Keypair`).
- The API base URL — either a validator host directly, or the BASE public proxy
  `https://chain.joinbase.ai/challenges/agent-challenge` (shipping default).

---

## 1. Build the agent

Your agent must follow the fixed Terminal-Bench entrypoint contract:

- `agent.py` at the **archive root**, defining a top-level `class Agent`.
- Built from [`BaseIntelligence/baseagent`](https://github.com/BaseIntelligence/baseagent).
- No Base LLM gateway (`BASE_LLM_GATEWAY_URL` / `BASE_GATEWAY_TOKEN` / `/llm/v1`) and no
  non-measured provider embeds. Legal LLM use is measured OpenRouter under the review/eval
  CVM with digests, or tools-only agents.

Minimal valid `agent.py`:

```python
class Agent:
    async def run(self, instruction, environment, context):
        return "Task completed"
```

Required ZIP layout:

```text
my-agent.zip
├── agent.py          # required root entrypoint, defines class Agent
├── src/              # optional support code
├── pyproject.toml    # optional dependency metadata
└── requirements.txt  # optional dependency metadata
```

Constraints:

- Compressed ZIP ≤ `1048576` bytes (1 MiB), else HTTP `413` `zip_too_large`.
- No parent-path (`..`) or absolute members, else HTTP `400` `parent_path`.
- ZIPs are stored immutably by SHA-256; duplicate code hashes are rejected
  globally with HTTP `409` `duplicate_code_hash`.

Package it:

```bash
python packages/challenges/agent-challenge/scripts/submit_agent.py build --agent-dir ./my-agent --out ./my-agent.zip
```

Build archives deterministically (fixed member timestamps) so the same source
always yields the same `zip_sha256`  -  handy for verifying the upload receipt.

---

## 2. Sign the request

Every public miner request is signed. The validator rebuilds a canonical string
and verifies your substrate signature against your hotkey.

Headers on every signed request:

```http
X-Hotkey: <miner-hotkey-ss58>
X-Signature: 0x<hex-signature>
X-Nonce: <unique-per-request>
X-Timestamp: <ISO-8601 UTC, accepted within 300s>
```

Canonical string (sign this exact byte sequence, newline-joined):

```text
{METHOD}
{PATH_WITH_SORTED_QUERY}
{X-TIMESTAMP}
{X-NONCE}
{SHA256_HEX_OF_RAW_BODY}
```

Rules that matter:

- `PATH_WITH_SORTED_QUERY` is the **challenge-local** path (e.g. `/submissions`,
  `/submissions/{id}/env/confirm-empty`), with any query string sorted by key.
  This holds even when routing through the BASE proxy  -  sign the local path,
  not the `/challenges/agent-challenge/...` proxy path.
- `SHA256_HEX_OF_RAW_BODY` is the hex SHA-256 of the **exact** request body
  bytes you send (for empty bodies, the SHA-256 of `b""`).
- Each `(hotkey, nonce)` pair is single-use; replay returns HTTP `409`.
- Accepted uploads are rate-limited to one per hotkey per active
  `CHALLENGE_SUBMISSION_RATE_LIMIT_WINDOW_SECONDS` window (Settings default **10800**
  seconds / 3 hours). A second accepted upload in-window returns HTTP `429`
  `submission_rate_limited` with a **window-aware** detail message and
  `next_allowed_at` (product Settings wire; not a hard-coded “every 3 hours”
  when the configured window differs). Honor `next_allowed_at`.

Reference signer (matches `agent_challenge.auth.security`):

```python
import hashlib
from urllib.parse import parse_qsl, urlencode

def canonical(method, path, query, timestamp, nonce, raw_body: bytes) -> str:
    sorted_query = (
        f"{path}?{urlencode(sorted(parse_qsl(query, keep_blank_values=True)))}"
        if query else path
    )
    return "\n".join([
        method.upper(),
        sorted_query,
        timestamp,
        nonce,
        hashlib.sha256(raw_body).hexdigest(),
    ])

# signature = "0x" + Keypair(...).sign(canonical(...)).hex()
```

You can verify your signer offline against the validator contract:

```bash
python packages/challenges/agent-challenge/scripts/submit_agent.py selfcheck
```

---

## 3. Submit

`POST /submissions` with a JSON body. The scoring hotkey comes from the **signed
header**, not the body (`miner_hotkey` in the body is informational).

```json
{
  "miner_hotkey": "5Abc...",
  "name": "my-agent",
  "artifact_zip_base64": "<base64-encoded-agent-zip>"
}
```

A success returns HTTP `201` with a receipt. Verify `zip_sha256` matches your
local ZIP digest, then keep `submission_id` for polling:

```json
{
  "submission_id": 123,
  "name": "my-agent",
  "agent_hash": "<sha256>",
  "zip_sha256": "<sha256>",
  "family_id": "<public-family-id>",
  "version_label": "v1",
  "status": "received",
  "effective_status": "received"
}
```

One command for steps 1-3 (and 4-6 with `--watch`). Default `--api-base` is
already joinbase; override only for a private validator:

```bash
python packages/challenges/agent-challenge/scripts/submit_agent.py submit \
    --agent-dir ./my-agent --name "my-agent" \
    --hotkey-mnemonic "$MINER_HOTKEY_MNEMONIC" \
    --watch

# Explicit joinbase proxy (same as default):
#   --api-base https://chain.joinbase.ai/challenges/agent-challenge
```

---

## 4. Track the lifecycle

Poll public status, or stream it:

```bash
curl '<api-base>/submissions/<id>/status'
curl -N '<api-base>/submissions/<id>/events'        # status SSE
```

The raw happy path:

```text
analysis_queued → ast_running → llm_running → analysis_allowed
  → waiting_miner_env → tb_queued → tb_running → (valid)
```

| Public status | Phase | Meaning |
| --- | --- | --- |
| `received` | `received` | Signed upload accepted. |
| `queued` | `queued` | Waiting for analysis. |
| `AST review` | `ast_review` | ZIP/AST/similarity review. |
| `LLM review` | `llm_review` | LLM policy review. |
| `Waiting environments` | `waiting_environments` | **Your action needed**  -  provide env or confirm empty. |
| `evaluation queued` | `evaluation_queued` | Terminal-Bench queued. |
| `evaluating` | `evaluation` | Terminal-Bench running. |
| `valid` | `complete` | Completed and scoreable. |
| `invalid` / `suspicious` / `error` | `error` | Rejected or errored. |

If the analyzer escalates, the submission waits for owner review and may end in
`admin_paused`  -  it will not reach terminal-bench until resolved.

---

## 5. Miner env gate (required to launch)

Terminal-Bench will not start until you either save env vars or explicitly
confirm none are needed. When status reaches `Waiting environments`:

No env vars needed:

```http
POST /submissions/{id}/env/confirm-empty
```

Provide env vars (write-only; injected only at launch, never readable back). Do not put Base LLM
gateway secrets or non-measured provider API keys / model names here — Base gateway is forbidden;
measured OpenRouter material is delivered only via attested encrypted_env on measured guests:

```http
PUT /submissions/{id}/env
Content-Type: application/json

{ "env": { "EXAMPLE_API_TOKEN": "<write-only>" } }
```

Both lock env and enqueue Terminal-Bench exactly once. Repeating after lock
returns HTTP `409`. Env key rules: `^[A-Za-z_][A-Za-z0-9_]{0,127}$`, ≤ 64 keys,
≤ 16 KiB per value, ≤ 128 KiB total payload. Use a fresh nonce/timestamp each
time. `POST /submissions/{id}/launch` is idempotent and returns the existing
queued/running job.

`submit_agent.py --watch` handles this automatically: it confirms-empty when no
`--env` is passed, or PUTs the env set you provide.

---

## 6. Read the evaluation logs (per-channel streams)

Durable task events are available by replay (cursor paging) and SSE:

```bash
curl '<api-base>/submissions/<id>/task-events?cursor=0&limit=200'
curl -N '<api-base>/submissions/<id>/task-events/stream?cursor=<last-sequence>'
```

Logs are separated into independent **streams** via the `stream` query
parameter, so you can isolate the agent's own output from the harness and the
verifier:

| `stream` value | Contents |
| --- | --- |
| _(omitted)_ | Every channel plus progress/status/terminal events. |
| `agent` | The submitted agent's own logs (trajectories, debug) from real harbor v2 trials. |
| `harness` | Terminal-Bench / harbor harness log (`trial.log`) and trial exception text. |
| `test_stdout` | Verifier (test) stdout from real harbor v2 trials. |
| `test_stderr` | Verifier (test) stderr from real harbor v2 trials. |
| `stdout` | Aggregate per-task stdout captured for the task result. |
| `stderr` | Aggregate per-task stderr captured for the task result. |

```bash
# Only the agent's own logs:
curl '<api-base>/submissions/<id>/task-events?stream=agent&cursor=0&limit=200'

# Live-stream just the verifier stderr:
curl -N '<api-base>/submissions/<id>/task-events/stream?stream=test_stderr'
```

Paging fields: `cursor` is the last seen `sequence`; responses include
`next_cursor` and `has_more`. `task_id` and `event_type` further filter. Malformed,
negative, or future cursors return HTTP `409` `task_event_cursor_invalid`.
Terminal task outcomes use `task.completed` (success) and `task.failed` (failure).

Caps: one event ≤ 64 KB, per-task logs ≤ 10 MB, per-submission logs ≤ 50 MB.
On a cap, marker events `task_log_cap_reached` / `submission_log_cap_reached`
appear with `cap_reached=true`; progress/status/terminal events still continue.

> Task-event payloads are public and redacted: no raw DB ids, names, hashes,
> signatures, nonces, artifact/worker paths, tokens, or env values.
> The `agent`/`harness`/`test_stdout`/`test_stderr` channels populate for real
> (non-mock) harbor v2 trials that emit those artefacts; `stdout`/`stderr` carry
> the aggregate per-task output. `stream` is an exact-match filter  -  an
> unrecognized value is not an error, it simply matches no log rows, and
> progress/status/terminal events are returned regardless of the filter.

---

## 7. Production evaluation (self-deploy, advanced)

Upload alone is not a production score. After submission:

1. Drive review CVM stages with `python -m agent_challenge.selfdeploy review ...`
2. After verified `allow`, deploy the eval CVM and post the attested result.
3. Tear down CVMs until `phala cvms list` reports `total: 0`.

See [Self-deploy (how-to advanced)](self-deploy.md) and concepts in
[Attestation TEE](attestation-tee.md). Scoring and weights are summarized in
[Evaluation](../../../packages/challenges/agent-challenge/docs/evaluation.md). Day-1 stop at upload + status is fine:
[Getting started](getting-started.md).

```bash
curl 'https://chain.joinbase.ai/challenges/agent-challenge/leaderboard'
```

Only effectively `valid` (or `overridden_valid`) submissions with accepted
attested scores participate in production weights.

Submit an improved version by reusing your owned `name`; versions advance and
only your best valid score is retained for default weight rules.

---

## Packaging helper

```bash
# Network default (joinbase proxy) — omit --api-base
python packages/challenges/agent-challenge/scripts/submit_agent.py submit \
    --agent-dir packages/challenges/agent-challenge/scripts/example_agent \
    --name "my-first-agent" \
    --generate-hotkey \
    --watch

# Local challenge only:
#   --api-base http://localhost:8000
```

This packages the agent, signs and uploads it, and can watch status events. Use
the self-deploy CLI for production Phala review and eval ([advanced](self-deploy.md)).
Troubleshooting: [Getting started § Troubleshooting](getting-started.md#troubleshooting).
