# NO_PHALA mode (temporary host-local unattested execution)

**Status:** temporary operator escape hatch while Phala CVMs are disabled.
Safe to remove: one module (`evaluation/no_phala.py`) plus thin call-sites.

## What it does

When `NO_PHALA=true` (or `CHALLENGE_NO_PHALA=true`) on the **master** validator:

1. Benchmark/eval jobs run **on the master host** via the existing
   `own_runner` local Docker path (`evaluation/own_runner_backend.py`).
2. **No** Phala Cloud client call is made. `PhalaCloudClient` construction and
   eval/review `deploy()` refuse with `NoPhalaModeError`.
3. Result envelopes are **explicitly marked unattested**:
   - `attested: false` (hard-coded; cannot be set true by the runner path)
   - `attestation_status: "unattested"`
   - `execution_mode: "no_phala_host"`
   - any `execution_proof` / `attestation_binding` stripped
4. When the miner ZIP is available on disk, `guest_artifact_proof` records
   `expected_hash == download_hash == executed_hash` (SHA-256 of ZIP bytes).
   Mismatch fails closed.

## What it does **NOT** prove

| Claim | NO_PHALA |
|-------|----------|
| TEE / Intel TDX isolation | **No** |
| TDX quote / DCAP verification | **No** |
| RTMR / compose_hash / KMS digest bind | **No** |
| Guest measurement allowlist | **No** |
| That the host was not tampered with | **No** |

Scores produced in this mode are **host-trust only**. Do not treat them as
attested TEE results for production weight decisions that require attestation.

## Env vars and precedence

| Variable | Role |
|----------|------|
| `CHALLENGE_NO_PHALA` | Challenge-prefix form (pydantic-settings field `no_phala`) |
| `NO_PHALA` | Plain operator form on the master host |

**Precedence:**

1. If `CHALLENGE_NO_PHALA` is **set** in the environment → use it.
2. Else if `NO_PHALA` is **set** → use it.
3. Else → **off** (default).

Truthy tokens: `1`, `true`, `yes`, `on` (case-insensitive).  
Falsy tokens: `0`, `false`, `no`, `off`, empty.

**Never** inferred from a missing `PHALA_API_KEY` / failed Phala call.

## Contradiction (fail closed)

If `NO_PHALA` is on **and** either of:

- `CHALLENGE_PHALA_ATTESTATION_ENABLED=true`
- `CHALLENGE_ATTESTED_REVIEW_ENABLED=true`

…settings construction raises. Attested TEE path and host-local unattested
path are mutually exclusive. Turn attestation flags **off** before enabling
NO_PHALA.

## Enable on master

Operator override file (loaded by master entrypoint):

```bash
# /var/lib/base/challenges/agent-challenge/embed.env
NO_PHALA=true
CHALLENGE_PHALA_ATTESTATION_ENABLED=false
CHALLENGE_ATTESTED_REVIEW_ENABLED=false
```

Master runs:

```text
uvicorn agent_challenge.app:app --host 127.0.0.1 --port 18081
```

Restart the master (or AC child) after editing `embed.env`.

## Disable

```bash
# embed.env — remove NO_PHALA or set:
NO_PHALA=false
# unset CHALLENGE_NO_PHALA if present
```

Restart. Attested path is unchanged when NO_PHALA is off.

## Confirm live mode

```bash
curl -sS http://127.0.0.1:18081/health
# {"status":"ok","slug":"agent-challenge","version":"…",
#  "no_phala":true,"attestation_mode":"no_phala_host"}

curl -sS http://127.0.0.1:18081/version
# … "no_phala":true, "capabilities":[…,"no_phala_host"]
```

Startup logs a multi-line `CRITICAL` banner when the mode is active.

Via master proxy (if embedded):  
`GET /challenges/agent-challenge/health` — same fields.

## Code map (for removal)

| Path | Role |
|------|------|
| `src/agent_challenge/evaluation/no_phala.py` | **Single module** — mode resolve, mark, proof, refuse |
| `src/agent_challenge/sdk/config.py` | `no_phala` field + contradiction validator |
| `src/agent_challenge/sdk/schemas.py` | `/health` + `/version` fields |
| `src/agent_challenge/sdk/app_factory.py` | Startup banner + health/version wiring |
| `src/agent_challenge/selfdeploy/phala.py` | `PhalaCloudClient` refuse |
| `src/agent_challenge/selfdeploy/eval.py` / `review.py` | deploy refuse |
| `src/agent_challenge/evaluation/own_runner_backend.py` | Mark unattested on legacy emit when mode on |

The **attested** emit branch in `own_runner_backend._emit_job_result` is not
modified when NO_PHALA is off.
