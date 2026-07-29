> **T40 product path:** Phala TEE dual flags and CVM attestation are **removed**.
> This document describes the **host-trust unattested** product mode.
> Never claim TEE, tamper-proof execution, or independent verification.

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
CHALLENGE_NO_PHALA=true
CHALLENGE_PHALA_ATTESTATION_ENABLED=false
CHALLENGE_ATTESTED_REVIEW_ENABLED=false
# Analyzer LLM review via OpenRouter (NO_PHALA only; gateway path unchanged when off)
CHALLENGE_LLM_PROVIDER=openrouter
CHALLENGE_LLM_MODEL=x-ai/grok-4.5
# Never commit this key. Prefer env injection / secret file over embed.env on shared hosts.
CHALLENGE_OPENROUTER_API_KEY=…
# Optional USD ceiling for analyzer OpenRouter spend (fail closed when exceeded)
# CHALLENGE_LLM_COST_LIMIT_USD=2.0
```

Master runs:

```text
uvicorn agent_challenge.app:app --host 127.0.0.1 --port 18081
```

Restart the master (or AC child) after editing `embed.env`.

### Analyzer LLM provider (NO_PHALA)

| Variable | Role |
|----------|------|
| `CHALLENGE_LLM_PROVIDER` | `openrouter` (default under NO_PHALA) or `gateway` |
| `CHALLENGE_LLM_MODEL` | OpenRouter model id (default `x-ai/grok-4.5`) |
| `CHALLENGE_OPENROUTER_API_KEY` | OpenRouter bearer key (never log/commit) |
| `OPENROUTER_API_KEY` | Fallback env if challenge-prefixed key unset |
| `CHALLENGE_LLM_COST_LIMIT_USD` | Optional spend ceiling; fail closed when exceeded |

Key resolution order for OpenRouter: explicit settings field →
`CHALLENGE_OPENROUTER_API_KEY` → `OPENROUTER_API_KEY` →
`~/.local/share/opencode/auth.json` (`openrouter.key`) → small
`~/.factory/*.json` configs.

When `NO_PHALA` is **off**, the analyzer always uses the BASE LLM gateway
(`CHALLENGE_LLM_GATEWAY_*`); OpenRouter settings are ignored.

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
| `src/agent_challenge/analyzer/openrouter_review_provider.py` | OpenRouter analyzer provider + key resolve (NO_PHALA only) |
| `src/agent_challenge/analyzer/lifecycle.py` | Provider selection under NO_PHALA |

The **attested** emit branch in `own_runner_backend._emit_job_result` is not
modified when NO_PHALA is off.


## Full offline pipeline (master host)

With `NO_PHALA=true` and both attestation flags **off**, submit uses the
pre-Phala analysis chain (not Phala review CVMs):

```text
submit → analysis_queued → AST + LLM review → analysis_allowed
      → waiting_miner_env (until env confirm / PUT)
      → tb_queued → tb_running (own_runner on host Docker)
      → tb_completed → EvaluationJob.score
      → get_weights → authenticated raw-weight push (when master_base_url set)
```

Requirements on the master process:

| Need | Env / setting |
|------|----------------|
| Mode | `NO_PHALA=true` or `CHALLENGE_NO_PHALA=true` |
| Attestation off | `CHALLENGE_PHALA_ATTESTATION_ENABLED=false`, `CHALLENGE_ATTESTED_REVIEW_ENABLED=false` |
| Worker | `CHALLENGE_COMBINED_WORKER=true` **or** run `agent-challenge-worker` |
| LLM review | OpenRouter key + `CHALLENGE_LLM_PROVIDER=openrouter` (default under NO_PHALA); or gateway base URL + token if provider forced to `gateway` |
| Benchmark | Docker + own_runner task cache |
| Weight push | `CHALLENGE_MASTER_BASE_URL` + shared token (optional loop) |

**embed.env note:** master entrypoint only forwards keys matching
`CHALLENGE_` / `PHALA_` / … prefixes. Prefer `CHALLENGE_NO_PHALA=true` in
`/var/lib/base/challenges/agent-challenge/embed.env` so the child sees it.

Unattested provenance is visible on:

- result envelopes (`attested:false`, `attestation_status:unattested`, `execution_mode:no_phala_host`)
- `tb_completed` status-event metadata (same fields)
- CRITICAL log on every raw-weight push while the mode is on
- `/health` → `no_phala:true`, `attestation_mode:no_phala_host`
