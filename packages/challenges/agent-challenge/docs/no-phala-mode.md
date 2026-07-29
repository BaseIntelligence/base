# NO_PHALA / host-trust mode (current production product path)

> **Current production product path.** Phala TEE dual flags and CVM attestation are
> **not** used for live scoring. Agent Challenge production runs **host-trust
> unattested** execution on the challenge host (broker / own_runner).
>
> Never claim TEE, tamper-proof execution, or independent hardware verification
> for scores produced in this mode.
>
> Miner day-1: signed ZIP on [joinbase.ai](https://joinbase.ai). Walkthrough:
> [agent-challenge miner getting-started](https://github.com/BaseIntelligence/agent-challenge/blob/main/docs/miner/getting-started.md).
> Canonical short pin: [host-trust.md](host-trust.md).

## What it does

When `NO_PHALA=true` (or `CHALLENGE_NO_PHALA=true`) on the **master** / embedded
Agent Challenge process:

1. Benchmark/eval jobs run **on the challenge host** via the existing
   `own_runner` local Docker path (`evaluation/own_runner_backend.py`) and/or
   broker backend wiring used in production.
2. **No** Phala Cloud client call is made for production scoring. `PhalaCloudClient`
   construction and eval/review `deploy()` refuse with `NoPhalaModeError` when
   the mode is active.
3. Result envelopes are **explicitly marked unattested**:
   - `attested: false` (hard-coded; cannot be set true by the runner path)
   - `attestation_status: "unattested"`
   - `execution_mode: "no_phala_host"`
   - any `execution_proof` / `attestation_binding` stripped
4. When the miner ZIP is available on disk, `guest_artifact_proof` records
   `expected_hash == download_hash == executed_hash` (SHA-256 of ZIP bytes).
   Mismatch fails closed.
5. Integrity still uses `package_tree_sha` + AGATE residual (host residual kinds)
   where configured. That is **host-trust integrity**, not TEE attestation.

## What it does **NOT** prove

| Claim | Host-trust / NO_PHALA |
|-------|------------------------|
| TEE / Intel TDX isolation | **No** |
| TDX quote / DCAP verification | **No** |
| RTMR / compose_hash / KMS digest bind | **No** |
| Guest measurement allowlist | **No** |
| That the host was not tampered with | **No** |

Scores produced in this mode are **host-trust only**. Do not treat them as
attested TEE results. UI honesty may show **Unattested · Host trust**. STATUS on
joinbase is the submission **lifecycle**, not a TEE badge.

## Env vars and precedence

| Variable | Role |
|----------|------|
| `CHALLENGE_NO_PHALA` | Challenge-prefix form (pydantic-settings field `no_phala`) |
| `NO_PHALA` | Plain operator form on the master host |
| `CHALLENGE_UNATTESTED_EXECUTION` | Related unattested product switch (see settings / host-trust.md) |

**Precedence:**

1. If `CHALLENGE_NO_PHALA` is **set** in the environment → use it.
2. Else if `NO_PHALA` is **set** → use it.
3. Else → **off** (default in code; production operators set it **on**).

Truthy tokens: `1`, `true`, `yes`, `on` (case-insensitive).  
Falsy tokens: `0`, `false`, `no`, `off`, empty.

**Never** inferred from a missing `PHALA_API_KEY` / failed Phala call.

## Contradiction (fail closed)

If `NO_PHALA` is on **and** either of:

- `CHALLENGE_PHALA_ATTESTATION_ENABLED=true`
- `CHALLENGE_ATTESTED_REVIEW_ENABLED=true`

…settings construction raises. Attested TEE path and host-local unattested
path are mutually exclusive. Turn attestation flags **off** before enabling
NO_PHALA. Production scoring uses the unattested path with attestation flags off.

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
(`CHALLENGE_LLM_GATEWAY_*`); OpenRouter settings are ignored. Production product
path keeps NO_PHALA **on**.

## Disable (leaves production host-trust path)

```bash
# embed.env — remove NO_PHALA or set:
NO_PHALA=false
# unset CHALLENGE_NO_PHALA if present
```

Restart. Only do this for local experiments that intentionally leave host-trust
production mode. Do not re-enable Phala dual flags for “current prod” docs.

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
`GET /challenges/agent-challenge/health` — same fields (may be blocked on public proxy).

## Code map (for operators)

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

## Full host-trust pipeline (production)

With `NO_PHALA=true` and both attestation flags **off**, submit uses the
host-trust analysis chain (not Phala review CVMs):

```text
submit → analysis_queued → AST + LLM review → analysis_allowed
      → waiting_miner_env (until env confirm / PUT)
      → tb_queued → tb_running (own_runner / broker on host)
      → tb_completed → EvaluationJob.score
      → get_weights → authenticated raw-weight push (when master_base_url set)
```

Miner-facing CLI (agent-challenge repo):

```bash
python scripts/submit_agent.py build --agent-dir ./my-agent --out ./agent.zip
python scripts/submit_agent.py submit \
  --api-base https://chain.joinbase.ai/challenges/agent-challenge \
  --zip ./agent.zip --name "my-agent" --confirm-empty --watch
```

Requirements on the master process:

| Need | Env / setting |
|------|----------------|
| Mode | `NO_PHALA=true` or `CHALLENGE_NO_PHALA=true` |
| Attestation off | `CHALLENGE_PHALA_ATTESTATION_ENABLED=false`, `CHALLENGE_ATTESTED_REVIEW_ENABLED=false` |
| Worker | `CHALLENGE_COMBINED_WORKER=true` **or** run `agent-challenge-worker` |
| LLM review | OpenRouter key + `CHALLENGE_LLM_PROVIDER=openrouter` (default under NO_PHALA); or gateway base URL + token if provider forced to `gateway` |
| Benchmark | Docker + own_runner task cache / broker |
| Weight push | `CHALLENGE_MASTER_BASE_URL` + shared token (optional loop) |

**embed.env note:** master entrypoint only forwards keys matching
`CHALLENGE_` / `PHALA_` / … prefixes. Prefer `CHALLENGE_NO_PHALA=true` in
`/var/lib/base/challenges/agent-challenge/embed.env` so the child sees it.

Unattested provenance is visible on:

- result envelopes (`attested:false`, `attestation_status:unattested`, `execution_mode:no_phala_host`)
- `tb_completed` status-event metadata (same fields)
- CRITICAL log on every raw-weight push while the mode is on
- `/health` → `no_phala:true`, `attestation_mode:no_phala_host`
- joinbase UI honesty: **Unattested · Host trust**; STATUS = lifecycle
