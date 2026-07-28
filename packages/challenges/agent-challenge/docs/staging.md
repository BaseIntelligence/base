# Agent Challenge — local staging (real Phala)

One-command local validator stack with **real** Phala TDX CVMs, dual attestation
flags ON, and fail-closed measurement allowlists. Isolated from production
(port `127.0.0.1:18082`, named volume `ac-staging-data`).

## Prerequisites

- Docker + BuildKit
- `uv` workspace at monorepo root
- Phala Cloud API key (`~/.phala/config.json` profile `echobts-projects`, or
  `PHALA_CLOUD_API_KEY`)
- OpenRouter key for review (`OPENROUTER_API_KEY` or OpenCode auth.json)
- Public HTTPS reachability for CVM callbacks (script starts `cloudflared` tunnel)
- Host key-release RA-TLS on `0.0.0.0:8701` (script can start a staging KR)
- `dcap-qvl` on PATH or baked into the runtime image

**Do not** point this stack at production master. **Do not** leave CVMs running.

## Quick start

```bash
cd /work/baseintelligence/base   # or your monorepo root

# Full loop: build → up → submit → review CVM → review_allowed
#            → eval CVM → guest_artifact_proof → teardown owned CVMs
./packages/challenges/agent-challenge/scripts/staging/run_staging.sh

# Review only (still tears down review CVM this run owns)
./packages/challenges/agent-challenge/scripts/staging/run_staging.sh --review-only

# Tear down local compose + owned CVMs from work/owned_cvms.txt
./packages/challenges/agent-challenge/scripts/staging/run_staging.sh --down

# Plan owned deletes only (never touches foreign/prod CVM ids)
./packages/challenges/agent-challenge/scripts/staging/run_staging.sh --dry-run-teardown
```

Evidence lands under `/var/lib/base/e2e/ac-staging/run-<UTC>/` (override with
`AC_STAGING_EVIDENCE_DIR`).

## What the runner does

1. Loads Phala + OpenRouter credentials (never prints them).
2. **Does not** account-sweep pre-existing Phala CVMs (owned-only policy). Warns if the account already has live CVMs.
3. Builds/starts `docker-compose.staging.yml` → `http://127.0.0.1:18082`.
4. Opens a temporary public HTTPS tunnel to that loopback (CVM callbacks).
5. Submits `scripts/miner_agent/dist/miner_agent.zip` (hash pin
   `61cca9bc…`).
6. `selfdeploy review deploy` (real `tdx.small` CVM) → poll until
   `review_allowed` → teardown review CVM.
7. `selfdeploy eval deploy` (real `tdx.small` CVM, RA-TLS KR + artifact grant)
   → poll until accepted result with `guest_artifact_proof` hash match.
8. Teardown **only owned** eval/review CVM ids tracked in this run
   (`cvms.txt` + `work/owned_cvms.txt`). Foreign/prod CVMs on the same Phala
   account are never selected. Use `--dry-run-teardown` to print the plan.

Flags: `--skip-build`, `--keep-up`, `--money-cap`, `--runtime-hours`,
`--submission-id` (with `--eval-only`), `--dry-run-teardown`, `--account-sweep`
(loud no-op expander — still owned-only).

## Pins and allowlists

| Surface | Source |
|---------|--------|
| Review image/compose/KMS/measurement | `scripts/staging/config/challenge.env` + `pins.json` |
| Eval image/compose/KMS/measurement | same |
| Frozen Terminal-Bench digest | `golden/dataset-digest.json` mounted at `/app/golden` |
| Benchmark backend | `CHALLENGE_BENCHMARK_BACKEND=terminal_bench` (compose) |
| KR allowlist | `scripts/staging/config/kr/eval-allowlist.json` |
| Provenance notes | `scripts/staging/config/measurements_source.md` |

Empty measurement allowlist = fail-closed (no CVM admission). Missing
`dataset-digest.json` → eval/prepare `503` `eval_dataset_unavailable`.
Recompute `compose_hash` offline when image or measured compose changes; do
not invent registers.

## Local topology

```text
Host
  ├─ ac-staging-validator :127.0.0.1:18082  (AC API + workers)
  ├─ cloudflared tunnel → public https://*.trycloudflare.com
  └─ staging KR RA-TLS :0.0.0.0:8701       (eval key release)
Phala Cloud
  └─ review CVM then eval CVM (tdx.small, max 1–2, always torn down)
```

Staging enables `CHALLENGE_ALLOW_DEV_URLS=1` and
`SELFDEPLOY_ALLOW_INSECURE_LOOPBACK=1` on the **miner CLI host** only so
non-joinbase callback bases work. Production pins stay joinbase.

## Spend controls

- Default money cap `$8`, runtime `1h`, shape `tdx.small`
- `CHALLENGE_EVALUATION_TASK_COUNT=1` / `CHALLENGE_EVAL_K=1` in compose
- Golden digest mount required for eval plan binding
- Always teardown on EXIT/INT/TERM; `--down` sweeps account CVMs

## Related

- Miner self-deploy: [`miner/self-deploy.md`](miner/self-deploy.md)
- Validator surfaces: [`validator/self-deploy.md`](validator/self-deploy.md)
- OpenAPI: challenge `/openapi.json` (local or production)
