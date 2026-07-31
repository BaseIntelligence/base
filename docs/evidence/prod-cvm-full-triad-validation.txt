# Full-prod CVM validation — base-miner-dev (pack triad)

**Date:** 2026-07-31  
**CVM:** `96238355-a3a7-4732-a610-eb4529c303bc`  
**App ID:** `7345719d5c152d26c73330ba07bf0f3dd4eb29c1`  
**Endpoint:** https://7345719d5c152d26c73330ba07bf0f3dd4eb29c1-8080.dstack-pha-prod5.phala.network  
**Git tip (pack residual):** `fd80148f`  
**Agent image:** `ghcr.io/baseintelligence/base/base-agent@sha256:b5e6081f352c44cf9374c0b5d5d15c938892e0a05bfa0cc36c04ac19d2ae3de2`  
**compose_hash (live Phala):** `5dbbf77467cae6f85242e69d70ede9e246ed6c87e9e09f5e1128025eac8e7015`  
**Mode:** FULL TRIAD (not stub)

## Surface checks

| Check | Result |
|-------|--------|
| CVM status running | PASS |
| GET /healthz → 200 ok | PASS |
| GET /readyz → 200 ready | PASS |
| GET /v1/capacity ×5 → 200 max_concurrency=1 | PASS |
| POST /v1/task unsigned → 401 unauthorized | PASS |
| GET unknown task → 404 task_not_found | PASS |
| Agent logs: agent_runner_listening auth_enabled=true | PASS |
| Live compose BASE_DOCKER_BASE | PASS |
| Live compose BASE_ENVIRONMENT_IMAGE (bash@sha256:3bee76a96…) | PASS |
| Live compose BASE_PACK_ROOT + packs volume | PASS |
| Live compose BASE_PACK_CATALOG_URL | PASS |
| Live compose BASE_TRUSTED_CHALLENGE_PUBKEY | PASS |
| No GBASE_ prefixes | PASS |
| Attestation is_online + event_log compose-hash matches live | PASS |
| MRTD/RTMR0/1/2 stable vs prior live; RTMR3 updated for new compose | PASS |
| measurements.toml dual-entry re-pin + owner sig verify | PASS |

## Measurements (live full-triad row)

- mr_td: `f06dfda6dce1cf904d4e2bab1dc370634cf95cefa2ceb2de2eee127c9382698090d7a4a13e14c536ec6c9c3c8fa87077`
- rtmr0: `68102e7b524af310f7b7d426ce75481e36c40f5d513a9009c046e9d37e31551f0134d954b496a3357fd61d03f07ffe96`
- rtmr1: `07e6f51aa763abfe75c3ddfbf4f425fe3f0ceff66d807a75e049303dce9addf68e7218729bd419638af63a370f65878c`
- rtmr2: `a2a58c9a959a4fa44bd6da0c97a2270c051faf12084cfe91ae900e4fdff6cdd4f69a82005e04ee920f231497894d677f`
- rtmr3: `2439cee0dc08193a1809a92dd6a92d757fcd9f62a35496fdfefc31159d2c2ebfc675c1c336650fd3dbf7c1c1f6390dfc`
- compose_hash: `5dbbf77467cae6f85242e69d70ede9e246ed6c87e9e09f5e1128025eac8e7015`

## Notes

- Pack **catalog** default is `http://127.0.0.1:8090` inside CVM (no in-CVM challenge yet). Agent stays up; pack fetch fails only when a task needs a missing pack without catalog. Full end-to-end pack pull needs gateway/challenge reachable URL overlay later.
- Phala live compose_hash (`5dbbf774…`) differs from miner offline app-compose-only hash (`f031992c…`) because Phala measures the full app-compose document (pre_launch_script, features, etc.). Trust root pins the **live Phala** hash from attestation event_log.
- Billable CVM remains running (tdx.small, ~$0.058/hr).

## Verdict

**FULL PROD CVM LAUNCH: PASS** (surface + triad env + attestation + dual-entry pin).
