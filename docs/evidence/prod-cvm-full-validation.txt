# Production CVM validation — base-miner-dev
UTC: 2026-07-31T06:13:29Z

## Target
- name: base-miner-dev
- vm_uuid: 96238355-a3a7-4732-a610-eb4529c303bc
- app_id: 7345719d5c152d26c73330ba07bf0f3dd4eb29c1
- node: prod5 ONLINE (US-WEST-1)
- endpoint: https://7345719d5c152d26c73330ba07bf0f3dd4eb29c1-8080.dstack-pha-prod5.phala.network
- agent image: ghcr.io/baseintelligence/base/base-agent@sha256:b92468cc4e619e6975c3b4d8774547323a2a9efcb9a6140c49774f6e2b0c1102
- compose_hash: 637a961cec974de7be300eb9a1e51585f995100c8845a97e42684dd56b0b4569
- exec mode: STUB (docker triad intentionally omitted; socket-proxy still measured)

## Results
| Check | Result |
|-------|--------|
| Phala status running | PASS |
| Node ONLINE | PASS |
| 3/3 containers running (agent, attest-helper, socket-proxy) | PASS |
| compose_hash matches measurements.toml dual-entry | PASS |
| GET /healthz → 200 ok | PASS |
| GET /readyz → 200 ready | PASS |
| GET /v1/capacity → 200 max=1 load=0 (5× stable) | PASS |
| Attestation is_online + is_public | PASS |
| Quote contains mr_td dual-entry | PASS |
| Quote contains rtmr0 | PASS |
| Quote contains rtmr1 | PASS |
| Quote contains rtmr2 | PASS |
| Quote contains rtmr3 | PASS |
| Quote contains compose_hash | PASS |
| GET /v1/task/unknown → 404 task_not_found | PASS |
| POST /v1/task unsigned → 401 unauthorized | PASS |
| BASE_TRUSTED_CHALLENGE_PUBKEY in measured compose | PASS |
| Full Docker pack triad live (ENV+PACK_ROOT+catalog+fetch) | NOT DEPLOYED (stub by design on this CVM) |

## Verdict
LIVE_PRODUCTION_MINER_SURFACE: PASS
PACK_DOCKER_PATH: residual local only (uncommitted); not on this CVM

## Evidence
- docs/evidence/prod-cvm-validate/
- docs/evidence/prod-cvm-full-validation.txt
