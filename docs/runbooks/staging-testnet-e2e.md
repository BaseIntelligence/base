# Staging testnet E2E runbook

End-to-end testnet 541 procedure on the 2-host staging pair.

## Prerequisites

- Staging master (`base-staging`, 68.183.23.51 / 10.116.0.2) running master role
- Staging validator (`base-staging-validator`, 142.93.197.253 / 10.116.0.4) running validator role
- Both deployed from the same `dev` commit via `deploy-staging.yml` or manual `remote-deploy.sh`
- `deploy/secrets/challenge_sk` and `deploy/secrets/gateway_sk` present on master (mode 0400, uid 65532)
- `deploy/env/*.env` materialized on both hosts (mode 0600)

## Verify staging master

```bash
ssh root@68.183.23.51
cd /opt/base
docker compose -f docker-compose.yml -f deploy/compose/role-master.yml -f deploy/compose/env-staging.yml --profile master ps
# All services Up, postgres + validator healthy
curl -fsS http://127.0.0.1:18080/healthz   # validator tunnel
curl -fsS http://127.0.0.1:18090/healthz   # agent-challenge tunnel
```

## Verify staging validator

```bash
ssh root@142.93.197.253
cd /opt/base
docker compose -f docker-compose.yml -f deploy/compose/role-validator.yml -f deploy/compose/env-staging.yml ps
# validator + agent-challenge + postgres + socket-proxy Up (no gateway)
docker logs $(docker ps -q --filter name=validator) 2>&1 | tail -20
# Look for: "Match epoch=" lines (bundle signature valid → coordination loop healthy)
```

If validator logs show `bundle gateway signature invalid`:
1. Confirm both hosts run the same commit: `git -C /opt/base rev-parse HEAD`
2. Confirm gateway_sk on master matches the key used to sign the bundle
3. Redeploy master: `remote-deploy.sh --host root@68.183.23.51 --role master --env staging`

## Verify bundle seal (master)

```bash
ssh root@68.183.23.51
curl -fsS http://127.0.0.1:18081/healthz   # gateway tunnel (if using master-net compose)
# Or in-container:
docker exec $(docker ps -q --filter name=gateway) curl -fsS http://127.0.0.1:8080/v1/weights/latest
# Returns SCALE-encoded sealed bundle
```

## Verify agent-challenge identity

```bash
ssh root@68.183.23.51
docker exec $(docker ps -q --filter name=agent-challenge) /usr/local/bin/agent-challenge identity
# Prints: challenge_id=agent-v1 scoring_version=2 public_key=...
```

## Testnet chain (read-only smoke)

The validator uses `FakeChain` by default. To verify live testnet connectivity:

```bash
# On operator machine (not on droplet — requires cargo)
cargo run -p xtask -- metadata-snapshot --check
# Verifies metadata/testnet.lock matches live Finney testnet
```

## Deploying a new commit to staging

1. Push to `dev` — `ci.yml` runs, then `deploy-staging.yml` auto-deploys both hosts.
2. Or manual:
   ```bash
   ./deploy/scripts/remote-deploy.sh --host root@68.183.23.51 --role master --env staging --build-from source
   ./deploy/scripts/remote-deploy.sh --host root@142.93.197.253 --role validator --env staging --build-from source
   ```
3. Post-deploy: CI checks validator `/healthz` (fail-closed) and greps for `Match epoch=` within 180s.

## Rollback

```bash
# Redeploy previous known-good commit
git checkout <good-sha>
./deploy/scripts/remote-deploy.sh --host root@68.183.23.51 --role master --env staging --build-from source
./deploy/scripts/remote-deploy.sh --host root@142.93.197.253 --role validator --env staging --build-from source
```

## Known limitations (see docs/COMPLETENESS.md)

- `FakeChain` is the default backend; `BASE_CHAIN_BACKEND=live` switches to `chain-live` (Phase 6).
- `agent-challenge` epoch dispatch is opt-in via `BASE_CHALLENGE_DISPATCH=1` (Phase 7).
- CRV4 tlock encryption is deferred; `set_weights` works when commit-reveal is off.
