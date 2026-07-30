# Runbook: manual gateway failover (R9)

**HA is not claimed.** A dead gateway takes down registry, reverse proxy, and bundle serving until recovery. Mitigations:

1. Docker `restart: unless-stopped` + healthcheck (auto-restart target **< 60s** when the daemon is healthy).
2. This manual failover runbook.
3. Validators can **mirror** bundles from peers (content-addressed by merkle root) so temporary gateway loss does not always halt verify if peers already have the bundle.

---

## 1. Symptoms

- `/healthz` or `/readyz` on the gateway listener fails.
- Validators log fetch errors for the bundle endpoint.
- Challenge proxy paths `/challenge/*` time out.
- `docker compose ps` shows `gateway` exited or unhealthy.

---

## 2. Fast path: let Docker restart

On the **master** host:

```bash
cd /opt/gbase   # or your checkout path
docker compose --profile master ps gateway
docker inspect "$(docker compose ps -q gateway)" \
  --format 'status={{.State.Status}} restart_policy={{.HostConfig.RestartPolicy.Name}}'
```

Expect `RestartPolicy.Name=unless-stopped`. If the container was killed, wait up to 60s for `running`.

Automated check script (skips cleanly when gateway is not up):

```bash
./deploy/scripts/gateway-kill-restart-check.sh
```

Offline CI proof of the policy lives in validator adversarial tests (`a48_ops_gateway_restart_policy_unless_stopped`).

---

## 3. Manual restart

```bash
cd /opt/gbase
docker compose --profile master up -d gateway
docker compose --profile master ps gateway
# replace listen port if your env differs
curl -fsS "http://127.0.0.1:${GBASE_GATEWAY_HEALTH_PORT:-8080}/healthz"
```

If the process exits **2** immediately: hotkey ≠ on-chain `SubnetOwnerHotkey` (D3). Fix `GBASE_GATEWAY_HOTKEY` / wallet; do not bypass the check.

---

## 4. Fail over to a standby master host (manual)

There is **no** automatic multi-master. Procedure:

1. **Stop** gateway on the failed host so two gateways never split-brain seal:

   ```bash
   docker compose --profile master stop gateway
   ```

2. Confirm DNS / VIP for `prod.$GBASE_DOMAIN` (or staging) points at the **standby** host you control. TLS still terminates **in** the gateway process on that host (D20).

3. On the standby, materialize env, ensure owner hotkey and age secrets are present, then:

   ```bash
   cd /opt/gbase
   ./deploy/scripts/materialize-env.sh
   docker compose --profile master up -d
   docker compose ps
   curl -fsS "https://prod.${GBASE_DOMAIN}/healthz"   # when DNS+TLS live
   ```

4. Confirm validators can fetch the current epoch bundle and that peer cross-check still meets `min_peer_sample`.

5. After the old host is repaired, keep gateway **stopped** there until you intentionally fail back (repeat steps with roles reversed).

---

## 5. Validator behaviour while gateway is down

| Situation | Expected |
|-----------|----------|
| Bundle already mirrored + peers reachable | Verify/recompute may continue from local/peer copy |
| No bundle, no peers | No submit; alarms; class B / degraded per policy |
| Peer sample below minimum | `Degraded`, no submit (D26) |

Do not run a second unsigned "emergency" gateway with a different hotkey. Master-only is a safety property, not a suggestion.

---

## 6. Spot-check (clean shell)

```bash
./deploy/scripts/assert-evil-gateway-not-default.sh
./deploy/scripts/gateway-kill-restart-check.sh
```

Both exit 0 (`gateway-kill-restart-check` may print `SKIP` when no live gateway; that is still exit 0).
