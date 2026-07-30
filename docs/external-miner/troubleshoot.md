# Miner troubleshooting

<!-- protocol_version: 1 -->

**Bundle `protocol_version`:** `1` · **Challenge `scoring_version`:** `2`

---

## `install.sh` (one-command self-deploy)

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Exit **2**, `missing prerequisite: \`docker\`` | Docker not installed / not on PATH | Install Docker Engine; re-run `./install.sh` |
| Exit **2**, daemon not reachable | Docker installed but service down / permissions | Start Docker; add user to `docker` group |
| Exit **2**, Compose missing | No `docker compose` / `docker-compose` | Install Compose v2 |
| Exit **3**, model key not found / not readable / empty | Bad `GBASE_MODEL_KEY_FILE` | Create mode-`0600` file with provider key; pass path only (Q3=A) |
| Exit **3**, invalid hotkey | Not 64 lowercase hex | Export 32-byte **public** hotkey as lowercase hex |
| Exit **3**, invalid max_concurrency | Outside `1..5` | Set `GBASE_MAX_CONCURRENCY` / `--max-concurrency` to 1–5 |
| Exit **3**, image not digest-pinned | `:latest` or bare tag | Use `repo@sha256:<64 hex>` |
| Exit **4**, image pull / not present | Registry unreachable or `GBASE_SKIP_PULL=1` without local image | Load `gbase/gbase-agent:test` or pull the pin; offline: `GBASE_SKIP_PULL=1` |
| Exit **4**, capacity timeout | Runner failed to start | `docker compose -f <install-dir>/state/docker-compose.runner.yml logs` |
| Secrets appear in logs | Misconfiguration | `install.sh` never echoes key/hotkey bytes — do not `cat` secret files into tickets |

```bash
# Fail-closed smoke (expect non-zero; nothing half-installed)
./install.sh --model-key-file /no/such/key 2>&1 | head -5; echo exit=$?

# Happy path capacity (after successful install)
curl -sS -o /tmp/cap.json -w '%{http_code}\n' http://127.0.0.1:8080/v1/capacity
cat /tmp/cap.json
```

Re-run is **idempotent**: same env refreshes compose and restarts cleanly.

---

## Deploy

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| `deploy --no-deploy` non-zero | Bad image ref / hash format | Use `repo@sha256:` + 64 hex; check `--launch-token-hash` is 64 hex |
| `phala` not found with `--deploy` | CLI missing | Install Phala CLI; or pass `--phala-bin` |
| Deploy fails: insufficient balance | Unfunded account | [funding-phala.md](./funding-phala.md) |
| Compose-hash differs from operator expect | Image/tag drift or template skew | Rebuild from same gbase commit; compare AGENT_CHALLENGE image pins |

```bash
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 1
echo exit=$?
# or after install.sh:
grep '^compose-hash=' miner-runtime/state/compose-hash.txt
```

---

## Certify

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| HTTP errors to validator | Wrong URL / TLS / firewall | Confirm operator URL; curl `/healthz` if exposed |
| `Rejected` | Quote/event-log/policy | Check measurements allowlist rotation; redeploy measured image |
| `Parked` | Collateral/TCB outage | Wait; do not expect prior Verified to carry forward |
| Missing `--agent-url` | Live mode without URL | Pass `--agent-url` or use `--fixture-mode` only for smoke |
| Hotkey parse error | Not 64 hex | Export raw 32-byte public key as lowercase hex |

---

## Scoring / packs (scoring_version 2)

| Symptom | Notes |
|---------|--------|
| You are up but weight is zero | Challenge may emit `NoScore`; attestation Parked; or you are outside expected set (stake/policy) |
| Expecting echo-answer / latency decay | **Retired** (scoring_version 1). Live path is Harbor packs → `model.patch` → operator grade |
| Pack env cannot pull images | Agent must use measured **socket-proxy** only — raw `docker.sock` on agent is forbidden |
| Model calls fail | Miner-funded key missing/invalid (`GBASE_MODEL_KEY_FILE`); subnet does not pay LLM bills (Q3=A) |
| Gateway "looks wrong" | Validators recompute from local trust roots; deviant gateway is validator-side (D19) |

Miners do not re-run aggregation. Bundle math is validator-side per BUNDLE_SPEC (**protocol_version 1**).

Default egress is **OPEN**; stripping protects grading-channel integrity, not miner honesty (D19).

---

## Protocol version mismatch

If operators announce a new bundle `protocol_version` and your docs/binary lag:

1. Upgrade gbase to the release validators run.
2. Confirm badge in [`README.md`](./README.md) matches (**bundle stays 1** unless leaf bytes change).
3. Redeploy CVM if the challenge compose contract changed (`challenge_scoring_version` bump — currently **2**).

```bash
# From repo: must exit 0
cargo run -q -p xtask -- external-docs-check
```

---

## Getting help

Provide: gbase git SHA, `compose-hash=`, certify `outcome=`, netuid, epoch, **public** hotkey hex only.  
Never send mnemonics, age identities, Phala API keys, model keys, receipt secrets, or coldkey files.
