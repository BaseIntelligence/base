# Runbook: measurement re-pin after socket-proxy CVM compose (todo 23)

Todo 22 put the allowlisted Docker socket-proxy **inside** the measured miner
`app-compose`. That changes the canonical **compose-hash**. The owner-signed
measurement allowlist (`config/measurements.toml`) is fail-closed: a quote whose
`(mr_td, rtmr0–3, compose_hash)` tuple is not listed is
`Rejected { reason: MeasurementNotAllowlisted }`, and scoring treats the miner as
**not** `Verified` (`AttestationNotVerified` / no credit).

This runbook is the **only** safe cutover order. **Do not invert the steps.**

Normative ceremony: [`../../config/CEREMONY.md`](../../config/CEREMONY.md).  
Generic dual-file rotation: [`trust-root-rotation.md`](./trust-root-rotation.md).

---

## 0. Why this exists (Metis S6)

| State | Compose-hash (example) |
|-------|------------------------|
| Pre–socket-proxy (task-22 fixed receipt key) | `721c2104e9a1379a403fa4e726230d6cdae07d6b62d437e691381e3700acd911` |
| Post–socket-proxy (same receipt key) | `15d1c5f4a52c5690eb1d7e852351776fe5f885e147ec9d5190964173d3621f7b` |
| Post–socket-proxy (`DeployParams::default` / library default receipt key) | `95089ce1b1ccb528e3309acc6dc304835c1634f540663784a99e700364c331ed` |
| Historical task-34/35 fixture still in allowlist | `1b8a63efb0f7afda1e52c823f39cc0a79d6d75c2e7a086b58e0e6a2db548524b` |

Compose-hash also depends on measured fields (agent/attest-helper/socket-proxy
image digests, launch-token **hash**, receipt **public** key, netuid, …). Always
recompute for the **exact** template you deploy:

```bash
cargo run -q -p miner-bin -- deploy --no-deploy --netuid 541 \
  --receipt-sk-host-path /path/to/receipt_sk
# → compose-hash=<64 hex>
```

---

## 1. Cutover order (MUST NOT invert)

### Step 1 — Deploy the new miner template **without** requiring `Verified` for scoring

1. Ship the todo-22 template (socket-proxy measured; agent uses
   `GBASE_DOCKER_BASE=http://socket-proxy:2375`; no raw docker.sock on the agent).
2. Keep validator / scoring in a mode that **does not** hard-fail the epoch when
   attestation is not yet `Verified` (park / prior policy as already deployed).
3. Miners redeploy CVMs from the new compose.

**Do not** flip “require Verified for scoring” in this step.

### Step 2 — Capture real measurements from a live CVM

From a CVM running the **new** compose (Phala CLI or existing capture path used in
task-34 / spikes under `docs/spikes/task-02-*`):

| Field | Source |
|-------|--------|
| `mr_td`, `rtmr0`–`rtmr3` | TDX quote body / `tdx_attest` report |
| `compose_hash` | RTMR3 event-log replay **and** `compose_hash(app-compose.json)` — must match |
| `mr_config_id` | Must equal `mr_config_id(compose_hash)` |

Store quote + event log + app-compose beside the capture (same layout as
`crates/attest-parse/tests/fixtures/real/` and
`/root/.omo/evidence/gbase-rust-subnet/phala-fixtures/`).

**If live deploy is blocked:** you may still land dual-entry **structure** in git
(tests + provisional compose-hash pin) and this runbook. You **must not** claim
production `Verified` for the new template until live MRTD/RTMR values replace any
provisional row. Fixture-only rows that pair old registers with a new
`compose_hash` will never match a real TDX quote (RTMR3 embeds compose events).

### Step 3 — Publish the updated owner-signed allowlist

Prefer **dual-entry** inside one body during the window (same spirit as D21
dual-accept of `v(n)` + `v(n+1)` files):

1. Keep the previous `[[measurements]]` row (old compose) so already-deployed
   miners do not brick mid-window.
2. Append a new `[[measurements]]` row with the **live** registers + new
   `compose_hash`.
3. Bump `version` / `introduced_epoch` as needed; resign:

```bash
cargo run -q -p trustroot-bin -- sign \
  --key ~/.gbase-secrets/owner-throwaway.age \
  --age-identity ~/.gbase-secrets/age-identity.txt \
  --input config/measurements.toml \
  --kind measurements

cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/measurements.toml \
  --kind measurements
```

4. Merge + promote config (see [`trust-root-rotation.md`](./trust-root-rotation.md)
   and [`promote-rollback-restore.md`](./promote-rollback-restore.md)).

After `rotation_epochs` (default 3) of dual-accept, drop the old row in a
follow-up release (hard-cut to new-only).

### Step 4 — Only then require `Verified` for scoring

Enable the gate that refuses credit / weight for miners without a current-epoch
`Verified` attestation. Do this **only after** step 3 is live on validators and
at least one new-compose quote has certified successfully against the updated
allowlist.

---

## 2. Failure symptom of doing it backwards

| Wrong order | What operators see |
|-------------|--------------------|
| Require `Verified` for scoring **before** publishing the new allowlist | Every miner on the new compose → `MeasurementNotAllowlisted` → no credit → network-wide `AttestationNotVerified` / empty or unfair weights |
| Publish **new-only** allowlist **before** miners finish redeploying | Old-compose miners brick until they upgrade |
| Edit `measurements.toml` without resigning | Loaders fail closed (`NonOwner` / missing sig) — validators refuse the root |
| Empty `measurements` array | Every quote rejected (`EmptyAllowlist`) |

Automated proof of the stale-allowlist outage lives in
`crates/attest-policy` re-pin tests and evidence
`task-23-stale-allowlist-rejected.txt`.

---

## 3. Spot-check after cutover

```bash
# Allowlist still owner-signed
cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/measurements.toml --kind measurements

# Offline compose-hash of what you intend to run
cargo run -q -p miner-bin -- deploy --no-deploy --netuid <NETUID> \
  --receipt-sk-host-path <receipt_sk>

# Certify a live (or fixture) quote against a validator loaded with the new root
# Expect: outcome=verified for new compose after step 3
# Expect: MeasurementNotAllowlisted if validator still has pre-repin body only
```

---

## 4. Abort

- Signature verify fails → **do not promote**; fix body or resign offline.
- Bad root already on staging → roll back config digest; restore previous
  `measurements.toml` + `.sig` from git tag.
- Live capture mismatch (compose-hash ≠ event-log replay) → do not allowlist;
  fix the deployed compose first.

---

## 5. Related evidence / tests

| Artifact | Meaning |
|----------|---------|
| `task-22-compose-hash-measured-proxy.txt` | Old vs new compose-hash after socket-proxy |
| `task-23-measurement-repin-verified.txt` | New measurements accepted by updated allowlist |
| `task-23-stale-allowlist-rejected.txt` | New measurements rejected by old allowlist (wrong-order outage) |
| `crates/attest-policy` `repin_*` tests | Dual-entry, hard-cut, stale-body matrix |
