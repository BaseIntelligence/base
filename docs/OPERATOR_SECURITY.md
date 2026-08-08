# Operator security checklist

Use this before every promote and after every incident. Architecture: [`ARCHITECTURE.md`](./ARCHITECTURE.md). Threat bounds: [`THREAT_MODEL.md`](./THREAT_MODEL.md).

---

## 1. Secrets and keys

- [ ] No secrets in git (`git log -p` / `git grep` clean of `dop_v1_`, `secretPhrase`, PEM private keys).
- [ ] Coldkeys and age identities are mode `600`; secret dirs mode `700`.
- [ ] Compose env files under `deploy/env/*.env` are mode `0600` after `materialize-env.sh`.
- [ ] Age identity delivered **out of band** to `/etc/base/age-identity.txt` (or `AGE_IDENTITY`). Never in terraform state or cloud-init user-data.
- [ ] Challenge signing secrets are **files** mounted into the challenge service, not env values (D11).
- [ ] Owner and challenge mini-secrets never committed; only `*.pubkey` / TOML bodies + detached `.sig` in git.
- [ ] Cloudflare / DO / Phala tokens live only in operator secret stores, not in docs or CI logs.

---

## 2. Images and compose

- [ ] Every image reference is digest-pinned (`repo@sha256:<64 hex>`). No `:latest`.
- [ ] Exactly one mount of `/var/run/docker.sock`: on `socket-proxy` (read-only).
- [ ] socket-proxy allowlist matches updater needs (`CONTAINERS`, `IMAGES`, `POST` as configured).
- [ ] Gateway service uses compose profile **`master`** only on the owner host.
- [ ] Profile `evil-gateway` is **absent** from prod hosts. Spot-check:

```bash
./deploy/scripts/assert-evil-gateway-not-default.sh
```

---

## 3. Trust roots

- [ ] Validators load `config/challenges.toml` and `config/measurements.toml` from **local disk** only.
- [ ] Detached signatures verify under `config/owner.pubkey`:

```bash
cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/challenges.toml --kind challenges

cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/measurements.toml --kind measurements
```

- [ ] Rotation follows dual-accept (D21); never hot-push a new root without a signed release. See [`runbooks/trust-root-rotation.md`](./runbooks/trust-root-rotation.md).
- [ ] You understand R12: owner signs roots **and** runs the gateway.

---

## 4. Gateway and TLS

- [ ] Gateway hotkey equals on-chain `SubnetOwnerHotkey` (else process exits 2).
- [ ] `BASE_GATEWAY_ADMIN_TOKEN_FILE` (or `BASE_GATEWAY_ADMIN_TOKEN`) is set whenever `BASE_GATEWAY_REQUIRE_OWNER=1` — `/v1/admin/*` must not be open on a public listener.
- [ ] Spot-check from the public Internet: `GET /v1/admin/backends` → **401/403** (not 200). Localhost/VPC seal scripts still work with the bearer.
- [ ] TLS terminates **only** in the gateway process (D20). No second reverse proxy claiming TLS.
- [ ] `BASE_DOMAIN` is a real delegated zone when ACME is enabled (D25).
- [ ] Manual failover procedure is known: [`runbooks/gateway-failover.md`](./runbooks/gateway-failover.md). HA is **not** claimed (R9).

---

## 5. Promote / backup

- [ ] `pg_dump` taken **before** every production promote.
- [ ] Rollback path tested or at least dry-run documented: [`runbooks/promote-rollback-restore.md`](./runbooks/promote-rollback-restore.md).
- [ ] Updater desired image is digest-pinned; health gate must pass or auto-rollback.
- [ ] Self-update of the updater is **operator one-shot**, never automatic in prod (D14).

---

## 6. Claims discipline (read aloud)

- [ ] Pitch / status updates use D19 wording; no "master cannot lie about scores" or "publicly auditable merkle on-chain".
- [ ] Merkle root is **not** in `WeightsTlockPayload`.
- [ ] Attestation does **not** prove env values (D11).
- [ ] Zero emission can still be a **pass** for extrinsic/reveal criteria (R4).

---

## 7. Quick hygiene commands

```bash
# Workspace gates (from repo root)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo run -q -p xtask -- loc-cap
cargo run -q -p xtask -- consensus-lint
cargo run -q -p xtask -- spec-check
cargo run -q -p xtask -- agent-challenge-check
cargo run -q -p xtask -- external-docs-check

# No accidental secret patterns in tracked docs (expect empty)
git grep -nEi 'dop_v1_|secretPhrase|BEGIN [A-Z ]*PRIVATE KEY' -- '*.md' || true
```
