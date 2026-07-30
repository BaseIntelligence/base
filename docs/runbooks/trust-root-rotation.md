# Runbook: trust-root rotation (D21)

Rotate owner-signed `challenges.toml` / `measurements.toml` (or challenge keys inside them) without a hot push and without bricking validators mid-epoch.

Normative ceremony background: [`../../config/CEREMONY.md`](../../config/CEREMONY.md).  
Threat bounds: owner remains the trust root (R12 / D19).

---

## 1. Rules

1. Rotation is a **signed release**, never an unsigned file drop on a live host.
2. Publish **`v(n+1)` beside `v(n)`**. Loaders accept **either** signature/body pair for `rotation_epochs` (default **3**) after the newer file's `introduced_epoch`.
3. After the window, drop `v(n)` in a follow-up release.
4. Challenge **secrets** stay off-git (age-encrypted). Only public keys and TOML bodies + `.sig` enter the repo.

---

## 2. Prepare (offline)

```bash
# From repo root. Secrets outside git.
mkdir -p ~/.gbase-secrets
chmod 700 ~/.gbase-secrets

# Optional: ensure age identity exists (do not commit)
# age-keygen -o ~/.gbase-secrets/age-identity.txt

# Edit the next version body (example paths; keep v(n) files until window ends)
cp config/challenges.toml config/challenges.vNEXT.toml
# ... edit public keys / bps / participant policy ...
# Ensure emission bps sum to 10000.

# Sign with owner mini-secret (age-encrypted path shown)
cargo run -q -p trustroot-bin -- sign \
  --key ~/.gbase-secrets/owner-throwaway.age \
  --age-identity ~/.gbase-secrets/age-identity.txt \
  --input config/challenges.vNEXT.toml \
  --kind challenges \
  --out config/challenges.vNEXT.toml.sig
```

If you do not have the production owner secret in this environment, stop. Do not invent a new owner key for prod.

---

## 3. Verify before merge

```bash
cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/challenges.toml \
  --kind challenges

cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/measurements.toml \
  --kind measurements
```

Both must exit 0 against the committed owner pubkey.

When dual files are wired in config, verify **both** `v(n)` and `v(n+1)` the same way.

---

## 4. Roll out

1. Open a PR that adds `v(n+1)` artifacts and bumps release notes. Do not delete `v(n)` yet.
2. Merge to `reborn` after CI green (`external-docs-check`, `spec-check`, tests).
3. Promote images / config to **staging** first ([`promote-rollback-restore.md`](./promote-rollback-restore.md)).
4. Confirm validators log acceptance of either root during the dual-accept window.
5. After `rotation_epochs` epochs on prod, PR to remove `v(n)`.

---

## 5. Abort

- If signature verify fails anywhere: **do not promote**. Fix the body or resign offline.
- If a bad root already reached staging: roll back the release digest and restore previous config files from git tag / backup.
- Compromised challenge key: quarantine absorbs garbage scores (D6); still rotate the key via this runbook (R10). Honesty of past scores is not restored (D19).

---

## 6. Spot-check commands (clean shell)

These must exit 0 on a clean checkout with the committed throwaway owner key:

```bash
cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/challenges.toml --kind challenges

cargo run -q -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/measurements.toml --kind measurements
```
