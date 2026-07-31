# Trust-root signing ceremony (task 18 / D12 / D18 / D21)

Offline, operator-only. Never commit secrets. Prefer `/root/.base-secrets/` (mode `0700`).

## Artifacts in git (public only)

| Path | Contents |
|------|----------|
| `config/owner.pubkey` | 32-byte owner public key (hex). **Throwaway for tests** — not the production `base-owner` coldkey mnemonic. |
| `config/challenges.toml` | Challenge id, public key, emission bps (sum = 10000), participant policy. |
| `config/challenges.toml.sig` | Detached sr25519 signature under `base-trustroot-v1`. |
| `config/measurements.toml` | Measurement allowlist (real Phala fixtures, task 35); empty = fail-closed. |
| `config/measurements.toml.sig` | Detached owner signature. |

## Secret layout (never git)

| Path | Contents |
|------|----------|
| `~/.base-secrets/age-identity.txt` | age X25519 identity (mode 600). |
| `~/.base-secrets/owner-throwaway.age` | age-encrypted owner mini-secret. |
| `~/.base-secrets/challenge-*.age` | age-encrypted challenge mini-secrets. |

## Commands

```bash
# 1. age identity (once)
age-keygen -o ~/.base-secrets/age-identity.txt
RECIPIENT=$(grep 'public key:' ~/.base-secrets/age-identity.txt | awk '{print $4}')

# 2. Owner keypair (throwaway for CI; production uses offline HSM / air-gapped owner key)
cargo run -p trustroot-bin -- keygen \
  --out-pub config/owner.pubkey \
  --out-secret ~/.base-secrets/owner-throwaway.age \
  --age-recipient "$RECIPIENT"

# 3. Challenge keypair (secret stays off-git)
cargo run -p trustroot-bin -- keygen \
  --out-pub ~/.base-secrets/challenge-dummy.pub \
  --out-secret ~/.base-secrets/challenge-dummy.age \
  --age-recipient "$RECIPIENT"
# paste public_key into challenges.toml

# 4. Sign bodies (payload = scale(version, introduced_epoch, scale(body)))
cargo run -p trustroot-bin -- sign \
  --key ~/.base-secrets/owner-throwaway.age \
  --age-identity ~/.base-secrets/age-identity.txt \
  --input config/challenges.toml --kind challenges

cargo run -p trustroot-bin -- sign \
  --key ~/.base-secrets/owner-throwaway.age \
  --age-identity ~/.base-secrets/age-identity.txt \
  --input config/measurements.toml --kind measurements

# 5. Verify
cargo run -p trustroot-bin -- verify \
  --owner-pub config/owner.pubkey \
  --input config/challenges.toml --kind challenges
```

## Signature preimage

Domain tag: `base-trustroot-v1` (via `crypto`).

```text
payload = scale(version: u32, introduced_epoch: u64, body: Vec<u8>)
body    = scale(ChallengesBody | MeasurementsBody)
```

## D21 rotation

Publish `v(n+1)` beside `v(n)` (directory of versioned TOML files). Loaders accept both for `rotation_epochs` (default 3 from `config`) after `introduced_epoch` of the newer file, then drop the old version.

## Fail-closed rules

- Missing TOML or `.sig` → error
- Signature not under `owner.pubkey` → `NonOwner`
- Empty `measurements` → every quote rejected (pre-task-35 bootstrap)
- No HTTP: this crate never fetches trust roots over the network
