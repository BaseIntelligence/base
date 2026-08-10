# Bounty challenge — enable real backends + emission unlock

Bounty emission is owner-controlled via the trust root. Current committed
shares are **design = 3000 bps**, **prism = 4500 bps**, **bounty = 2500 bps**
(sum `10000`). See [`config/challenges.toml`](../../config/challenges.toml)
and [`BOUNTY_CHALLENGE.md`](../BOUNTY_CHALLENGE.md).

## A. Bring up bounty without fighting emission (staging / local)

1. Materialize env: `./deploy/scripts/materialize-env.sh`
2. Ensure secrets (mode **0400**, uid **65532**):
   - `deploy/secrets/bounty_sk` — challenge signing mini-secret
   - `deploy/secrets/openrouter/api_key` — similar-24h OpenRouter path
     (`SimAgent` if absent; **never** `BOUNTY_FORCE_SIM=true` on staging/prod)
   - `deploy/secrets/bounty/admin_tokens` — operator approve/reject bearers
     (one token per line; hashed at boot)
3. Confirm trust-root pubkey for `bounty` matches `bounty_sk` public key.
4. Start stack. Health:
   ```bash
   curl -sS http://127.0.0.1:28095/health          # local overlay
   curl -sS http://127.0.0.1:8095/health            # in-container / droplet
   ```
5. Register challenge with gateway after redeploy:
   ```json
   { "challenge_id": "bounty", "base_url": "http://bounty-challenge:8095", "weight": 1 }
   ```
   Or: `./deploy/scripts/register-challenge-backends.sh` (registers prism +
   design + bounty and smokes `/challenge/*/health`).
6. Emission share is independent of bringing the service up: leaves still emit
   (D24 exact-E) even at `0` bps. Current shares are 3000/4500/2500.

## B. Keygen (`bounty_sk`)

Dev throwaway (matches committed trust-root pubkey when using the ceremony
key under `~/.base-secrets/`):

```bash
RECIPIENT=$(grep 'public key:' ~/.base-secrets/age-identity.txt | awk '{print $4}')
cargo run -p trustroot-bin -- keygen \
  --out-pub ~/.base-secrets/challenge-bounty.pub \
  --out-secret ~/.base-secrets/challenge-bounty.age \
  --age-recipient "$RECIPIENT"
# paste public_key hex into config/challenges.toml [[challenges]] id = "bounty"
age -d -i ~/.base-secrets/age-identity.txt \
  -o deploy/secrets/bounty_sk ~/.base-secrets/challenge-bounty.age
chown 65532:65532 deploy/secrets/bounty_sk && chmod 0400 deploy/secrets/bounty_sk
```

Never bake the mini-secret into images or cloud-init. Do **not** mount
`bounty_sk` on gateway, validator, or other challenge services.

## C. Emission ceremony (owner)

Goal: keep `design + prism + bounty = 10000` bps after any rebalance.

1. Choose new shares (all ≥ 0, sum exactly **10000**).
2. Edit `config/challenges.toml` emission lines (plus pubkey if rotating).
   Mirror the same body into `config/challenges.staging.toml` when staging
   should track prod.
3. Re-sign **both** files with the throwaway (or production) owner key:
   ```bash
   cargo run -p trustroot-bin -- sign \
     --key ~/.base-secrets/owner-throwaway.age \
     --age-identity ~/.base-secrets/age-identity.txt \
     --input config/challenges.toml --kind challenges
   cargo run -p trustroot-bin -- sign \
     --key ~/.base-secrets/owner-throwaway.age \
     --age-identity ~/.base-secrets/age-identity.txt \
     --input config/challenges.staging.toml --kind challenges
   cargo run -p trustroot-bin -- verify \
     --owner-pub config/owner.pubkey \
     --input config/challenges.toml --kind challenges
   ```
   Full ceremony: [`../../config/CEREMONY.md`](../../config/CEREMONY.md) and
   [`trust-root-rotation.md`](./trust-root-rotation.md) (dual-accept if rotating).
4. Roll **all** validators with the new signed trust root before relying on the
   new shares.
5. Do **not** change bounty `challenge_scoring_version` in the same ceremony
   unless intentionally bumping the scoring contract.

Current committed default: **design = 3000**, **prism = 4500**,
**bounty = 2500**.

### Ceremony status (dev throwaway)

As of bounty enablement on branch `add/bounty-challenge`:

| Artifact | Status |
|----------|--------|
| `bounty` pubkey in `challenges.toml` | set (`challenge-bounty.age` under `~/.base-secrets/`) |
| `challenges.toml.sig` | re-signed under `config/owner.pubkey` |
| `challenges.staging.toml` + `.sig` | mirrored + re-signed |
| Production owner / prod `bounty_sk` rotation | **still pending** (ops; see CEREMONY.md) |

## D. Admin tokens + OpenRouter

- Admin bearers: `deploy/secrets/bounty/admin_tokens` (mode **0400**, uid
  **65532**). Optional override: `BOUNTY_ADMIN_TOKENS_FILE`.
- Approve path (master-local only):
  `GET /v1/admin/bugs?status=pending_admin` →
  `POST /v1/admin/bugs/{id}/approve` (or `/reject` with reason).
  See [`BOUNTY_CHALLENGE.md`](../BOUNTY_CHALLENGE.md) §5.
- Without OpenRouter key: similarity uses Sim backends in CI/local only;
  staging/prod must mount a real key and keep `BOUNTY_FORCE_SIM=false`.

## E. Rollback emission

Re-sign `challenges.toml` with previous bps (e.g. design 5000 / prism 5000 /
bounty 0, or drop the bounty row only after dual-accept planning), roll
validators. Challenge service can keep running; emission share alone changes
aggregation weight.

## F. Compose / image notes

| Item | Value |
|------|-------|
| Service | `bounty-challenge` |
| Port | `8095` (local host `28095`) |
| Image / Dockerfile target | `bounty-challenge:0.1.0` / `bounty-challenge` |
| Volume | `bounty-artifacts` → `/var/lib/bounty` |
| Env example | `deploy/env/bounty-challenge.env.example` |

Registry pin ladder (`deploy/pins/*.json` / `images.yml`) for
`bounty-challenge` is a follow-up once GHCR builds include the new target —
`--build-from source|prebuilt` works without pin rows.
