# Deploy secrets (NEVER commit secret bytes)

Containers run as `base` (uid **65532**). Host secret files MUST be:

```bash
chown 65532:65532 deploy/secrets/gateway_sk deploy/secrets/prism_sk deploy/secrets/design_sk
chmod 0400 deploy/secrets/gateway_sk deploy/secrets/prism_sk deploy/secrets/design_sk
```

Bind-mounts use the file inode; directory mode 0700 is OK.

## Challenge / gateway keys

| Path | Used by | Notes |
|------|---------|-------|
| `gateway_sk` | gateway | Bundle seal mini-secret (`BASE_GATEWAY_SK_FILE`) |
| `gateway_admin_token` | gateway + seal scripts | Bearer for `/v1/admin/*` (`BASE_GATEWAY_ADMIN_TOKEN_FILE`). **Required** when `BASE_GATEWAY_REQUIRE_OWNER=1`. Mode **0400**, uid **65532** |
| `prism_sk` | prism-challenge | PRISM challenge mini-secret |
| `design_sk` | design-challenge **only** | Design challenge mini-secret; never mount on egress proxy |
| `challenge_sk` | legacy placeholder | Prefer `prism_sk` / `design_sk`; do not reuse across challenges |

```bash
# Generate once per environment; never commit the bytes.
openssl rand -hex 32 > deploy/secrets/gateway_admin_token
chown 65532:65532 deploy/secrets/gateway_admin_token
chmod 0400 deploy/secrets/gateway_admin_token
```

Local dummy for development: decrypt with age:
`age -d -i ~/.base-secrets/age-identity.txt -o deploy/secrets/design_sk ~/.base-secrets/design-dummy.age`

## Design challenge

| Path | Used by | Notes |
|------|---------|-------|
| `design/annotator_tokens` | design-challenge | One bearer token per line; hashed (SHA-256) at boot. Mode **0400**, uid **65532** |
| `openrouter/api_key` | design-egress-proxy (**proxy only**); also prism reviewer | Never mount OpenRouter key into design-challenge or sandbox containers |

```bash
mkdir -p deploy/secrets/design deploy/secrets/openrouter
touch deploy/secrets/design/annotator_tokens deploy/secrets/openrouter/api_key
chown -R 65532:65532 deploy/secrets/design deploy/secrets/openrouter
chmod 0400 deploy/secrets/design/annotator_tokens deploy/secrets/openrouter/api_key
```

## Other

- `lium/` — prism Lium API + SSH keys (see prism runbook)
- `wallets/` — btcli wallet trees for gateway owner / validator hotkeys
- `github/token` — prism-challenge top-model publisher: fine-grained GitHub
  token with **contents:write** (and **contents:write** on Releases for large
  checkpoints) on `BaseIntelligence/prism` only. Read via
  `PRISM_TOPMODEL_GITHUB_TOKEN_FILE` (`/run/base/github/token`); missing or
  empty file = top-model publish silently disabled. Mode **0400**, uid
  **65532** — never commit it.
- `prism/admin_tokens` — one operator bearer per line for Prism
  `/v1/submissions/{id}/retry`, `POST /v1/admin/playground/complete`,
  `POST /v1/admin/gating/{hotkey}/reset`, and
  `POST|GET /v1/admin/artifacts/...`. Read via `PRISM_ADMIN_TOKENS_FILE`
  (`/run/base/prism/admin_tokens`). Empty/missing → those routes answer
  **503 `auth_unconfigured`** (fail-closed). Mode **0400**, uid **65532**.
- `prism/payer_vault_key` — 32-byte (or 64-hex) key for short-TTL encrypted
  miner BYOK seals (`PRISM_PAYER_VAULT_KEY_FILE`). Host dir
  `/var/lib/prism/payer-vault` is mounted RW for `*.seal` files. **Never
  commit the key.** Generate once:

```bash
mkdir -p deploy/secrets/prism /var/lib/prism/payer-vault
openssl rand -hex 32 > deploy/secrets/prism/payer_vault_key
chown 65532:65532 deploy/secrets/prism/payer_vault_key /var/lib/prism/payer-vault
chmod 0400 deploy/secrets/prism/payer_vault_key
chmod 0700 /var/lib/prism/payer-vault
```
