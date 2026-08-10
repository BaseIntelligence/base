# Deploy secrets (NEVER commit secret bytes)

Containers run as `base` (uid **65532**). Host secret files MUST be:

```bash
chown 65532:65532 deploy/secrets/gateway_sk deploy/secrets/prism_sk \
  deploy/secrets/design_sk deploy/secrets/bounty_sk
chmod 0400 deploy/secrets/gateway_sk deploy/secrets/prism_sk \
  deploy/secrets/design_sk deploy/secrets/bounty_sk
```

Bind-mounts use the file inode; directory mode 0700 is OK.

## Challenge / gateway keys

| Path | Used by | Notes |
|------|---------|-------|
| `gateway_sk` | gateway | Bundle seal mini-secret (`BASE_GATEWAY_SK_FILE`) |
| `prism_sk` | prism-challenge | PRISM challenge mini-secret |
| `design_sk` | design-challenge **only** | Design challenge mini-secret; never mount on egress proxy |
| `bounty_sk` | bounty-challenge **only** | Bounty challenge mini-secret; never mount on other services |
| `challenge_sk` | legacy placeholder | Prefer `prism_sk` / `design_sk` / `bounty_sk`; do not reuse across challenges |

Local dummy for development: decrypt with age:
`age -d -i ~/.base-secrets/age-identity.txt -o deploy/secrets/design_sk ~/.base-secrets/challenge-design.age`

Bounty (throwaway ceremony key under `~/.base-secrets/`):
`age -d -i ~/.base-secrets/age-identity.txt -o deploy/secrets/bounty_sk ~/.base-secrets/challenge-bounty.age`

## Design challenge

| Path | Used by | Notes |
|------|---------|-------|
| `design/annotator_tokens` | design-challenge | One bearer token per line; hashed (SHA-256) at boot. Mode **0400**, uid **65532** |
| `openrouter/api_key` | design-egress-proxy (**proxy only**); also prism reviewer; bounty similarity | Never mount OpenRouter key into design sandboxes |

```bash
mkdir -p deploy/secrets/design deploy/secrets/openrouter
touch deploy/secrets/design/annotator_tokens deploy/secrets/openrouter/api_key
chown -R 65532:65532 deploy/secrets/design deploy/secrets/openrouter
chmod 0400 deploy/secrets/design/annotator_tokens deploy/secrets/openrouter/api_key
```

## Bounty challenge

| Path | Used by | Notes |
|------|---------|-------|
| `bounty/admin_tokens` | bounty-challenge | One bearer token per line; hashed at boot. Mode **0400**, uid **65532** |
| `openrouter/api_key` | bounty-challenge (agentic similar-24h) | Same file mount as other challenges; missing key → SimAgent in CI/local only |

```bash
mkdir -p deploy/secrets/bounty deploy/secrets/openrouter
touch deploy/secrets/bounty/admin_tokens
# ensure openrouter/api_key exists (see Design section)
chown -R 65532:65532 deploy/secrets/bounty
chmod 0400 deploy/secrets/bounty/admin_tokens
```

## Other

- `lium/` — prism Lium API + SSH keys (see prism runbook)
- `wallets/` — btcli wallet trees for gateway owner / validator hotkeys
- `github/token` — prism-challenge top-model publisher: fine-grained GitHub
  token with **contents:write** on `BaseIntelligence/prism` only. Read via
  `PRISM_TOPMODEL_GITHUB_TOKEN_FILE` (`/run/base/github/token`); missing or
  empty file = top-model publish silently disabled. Mode **0400**, uid
  **65532** — never commit it.
