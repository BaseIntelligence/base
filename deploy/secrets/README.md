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
| `prism_sk` | prism-challenge | PRISM challenge mini-secret |
| `design_sk` | design-challenge **only** | Design challenge mini-secret; never mount on egress proxy |
| `challenge_sk` | legacy placeholder | Prefer `prism_sk` / `design_sk`; do not reuse across challenges |

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

- `lium/` — Lium API key + funding deposit wallet placeholders (see [`lium/README.md`](lium/README.md) and [`docs/LIUM_FUNDING.md`](../../docs/LIUM_FUNDING.md)); also prism SSH keys (prism runbook)
- `wallets/` — btcli wallet trees for gateway owner / validator hotkeys
- `github/token` — prism-challenge top-model publisher: fine-grained GitHub
  token with **contents:write** on `BaseIntelligence/prism` only. Read via
  `PRISM_TOPMODEL_GITHUB_TOKEN_FILE` (`/run/base/github/token`); missing or
  empty file = top-model publish silently disabled. Mode **0400**, uid
  **65532** — never commit it.
