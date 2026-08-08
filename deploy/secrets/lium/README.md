# Lium secrets (NEVER commit secret bytes)

Operator Lium account + GPU funding deposit wallet for shared challenge prepay
([`docs/LIUM_FUNDING.md`](../../../docs/LIUM_FUNDING.md)).

Host files MUST be mode **0400**, owner uid **65532** (`base`), same as other
`deploy/secrets/` material. Bind-mount into the master challenge containers only.

## Placeholders (create empty files locally; fill via age)

| Path | Used by | Notes |
|------|---------|-------|
| `lium/api_key` | `prism-challenge` / funding | Lium `X-API-Key` for operator account (`LIUM_API_KEY_FILE=/run/base/lium/api_key`). Same key as pod rent. |
| `lium/deposit_ss58` | funding quote | Operator **deposit coldkey SS58** miners pay TAO to (`LIUM_FUNDING_DEPOSIT_ADDRESS_FILE`). |
| `lium/deposit_coldkey` | operator only | Bittensor coldkey material for the deposit wallet — **never** mount into containers unless a future watcher needs it; prefer host-side watcher. |
| `lium/deposit_hotkey` | operator / `lium fund` | Hotkey used when sweeping TAO into Lium via CLI `lium fund` (operator runbook). |
| `lium/funding_admin_token` | funding admin routes | Bearer for `GET /v1/funding/admin/credits` (`LIUM_FUNDING_ADMIN_TOKEN_FILE`). |

```bash
mkdir -p deploy/secrets/lium
touch deploy/secrets/lium/api_key \
      deploy/secrets/lium/deposit_ss58 \
      deploy/secrets/lium/deposit_coldkey \
      deploy/secrets/lium/deposit_hotkey \
      deploy/secrets/lium/funding_admin_token
chown -R 65532:65532 deploy/secrets/lium
chmod 0400 deploy/secrets/lium/*
```

## Env knobs (non-secret)

| Env | Default | Meaning |
|-----|---------|---------|
| `PRISM_REQUIRE_LIUM_FUNDING` | `0` | When `1`, orchestrator refuses Lium rent without an unspent credit |
| `PRISM_FUNDING_RATE_USD_PER_HOUR` | `0.67` | Prism GPU USD/h |
| `PRISM_FUNDING_HOURS` | `6` | Prism billable hours |
| `LIUM_FUNDING_BUFFER` | `0.10` | +10% buffer |
| `LIUM_FUNDING_TAO_USD` | (oracle) | Fixed/oracle USD per TAO for quotes |
| `LIUM_FUNDING_QUOTE_TTL_SECS` | `900` | Quote lifetime |

**Do not enable `PRISM_REQUIRE_LIUM_FUNDING` in prod** until the deposit wallet,
live oracle, and on-chain watcher are validated on staging.
