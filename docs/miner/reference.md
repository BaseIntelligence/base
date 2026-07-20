# Reference (miners)

Copy-paste oriented. Challenge OpenAPI remains authoritative for body schemas.

## Hosts

| Name | Value |
|------|-------|
| Website | `https://joinbase.ai` |
| Master API | `https://chain.joinbase.ai` |

## Discovery and weights

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/health` | Master liveness; expect `role=master`, `ready=true` |
| `GET` | `/version` | Master build identity |
| `GET` | `/v1/registry` | Challenges, status, **`emission_percent`** |
| `GET` | `/v1/weights/latest` | Sealed final vector for validators; may 404 before first seal |
| `GET` | `/v1/validators/public` | Public validator inventory (ops / transparency) |

## Challenge proxy pattern

Replace `{slug}` with `prism` or `agent-challenge`.

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/challenges/{slug}/openapi.json` | Public readiness (prefer over `/health`) |
| `GET` | `/challenges/{slug}/docs` | Swagger UI |
| `GET` | `/challenges/{slug}/leaderboard` | Challenge leaderboard |
| `*` | `/challenges/{slug}/...` | Generic public proxy to challenge routes |
| `POST` | `/v1/challenges/{slug}/submissions` | Raw ZIP **bridge** (BASE verifies/forwards when enabled) |

Blocked on public proxy (expect **403**, not miner bugs):

- `/challenges/{slug}/health`
- `/challenges/{slug}/version`
- `/challenges/{slug}/internal/*`
- Other challenge-private / capability surfaces documented as blocked

## Prism (common)

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/challenges/prism/submissions` | Signed ZIP upload bridge |
| `GET` | `/challenges/prism/leaderboard` | Rankings |
| `GET` | `/challenges/prism/openapi.json` | Schemas |

Exact status and report paths: Prism OpenAPI.

## Agent Challenge (common)

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/registry` | Confirm `agent-challenge` active + emission |
| `GET` | `/challenges/agent-challenge/benchmarks` | Benchmark listing |
| `POST` | `/v1/challenges/agent-challenge/submissions` | Raw ZIP bridge |
| `POST` | `/challenges/agent-challenge/submissions` | JSON/base64 generic proxy path (signed local) |
| `GET` | `/challenges/agent-challenge/submissions/{id}/status` | Status |
| `GET` | `/challenges/agent-challenge/submissions/{id}/events` | Event stream / history |
| `GET` / `PUT` | `/challenges/agent-challenge/submissions/{id}/env` | Env metadata (values write-only) |
| `POST` | `/challenges/agent-challenge/submissions/{id}/env/confirm-empty` | Explicit empty env |
| `POST` | `/challenges/agent-challenge/submissions/{id}/launch` | Lock env and queue eval |
| `GET` | `/challenges/agent-challenge/leaderboard` | Best row per hotkey (v1) |

v1 list notes: submissions list may return latest 100 newest-first; leaderboard one best
row per hotkey. Pagination deferred.

## Signature headers (miner-signed routes)

Use placeholders only in docs and examples:

```http
X-Hotkey: <miner-hotkey>
X-Signature: <signature>
X-Nonce: <nonce>
X-Timestamp: <timestamp>
```

Canonical payload bytes are **challenge-defined**. Sign with the same hotkey you mine under.
Never commit real signatures, mnemonics, or admin tokens.

## Emission shares (read path)

```bash
curl -fsS https://chain.joinbase.ai/v1/registry
```

Interpret `emission_percent` as absolute contribution of that challenge’s normalized raw
weights into the sealed vector. Default network policy: **Prism 50 + Agent Challenge 50**.

## Env action limits (Agent Challenge)

As enforced by the challenge (summary for integrators):

| Constraint | Typical bound |
|------------|---------------|
| Env key charset | `^[A-Za-z_][A-Za-z0-9_]{0,127}$` |
| Keys per request | ≤ 64 |
| Value size | ≤ 16 KiB each |
| Total payload | ≤ 128 KiB |
| `PUT /env` | Replaces full set |
| Values | Write-only; responses expose metadata only |

## Related

- [Getting started](getting-started.md)
- [Troubleshooting](troubleshooting.md)
- [../challenge-integration.md](../challenge-integration.md) — owner-facing challenge contract
