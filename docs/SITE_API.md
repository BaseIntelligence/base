# Site API (`GET /v1/site/*`)

Marketing aggregator on the **master gateway**. CamelCase JSON matching the
frontend `BaseApi` contract (`types.ts` / `contract.ts`).

## Sources

| Site path | Upstream |
|-----------|----------|
| `/v1/site/arenas`, `/arenas/design/*` | Registry pick `challenge_id=design` → `/v1/dashboard`, `/v1/rounds/{id}/leaderboard`, `/v1/harness/{id}`, `/v1/runs/{id}` |
| `/v1/site/arenas/prism/*` | Registry pick `challenge_id=prism` → `/v1/status`, `/v1/submissions`, `/v1/recipe` |
| `/v1/site/network`, `/validators` | Chain tip / metagraph when available; numeric unknowns are `0` or omitted — never invented TAO price/emission |
| `/v1/site/arenas/design/duels` | Always `[]` (admin winners model; no fabricated matchups) |
| Coding arena | `status: "paused"`, empty submissions / matrix / leaderboard |

Design submissions carry **no `url`** (produced HTML is never served); the
public preview is `screenshotUrl` → `/challenge/design/v1/view/{runId}/index.png`,
and runs without a captured screenshot are excluded from the submissions list.
Leaderboard `elo` is the design
`rating` field. Prism window series use real terminal `bpb` with a single
`[final]` point when no step curve is stored.

Backends must be registered (same as challenge proxy), e.g.
`deploy/scripts/register-challenge-backends.sh`.
