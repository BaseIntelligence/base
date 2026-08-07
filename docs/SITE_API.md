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
| `/v1/site/activity` | Design `recent_runs` + round winners + prism submissions → ops-style English lines (deduped) |
| Coding arena | `status: "paused"`, empty submissions / matrix / leaderboard |

Design submissions carry **no `url`** (produced HTML is never served); the
public preview is `screenshotUrl` → `/challenge/design/v1/view/{runId}/index.png`
(relative path on the gateway). Marketing clients should resolve that path to the
**absolute** gateway host for `<img src>` (e.g.
`https://chain.joinbase.ai/challenge/design/v1/view/{runId}/index.png`) so PNG
bytes are not proxied through the site's Vercel `/gbase-api` rewrite. JSON
`/v1/site/*` calls may keep using the same-origin proxy. Runs without a captured
screenshot are excluded from the submissions list. Design `GET /v1/dashboard`
`recent_runs` therefore prioritizes post-sanitize stages (`awaiting_admin`,
`scored`, …) over a flood of brand-new `queued` rows so the site gallery is not
starved.

Leaderboard `elo` is the design `rating` field. When the current round has no
winners yet (`ratings: []`), `/v1/site/arenas/design/leaderboard` surfaces the
previous round's standings (`roundId` = previous) rather than an empty board.
Prism window series use real terminal `bpb` with a single `[final]` point when
no step curve is stored.

`GET /v1/site/arenas/{slug}/submissions` and `/leaderboard` accept optional
`?q=` — case-insensitive substring over miner hotkey (SS58 or hex), handle,
slug, operator, and (for submissions) prompt title / id / run id.

Backends must be registered (same as challenge proxy), e.g.
`deploy/scripts/register-challenge-backends.sh`.
