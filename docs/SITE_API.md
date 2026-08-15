# Site API (`GET /v1/site/*`)

Marketing aggregator on the **master gateway**. CamelCase JSON matching the
frontend `BaseApi` contract (`types.ts` / `contract.ts`).

## Sources

| Site path | Upstream |
|-----------|----------|
| `/v1/site/arenas`, `/arenas/design/*` | Registry pick `challenge_id=design` → `/v1/dashboard`, `/v1/rounds/{id}/leaderboard`, `/v1/harness/{id}`, `/v1/runs/{id}` |
| `/v1/site/arenas/prism/*` | Registry pick `challenge_id=prism` → `/v1/status`, `/v1/submissions`, `/v1/submissions/{id}`, `/v1/submissions/{id}/diff`, `/v1/recipe` |
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
no step curve is stored. `PrismWindow.tokenBudget` is **0** unless a recipe
publishes a fixed token quota (prism ≥1.2 does not — caps are wall-clock /
steps / params). Chart x-values still come from miner telemetry
(`layer_stats.tokens` when present); clients must not label the max observed
x as an egalitarian “token window.” Prism **leaderboard** lists **champions
only** (`score.kind=score` and `value > 0` — current top and historical
ex-tops). Prism **`/submissions`** defaults to **`scope=all`** (in-flight +
terminal, including Score=0 / failed) so the marketing FE can show live
progress; pass `?scope=champions` for the scored gallery only. Missing
`recipeEra` on a row is filled as **`legacy`** after detail fan-out (or when
no AutoModel signals exist).

### Prism era, benches, detail, GPT-2 references

Champion list / leaderboard rows may carry (when detail fan-out succeeds):

| Field | Meaning |
|-------|---------|
| `recipeEra` | `"automodel"` \| `"legacy"` — AutoModel if `pin_id` / recipe major ≥ 2 / diff shape; else legacy |
| `pinId` | AutoModel pin (`automodel@…`) when era is automodel |
| `evalGroups` | `{ group: "g1"…"g8", g }` from composite `eval.groups` |
| `benchmarks` | Public G2 subset: `hellaswag`, `arcEasy`, `arcChallenge`, `piqa`, `winogrande`, `boolq` from `org.g2.*` / battery keys (Zone-A `/metrics?zone=a` fan-in when detail omits them) |
| `submissionId` | (leaderboard) best-BPB champion id for the detail modal |

Additional Prism routes:

| Path | Response |
|------|----------|
| `GET /v1/site/arenas/prism/submissions/{id}` | `PrismSubmissionDetail` — list fields + `eval` summary (status, groups, gates, composite) + telemetry + public `review` / `similarity` (quality/kind only). **No** raw patch text. |
| `GET /v1/site/arenas/prism/references` | `PrismReferenceBaseline[]` — frozen **Prism-protocol** GPT-2 references (Large 774M **and** Small 124M): measured val **`bpb`** + G2 benches from 1×RTX 5090 eval-only runs on the public pack (`gpt2-large` + `openai-community/gpt2`). Includes `sourceUrl` / `disclaimer`. |
| `GET /v1/site/arenas/prism/submissions/{id}/telemetry` | Existing loss-curve payload (also embedded on detail). |

GPT-2 Large + Small constants live in `crates/site-api` (`prism_enrich`) so API and FE stay aligned; they are **measured Prism-protocol** numbers (eval-only, public pack), not Eleuther literature tables. List/leaderboard row shells still map in `crates/site-data`.

`GET /v1/site/arenas/{slug}/submissions` and `/leaderboard` accept optional
`?q=` — case-insensitive substring over miner hotkey (SS58 or hex), handle,
slug, operator, and (for submissions) prompt title / id / run id.

Backends must be registered (same as challenge proxy), e.g.
`deploy/scripts/register-challenge-backends.sh`.
