# Bounty Challenge (Base)

**Status: live contract** for `challenge_id = "bounty"`, `challenge_scoring_version` **u16 = 1**.

Normative contract for the bounty (video bug-report) challenge on Base.
Byte-level epoch bundle rules live in [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md)
(`protocol_version = 1`). Enablement / emission ceremony:
[`runbooks/bounty-enable-and-emission.md`](./runbooks/bounty-enable-and-emission.md).
Miner-facing mirror: [`external-miner/bounty.md`](./external-miner/bounty.md).

Miners upload a **video + structured bug report**. The master pipeline compresses
the video, runs an agentic **similar-24h** check (DeepSeek V4 Flash via
OpenRouter), and queues novel reports for **admin approve / reject**. Scoring
targets **50 approved bugs per epoch** for the full bounty emission share;
shortfall burns via a UID-0 leaf (see §6).

---

## 1. What runs where (topology)

```text
Miner  --POST /v1/bugs (multipart)-->  bounty-challenge (:8095)
                                          |
                              store raw video + pending row
                                          |
                         worker: ffmpeg compress + fingerprint
                                          |
                    OpenRouter DeepSeek V4 Flash (similar-24h)
                         |                        |
                    duplicate → rejected     novel → pending_admin
                                                      |
                                         admin approve / reject
                                                      |
                              epoch emitter → D24 leaves → gateway
```

| Process | Host | Holds `bounty_sk`? | Holds OpenRouter key? |
|---------|------|--------------------|------------------------|
| `bounty-challenge` | **master only** | **yes** (file mount) | **yes** (similar-24h; optional Sim) |
| gateway | master | no | no |
| validator | validators | no | no — **no challenge exec**; fetch sealed weights only |

Evaluation (compress, agentic similarity, admin approve, leaf emit) is
**master-only**. Validators never run `bounty-challenge`.

Admin routes are **not exposed via gateway** (`/challenge/bounty/v1/admin/*`
→ `403`; gateway `is_admin_path` blocks all `v1/admin/*`). Use the master-local
challenge port (compose expose `8095`, local overlay `28095`).

---

## 2. Identifiers and versions

| Field | Value |
|-------|-------|
| `challenge_id` | `bounty` |
| `challenge_scoring_version` | **u16 = 1** |
| `SCORE_MAX` | `1_000_000` |
| `TARGET_BUGS` | `50` (approved bugs = full bounty share this epoch) |
| Listen port | `8095` (local overlay `28095`) |
| Gateway proxy prefix | `/challenge/bounty/*` |
| Bundle `protocol_version` | `1` ([`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md)) |
| `emission_share_bps` | **2500** (with design `3000` / prism `4500`; sum `10000`) |
| Policy | `all_metagraph_hotkeys` |
| Domain bug id | `base-bounty-bug-id-v1` |
| Domain submission | `base-bounty-submission-v1` |
| Raw-weight domain | `base-rawweight-v1` (via bundle) |
| Default OpenRouter model | `deepseek/deepseek-v4-flash` (`BOUNTY_OPENROUTER_MODEL`) |

Emission posture: design **3000** / prism **4500** / bounty **2500** bps.
Rebalance via the owner ceremony in
[`runbooks/bounty-enable-and-emission.md`](./runbooks/bounty-enable-and-emission.md)
and [`config/challenges.toml`](../config/challenges.toml).

---

## 3. Miner submit contract

### `POST /v1/bugs`

`multipart/form-data` fields:

| Field | Required | Notes |
|-------|----------|-------|
| `video` | yes | `mp4` / `webm` / `mov`; raw size cap ~100 MiB |
| `title` | yes | Short human title |
| `description` | yes | Bug narrative |
| `app_id` | yes | Target app slug |
| `steps` | no | Repro steps text |

Headers: `X-Miner-Hotkey` (64 lowercase hex). Shared metagraph / 1-max gating
via `submission-gating` (same family as design/prism).

Gateway path: `POST /challenge/bounty/v1/bugs`.

### Read APIs

| Route | Notes |
|-------|-------|
| `GET /v1/bugs/{id}` | Status, similarity verdict, metadata; compressed video URL |
| `GET /v1/bugs/{id}/video` | Stream compressed `video.mp4` |
| `GET /v1/bugs?status=&mine=1` | List (filter by status / own hotkey) |
| `GET /health` | Liveness |
| `GET /v1/status` | Epoch, approved/pending counts, burn preview |

---

## 4. Processing pipeline

1. **Intake** — validate MIME/size; write staging artifact; row `uploaded`.
2. **Compress** — `ffmpeg` in the bounty container (`libx264`/`libx265`, max
   720p, CRF ~28, aac) → `video.mp4` on volume `bounty-artifacts`; delete raw;
   store sha256 + bytes.
3. **Corpus 24h** — bugs in `approved|pending_admin|rejected` with
   `created_at >= now-24h` (exclude same hotkey/coldkey).
4. **Agentic similar-24h** — `OpenRouterAgent` with
   `BOUNTY_OPENROUTER_MODEL` (default DeepSeek V4 Flash). Structured verdict
   `novel | duplicate` + `nearest_id` + `similarity_bps` + rationale.
   Fail-closed on infra errors (retry / `NoScore` path — never invent `novel`).
5. **Duplicate** → terminal `rejected` (0 points). **Novel** → `pending_admin`.
6. **Admin approve** → +1 epoch point for the miner hotkey. **Reject** →
   terminal with reason.

CI / local: `BOUNTY_FORCE_SIM=1` → deterministic `SimAgent` (no OpenRouter /
host ffmpeg required when fixture is pre-compressed). **Never** enable Sim on
staging/prod droplets.

---

## 5. Admin API (master-local)

Bearer hashes from `deploy/secrets/bounty/admin_tokens` (one token per line;
hashed at boot). Empty file → all admin routes reject.

| Route | Body / notes |
|-------|----------------|
| `GET /v1/admin/bugs?status=pending_admin` | Queue |
| `GET /v1/admin/bugs/{id}` | Detail + video + nearest_duplicate |
| `POST /v1/admin/bugs/{id}/approve` | Award +1 point |
| `POST /v1/admin/bugs/{id}/reject` | `{ "reason": "…" }` |

---

## 6. Epoch scoring + burn sink

Aggregator renormalizes positive scores inside a challenge — a lone
`Score(n)` does **not** burn the remainder. Bounty uses an explicit UID-0
burn leaf (compatible with [`BUNDLE_SPEC`](./BUNDLE_SPEC.md) §6.5):

```text
TARGET = 50
approved_points[m] = # bugs approved for miner m this epoch
total = sum(approved_points)
capped = min(total, TARGET)
miner_pool = SCORE_MAX * capped / TARGET          # integer
burn_units = SCORE_MAX - miner_pool                 # 0 when total >= TARGET

if total > 0:
  each miner: Score(floor(miner_pool * points_m / total))
else:
  participants Score(0) / NoScore(NotAttempted)

if burn_units > 0:
  Score(burn_units) on metagraph hotkey with uid == 0
  # dropped at aggregate → burns into bounty frac
```

- **Under 50** approved: missing mass → burn UID 0.
- **At/above 50**: `burn_units = 0`; proportional share among miners (dilution).

Leaves are D24-complete for the exact epoch (`exact-E`) and posted to
`POST /v1/weights/raw` with `bounty_sk` matching the trust-root pubkey.

---

## 7. Persistence

Migration `0017_bounty_challenge.sql`:

- `bounty_bug` — identity, miner keys, app/title/description/steps, status,
  agentic verdict, nearest_id, video metadata/path, epoch, timestamps
- `bounty_stage_event` — append-only journal
- `bounty_epoch_score` — points / final score per `(epoch, hotkey)`

Video bytes live on volume `bounty-artifacts` (not base64 in Postgres).

---

## 8. Deploy knobs

| Env / mount | Role |
|-------------|------|
| `BASE_CHALLENGE_BIND` | default `0.0.0.0:8095` |
| `BASE_CHALLENGE_SK_FILE` | `/run/base/challenge_sk` ← `deploy/secrets/bounty_sk` |
| `BOUNTY_ADMIN_TOKENS_FILE` | `/run/base/bounty/admin_tokens` |
| `OPENROUTER_API_KEY_FILE` | `/run/base/openrouter/api_key` |
| `BOUNTY_ARTIFACTS_ROOT` | `/var/lib/bounty` |
| `BOUNTY_FORCE_SIM` | CI/local only; staging/prod `false` |
| `BOUNTY_OPENROUTER_MODEL` | default `deepseek/deepseek-v4-flash` |
| `BASE_DATABASE_URL` | required in compose (`deploy/env/bounty-challenge.env`) |

Compose service: `bounty-challenge`. Dockerfile target: `bounty-challenge`
(ffmpeg in runtime). Backend registration:
`deploy/scripts/register-challenge-backends.sh`.

---

## 9. Out of scope

- Public miner GitHub repo (follow-up ops; monorepo mirror is
  [`external-miner/bounty.md`](./external-miner/bounty.md)).
- Product bug **fixes** — the challenge scores **approved reports**, not remediations.
- Aggregation algorithm changes / `algorithm_version` bumps.
