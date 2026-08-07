# Design Challenge (Base)

**Status: FROZEN** for `challenge_id = "design"`, `challenge_scoring_version` **u16 = 2**.

Normative contract for the design challenge on Base. Byte-level epoch bundle
rules live in [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) (`protocol_version = 1`).
Pin map for CI: [`DESIGN_CHALLENGE_CHECKLIST.md`](./DESIGN_CHALLENGE_CHECKLIST.md).
Gate: `cargo run -p xtask -- design-check`.

This challenge **replaces** agent-v1 (Phala CVM) and hypertraining. Miners submit
Python harness source over HTTP — no miner-provided Docker image, no Phala/CVM path.

---

## 1. What runs where (topology)

```text
Miner  --POST /v1/harness-->  design-challenge (:8093)
                                  |
                    round clock (10/day UTC) + quota
                                  |
                    +-------------v--------------+
                    | design-sandbox (two phase) |
                    |  install → run             |
                    |  image: design-runtime     |
                    |  net: design-sandbox-egress|
                    +------+---------------------+
                           | HTTP only
                    +------v---------------------+
                    | design-egress-proxy         |
                    |  open egress, internal      |
                    |  blocklist; OpenRouter key  |
                    +----------------------------+
                           |
                    /out/pages/*.html
                           |
                    design-sanitize → store (postgres)
                           |
          +----------------+----------------+
          |                                 |
   viewer (index.png          agentic review → admin winners (1|2)
   screenshots only;                       |
   HTML never served)         exact-E leaves → gateway /v1/weights/raw
```

| Process | Host | Holds `design_sk`? | Holds OpenRouter key? |
|---------|------|--------------------|------------------------|
| `design-challenge` | **master only** | **yes** (file mount) | **yes** (agentic review; optional Sim) |
| `design-egress-proxy` | **master only** | no | **yes** (sandbox LLM path) |
| sandbox container | master (ephemeral) | **never** | **never** |
| gateway | master | no | no |
| validator | validators | no | no — **no challenge exec**; fetch sealed weights only |

Evaluation (sandbox, sanitize, `AgenticReview`, admin winners, leaf emit) is
**master-only**. Validators never run design-challenge / egress / socket-proxy.

Sandbox containers attach only to the internal Docker network
`design-sandbox-egress`. The sole reachable peer is `design-egress-proxy`.

---

## 2. Identifiers and versions

| Field | Value |
|-------|-------|
| `challenge_id` | `design` |
| `challenge_scoring_version` | **u16 = 3** |
| `SCORE_MAX` | `1_000_000` |
| Listen port | `8093` (local overlay `28093`) |
| Gateway proxy prefix | `/challenge/design/*` |
| Bundle `protocol_version` | `1` ([`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md)) |
| `emission_share_bps` | **5000** (equal split with prism; sum `10000`) |
| Policy | `all_metagraph_hotkeys` |
| Round length | `ROUND_SECS = 8_640` (10 rounds / UTC day) |
| Agent run timeout | `AGENT_RUN_TIMEOUT_SECS = 1_800` (30 min; distinct from round length) |
| Scoring window | `SCORING_WINDOW_ROUNDS = 10` (rolling round-win points share) |
| `round_id` | `floor(unix_secs / 8640)` |
| Domain `round_id` | `base-design-round-id-v1` |
| Domain submission | `base-design-submission-v1` |
| Domain pair | `base-design-pair-id-v1` |
| Domain run | `base-design-run-id-v1` |
| Raw-weight domain | `base-rawweight-v1` (via bundle) |

Staging knobs (defaults above are the prod pins and never change implicitly):
`DESIGN_ROUND_SECS`, `DESIGN_AGENT_RUN_TIMEOUT_SECS`, `DESIGN_PROMPTS_PER_ROUND`
override round length / run timeout / prompts per round (staging uses ~15-minute
rounds; `env-staging.yml`).

Emission posture: `emission_share_bps = 5000` for design and `5000` for prism
(50/50; sum `10000`). Rebalance via the owner ceremony in
[`runbooks/design-enable-and-emission.md`](./runbooks/design-enable-and-emission.md)
and [`config/challenges.toml`](../config/challenges.toml).

---

## 3. Miner harness contract

Miners submit **source**, not images. Preferred transport is **ZIP**
(`Content-Type: application/zip` + `X-Miner-Hotkey`, or JSON `zip_base64`).
JSON with inline `agent_py` / `pyproject_toml` remains accepted for local/CI.

Optional `env_vars` (JSON map or `X-Env-Json` on ZIP) are injected into the
sandbox **run** phase only. Keys must match `[A-Z][A-Z0-9_]*`; operator /
proxy prefixes (`DESIGN_`, `HTTP_`, `PYTHON…`, …) are rejected. Values are
**never logged**.

### Bundle

| File | Rule |
|------|------|
| `agent.py` | Must define `def run(task, llm, out) -> None` |
| `pyproject.toml` | Required; deps installed in sandbox install phase |
| Extra files | ≤ 16 files, ≤ 256 KiB each, total bundle ≤ 1 MiB |

`harness_id = sha256(base-design-submission-v1 || hotkey || agent || pyproject || extras || env)`.
`POST /v1/harness` is idempotent on that digest.

### Submission gating (1-max)

- The miner hotkey must be **in the metagraph** (cached snapshot; no signature
  added). Unknown hotkey → `403 hotkey_not_in_metagraph`; snapshot not ready →
  `503 metagraph_unavailable`.
- **One accepted submission per `(challenge, hotkey)`** (`submission_gating`
  table). While the row is `registered` / `blocked` / `rejected`, a *different*
  harness from the same hotkey → `409 submission_gated`. The identical re-POST
  stays idempotent (`200 already-queued`).
- **Auto-retry**: infra-class failures (`install`, `ast_infra`, `llm_infra`)
  requeue automatically up to **3** times; `cheat` / `rejected` are terminal.
  Budget exhaustion → `failed` + gating `blocked`. Manual
  `POST /v1/runs/{id}/retry` is unchanged.
- A metagraph **watcher** reopens eligibility (`open`) when the hotkey leaves
  the metagraph (uid deregistered or hotkey replaced), so the owner may
  resubmit under a new uid.
- Miner env (`env_vars`) is **locked at submission** into the stored bundle;
  changing it requires a new digest (and a free gating slot).

### Required output (`/out/pages/`)

- `index.html`
- `pricing.html`
- `components.html`
- `manifest.json` (present in operator harness layout; pages gated by sanitize)

### Injected SDK (`base_design`, not miner-modifiable)

- `task.prompt`, `task.round_id`, `task.pages_required`, `task.budget`
- `llm.chat(messages, model)` → HTTP to egress proxy (no key in harness)
- `out.write_page(name, html)`, `out.write_asset(name, bytes)` (size-capped)

Operator entrypoint `design_harness.py` loads miner `agent.py` after install.

---

## 4. Sandbox hardening

Two phases on pinned `design-runtime`:

1. **install** — `pip install --no-cache-dir -e /work` into a work-root venv,
   resolving `pyproject.toml` deps from PyPI via the egress proxy (open
   Internet; internal-target blocklist). Miner env is **not** injected in this
   phase. Timeout `DESIGN_INSTALL_TIMEOUT_SECS` (default 300s); non-zero exit
   or timeout → error class `install` (auto-retry ≤ 3, then `failed` with
   `NoScore(ChallengeInternal)`).
2. **run** — same work root; open egress via the proxy for external APIs /
   MCP servers (same blocklist); `llm.chat` rides the budgeted `/v1/chat`
   path; miner env injected; loads `design_harness.py`.

The egress proxy blocklist refuses cloud metadata (`169.254.169.254`),
loopback, RFC1918/VPC (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`),
CGNAT (`100.64.0.0/10`), and control-plane service names, enforced **after
DNS resolution** (DNS-rebinding safe).

`docker-engine` `RunSpec` hardening for design (prefix `base-design-`):

- `ReadonlyRootfs: true` + bounded tmpfs `/tmp`
- `CapDrop: ["ALL"]`
- `SecurityOpt: ["no-new-privileges:true"]`
- `PidsLimit`, `Memory`, `MemorySwap`, `NanoCpus` set
- `NetworkMode: design-sandbox-egress` (pre-created; socket-proxy denies Networks API)
- `User: 65532:65532`
- Wall-clock timeout → stop/rm

Host `SimSandbox` is fail-closed outside explicit non-prod/CI opt-in
(`BASE_ALLOW_HOST_SIM=1` + non-prod, typically via `env-local.yml` or e2e).
Staging/prod paths are Docker-only via `socket-proxy` — no silent fallback.

Floating tags (`:latest`) are **forbidden** for `design-runtime` / challenge images in prod pins — digest-only.

---

## 5. Sanitize rules

Ingestion via `design-sanitize` (ammonia + CSS filter). **Produced HTML is
never served** — sanitized pages are orchestrator input only (screenshot
capture, anti-cheat review); the public viewer serves PNG screenshots
only (§6).

### Stripped / rejected

- Tags: `script`, `iframe`, `object`, `embed`, `applet`, `base`, `form`, `meta[http-equiv=refresh]`, `link[rel=import]`, scriptable SVG
- All `on*` event attributes
- URL schemes: reject `javascript:`, `vbscript:`, `data:text/html`; allow `http`, `https`, `mailto`, `data:image/*`
- CSS: soft-strip `@import …;` rules (remainder of the `<style>` block is kept);
  hard-reject (drop the whole block/attr) for `expression(`, `url(javascript:`,
  IE-only `behavior:` (not `scroll-behavior:`), `-moz-binding`
- External `<link rel=stylesheet>` is stripped (no CDN / Tailwind CDN); miners
  must embed presentation CSS in `<style>` or inline `style=` for screenshots
  and the sandboxed viewer to look styled

### Annotator signal

`sanitize_report` (including `js_stripped`) is stored and shown to annotators.
JS stripped is **not** an automatic `Score(0)` — only a visible signal. Invalid /
missing required pages → automatic `Score(0)` at scoring gates.

---

## 6. Viewer headers and CSP

`GET /v1/view/{run_id}/{page}` serves **PNG screenshots only** (`image/png`,
`private, no-store`, `nosniff`). **Produced HTML is never served**: requests
for `.html` pages (or bare page names) return `410 Gone` with a short JSON
error, and `GET /v1/runs/{id}/bundle.json` no longer embeds page HTML (same
`410 Gone` contract — use `/v1/runs/{id}/pages` for page metadata). Miner
output reaches browsers exclusively as the captured `index.png` screenshot.

The full lockdown header set remains as the **gateway-enforced floor** on
every `/challenge/{id}/v1/view/*` response (defense in depth — below):

```
Content-Security-Policy: sandbox; default-src 'none'; img-src data: https:; style-src 'unsafe-inline' https:; font-src data: https:; base-uri 'none'; form-action 'none'; frame-ancestors <allowlist>
X-Content-Type-Options: nosniff
Referrer-Policy: no-referrer
Cross-Origin-Resource-Policy: same-origin
Cross-Origin-Opener-Policy: same-origin
Permissions-Policy: accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()
Cache-Control: private, no-store
```

The `sandbox` directive is emitted **without** `allow-scripts` and without
`allow-same-origin`: even if a stale or misbehaving challenge upstream served
miner HTML through the gateway, it would run in an **opaque origin with script
execution disabled**, unable to read the serving origin's cookies,
storage, or DOM — even though joinbase.ai embeds view responses
**same-origin** through the `/gbase-api` proxy. These responses **never carry
`Set-Cookie`** (the gateway strips it); the endpoint stays public (`run_id` is
the capability) and reads no auth cookie.

`frame-ancestors <allowlist>` defaults to
`'self' https://joinbase.ai https://*.vercel.app http://localhost:*`
(`DESIGN_FRAME_ANCESTORS` override): the public site, Vercel preview deploys,
and local dev may embed the viewer; everyone else is refused. `'self'` covers
same-origin proxy embedding at any host (including staging consoles).

**Defense in depth — gateway re-injection.** The gateway proxy re-applies the
full lockdown header set (and strips any `Set-Cookie`) on every
`/challenge/{id}/v1/view/*` response
(`BASE_GATEWAY_VIEW_FRAME_ANCESTORS` override, same default), so even a stale
or misbehaving challenge upstream cannot serve miner HTML through the gateway
without the sandbox floor. `Cross-Origin-Resource-Policy: same-origin` means
cross-origin embedders get nothing: integrations must load screenshots through
a same-origin proxy (as joinbase.ai does), not the bare gateway origin.

CSP `sandbox` (without `allow-scripts`) neutralizes script even if an integrator
omits the iframe `sandbox` attribute. Screenshots-only serving makes miner
HTML unreachable in the first place; the header floor is the second line.

### Full-page screenshot (`index.png`)

After sanitize, the orchestrator renders the sanitized `index.html` **artifact**
(store → temp file → `file://`) in headless Chromium inside the
design-challenge container — never the public `/v1/view` URL (screenshots-only;
the stored artifact is the capture source of truth).
Chromium runs `--no-sandbox` because the renderer sandbox needs user
namespaces / `CAP_SYS_ADMIN`, which Docker does not grant; the container
boundary plus the scriptless sanitized artifact is the sandbox. Two passes
(measure height via `--dump-dom`, then `--screenshot --window-size=1280×H`),
hard process timeout, one retry; failure never fails the run. The PNG is
stored as the `index.png` artifact (base64) and served at
`GET /v1/view/{run_id}/index.png` (`image/png`, `private, no-store`,
`nosniff`); run detail exposes `screenshot_url` when the artifact exists.

Backfill (idempotent; upserts on `(run_id, path)` so it can be re-run and can
race a live capture safely):

```bash
# on the master host, from the repo checkout (env comes from the container):
docker compose -f docker-compose.yml -f deploy/compose/role-master.yml \
  -f deploy/compose/env-prod.yml exec design-challenge \
  design-challenge backfill-screenshots --limit 500
```

When historical rows were sanitized by a buggy filter that wiped `<style>`
(raw still has CSS; sanitized does not), `backfill-screenshots` is not enough —
it only re-renders the already-broken sanitized HTML and skips runs that already
have `index.png`. Use re-sanitize from `raw_html` (throttled Chromium):

```bash
docker compose -f docker-compose.yml -f deploy/compose/role-master.yml \
  -f deploy/compose/env-prod.yml exec design-challenge \
  design-challenge backfill-resanitize --limit 500 --sleep-ms 2000
# or pin specific runs:
#   design-challenge backfill-resanitize --run-id <id> --run-id <id2> --sleep-ms 2000
```

---

## 7. Rounds and quotas

- **Round** every `ROUND_SECS = 8_640` (10 rounds / UTC day):
  `round_id = floor(unix_secs / 8640)`.
- **Registration waits for the next round**: an accepted harness is scheduled
  into `round_id(now) + 1` and its runs only become claimable once that round
  opens — never into the round already in flight.
- **Agent run timeout**: `AGENT_RUN_TIMEOUT_SECS = 1_800` (30 minutes wall clock
  for the sandbox run phase — not the round length).
- **Prompts**: repo-pinned bank (`bank_v1.json`, no human approval API);
  deterministic weighted draw
  `SHA256(domain || round_id || bank_digest)` → 3 prompts per round;
  identical for every harness in that round.
- **Quota (two buckets, per hotkey per UTC day)** — an honest harness that runs
  every round must never be locked out, so organizer-scheduled work does not
  draw on the miner's anti-spam allowance:

  | Bucket | Charged by | Cap | Override |
  |--------|-----------|-----|----------|
  | Manual | `POST /v1/harness` (miner-initiated) | `MANUAL_DAILY_RUN_QUOTA = 10` | `DESIGN_MANUAL_DAILY_RUN_QUOTA` |
  | Scheduled | Round dispatch + `admin/rounds/current/requeue` | `rounds/day × prompts/round × SCHEDULED_DAILY_RUN_HEADROOM` (= **60** at 10 rounds × 3 prompts) | `DESIGN_SCHEDULED_DAILY_RUN_CAP` (clamped ≥ the day's own schedule) |

  Enforcement is **per bucket**: exhausting manual submissions never stops the
  round scheduler, and the scheduled cap is a runaway-scheduler guard, not a
  participation limit. `GET /v1/quota/{hotkey}` reports both.
- **Auto-retry**: infra-class failures (`install` / `ast_infra` / `llm_infra`)
  requeue up to 3 times (`DESIGN_AUTO_RETRY_MAX`), then terminal
  `NoScore(ChallengeInternal)` + gating `blocked`.
- Automatic gates → `Score(0)`: invalid bundle, missing pages, timeout, crash.
- Operator / infra fault → `NoScore(ChallengeInternal)`.

---

## 8. Admin winners + agentic anti-cheat

Stages after sanitize: **`AgenticReview` → `AwaitingAdmin`** (clean only).
Cheat / suspicious → immediate `Score(0)` (not admin-eligible).

Human role is **only** selecting **1 or 2** winner harnesses per round
via `POST /v1/admin/rounds/{id}/winners` (no prompt approval, no page-pair
Elo on the leaf path). Annotate endpoints are deprecated / unused for scoring.
Prompt bank `bank_v1.json` is fully automatic — no human prompt validation.

Shared verifier: `challenge-agentic` (tools + `challenge-ast` + pages /
`sanitize_report`; OpenRouter when keyed, `SimAgent` in CI). Fail-closed:
unparseable verdict → `NoScore(ChallengeInternal)`.

### Containerized review (`design-review` image)

The review itself runs in a one-shot Docker container (same hardening pattern
as the sandbox: `ReadonlyRootfs`, `CapDrop`, `no-new-privileges:true`, uid
65532) built from `deploy/Dockerfile` target `design-review`
(`challenge-agentic` + `challenge-ast`). The container mounts the submitted
agent **and** the most-similar harness (`_similar/`) read-only; the LLM may
use the sandboxed `run_command` tool (scrubbed env, cwd-pinned, 15s cap) for
diffs / grep / AST probes. `DESIGN_REVIEW_BACKEND=inline` keeps the legacy
in-process path for local/CI only.

### Pre-LLM copy gate (`created_at` ordered)

Before any LLM call the orchestrator scores the candidate against the recent
harness corpus (byte hash + AST fingerprint). A byte/AST copy
(`similarity ≥ 9500 bps`) of a harness with **strictly earlier `created_at`**
→ run status **`rejected`** (terminal, `Score(0)`, gating `rejected`) and the
LLM review is **skipped**. Unknown timestamps (baseline, legacy rows) fall
through to the LLM. Starting from the published miner **baseline** is never a
cheat signal (baseline-zeroing fix); copying another *miner's* harness is.

**Corpus rule (both the gate and the LLM review):** the comparison corpus is
**other hotkeys' prior art only** — entries owned by the candidate's own
`miner_hotkey` are excluded, and so is anything created at or after the
candidate. A miner iterating on their own harness is therefore never scored
against their own previous version. Selection lives in one place,
[`crates/design-challenge/src/corpus.rs`](../crates/design-challenge/src/corpus.rs),
so the gate and the review can never disagree.

### Allowed inspiration

Internet + PyPI via egress, external APIs / MCP servers at run time, Mobbin /
Dribbble / design refs, image generation, and UI libraries are **allowed**
when the output is substantially transformed. Network use itself is never a
cheat signal; copying another *miner's* harness or scrape-cloning a real site
is.

### Cheat → `Score(0)` (before admin)

| Code / pattern | Meaning |
|----------------|---------|
| `near_identical_harness_copy` | Near-identical harness vs corpus (AST + LLM) |
| `trivial_republish_wrapper` | Thin wrapper republishing another miner's HTML |
| `scraped_site_clone` | Fetch + republish identifiable real site without substance |
| `sanitize_bypass` | Sanitize bypass / JS exfil / phishing reinjection |
| `obfuscation_to_hide_copy` | Obfuscation whose only purpose is hiding a copy |

`suspicious` → also `Score(0)` (same policy as Prism); rationale stored for admin.

### Score semantics (`challenge_scoring_version = 3`)

Admin still picks **1 or 2** winner harnesses **per round**; each round win is
one **point**. The leaf projection uses a **rolling window of the last
`SCORING_WINDOW_ROUNDS = 10` rounds** (replaces the v2 daily ≥2-wins rule):

| Situation | Leaf |
|-----------|------|
| Miner with `p` window points (clean) | `Score(SCORE_MAX × p / total_window_points)` — proportional share, integer floor |
| Miner without window wins | `Score(0)` |
| Cheat / suspicious / copy-gate rejected | `Score(0)` (wins excluded from the pool) |
| Round timeout, no winners set | no new points; window projection unchanged |
| No harness | `NoScore(NotAttempted)` |
| Agentic / infra failure (retry budget spent) | `NoScore(ChallengeInternal)` |

Proportional share uses integer floor division; the remainder is unassigned.

---

## 9. Elimination

After round close: eliminate the **bottom 20%** of rated miners
(`ELIMINATION_BOTTOM_BPS = 2000`), at least **1** miner when the set is non-empty.

`eliminated_until_round = round + 10` (`ELIMINATION_COOLDOWN_ROUNDS = 10` → 1 day).

During cooldown: no new sandbox runs, no pairing — but D24 still requires a leaf:
emit `Score(0)`. Silence is a bug.

---

## 10. Declared participant set and `NoScore` reasons (D24)

Expected set `E` = all metagraph hotkeys for the pinned epoch (policy
`all_metagraph_hotkeys`). Leaf emit uses `challenge-common::emit_signed_leaf_set`:

- Exactly one signed leaf per `h ∈ E`
- **Refuses subset and superset** — Silence is a bug
- Emit at round close and at each epoch boundary via `POST /v1/weights/raw`

Absence codes used on this path include `NotAttempted`, `Timeout`,
`InvalidResponse`, `MinerError`, `RateLimited`, `ChallengeInternal` (bundle enum).

---

## 11. Key custody (challenge signing key)

| Secret | Mounted where | Notes |
|--------|---------------|-------|
| `design_sk` | `design-challenge` only | Signs leaves; never in sandbox/proxy |
| OpenRouter API key | `design-egress-proxy` only | Injected on LLM allowlist path |
| Admin bearer tokens (winners API) | hashed in challenge config | Raw tokens in `deploy/secrets/design/annotator_tokens` (optional override `DESIGN_ADMIN_TOKENS_FILE`) |
| OpenRouter (agentic) | `design-challenge` | Optional; missing key → `SimAgent` (CI/local only; never host Sim in staging/prod) |

Challenge signing key is **never** in miner harness, sandbox env, or gateway DB.
Gateway is routing only (D18/D23).

---

## 12. Compose services, ports, image contract

| Service | Port | Image target |
|---------|------|--------------|
| `design-challenge` | `8093` | `design-challenge` |
| `design-egress-proxy` | internal | `design-egress-proxy` |
| sandbox runtime | n/a | `design-runtime` (Python pin) |

Network: `design-sandbox-egress` (`internal: true`). Volume: `design-artifacts`.

Prod pins are **digest-only** — `:latest` forbidden. Rollable with
`prism-challenge` via updater `ROLLABLE_SERVICES` once deploy wiring lands.

Local health probe: `http://127.0.0.1:28093/health` (see
[`runbooks/local-testnet-e2e.md`](./runbooks/local-testnet-e2e.md)).

---

## 13. HTTP API surface

Proxied at `/challenge/design/*` → `:8093`.

Subnet frontends should **poll** (default hint `poll_hint_ms = 1000`); SSE is not required.

Stage journal (`design_stage_event`): every transition writes `queued` → `installing` →
`running` → `sanitizing` → `agentic_review` → `awaiting_admin` | `scored` | `failed`.
Harness stdout/stderr is appended as `stage = "log"` events with
`detail.{phase,stream,seq,text}` (install + run; truncated at 64 KiB per chunk).

### Miner

| Route | Purpose |
|-------|---------|
| `POST /v1/harness` | Submit harness JSON, `zip_base64`, or `application/zip` (idempotent by digest); optional `env_vars`; returns `run_ids` + poll paths |
| `GET /v1/harness/{id}` | Harness detail |
| `GET /v1/harness?miner=` | List by miner |
| `GET /v1/quota/{hotkey}` | Daily quota remaining |
| `GET /v1/miners/{hotkey}` | Per-miner harnesses, runs, quota, rating |
| `GET /v1/prompts` | Prompt set descriptor |
| `GET /v1/rounds` | Round list |
| `GET /v1/runs/{id}` | Run detail (stage, scores, pages summary, errors) |
| `GET /v1/runs/{id}/events` | Append-only stage events |
| `GET /v1/runs/{id}/logs` | Harness logs (`?since=` cursor, optional `?tail=`) |

### Viewer (screenshots-only)

Produced HTML is never served; the viewer exposes captured PNG screenshots
only (see §6).

| Route | Purpose |
|-------|---------|
| `GET /v1/runs/{id}/pages` | Page metadata list (path, bytes, sha256) |
| `GET /v1/view/{id}/{page}` | PNG screenshots only (`index.png`); `.html`/bare page → `410 Gone` |
| `GET /v1/runs/{id}/bundle.json` | `410 Gone` — no longer embeds produced HTML |

### Admin winners (operator bearer; master-local — not exposed via gateway)

Admin routes are **not exposed via gateway** (`/challenge/design/v1/admin/*`
returns 403). Operators hit `design-challenge:8093` on the master host
(SSH/VPC). Bearer tokens still required.

| Route | Purpose |
|-------|---------|
| `GET /v1/admin/rounds/{id}/candidates` | Clean `awaiting_admin` runs (pages + verdict) |
| `POST /v1/admin/rounds/{id}/winners` | Body `{ "harness_ids": ["…"] }` length 1 or 2; awards + emits leaves |
| `POST /v1/admin/rounds/current/requeue` | Schedule all active harnesses into the **current** open round (idempotent per harness; quota-blocked harnesses reported under `skipped`) |
| `GET /v1/rounds/{id}/leaderboard` | Round ratings |

### Annotation (deprecated; unused on leaf path)

| Route | Purpose |
|-------|---------|
| `GET /v1/annotate/next?annotator=` | Legacy pair fetch |
| `POST /v1/annotate` | Legacy vote |

### Ops / dashboard

| Route | Purpose |
|-------|---------|
| `GET /health` | Liveness |
| `GET /v1/status` | Backend mode, epoch, queues |
| `GET /v1/stats` | Aggregate queue + round clock + digest |
| `GET /v1/dashboard` | One-shot UI JSON (status, leaderboards, recent runs) |
| `GET /v1/jobs` | Active/recent jobs |
| `POST /v1/runs/{id}/retry` | Operator retry |

---

## Crates

| Crate | Role |
|-------|------|
| `design-challenge-task` | Identity, domains, quotas, round math |
| `design-harness` | Bundle contract + embedded Python harness/SDK |
| `design-prompts` | Pinned prompt bank + deterministic weighted selection |
| `design-sandbox` | Two-phase Docker + `SimSandbox` |
| `design-sanitize` | HTML/CSS sanitize + viewer headers |
| `design-rating` | Legacy Elo helpers (not on leaf path) |
| `design-store` | `DesignStore` trait, memory + DB adapter |
| `design-egress-proxy` | Open egress proxy (internal blocklist) + budgeted LLM path |
| `design-http` | Miner/viewer/admin winners/stats HTTP API |
| `design-challenge` | Orchestrator, agentic, scoring, leaf emit |
| `challenge-agentic` | Shared OpenRouter/Sim anti-cheat verifier |
| `challenge-ast` | Python AST fingerprint + similarity |
| `bins/design-challenge` | Operator binary `:8093` |
| `bins/design-egress-proxy` | Proxy binary |

Shared: `challenge-common` (exact-E), `challenge-keys`, `docker-engine`.

---

## Related

- Miner guide: [`external-miner/`](./external-miner/)
- Emission ceremony: [`runbooks/design-enable-and-emission.md`](./runbooks/design-enable-and-emission.md)
- Prism (sibling challenge): [`PRISM.md`](./PRISM.md)
- Architecture map: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
