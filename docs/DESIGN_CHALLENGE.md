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
                    |  install: PyPI allowlist    |
                    |  run: OpenRouter + key      |
                    +----------------------------+
                           |
                    /out/pages/*.html
                           |
                    design-sanitize → store (postgres)
                           |
          +----------------+----------------+
          |                                 |
   viewer (sanitized+CSP)     agentic review → admin winners (1|2)
                                            |
                              exact-E leaves → gateway /v1/weights/raw
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
| `challenge_scoring_version` | **u16 = 2** |
| `SCORE_MAX` | `1_000_000` |
| Listen port | `8093` (local overlay `28093`) |
| Gateway proxy prefix | `/challenge/design/*` |
| Bundle `protocol_version` | `1` ([`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md)) |
| `emission_share_bps` | **0** until owner ceremony (prism holds `10000`) |
| Policy | `all_metagraph_hotkeys` |
| Round length | `ROUND_SECS = 8_640` (10 rounds / UTC day) |
| Agent run timeout | `AGENT_RUN_TIMEOUT_SECS = 1_800` (30 min; distinct from round length) |
| Daily qualification | `MIN_DAILY_WINS = 2` |
| `round_id` | `floor(unix_secs / 8640)` |
| Domain `round_id` | `base-design-round-id-v1` |
| Domain submission | `base-design-submission-v1` |
| Domain pair | `base-design-pair-id-v1` |
| Domain run | `base-design-run-id-v1` |
| Raw-weight domain | `base-rawweight-v1` (via bundle) |

Emission posture: `emission_share_bps = 0` for design until the owner ceremony
documented in [`runbooks/design-enable-and-emission.md`](./runbooks/design-enable-and-emission.md).
Until then prism is the sole non-zero share (`10000` bps).

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

1. **install** — `pip install` into work-root venv; proxy in PyPI mode; no LLM; short timeout.
2. **run** — same work root; venv read-oriented; proxy in LLM mode with per-run budget; loads `design_harness.py`.

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

Ingestion via `design-sanitize` (ammonia + CSS filter). **Raw HTML is never served.**

### Stripped / rejected

- Tags: `script`, `iframe`, `object`, `embed`, `applet`, `base`, `form`, `meta[http-equiv=refresh]`, `link[rel=import]`, scriptable SVG
- All `on*` event attributes
- URL schemes: reject `javascript:`, `vbscript:`, `data:text/html`; allow `http`, `https`, `mailto`, `data:image/*`
- CSS: reject `@import`, `expression(`, `url(javascript:`, `behavior:`, `-moz-binding`

### Annotator signal

`sanitize_report` (including `js_stripped`) is stored and shown to annotators.
JS stripped is **not** an automatic `Score(0)` — only a visible signal. Invalid /
missing required pages → automatic `Score(0)` at scoring gates.

---

## 6. Viewer headers and CSP

`GET /v1/view/{run_id}/{page}` serves **sanitized** HTML only, with:

```
Content-Security-Policy: sandbox; default-src 'none'; img-src data: https:; style-src 'unsafe-inline'; font-src https: data:; base-uri 'none'; form-action 'none'; frame-ancestors <allowlist>
X-Content-Type-Options: nosniff
Referrer-Policy: no-referrer
Cross-Origin-Resource-Policy: same-site
Cross-Origin-Opener-Policy: same-origin
Permissions-Policy: accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=()
Cache-Control: private, no-store
```

CSP `sandbox` (without `allow-scripts`) neutralizes script even if an integrator
omits the iframe `sandbox` attribute. Integrators should still use
`<iframe sandbox="" src="...">` on a dedicated subdomain.

---

## 7. Rounds and quotas

- **Round** every `ROUND_SECS = 8_640` (10 rounds / UTC day):
  `round_id = floor(unix_secs / 8640)`.
- **Agent run timeout**: `AGENT_RUN_TIMEOUT_SECS = 1_800` (30 minutes wall clock
  for the sandbox run phase — not the round length).
- **Prompts**: repo-pinned bank (`bank_v1.json`, no human approval API);
  deterministic weighted draw
  `SHA256(domain || round_id || bank_digest)` → 3 prompts per round;
  identical for every harness in that round.
- **Quota**: **10** runs/day/hotkey (`DAILY_RUN_QUOTA = 10`; ~2–3 prompts/round × harness).
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

### Allowed inspiration

Internet + PyPI via egress, Mobbin / Dribbble / design refs, image generation,
and UI libraries are **allowed** when the output is substantially transformed.

### Cheat → `Score(0)` (before admin)

| Code / pattern | Meaning |
|----------------|---------|
| `near_identical_harness_copy` | Near-identical harness vs corpus (AST + LLM) |
| `trivial_republish_wrapper` | Thin wrapper republishing another miner's HTML |
| `scraped_site_clone` | Fetch + republish identifiable real site without substance |
| `sanitize_bypass` | Sanitize bypass / JS exfil / phishing reinjection |
| `obfuscation_to_hide_copy` | Obfuscation whose only purpose is hiding a copy |

`suspicious` → also `Score(0)` (same policy as Prism); rationale stored for admin.

### Score semantics (`challenge_scoring_version = 2`)

Admin still picks **1 or 2** winner harnesses **per round**. Leaf mass is
**not** WTA on that round alone:

| Situation | Leaf |
|-----------|------|
| Miner with ≥ `MIN_DAILY_WINS = 2` round wins that UTC day (clean) | share SCORE_MAX equally among that day's qualified winners |
| Miner with fewer than 2 round wins that day | `Score(0)` |
| Cheat / suspicious | `Score(0)` (never qualifies) |
| Round timeout, no winners set | no new win; day projection unchanged / zeros |
| No harness | `NoScore(NotAttempted)` |
| Agentic / infra failure | `NoScore(ChallengeInternal)` |

Equal share uses integer division (`SCORE_MAX / n_qualified`).

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

### Viewer

| Route | Purpose |
|-------|---------|
| `GET /v1/runs/{id}/pages` | Page list |
| `GET /v1/view/{id}/{page}` | Sanitized HTML + hardened headers |
| `GET /v1/runs/{id}/bundle.json` | Sanitized bundle JSON |

### Admin winners (operator bearer; master-local — not exposed via gateway)

Admin routes are **not exposed via gateway** (`/challenge/design/v1/admin/*`
returns 403). Operators hit `design-challenge:8093` on the master host
(SSH/VPC). Bearer tokens still required.

| Route | Purpose |
|-------|---------|
| `GET /v1/admin/rounds/{id}/candidates` | Clean `awaiting_admin` runs (pages + verdict) |
| `POST /v1/admin/rounds/{id}/winners` | Body `{ "harness_ids": ["…"] }` length 1 or 2; awards + emits leaves |
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
| `design-egress-proxy` | Allowlisted HTTP proxy |
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
