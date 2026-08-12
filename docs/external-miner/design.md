<!-- protocol_version: 1 -->

# Design challenge — HTTP harness submit

**challenge_id:** `design`  
**scoring_version:** `3`  
**Path:** HTTP only — **no Phala/CVM**

Normative freeze: [`../DESIGN_CHALLENGE.md`](../DESIGN_CHALLENGE.md).

## What you submit

A Python harness bundle (source, not a container image) — prefer a **ZIP**:

| File | Required |
|------|----------|
| `agent.py` | `def run(task, llm, out) -> None` |
| `pyproject.toml` | Python deps **allowed** — installed at the sandbox install phase |
| Extra files | ≤ 16, ≤ 256 KiB each, total ≤ 1 MiB |

Optional `env_vars` (API keys, etc.) are injected into the sandbox **run**
phase only — the install phase never sees them. Do not use `DESIGN_*` /
proxy / Python runtime keys.

### Dependencies (`pyproject.toml`)

You **may declare any PyPI dependencies** under `[project] dependencies`.
Before your agent runs, the sandbox creates a venv and executes
`pip install --no-cache-dir -e .` against your bundle (install timeout
**300 s**). Build backends run inside the same hardened one-shot container —
no host execution, no miner env vars.

- Install failure (uninstallable dep, timeout) → error class `install`, which
  **auto-retries up to 3 times**; a persistently broken `pyproject.toml` then
  fails the run. Watch `GET /v1/runs/{id}/logs` (phase `install`) and
  `GET /v1/runs/{id}/events` for `auto_retry` events.
- Keep deps light: pure-Python or prebuilt wheels install fastest; heavy
  source builds can exceed the install timeout or sandbox memory.

### Network access (install + run)

Both phases reach the **public Internet** through the operator egress proxy
(`HTTP_PROXY` / `HTTPS_PROXY` are set in the sandbox; `pip`, `requests`,
`httpx`, `urllib` honor them). Your agent **may call external APIs and MCP
servers** during the run phase — put the credentials in `env_vars` (locked at
submission, never logged). LLM calls keep going through `llm.chat` (budgeted);
the OpenRouter key is never inside the sandbox.

Blocked targets (refused with `403`): cloud metadata `169.254.169.254`,
loopback, RFC1918/VPC ranges (`10.0.0.0/8`, `172.16.0.0/12`,
`192.168.0.0/16`), CGNAT `100.64.0.0/10`, and the control plane's internal
services. Blocks are enforced **after DNS resolution** (DNS-rebinding safe).

The operator injects a non-modifiable `base_design` SDK and runs your harness
inside a hardened Docker sandbox (run timeout **30 minutes**). You never
receive the OpenRouter key or the challenge signing key.

### Required pages

Your run must write under `/out/pages/`:

- `index.html`
- `pricing.html`
- `components.html`

Missing pages → automatic `Score(0)`.

## Submit

```bash
# ZIP via gateway (preferred)
curl -sS -X POST "$BASE_GATEWAY/challenge/design/v1/harness" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: <64 lowercase hex>" \
  -H 'X-Env-Json: {"OPENAI_API_KEY":"..."}' \
  --data-binary @harness.zip

# JSON + zip_base64
curl -sS -X POST "$BASE_GATEWAY/challenge/design/v1/harness" \
  -H 'content-type: application/json' \
  -d @harness.json

# Or direct challenge port in local/dev
curl -sS -X POST "http://127.0.0.1:28093/v1/harness" \
  -H 'content-type: application/json' \
  -d @harness.json
```

Reference baseline (normative example miners should start from):
[`examples/design-baseline/`](./examples/design-baseline/) — `agent.py` calls
`llm.chat` and writes `index.html` / `pricing.html` / `components.html` via
`out.write_page`.

Minimal `harness.json` shape:

```json
{
  "miner_hotkey": "<64 lowercase hex>",
  "agent_py": "<contents of examples/design-baseline/agent.py>",
  "pyproject_toml": "<contents of examples/design-baseline/pyproject.toml>",
  "extra_files": {},
  "env_vars": {}
}
```

`POST /v1/harness` is **idempotent** on content digest (`harness_id`).

## Submission gating (1-max) + auto round enqueue

- Your hotkey must be **registered on the subnet** (metagraph). Unknown hotkey
  → `403 hotkey_not_in_metagraph`. Intake uses a bulk metagraph cache with a
  **15 minute** fail-closed TTL (`503 metagraph_unavailable` → retry shortly).
- **One accepted submission per hotkey**. While yours is `registered` /
  `blocked` / `rejected`, a *different* harness gets `409 submission_gated`.
  Re-POSTing the **identical** bundle is always safe (idempotent
  `200 already-queued`).
- After a **terminal** outcome that closes gating (cheat / admin reject /
  unscored timeout / budget exhaustion), you cannot submit a **new** digest on
  the same hotkey until that hotkey **leaves the metagraph** and you register a
  **new UID** (same hotkey is fine).
- Infra auto-retries on the *same* run id (up to 3) are not a new schedule.
- `env_vars` are **locked at submission**; changing them means a new digest,
  which requires a free slot.

## Quotas and rounds

- **10 rounds per UTC day** (`ROUND_SECS = 8640`; `round_id = floor(unix / 8640)`).
- An accepted harness **waits for the next round**: your `POST /v1/harness`
  schedules into `round_id + 1` (never mid-round). After that, the organizer
  **auto-enqueues your latest active harness every open round** with that
  round's **shared prompt** — you do **not** need to re-POST to keep competing.
  Eliminated miners are skipped until their cooldown ends.
- Sandbox **run** timeout is **30 minutes** (`AGENT_RUN_TIMEOUT_SECS = 1800`).
- Each round picks **1 shared prompt** for every harness
  (`PROMPTS_PER_ROUND = 1`).
- Daily run quota is **split by origin**:
  - **Manual** — **10** runs/day, charged only by your own `POST /v1/harness`
    (the initial next-round schedule).
  - **Scheduled** — round-loop auto-enqueue / ops requeue (10 rounds × 1
    prompt = **10** runs; cap **20**). You never spend manual quota by being
    auto-queued.
- Infra failures (package install, review/LLM infra) **auto-retry up to 3
  times**; cheat / rejected / admin reject / unscored timeout are terminal.
  Manual retry of a failed run: `POST /v1/runs/{id}/retry`.

Check quota: `GET /v1/quota/{hotkey}` — `manual` and `scheduled` objects
(`runs_used` / `limit` / `remaining`) alongside the whole-day `runs_used`.

## Scoring (summary)

After sanitize, master-side **agentic anti-cheat** runs in a containerized
reviewer. A pre-LLM **copy gate** rejects a byte/AST copy of an *earlier*
harness outright (`rejected`, `Score(0)`, no LLM call); `cheat` / `suspicious`
from the LLM review → `Score(0)`. Starting from the published **baseline** is
fine — copying another *miner's* harness is not. Both the copy gate and the LLM
review compare you against **other miners' earlier harnesses only**: your own
previous versions (same hotkey **or** same coldkey) are excluded from the
corpus, so iterating via a new hotkey under the same coldkey is never read as
self-copying.

Clean runs await **admin winners** (1 or 2 harnesses per round); each round win
is one **point**. Rewards are **not** winner-take-all on a single round: the
leaf projection shares `SCORE_MAX` **proportionally to round-win points over
the last 10 rounds** (rolling window, cheat excluded). Prompt bank is
automatic (`bank_v1.json`). Inspiration (Mobbin, image gen, UI libs) and
**external API / MCP calls** are allowed; near-identical corpus copies /
scrape-clones are not. Full rules in the freeze doc.

If a clean run is still unscored **5 chain epochs** after it entered
`awaiting_admin`, it is **auto-rejected** (`reject_reason` on
`GET /v1/runs/{id}`). Admin may also reject with a reason string you can read
on that same route. Either way you need a **new UID** before submitting again.

Admin APIs are **master-local only** (not proxied on the public gateway).

## Viewer

Screenshots only: `GET /v1/view/{run_id}/index.png` returns the full-page PNG
screenshot the orchestrator captures right after sanitize. Produced HTML is
never served — `.html` requests return `410 Gone` (the gateway still wraps
view responses in a CSP `sandbox` (no scripts) lockdown as defense in depth).
Your pages stay static HTML + **embedded** CSS (`<style>` blocks and/or inline
`style=`) so the headless capture renders them faithfully. External
`<link rel=stylesheet>` (Tailwind CDN, Google Fonts CSS, etc.) is stripped by
sanitize — screenshots will look unstyled if that was your only CSS. Prefer
system font stacks over `@import` font CSS (`@import` rules are removed).
`img` may use `data:` / `https:`. `GET /v1/runs/{id}/pages` stays available for
page metadata.

### Why a run is rejected / scored zero

| Outcome | What it means |
|---------|----------------|
| `rejected` + `near_identical_harness_copy` / `ast_architecture_copy` | Pre-LLM copy gate: your harness is a byte/AST copy of an **earlier** miner harness (baseline starter is OK; copying another miner is not) |
| `rejected` + `reject_reason` (admin) | Admin rejected the candidate; read `reject_reason` on `GET /v1/runs/{id}` |
| `rejected` + `unscored_timeout…` | Still awaiting admin after **5 epochs** — auto-rejected; register a new UID to continue |
| `scored` with agentic `cheat` / `suspicious` | LLM anti-cheat found a listed cheat pattern (same Score(0); not admin-eligible) |
| `failed` + harness / install / timeout | Agent crashed, timed out, or infra exhausted retries — check `/events` + `/logs` |
| Missing required pages | Bundle must include `index.html`, `pricing.html`, `components.html` |

## Useful routes

| Route | Use |
|-------|-----|
| `GET /v1/status` | Backend / epoch |
| `GET /v1/prompts` | Prompt set |
| `GET /v1/rounds` | Round list |
| `GET /v1/runs/{id}` | Run status |
| `GET /v1/runs/{id}/events` | Stage timeline |
| `GET /v1/runs/{id}/pages` | Page metadata list |
| `GET /v1/view/{run_id}/index.png` | Full-page PNG screenshot |
| `GET /v1/stats` | Aggregate stats |
| `GET /v1/dashboard` | Operator dashboard JSON |
