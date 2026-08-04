<!-- protocol_version: 1 -->

# Design challenge — HTTP harness submit

**challenge_id:** `design`  
**scoring_version:** `2`  
**Path:** HTTP only — **no Phala/CVM**

Normative freeze: [`../DESIGN_CHALLENGE.md`](../DESIGN_CHALLENGE.md).

## What you submit

A Python harness bundle (source, not a container image) — prefer a **ZIP**:

| File | Required |
|------|----------|
| `agent.py` | `def run(task, llm, out) -> None` |
| `pyproject.toml` | deps installed in the operator sandbox |
| Extra files | ≤ 16, ≤ 256 KiB each, total ≤ 1 MiB |

Optional `env_vars` (API keys, etc.) are injected into the sandbox **run**
phase only. Do not use `DESIGN_*` / proxy / Python runtime keys.

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

## Quotas and rounds

- **10 rounds per UTC day** (`ROUND_SECS = 8640`; `round_id = floor(unix / 8640)`).
- Sandbox **run** timeout is **30 minutes** (`AGENT_RUN_TIMEOUT_SECS = 1800`).
- **10** sandboxed runs per hotkey per UTC day.
- Each round picks **3** prompts via deterministic weighted draw for all harnesses.

Check quota: `GET /v1/quota/{hotkey}`.

## Scoring (summary)

After sanitize, master-side **agentic anti-cheat** runs; `cheat` /
`suspicious` → `Score(0)`. Clean runs await **admin winners** (1 or 2 harnesses
per round). Rewards are **not** winner-take-all on a single round: miners need
**≥ 2 round wins in the UTC day** to qualify, then **share `SCORE_MAX` equally**
among that day's qualified winners. Prompt bank is automatic (`bank_v1.json`).
Inspiration (Mobbin, image gen, UI libs) is allowed; near-identical corpus
copies / scrape-clones are not. Full rules in the freeze doc.

Admin APIs are **master-local only** (not proxied on the public gateway).

## Viewer

Sanitized HTML only: `GET /v1/view/{run_id}/{page}` with CSP `sandbox` (no
scripts). Raw HTML is never served.

## Useful routes

| Route | Use |
|-------|-----|
| `GET /v1/status` | Backend / epoch |
| `GET /v1/prompts` | Prompt set |
| `GET /v1/rounds` | Round list |
| `GET /v1/runs/{id}` | Run status |
| `GET /v1/runs/{id}/events` | Stage timeline |
| `GET /v1/runs/{id}/pages` | Sanitized page list |
| `GET /v1/view/{run_id}/{page}` | CSP viewer (sanitized HTML) |
| `GET /v1/stats` | Aggregate stats |
| `GET /v1/dashboard` | Operator dashboard JSON |
