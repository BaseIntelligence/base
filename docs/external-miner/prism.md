<!-- protocol_version: 1 -->

# Prism challenge — HTTP script submit

**challenge_id:** `prism`  
**scoring_version:** `2` (bpb-only; LLM review is an anti-cheat gate, not a grader)  
**Path:** HTTP only — **no Phala/CVM**

Normative docs: [`../PRISM.md`](../PRISM.md), recipe [`../PRISM_RECIPE.md`](../PRISM_RECIPE.md).

## What you submit

A **ZIP** (preferred) containing two Python scripts under the official recipe
contract, or JSON with the same fields / `zip_base64`:

- `architecture.py`
- `training.py`

Models must stay **≤ 350M parameters** after `build_model` (hard fail otherwise).

**Telemetry hooks (required, recipe ≥ 1.1.0).** Your `training.py` MUST import
the harness-provided `prism_telemetry` module and call
`prism_telemetry.report(loss=..., step=..., ...)` during training plus
`prism_telemetry.finish_evaluation()` to end the eval early. Missing hooks
are a hard contract violation — the review fails the submission
(`missing_telemetry_hooks`, zero score, terminal). See the baseline for the
exact pattern (it includes an offline fallback stub you can copy).

**Training-only entries (architecture competition, recipe ≥ 1.2.0).** To
compete on an already-published architecture, submit `training.py` +
`arch_id` (JSON field, or `X-Prism-Arch-Id` header with a `training.py`-only
ZIP). Do **not** include `architecture.py` — the source is pulled from the
registry. Published archs and their best bpb: `GET /v1/architectures`.

Evaluation runs on operator-rented Lium GPU pods (or `SimLiumBackend` in CI).
You do **not** deploy a miner CVM.

## GPU funding (optional / rolling out)

When the operator enables TAO prepay, request a quote before your first
submission (`POST`/`GET /v1/funding/quote?challenge_id=prism&hotkey=…`), send
the quoted TAO to the deposit address (memo included), then poll
`GET /v1/funding/status`. Eligibility: registered hotkey with **no prior**
Prism submission. Normative design: [`../LIUM_FUNDING.md`](../LIUM_FUNDING.md).
Until `PRISM_REQUIRE_LIUM_FUNDING=1`, funding is informational scaffolding only.

## Submit

```bash
# ZIP via gateway (preferred)
curl -sS -X POST "$BASE_GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/zip' \
  -H "X-Miner-Hotkey: <64 lowercase hex>" \
  --data-binary @submission.zip

# JSON sources (local/CI convenience)
curl -sS -X POST "$BASE_GATEWAY/challenge/prism/v1/submissions" \
  -H 'content-type: application/json' \
  -d @submission.json

# Local / direct
curl -sS -X POST "http://127.0.0.1:28092/v1/submissions" \
  -H 'content-type: application/json' \
  -d @submission.json
```

Inspect recipe pins before coding:

```bash
curl -sS "$BASE_GATEWAY/challenge/prism/v1/recipe"
curl -sS "$BASE_GATEWAY/challenge/prism/v1/recipe/baseline"
```

Live production (recipe **1.2.0**) advertises `train_rows: 2048`,
`val_rows: 256`, `train_hours_cap: 6.0`, `max_train_steps: 20000`,
`max_params: 350000000`, and `pin_hex` (sha over version + caps + dataset +
harness). The sealed baseline only trains on the 2048-row cut (~2M GPT-2
tokens) and scores poorly by design; competitive entries may stream the full
pinned FineWeb-Edu shard for up to 6h. Site chart labels that show “~2.6B
tokens · single pass” were **observed leader telemetry**, not a fixed recipe
quota — trust `/v1/recipe`, not the chart meta line.

`POST /v1/submissions` is idempotent by `submission_id`.

## Submission gating (1-max)

- Your hotkey must be **registered on the subnet** (metagraph). Unknown hotkey
  → `403 hotkey_not_in_metagraph`; a fresh registration may lag the snapshot
  (`503 metagraph_unavailable` → retry shortly).
- **One accepted architecture submission per hotkey.** While yours is
  `registered` / `blocked` / `rejected`, a *different* architecture submission
  gets `409 submission_gated`. Re-POSTing the **identical** sources is always
  safe (idempotent `200 already-queued`).
- **Training-only entries are separate**: one accepted entry per
  `(hotkey, arch_id)`, same retry rules — you may train on many published
  archs, one script per arch.
- If your hotkey **leaves the metagraph**, the watcher reopens your slot(s)
  automatically — resubmit under your new uid.
- Infra failures (Lium pod, review/similarity/LLM infra) **auto-retry up to 3
  times**; cheat / rejected verdicts are terminal. Manual retry:
  `POST /v1/submissions/{id}/retry`.

## Anti-copy rule (architecture-only)

Copying another miner's `architecture.py` (byte-for-byte or renamed/shuffled)
from an **earlier** submission is terminal `rejected` with zero score — judged
automatically before any GPU time is spent, no appeal. Similarity is judged on
`architecture.py` **only**: reusing a known training loop on your own novel
architecture is fine, and training-only entries on a published arch are never
"copies" by construction. Starting from the published baseline is always
allowed.

## Scoring (summary)

Final leaf score is pure bits-per-byte (bpb) on the lattice `[0, SCORE_MAX]`.
Cheap similarity plus the shared **agentic** gate (AST + metrics/receipt) force
hard-zero on `cheat` / `suspicious` (and cheap `Copied` / `Suspicious`).
Copy/similarity corpora exclude your own prior art (same hotkey **or** same
coldkey), so iterating via a new hotkey under the same coldkey is not treated
as a cross-miner copy. LLM quality is coherence-only, not a grader.
**Competition:** per epoch you are
credited the max of (a) your own best training result and (b) for each arch you
own, that arch's best result by *any* trainer — architecture owners are rewarded
for architectures people win with. Emission is **winner-take-all**: only the
single highest credit that epoch receives Prism's share (50% of the subnet);
ties break by lexicographically smallest hotkey. Scores first land in the leaf
set emitted at the first chain-epoch boundary **after** your run finalizes (a
long train that crosses epochs is normal — outbox assignment is exactly once).
Positive scores then keep participating in later epochs' competition sets until
a better valid score supersedes them (WTA still collapses to one leaf winner).
The global-best model is published to
[`BaseIntelligence/prism`](https://github.com/BaseIntelligence/prism)
`top-model/`. See [`PRISM.md`](../PRISM.md).

## Useful routes

| Route | Use |
|-------|-----|
| `GET /v1/status` | Backend mode, epoch, queue |
| `GET /v1/submissions/{id}` | Detail + receipt + scores |
| `GET /v1/submissions/{id}/events` | Stage timeline |
| `GET /v1/architectures` | Published archs + per-arch best bpb |
| `GET /v1/site/arenas/prism/submissions/{id}/telemetry` | Miner-reported loss curve / gradients / layer stats (from `prism_telemetry.report`) |
| `GET /v1/jobs` | Active/recent pods (ops) |
| `GET /health` | Liveness |

Emission share for prism is owner-controlled via the trust root. Current split is
`5000` bps prism / `5000` bps design (50/50) — see
[`../runbooks/prism-enable-lium-and-emission.md`](../runbooks/prism-enable-lium-and-emission.md)
and [`../runbooks/design-enable-and-emission.md`](../runbooks/design-enable-and-emission.md).
