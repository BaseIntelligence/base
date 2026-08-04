<!-- protocol_version: 1 -->

# Prism challenge — HTTP script submit

**challenge_id:** `prism`  
**scoring_version:** `2` (bpb-only; LLM review is an anti-cheat gate, not a grader)  
**Path:** HTTP only — **no Phala/CVM**

Normative docs: [`../PRISM.md`](../PRISM.md), recipe [`../PRISM_RECIPE.md`](../PRISM_RECIPE.md).

## What you submit

Two Python scripts under the official recipe contract:

- `architecture.py`
- `training.py`

Evaluation runs on operator-rented Lium GPU pods (or `SimLiumBackend` in CI).
You do **not** deploy a miner CVM.

## Submit

```bash
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

`POST /v1/submissions` is idempotent by `submission_id`.

## Scoring (summary)

Final leaf score is pure bits-per-byte (bpb) on the lattice `[0, SCORE_MAX]`.
Cheap similarity plus the shared **agentic** gate (AST + metrics/receipt) force
hard-zero on `cheat` / `suspicious` (and cheap `Copied` / `Suspicious`);
LLM quality is coherence-only, not a grader. See [`PRISM.md`](../PRISM.md).

## Useful routes

| Route | Use |
|-------|-----|
| `GET /v1/status` | Backend mode, epoch, queue |
| `GET /v1/submissions/{id}` | Detail + receipt + scores |
| `GET /v1/submissions/{id}/events` | Stage timeline |
| `GET /v1/jobs` | Active/recent pods (ops) |
| `GET /health` | Liveness |

Emission share for prism is owner-controlled via the trust root. Until the design
enablement ceremony, prism typically holds the full `10000` bps share — see
[`../runbooks/prism-enable-lium-and-emission.md`](../runbooks/prism-enable-lium-and-emission.md)
and [`../runbooks/design-enable-and-emission.md`](../runbooks/design-enable-and-emission.md).
