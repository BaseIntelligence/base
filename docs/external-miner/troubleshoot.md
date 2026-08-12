<!-- protocol_version: 1 -->

# External miner — troubleshoot (HTTP)

**Path:** HTTP submit to **design** / **prism** only — **no Phala/CVM**

## Design

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| `400` on `POST /v1/harness` | Invalid bundle | `agent.py` defines `run`, `pyproject.toml` non-empty, size limits |
| `409 schedule` "daily manual run quota exceeded" | Manual anti-spam cap (10/day) — round-loop auto-enqueue does **not** spend it | `GET /v1/quota/{hotkey}` → `manual.remaining`; wait until next UTC day |
| Active harness but no runs this round | Checked before round-tick enqueue; not latest active harness on hotkey; scheduled quota; or queried with SS58 instead of lowercase hex | `GET /v1/miners/{hotkey}` and `GET /v1/quota/{hotkey}` with **64-char lowercase hex** (SS58 lowercases to a different key and shows empty); confirm harness `id` is the only `active: true` on that hotkey; `scheduled.remaining`; ops `admin/rounds/current/requeue` → `skipped[].reason` |
| `auto_retry` events, class `install` | Dep won't install (bad name/version, heavy source build) | `GET /v1/runs/{id}/logs` phase `install`; fix `pyproject.toml` deps |
| Run `failed` / Score 0 | Missing pages, timeout, crash | `GET /v1/runs/{id}/events`; ensure three required HTML pages |
| External call refused (`403`) | Target is internal-blocklisted (metadata IP, loopback, RFC1918/VPC, control plane) | Call public endpoints only; egress is otherwise open |
| Pages look empty in viewer | Sanitize stripped content | Scripts/`on*` handlers are removed; use static HTML/CSS |
| Eliminated | Bottom 20% last round | Cooldown 4 rounds; leaves are still `Score(0)` |
| `503` / ChallengeInternal | Operator infra | Retry later; not a miner signing issue |

## Prism

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| Rejected submit | Recipe contract | `GET /v1/recipe` (`automodel_pin_id` + caps); follow [`PRISM_RECIPE.md`](../PRISM_RECIPE.md) § 2.0 |
| `400 unsupported_layout` | Legacy 1.x ZIP or missing AutoModel members | Ship `automodel.base` + `automodel.patch` (+ optional `prism.toml`). Two-script / source-tree / `arch_id` layouts are rejected on live 2.0 |
| `400 recipe_version` | Payload implies recipe 1.x while live advertises ≥ 2.0 | Re-pack as AutoModel patch ZIP; do not send `architecture.py`/`training.py` |
| Patch apply failure / conflict | Diff not against the live pin, or stale rebase | Checkout exact `automodel_git_commit` from `/v1/recipe`; regenerate `git diff <commit>`; ensure `automodel.base` == `automodel_pin_id` |
| Wrong / unknown pin id | `automodel.base` ≠ recipe `automodel_pin_id` | Copy `automodel_pin_id` (live: `automodel@v0.5.0`) byte-identical from `/v1/recipe` |
| Binary / path-escape / oversized patch | Fail-closed apply rules | Text-only unified diff; no path escape outside allowlisted roots; keep diff within intake budgets |
| Tokenizer / hub errors on pod | No network; Hub download from miner code | Stay offline; use pin/harness tokenizer paths — do not `from_pretrained("<hub id>")` |
| `CAP_EXCEEDED` / Score 0 | Model > 350M params | Terminal — resize model config in your patch; not auto-retried |
| Score 0 after review | `Copied` / high-confidence `Suspicious` (≥0.9, non-trope) | Similarity on **your delta**; rewrite unique hunks; tropes alone are not plagiarism |
| `similar: true` on precheck | Would hit intake copy gate | Change the patch vs prior champions; starting from the operator pin is fine |
| `429 precheck_quota_exceeded` | 3 prechecks/coldkey/UTC day used | Wait until next UTC day; rotating hotkeys does not reset |
| `400 missing_lium_api_key` | Live path needs miner-funded Lium | Pass `X-Lium-Api-Key` (your Lium account); see [`prism.md`](prism.md) |
| Stuck `Provisioning` | Lium market / underfunded key | Check your Lium balance; watch `GET /v1/jobs` / events |
| Idempotent replay | Same `submission_id` (pin id + patch bytes) | Expected — returns prior row |

## Shared

- Wrong host: use gateway `/challenge/{id}/…` in staging/prod; direct `:2809x` only for local.
- Auth: miner routes are hotkey-identified in the JSON body — do not send challenge keys.
- Bundle axis: leaf bytes follow [`BUNDLE_SPEC.md`](../BUNDLE_SPEC.md) `protocol_version = 1` regardless of challenge scoring version.
- If docs still mention agent-v1 CVM steps, they are stale — this tree is HTTP-only.
