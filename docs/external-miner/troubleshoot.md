<!-- protocol_version: 1 -->

# External miner — troubleshoot (HTTP)

**Path:** HTTP submit to **design** / **prism** only — **no Phala/CVM**

## Design

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| `400` on `POST /v1/harness` | Invalid bundle | `agent.py` defines `run`, `pyproject.toml` non-empty, size limits |
| `409 schedule` "daily manual run quota exceeded" | Manual anti-spam cap (10/day) — being scheduled into rounds does **not** spend it | `GET /v1/quota/{hotkey}` → `manual.remaining`; wait until next UTC day |
| `auto_retry` events, class `install` | Dep won't install (bad name/version, heavy source build) | `GET /v1/runs/{id}/logs` phase `install`; fix `pyproject.toml` deps |
| Run `failed` / Score 0 | Missing pages, timeout, crash | `GET /v1/runs/{id}/events`; ensure three required HTML pages |
| External call refused (`403`) | Target is internal-blocklisted (metadata IP, loopback, RFC1918/VPC, control plane) | Call public endpoints only; egress is otherwise open |
| Pages look empty in viewer | Sanitize stripped content | Scripts/`on*` handlers are removed; use static HTML/CSS |
| Eliminated | Bottom 20% last round | Cooldown 4 rounds; leaves are still `Score(0)` |
| `503` / ChallengeInternal | Operator infra | Retry later; not a miner signing issue |

## Prism

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| Rejected submit | Recipe contract | `GET /v1/recipe` + baseline; follow [`PRISM_RECIPE.md`](../PRISM_RECIPE.md) |
| Score 0 after review | `Copied` / `Suspicious` | Similarity gate; rewrite; do not paste baseline wholesale |
| Stuck `Provisioning` | Lium market thinness | Ops-side; watch `GET /v1/jobs` / events |
| Idempotent replay | Same `submission_id` | Expected — returns prior row |

## Shared

- Wrong host: use gateway `/challenge/{id}/…` in staging/prod; direct `:2809x` only for local.
- Auth: miner routes are hotkey-identified in the JSON body — do not send challenge keys.
- Bundle axis: leaf bytes follow [`BUNDLE_SPEC.md`](../BUNDLE_SPEC.md) `protocol_version = 1` regardless of challenge scoring version.
- If docs still mention agent-v1 CVM steps, they are stale — this tree is HTTP-only.
