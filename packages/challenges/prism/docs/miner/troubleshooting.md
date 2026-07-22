> **API truth is OpenAPI** (`https://chain.joinbase.ai/challenges/prism/openapi.json`, `/docs`).
> Day-1 miners: repo-root [`docs/miner/getting-started.md`](../../../../docs/miner/getting-started.md).
> This page is a short product pin note, not a route dump.

# Troubleshooting (Prism miners)

Fast diagnosis for the public joinbase path. Prefer Prism OpenAPI error bodies when present.

## Quick matrix

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| **401** / **403** on submit | Bad/missing signature, wrong hotkey, stale timestamp, replay/nonce | Re-sign canonical `prism:{hotkey}:{nonce}:{timestamp}:{sha256(body)}`; fresh nonce + timestamp; confirm ss58 |
| **409** on submit | Nonce already used | Generate a new unique `X-Nonce` |
| **422** | Body/schema validation | Confirm raw ZIP (`Content-Type: application/zip`) and two-script layout |
| **429** | Rate limit | Back off with jitter; do not parallel-spam create |
| **502** on `/challenges/prism/...` | Proxy transport / Prism container down | See [502](#502-prism-unavailable); check openapi **200** first |
| **403** on `/challenges/prism/health` | Expected public block | Use `/openapi.json`, `/docs`, `/leaderboard` instead |
| **404** on direct `/v1/submissions` via public | Public direct submit may be disabled | Use bridge `POST /v1/challenges/prism/submissions` |
| **404** on `/v1/weights/latest` | No sealed epoch yet | Normal early/quiet network; shares apply on next seal |
| Static reject / `rejected` | AST sandbox, single-module idiom, similarity, ladder | Fix two-script contract; start from `examples/tiny-1m` |
| Score ~0 / fail | Step-0 anomaly, memorization gap, missing data contract | No pretrained weights; read only `ctx.data_dir`; write only `ctx.artifacts_dir` |

## 401 Unauthorized / 403 Forbidden

Common on:

```text
POST https://chain.joinbase.ai/v1/challenges/prism/submissions
```

Checklist:

1. Headers exact: `X-Hotkey`, `X-Signature`, `X-Nonce`, `X-Timestamp`.
2. Sign the **canonical** UTF-8 bytes:
   `prism:{hotkey}:{nonce}:{timestamp}:{sha256_hex(raw_zip_body)}`.
3. Hotkey in the message must match `X-Hotkey` and the keypair that signed.
4. Timestamp within signature TTL (clock skew kills otherwise-valid sigs).
5. Nonce unique per attempt within the replay window (**409** if reused).
6. Validator hotkeys are not allowed to submit (self-submission guard → **403**).
7. Worker-plane networks: `403` / `NO_ACTIVE_WORKER` means bind a worker first
   (advanced; see BASE [worker-plane](https://github.com/BaseIntelligence/base/blob/main/docs/miner/worker-plane.md)), not always a signature bug.

Unsigned smoke probes should fail **auth-class** (401/403/422), never **502**.

## 409 Conflict (nonce)

- Each `(hotkey, nonce)` pair is single-use.
- Regenerate `X-Nonce` (uuid hex is fine) on every attempt.
- Do not retry the exact same signed headers after a transport glitch; re-sign.

## 422 Unprocessable

- Body must be the **raw zip bytes** on the bridge (not double-encoded JSON) unless you
  are on a local JSON route operators use for tests.
- Zip must extract to two distinct scripts with `build_model(ctx)` and `train(ctx)`.
- Path traversal, symlinks, or oversized archives fail closed before GPU work.

## 429 Too Many Requests

- Shared capacity protection on BASE and/or Prism.
- Exponential backoff + jitter; avoid create storms from one hotkey.
- Miners cannot raise product rate limits.

## 502 Prism unavailable

A **502** under `/challenges/prism/...` is often a **safe unavailable** rewrite when the
challenge did not answer (connection refused, timeout), not your ZIP contents.

Miner checks:

```bash
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/prism/openapi.json
curl -fsS -o /dev/null -w '%{http_code}\n' \
  https://chain.joinbase.ai/challenges/prism/leaderboard
```

- OpenAPI **200** but submit **401** → fix signatures, not infrastructure.
- OpenAPI **502** → operator/challenge crash-loop; wait or escalate; do not spam POST.
- Frontends should show friendly unavailable copy, not raw transport text.

## Rejected after admit (static / similarity)

| Gate | Typical fix |
|------|-------------|
| Single-module idiom | Split into `architecture.py` + `training.py` |
| AST sandbox | Remove blocked imports / dynamic code loads |
| Param ladder | Stay ≤ **124M** explore (or promote pin ≤ **350M**) |
| Similarity / anti-cheat | Do not ship near-duplicates of prior zips |
| Multi-GPU static | Keep distributed primitives correct at `world_size=1` |

## Scored poorly or zeroed

- **Forced random init** makes smuggled pretrained weights inert; anomalous step-0 loss
  zeroes the score.
- Secret val/test never leave the master; huge train-vs-held-out gaps are penalized.
- Ranking is compute-normalized; wall-clock bragging does not help emission.
- Challenge recomputes metrics; miner-reported numbers are ignored.

## Master healthy but no rewards

1. Confirm your hotkey on the **Prism** leaderboard, not only BASE health.
2. Confirm registry `emission_percent` for `prism` (expect **50**).
3. `GET /v1/weights/latest` **404** ⇒ no seal yet; patience until a vector publishes.
4. Wrong hotkey or rejected submission ⇒ no raw weight contribution.

## Honesty reminders

- No Base LLM gateway to “fix” scores from.
- Prism product path is **NO-TEE** (provider trust + IMAGE_PIN / deterministic scoring).
- REAL-PROVIDER TEE PASS is not a Prism day-1 or production claim.
- Never paste admin tokens, mnemonics, or provider API keys into issues.

## Still stuck?

1. Re-run [Getting started](getting-started.md) probes and checklist.
2. Read Prism OpenAPI error schema: https://chain.joinbase.ai/challenges/prism/openapi.json
3. Contract details: [Submission format](../submissions.md), [Security](../security.md).
4. BASE hub troubleshooting: https://github.com/BaseIntelligence/base/blob/main/docs/miner/troubleshooting.md
