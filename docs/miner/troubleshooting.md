# Troubleshooting (miners)

Fast diagnosis for the public joinbase path. Prefer challenge OpenAPI error bodies when present.

## Quick matrix

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| **401** / **403** on submit | Missing/bad signature, wrong hotkey, replay/nonce, or policy deny | Re-sign with correct canonical payload; fresh nonce + timestamp; confirm hotkey matches wallet |
| **422** | Body/schema validation | Fix ZIP layout or JSON fields per challenge OpenAPI |
| **429** | Rate limit | Back off; wait for window; do not hammer create endpoints |
| **502** on `/challenges/{slug}/...` | Proxy transport / challenge container down | See [502](#502-challenge-unavailable); check openapi 200 first |
| **403** on `/challenges/{slug}/health` | Expected public block | Use `/openapi.json`, `/docs`, `/leaderboard` instead |
| **404** on `/v1/weights/latest` | No sealed epoch yet | Normal early / quiet network; shares still apply on next seal |
| OpenAPI/docs **502** | Challenge service crash-loop or not adopted | Operator issue; miners cannot fix via retry spam |
| Dashboard shows “unavailable” | Safe frontend copy for proxy 502 | Retry later; do not paste raw `BASE request failed with status 502` to users |

## 401 Unauthorized / 403 Forbidden

Common on:

```text
POST /v1/challenges/{slug}/submissions
POST /challenges/{slug}/submissions
…/env, …/launch, other signed miner actions
```

Checklist:

1. Send all required headers: `X-Hotkey`, `X-Signature`, `X-Nonce`, `X-Timestamp` (names exact).
2. Sign the **challenge-canonical** bytes (not an ad-hoc concatenation you invented).
3. Nonce must be unique within the replay window; timestamp within allowed skew.
4. Hotkey must be the ss58 you intend to earn on.
5. Prism worker-plane networks: `403 NO_ACTIVE_WORKER` means bind/deploy a worker first
   ([worker-plane.md](worker-plane.md)), not a signature bug.
6. Agent Challenge advanced attestation denials are challenge policy; read the JSON
   `detail` / `code` when present.

Unsigned probes used for smoke should fail **auth-class** (401/403/422), never 502.

## 429 Too Many Requests

- Challenge or BASE rate limits protect shared capacity.
- Exponential backoff; jitter; avoid parallel create storms from one hotkey.
- Agent Challenge has product rate knobs on the challenge service (window / attempts).
  Miners cannot raise them; wait or reduce attempt rate.
- Do not confuse product 429 with upstream OpenRouter 429 inside attested evals
  (that path is challenge/agent-side after admit).

## 502 Challenge unavailable

A **502** under `/challenges/{slug}/...` is often a **safe unavailable** rewrite from the
BASE proxy when the challenge did not answer (DNS, connection refused, timeout), not your
ZIP contents.

Operator / environment checklist (miners: verify publicly, then wait or escalate):

1. Ingress routes **`/challenges`** (not only `/v1/challenges`) to the BASE master proxy.
2. Challenge long-lived container is **Up/healthy** in the master Compose project
   (`challenge-prism`, `challenge-agent-challenge`), not restarting.
3. Public readiness is green:
   ```bash
   curl -fsS -o /dev/null -w '%{http_code}\n' \
     https://chain.joinbase.ai/challenges/prism/openapi.json
   curl -fsS -o /dev/null -w '%{http_code}\n' \
     https://chain.joinbase.ai/challenges/agent-challenge/openapi.json
   ```
4. Separate **transport 502** from **challenge-origin 4xx** (auth, validation, rate limit),
   which should pass through with the challenge status code.
5. Frontends should show friendly unavailable copy, not raw transport text.

## Master health but empty rewards

1. Confirm you appear on the **challenge** leaderboard, not only BASE health.
2. Confirm registry `emission_percent` for your slug (expect **50** on both active majors).
3. `GET /v1/weights/latest` 404 ⇒ no seal yet; patience until validators have a vector.
4. Wrong challenge or inactive slug ⇒ zero contribution regardless of local scores.

## Env / launch stuck (Agent Challenge)

Public state may wait on miner action after analysis allow:

```http
GET  /challenges/agent-challenge/submissions/{id}/env
PUT  /challenges/agent-challenge/submissions/{id}/env
POST /challenges/agent-challenge/submissions/{id}/env/confirm-empty
POST /challenges/agent-challenge/submissions/{id}/launch
```

- Need zero secrets → call **`confirm-empty`**, do not leave the submission hanging.
- Values are write-only; you cannot read them back.
- Still stuck after launch → check status/events endpoints; advanced TEE residuals are
  challenge-side (see agent-challenge docs), not BASE gateway.

## Honesty reminders

- No Base LLM gateway to “fix” scores from.
- Prism product path is **NO-TEE** (provider trust + pin / deterministic scoring).
- Agent Challenge Phala/KR is advanced attestation, not a day-1 requirement for reading
  this hub.
- Never paste admin tokens, mnemonics, or live provider API keys into issues.

## Still stuck?

1. Re-run [Getting started](getting-started.md) probes.
2. Read challenge OpenAPI error schema.
3. Challenge-specific guides: [How-to](how-to.md).
4. Architecture / ops: [../architecture.md](../architecture.md), [../compose.md](../compose.md).
