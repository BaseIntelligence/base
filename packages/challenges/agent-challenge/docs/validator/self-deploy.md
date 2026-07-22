> OpenAPI: https://chain.joinbase.ai/challenges/agent-challenge/openapi.json · Day-1: root docs/miner/getting-started.md

# Validator / operator self-deploy surfaces

Miners fund CVMs. **You** own the trust root: dual measurement allowlists, golden
AES-256 key-release, quote verification, production flags, and score admission.
Validators are **not** the parties who deploy miner production scored jobs. This
document covers the validator-operated surfaces that back the miner
[`self-deploy` flow](../miner/self-deploy.md).

The validator host may have **no local TDX**. Review and eval guests run on Phala
Cloud Intel TDX; you verify quotes and measurements online and re-check offline
when auditing.

## Production configuration (mandatory)

Production requires both flags ON. Mixed settings fail closed at startup.

```yaml
phala_attestation_enabled: true   # CHALLENGE_PHALA_ATTESTATION_ENABLED
attested_review_enabled: true
```

| Mode | Flags | Interpretation |
| --- | --- | --- |
| **Production** | both true | Miner self-deploy is the scored path; attestation-only grading |
| **Offline / compat** | both false | Local CI without Phala; not production scoring |
| **Mixed** | one true | Rejected at startup (fail closed) |

Do not document flag-off as a supported production scored deployment.

## Ordered review and eval lifecycle

Intake alone creates no CVM and spends no Phala credits. The miner explicitly
performs the review stages first. Only a validator-**re-verified** `allow` for the
immutable submission version (fresh bound times, not cache-only DB phase bits)
permits `eval/prepare`, task selection, eval deployment, key release, result
acceptance, or weights. Reject, escalate, expiry, cancellation, provider failure,
and trust failure expose no benchmark work and no score.

Signed miner routes:

```http
POST /submissions/{submission_id}/review/prepare
POST /submissions/{submission_id}/review/retry
POST /submissions/{submission_id}/review/cancel
POST /submissions/{submission_id}/review/deployed
GET /submissions/{submission_id}/review/history
GET /submissions/{submission_id}/review/report
POST /submissions/{submission_id}/eval/prepare
POST /submissions/{submission_id}/eval/retry
POST /submissions/{submission_id}/eval/cancel
POST /submissions/{submission_id}/eval/failure
GET /submissions/{submission_id}/eval/status
POST /evaluation/v1/runs/{eval_run_id}/result
```

The direct challenge-owned result route is bearer-scoped to that eval run and is
never BASE-public-proxied. Review capability routes (`/review/v1/assignments/...`)
and validator routes (`/internal/v1/...`) are challenge-direct or internal only.
BASE public aliases for capability, internal, and result-ingestion routes are
BASE-blocked. ExecutionProof Phala-tier carry-through lives in
[`BaseIntelligence/base`](https://github.com/BaseIntelligence/base) (available after PR merge) (cross-repo).

### Review status and history

`GET /submissions/{submission_id}/status` includes a safe current review projection.
`review/history` and `review/report` use stable cursor pagination (default 10,
maximum 16) and retain cancelled, expired, failed, rejected, escalated, and
superseded attempts. Safe fields include session and assignment identity, phase,
terminal, verdict, verified, retryable, bounded `reason_code`, issue/finish times,
`report_available`, and projection digests. A report read before a projection
exists returns HTTP 404 with `review_report_not_available`.

Review phases: `review_queued`, `review_cvm_running`, `review_provider_standby`,
`review_verifying`, `review_allowed`, `review_rejected`, `review_escalated`,
`review_expired`, `review_cancelled`, `review_error`. Public report projections
contain digests, model identity, bounded reason codes, quote fingerprint,
measurement allowlist state, and verification state. They never contain plaintext
credentials, session capabilities, nonce values, raw model IO, unrestricted source,
or unrestricted evidence.

### Eval status, receipt, and rejection

`GET /submissions/{submission_id}/eval/status` returns attempt-ordered history with
stable cursor pagination (default 10, maximum 16). Each item exposes
`eval_run_id`, attempt and predecessor, `receipt_id`, `body_sha256`, phase,
terminal/verified/retryable flags, bounded `reason_code`, `key_grant_state`,
`key_release_nonce_state`, `score_nonce_state`, timestamps, and `result_available`.
Run tokens, nonce values, selected task contents, quotes, and raw evidence are not
returned.

Phases: `eval_prepared`, `eval_running`, `eval_verifying`, `eval_expired`,
`eval_cancelled`, `eval_error`, `eval_rejected`, `eval_accepted`. Pre-receipt
failure reasons: `eval_deploy_failed`, `eval_tunnel_failed`,
`eval_key_release_unavailable`, `eval_no_result`.

A result request is durably receipted before expensive verification. Receipt can
be `received`, `verifying`, `verified`, `rejected`, or `verifier_unavailable`.
Invalid or rejected results write no accepted score. Verifier-unavailable is
retryable and does not consume the score nonce. A conflicting body digest is a
conflict.

Eval retry is allowed only for a pre-receipt, pre-key-grant retryable state,
cancellation, or expiry. A key-granted, receipted, accepted, or permanent failure
state cannot retry. Cancellation and pre-receipt failure revoke the run capability
and nonces. Every attempt remains in history.

## Golden key-release endpoint

The golden key is the validator-held **AES-256** secret used with AES-256-GCM to
package and unseal encrypted oracle / golden task material. Production releases that
key only to a genuine, allowlisted eval CVM over **raw TCP RA-TLS**, never via
public L7 HTTP `/release`.

### Offline HTTP fixture (port 8700)

Port `8700` is the validator-local offline HTTP decision fixture and health
endpoint (lab and CI helpers). Start it with:

```bash
KEY_RELEASE_PORT=8700 uv run python -m agent_challenge.keyrelease.server
```

Routes:

- `GET /health` returns `{"status": "ok"}`.
- `GET /nonce` and `POST /nonce` are offline fixture helpers for a fresh,
  single-use, time-bounded nonce.
- Production does **not** release a key over HTTP. `POST /release` is disabled
  by the process entrypoint and returns 404. A denial or infrastructure error
  therefore returns no key and no score.

Health check:

```bash
curl -sf http://localhost:8700/health
```

### Production RA-TLS listener (port 8701)

The production listener is raw TCP on `KEY_RELEASE_RA_TLS_HOST` and
`KEY_RELEASE_RA_TLS_PORT` (default `127.0.0.1:8701`). It requires TLS 1.3,
client certificates, and dstack RA-TLS certificate extensions. Configure:

| Env (conceptual) | Role |
| --- | --- |
| `KEY_RELEASE_RA_TLS_HOST` | Bind host for raw listener |
| `KEY_RELEASE_RA_TLS_PORT` | Bind port (default 8701) |
| `KEY_RELEASE_RA_TLS_CERT_FILE` | Server certificate for the listener |
| `KEY_RELEASE_RA_TLS_KEY_FILE` | Server private key |
| `KEY_RELEASE_RA_TLS_CA_FILE` | **Client-trust** CA / chain (dstack guest issuer) used to verify guest mTLS clients |

The measured eval CVM obtains **GetTlsKey** guest client cert material, then dials
the raw listener. Guest-side mTLS files map to
`CHALLENGE_PHALA_RA_TLS_CERT_FILE`, `CHALLENGE_PHALA_RA_TLS_KEY_FILE`, and the
**server CA** inject (`CHALLENGE_PHALA_RA_TLS_CA_FILE` or
`CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM`). Production compose has no HTTP `/release`
fallback.

**Client-trust CA** (host verifies guest) is separate from **server CA** (guest
verifies host). Immediately after materials are ready, the guest can export a
**public-only** client fullchain (leaf + intermediates, never private keys) so
operators can harvest client-trust install material when remote file pull is
disabled. Never copy private key PEMs into host trust stores or docs.

The external tunnel must preserve raw TCP and end-to-end client-certificate
identity. Do not put an L7 TLS terminator in front of this listener and do not
trust a caller-provided `X-RA-TLS-Peer-Key` header.

After TLS, the client sends one 4-byte big-endian length followed by canonical
JSON with exactly `schema_version`, `eval_run_id`, `nonce`, `quote_hex`, and
`event_log`:

```json
{
  "schema_version": 1,
  "eval_run_id": "<eval-run-id>",
  "nonce": "<key-release-nonce>",
  "quote_hex": "<quote-hex>",
  "event_log": []
}
```

The response is another canonical frame with `schema_version`, `released`, and
either `key_b64` on success or `reason_code` on denial. The frame cap is 3 MiB,
the quote cap is 64 KiB, the event-log cap is 2 MiB / 4096 entries, the TLS
handshake deadline is 10 seconds, the full exchange deadline is 30 seconds,
the rate limit is 10 attempts per run per minute, and at most 8 verifications
run concurrently. There is no HTTP status framing on the raw socket.

Additional key-release configuration:

| Env | Role |
| --- | --- |
| `KEY_RELEASE_PORT` / `KEY_RELEASE_HOST` | Offline HTTP fixture bind (default `8700` / `127.0.0.1`) |
| `CHALLENGE_KEY_RELEASE_ALLOWLIST_FILE` | Validator-owned measurement allowlist (JSON list or `{"entries": [...]}`). Empty fails closed |
| `CHALLENGE_GOLDEN_KEY_FILE` | Path to the AES-256 golden-test key file, readable only by the validator process |
| `CHALLENGE_KEY_RELEASE_ACCEPTABLE_TCB` | Comma-separated acceptable TCB statuses (default `UpToDate`) |
| `CHALLENGE_KEY_RELEASE_NONCE_TTL_SECONDS` | Nonce validity window (default `120`) |

Server-side release checks (conjunction):

1. mTLS peer certificate vs client-trust CA and RA-TLS extensions
2. Quote `report_data` in key-release domain; SPKI digest binding
3. Event log / measurement allowlist for the **eval** image
4. Nonce freshness, rate limits, concurrent verify budget

On any failure the process returns no key bytes. The allowlist authority and key
release are **validator-owned**, never miner-owned.

## Measurement allowlist

Pin dual images (review vs canonical/eval) by reproducible measurement records:

`{mrtd, rtmr0, rtmr1, rtmr2, compose_hash, os_image_hash}`

Product `os_image_hash` is the SHA-256 of the concatenation of hex-decoded
MRTD + RTMR1 + RTMR2 (product formula). Event-log replay still recovers
`compose_hash` and `key_provider` from RTMR3 events. A miner reproduces the same
record with `python -m agent_challenge.selfdeploy measurements`; you can check a
reported measurement against the allowlist with:

```bash
python -m agent_challenge.selfdeploy verdict \
  --measurement ./measurement.json --allowlist ./allowlist.json
```

The command prints the measurement's six canonical fields and an `IN-LIST` /
`NOT-IN-LIST` verdict; a single-field difference is `NOT-IN-LIST`. Dry-run deploy
paths report the same verified result or `UNKNOWN` when no allowlist is provided;
they never fabricate `IN-LIST` membership.

Ops pin drift (wrong compose digest after image rebuild, mismatched dstack OS
catalog pin, or stale RTMR seal) fails closed until you refresh the allowlist.

## Quote verification and acceptance

Before a task score is written the challenge verifies the attested result's TDX
quote (signature/cert chain + acceptable TCB), replays the event log to recover
compose identity, checks the reconstructed measurement is on the **eval**
allowlist, checks `report_data` binds the exact run (measurement, agent hash,
task ids, scores digest, and the fresh validator score nonce), confirms nonces
are fresh and single-use, and confirms the matching **durable key-grant**.
Acceptance is a conjunction of binding, quote, measurement, nonce, and key-grant.
Any failing check parks the result with a retrievable reason and writes no score;
weight eligibility requires a verified attestation with grant.

Eval prepare and CVM spend also require **fresh re-verified review allow
materials** (bound times ≤24h). Cached DB phase labels alone are insufficient.

A quote can be re-checked offline with `dcap-qvl`:

```bash
dcap-qvl verify --hex ./quote.hex
```

or against the hosted Phala verifier
(`POST https://cloud-api.phala.com/api/v1/attestations/verify`). Offline tools
support trust-but-audit re-verification; they do not by themselves replace online
acceptor conjunction checks.

## Attested-result verification (base repo)

The attested-result envelope reuses the base `ExecutionProof` schema with a
Phala tier, and the validator-adapter / master carry-through (R=1 for attested
units) live in the separate base repository
([`BaseIntelligence/base`](https://github.com/BaseIntelligence/base) (available after PR merge)).
The report_data binding is single-sourced in base
`src/base/worker/proof.py` and replicated byte-identically here for the in-image
emitter.

## Money cap and teardown (operator awareness)

Any review or eval CVM created for live verification is miner-funded and subject
to the cumulative review+eval money cap of **$20**. Prefer the smallest CPU shape
(`tdx.small` / `tdx.medium`) and never a GPU shape. Miners must delete every
attributable CVM after success, reject, expiry, provider failure, quote failure,
cancellation, interruption, or result failure. Confirm none remain:

```bash
phala cvms delete <id> -f
phala cvms list
```

`phala cvms list` must report `total: 0` after teardown. Provide Phala credentials
through environment variables only; never write them into a committed file. The
review key `OPENROUTER_API_KEY` is supplied only via Phala `encrypted_env` and
must never appear in compose, ordinary env, arguments, logs, reports, evidence,
or the eval CVM.

## Residual operator risks

Documented more fully in [security](../security.md):

| Risk | Operator action |
| --- | --- |
| TEE.fail-class research | Keep quotes and sealfaces auditable; do not claim absolute TEE immunity |
| Ops pin drift | Re-pin dual allowlists after deliberate image/OS changes; refuse ad hoc drift |
| Provider availability | Phala, OpenRouter, or DCAP unavailability must fail closed (retryable codes), never silent scores |
| Wrong CA wiring | Keep client-trust CA separate from server CA; never install private keys into trust stores |

## Related

- [Architecture](../architecture.md)
- [Evaluation](../evaluation.md)
- [Attestation TEE](../miner/attestation-tee.md)
- [Security](../security.md)
