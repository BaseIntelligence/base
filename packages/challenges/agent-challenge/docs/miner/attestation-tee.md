# Attestation on Intel TDX (Phala) — Concepts

> Day-1 miners: start at [Getting started](getting-started.md) (joinbase upload).
> This page is **Concepts** for the TEE trust chain; CLI ops are the advanced
> [Self-deploy how-to](self-deploy.md).

This page explains the TEE trust chain for **production miner self-deploy** on Phala Cloud
CPU Intel TDX CVMs. The validator host may have **no local TDX**; measurements and quotes are
produced inside Phala guests and re-verified on the challenge service. Operational CLI steps are in
[self-deploy](self-deploy.md). Residual risk is in [security](../security.md).

## Agent-driven order (package verify → tree SHA → TEE → eval)

Production evaluation is **agent-driven**. Trust is built in this **fixed order**. Skipping a step
fail-closes: **no eval prepare, no key-release grant, no score attestation**.

```text
submit ZIP
  → extract + validate package
  → package_tree_sha = canonical folder-tree SHA (content-addressed proof)
  → measured package residual: agent LLM rules under harness / .rules
       FAIL  → stop (no TEE-authorized eval)
       PASS  → bind (package_tree_sha, residual verdict, rules digests)
  → TEE authorization: fresh re-verified review allow + quote materials
       that bind tree SHA + residual digests
  → ONLY THEN eval prepare / deploy / KR / trials / score attestation
```

| Step | What must hold | If missing |
| --- | --- | --- |
| 1. Package + LLM rules residual | Measured review residual **allow** under `.rules` (agent-driven; host-static analyzer alone is **not** enough for TEE auth) | No eval authorizable |
| 2. Tree SHA proof | Durable `package_tree_sha` next to zip digest; bound into plan / review / guest | Mismatch or absent → refuse prepare / KR / score |
| 3. TEE auth | Fresh residual allow + tree SHA in authorizing materials; dual flags ON | No eval start |
| 4. Eval / attestation | Guest rechecks `package_tree_sha` before trials; KR + score require the same proof chain | No free attestation |

**Model rule (agent path):** there is **no closed catalog** of allowed agent model IDs. The only
product ban on agent models is **personal / custom finetunes**. The review judge pin
(`REVIEW_MODEL`) stays separate.

Host-only analyzer allow without measured residual + tree SHA is **insufficient** for
`eval/prepare`, key release, or production score under dual attestation flags.

## What is measured

A production app-compose plus OS image yield a canonical measurement record used on the allowlist:

| Field | Role |
| --- | --- |
| `mrtd` | TD measurement root |
| `rtmr0` | Runtime measurement register 0 |
| `rtmr1` | Runtime measurement register 1 |
| `rtmr2` | Runtime measurement register 2 |
| `compose_hash` | Hash of the measured compose definition (dstack-compatible) |
| `os_image_hash` | Product formula identity from MRTD + RTMR1 + RTMR2 (SHA-256 of decoded registers) |

`rtmr3` is treated as runtime and is excluded from the static allowlist pin used in score binding
helpers; event-log replay still recovers compose identity and `key_provider` from RTMR3 events.
Miners reproduce the six-field record with:

```bash
python -m agent_challenge.selfdeploy measurements \
  --metadata ./metadata.json --cpu 1 --memory 2G --compose ./deploy/app-compose.json
```

Validators publish **dual** allowlists (review image vs canonical/eval image) and enforce membership.
A single-field mismatch is `NOT-IN-LIST`.

## Separate `report_data` domains

TDX quotes carry a 64-byte `report_data` field. Agent Challenge binds a **domain-separated**
canonical JSON preimage into that field so a review quote cannot authorize key release or score
acceptance, and a key-release quote cannot authorize a score.

| Domain tag | Stage | Typical bound content |
| --- | --- | --- |
| `base-agent-challenge-review-v1` | Review report | Review-domain envelope: session / report digests, measurement subset, and cryptographically bound `issued_at` / `received_at` (challenge clock; ≤24h freshness on re-verify) |
| `base-agent-challenge-keyrelease-v1` | Key release | `eval_run_id`, key-release nonce, RA-TLS SPKI digest, schema_version |
| `base-agent-challenge-v1` | Score result | measurement, agent_hash, task_ids, scores_digest, eval_run_id, score_nonce |

Bindings are schema-closed. Mixing domains is a verification failure. Guest wall clock alone never
authorizes review freshness; unattested DB times or phase labels (`review_allowed`) are cache only
and never alone admit eval CVM or a production score.

Score-domain binding (schema version 2) conceptually:

```text
SHA256(canonical_json({
  domain, schema_version, canonical_measurement,
  agent_hash, eval_run_id, score_nonce, scores_digest, task_ids
}))
```

left-aligned into 64 bytes with zero padding. Source of truth: `agent_challenge.canonical.eval_wire`
and the sibling helper in [BASE](https://github.com/BaseIntelligence/base) (cross-repo).

## Review CVM vs eval CVM

| | Review CVM | Eval CVM |
| --- | --- | --- |
| Phala guest | CPU Intel TDX CVM on Phala Cloud | Separate CPU Intel TDX CVM on Phala Cloud |
| Image | Measured review compose (shipping script + agent ZIP under harness / `.rules`) | Separate measured canonical/eval compose with baked live-task-cache |
| Secrets | Real OpenRouter + review session via `encrypted_env` only (**no** Base LLM gateway tokens) | Eval capability / plan fields via `encrypted_env`; agent OpenRouter only when measured digests allow (**no** Base `/llm/v1`) |
| Work | Attested LLM review of submission (no golden / tasks) | k-trial benchmark with DooD isolation; decrypt golden only after KR; full score-chain re-verify on RESULT |
| Quote domain | review-v1 (`issued_at` / `received_at` bound; ≤24h) | keyrelease-v1 for GetTlsKey path; score-v1 for RESULT |
| Golden AES-256 key | Never | Only after successful RA-TLS release; KR grant is durable for the run |

Trust is **attestation-only** on the scored path: production does not restore a Base LLM gateway on
the host. Grading of agent quality is the measured eval workload after key-grant; review is the
attested gate before eval spend.

## RA-TLS key release and GetTlsKey

Production key release is **raw TLS 1.3** with client certificates and dstack RA-TLS extensions
on the validator listener (default bind `127.0.0.1:8701`, externalized via operator tunnel that
preserves raw TCP and peer identity). HTTP `POST /release` is disabled and returns 404. Do not place
a public L7 terminator in front of the listener.

### Guest materials (GetTlsKey)

Inside the measured eval guest, dstack **GetTlsKey** materializes mTLS client cert and key under the
production RA-TLS directory (for example `/run/secrets/ra_tls/client.crt` and `client.key`). The
guest also needs the **validator server CA** (operator-injected via
`CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM` / server CA file envs) so it can verify the KR listener
certificate. The guest never invents that server root.

### Host materials (client-trust vs server cert)

| Host path / env (conceptual) | Role |
| --- | --- |
| `KEY_RELEASE_RA_TLS_CERT_FILE` / `KEY_RELEASE_RA_TLS_KEY_FILE` | Listener server identity |
| `KEY_RELEASE_RA_TLS_CA_FILE` | **Client-trust** store: CA / chain that issued guest client certs (dstack guest issuer) |
| Guest `CHALLENGE_PHALA_RA_TLS_CA_FILE` or `CHALLENGE_PHALA_RA_TLS_SERVER_CA_PEM` | **Server CA** for the guest to verify the listener |

Client-trust CA is **not** the same blob as server CA inject. Operators may harvest the guest
**public-only** fullchain (leaf + intermediates, never private keys) from guest logs / known export
path for client-trust install when remote file pull is unavailable.

### Server checks

Client presents a quote whose `report_data` is the key-release domain. Server checks:

1. TLS peer certificate vs RA-TLS quote extensions and allowlisted measurement
2. Event log / dual-domain allowlist membership for the eval image
3. Nonce freshness and rate limits
4. SPKI digest binding (client-supplied peer headers are not trusted)

On denial: no golden key, no score path.

## Quote verification (trust-but-audit)

Operators and auditors can re-check quotes with:

```bash
dcap-qvl verify --hex ./quote.hex
```

or Phala hosted verify (`POST https://cloud-api.phala.com/api/v1/attestations/verify`). Challenge
acceptance is a **conjunction** of quote, measurement allowlist, event log, domain binding, nonces,
review freshness (bound times ≤24h on **re-verify**), and (for scores) durable key-grant state.
Eval CVM work units require a fresh re-verified review allow. A single failure rejects acceptance.
This is cryptographically-anchored trust-but-audit; it is not a claim that TEE hardware is free of
class attacks or that re-check tools replace every function of the online acceptor.

## Public TEE math (joinbase / self-deploy inspection)

After a verified review report is durable, anyone can inspect the **safe TEE math
subset** without miner signatures:

```http
GET /submissions/{id}/review/tee
```

Proxied public URL (joinbase):

```http
GET https://chain.joinbase.ai/challenges/agent-challenge/submissions/{id}/review/tee
```

- No report yet: HTTP **200** with exactly `{"available": false}` (not invented math).
- Report available: measurements (`mrtd`, `rtmr0`–`rtmr3`, `compose_hash`,
  `os_image_hash`, `key_provider`, `vm_shape`), size-capped `tdx_quote_hex`,
  `report_data_hex`, `report_data_preimage` **without** raw `review_nonce`
  (nonce hash only), public `verification_outcome`, and digests /
  `quote_fingerprint_sha256`.
- **Never** exposed on this surface: nonce plaintext, session tokens,
  capabilities, evidence bodies, model IO, or encryption KEY material.

This is separate from the miner-signed `GET .../review/report` digests path and
from internal full envelopes. Dual-flag `GET .../status` still carries
`review.phase` / `verdict` / `verified` / `report_available` independently of
list lifecycle status (FE STATUS badges prefer dual-flag).

## Operational bounds

- CPU TDX only for production self-deploy shapes (`tdx.small` / `tdx.medium`). GPUs refused.
- Hard projected spend cap (default **$20** for review+eval lifetime) before create.
- Mandatory teardown: `phala cvms list` must show `total: 0` after success or failure cleanup.
- Evaluations that never obtain key-grant still tear down; they never become accepted scores.

## Related

- [Self-deploy CLI](self-deploy.md)
- [Operator surfaces](../../../packages/challenges/agent-challenge/docs/validator/self-deploy.md)
- [Security residual risk](../security.md)
- [Evaluation gate narrative](../../../packages/challenges/agent-challenge/docs/evaluation.md)
