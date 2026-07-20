# Security

Agent Challenge treats miner code as untrusted, keeps secrets out of the public API, and accepts
scores only after cryptographically-anchored checks on the Phala Intel TDX path. The product model
is **trust-but-audit**, not absolute TEE immunity.

## Residual TEE and operations risk

Intel TDX and Phala guest isolation reduce operator visibility into measured code paths, but they
do not eliminate physical, side-channel, implementation, or operator-policy risk.

| Risk class | Notes | Mitigation in this design |
| --- | --- | --- |
| Hardware / TEE.fail-class | Public research on TEE classes continues | Allowlist plus quote verify; bounded acceptance; auditors re-check quotes (`dcap-qvl` or Phala verify) |
| Ops pin drift | Wrong compose_hash, os_image_hash, dstack OS catalog, or RTMR pin on shared allowlists | Dual review/eval allowlists; single-field mismatch denies key and score; dry-run `IN-LIST` / `NOT-IN-LIST` only |
| Provider availability | Phala / OpenRouter / DCAP verify outages | Fail closed (retryable or terminal reason_code); no silent accepted scores |
| Ungrounded image | Arbitrary miner-built image | Validator-owned measurement allowlist; unknown compose_hash fails closed |
| Domain mixup | Reuse quote across stages | Separate review / keyrelease / score report_data domains |
| Replay | Re-post old score or stale allow | Single-use nonces; bound eval_run_id; fresh review re-verify ≤24h (cache-only DB bits insufficient) |
| Key theft | Golden key on wrong peer | RA-TLS mTLS + SPKI binding in keyrelease domain; HTTP `POST /release` disabled in production |
| Residual CVM cost / data | Miner-funded leftover guests | Money cap and mandatory teardown to `phala cvms list` total 0 |

If continuous attestation verification or the DCAP path is unavailable, results stay unaccepted
(retryable or terminal according to reason_code); they never become silent scores.

## Isolation (eval)

Eval CVMs run measured workload with Docker-out-of-Docker style isolation for task trials. Task
containers typically run without host network privileges unless a task policy opts in. Terminal-Bench
trees for prepare-selected tasks are **baked** into the canonical image
(`/opt/agent-challenge/task-cache`); eval time does not network-fetch dataset content.

Review CVMs are a separate measured image: they call direct OpenRouter under the harness / `.rules`
only as configured by validator-pinned composition, and must never receive golden task material or
golden keys.

## Secrets and measured LLM policy (no Base gateway)

- Miners MUST NOT embed Base LLM gateway client material
  (`BASE_LLM_GATEWAY_URL`, `BASE_GATEWAY_TOKEN`, `/llm/v1`) or non-measured provider secrets /
  hard-coded emission model names in submissions (`base_gateway_forbidden`,
  `unauthorized_llm_provider`).
- Production legal LLM path is **measured OpenRouter** under the review harness / measured eval
  CVM with digests, or **tools-only** agents. Base master gateway is not restored on the scored path.
- OpenRouter review material for the review CVM is delivered only through Phala `encrypted_env`,
  never plain compose text, ordinary environment for the eval CVM, logs, or public reports.
- **Golden key** is a validator-held **AES-256** key for AES-256-GCM packaging of oracle / golden
  task material. Bytes live only on the validator key-release process and, after grants, only
  inside the measured eval guest that passed RA-TLS. Plaintext oracle must not ship at rest in
  miner-visible images.
- Guest **GetTlsKey** client cert material is distinct from operator **server CA** inject for the
  KR listener. Client-trust CA on the host is installable from harvested **public-only** guest
  fullchain export; private keys never leave guest mTLS paths.
- Miner env values (when used on non-TEE compat surfaces) are write-only after lock, encrypted at rest
  under challenge storage, and never returned by public status routes.
- Public proxy must not store or log raw env bodies, nonces as secrets, mnemonics, or golden keys.

## Measurement allowlist

The allowlist pins static measurement fields on product formula (plus event-log `compose_hash` /
`key_provider` replay):

`mrtd`, `rtmr0`, `rtmr1`, `rtmr2`, `compose_hash`, `os_image_hash`

Product `os_image_hash` is the SHA-256 of the concatenation of the hex-decoded MRTD, RTMR1, and
RTMR2 registers (same formula used when reconstructing identity from a quote). An empty allowlist
fails closed (no key, no accepted score). Miners can reproduce measurements with the self-deploy
CLI; they cannot edit the operator allowlist. Review and eval pins are separate lists.

## Score acceptance fail-closed

Score admission requires the conjunction of quote integrity, allowlisted measurement, domain
`report_data`, single-use nonces, **durable key-grant** for that eval run, and **fresh re-verified
review allow materials**. Failure or absence of key-grant / attestation materials writes **no**
accepted score and **no** weight eligibility.

## Public surface redaction

Status, SSE, review history, and eval status APIs expose phase, digests, bounded reason codes, and
progress counts. They do not expose source code, raw model transcripts, bearer tokens, run tokens,
quote blobs, full event logs, Swarm refs, or free-form internal diagnostics.

## 502 hygiene (BASE proxy)

A transport failure at the BASE public proxy becomes a safe 502. UIs should show unavailable copy,
not raw proxy error text. Challenge-origin client errors should pass through with safe bodies.
Operator checks: proxy routes `/challenges`, blocks `/internal/*`, strips sensitive non-signing
headers, preserves only signed miner headers where required.

## Honesty language

Prefer: cryptographically-anchored trust-but-audit, verifier-bound acceptance, residual risk documented.
Avoid: trustless, 100% sealed, anonymous-by-default as product claims.

## Related

- [Architecture](architecture.md)
- [Attestation TEE](miner/attestation-tee.md)
- [Operator self-deploy](validator/self-deploy.md)
