# gbase Agent Challenge Specification

**Status:** FROZEN (task 9 wave gate)  
**Normative for:** `agent-challenge` service, `miner` deploy/certify, validator attestation integration  
**challenge_id:** `agent-v1`  
**challenge_scoring_version:** `1`  
**Bundle leaf protocol:** [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) **`protocol_version = 1`**

This file is the single source of truth for the first gbase challenge: topology, wire protocol between the challenge service and the miner CVM, the offline-testable scoring rule, challenge key custody, D24 participant coverage, and the Phala compose image/port contract consumed by tasks 37/38.

Where this document and any other source disagree on **agent-challenge behaviour**, **this document wins**.  
Where this document and [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) disagree on **leaf bytes, signatures, aggregation, or bundle verify**, **BUNDLE_SPEC wins**.

Checklist map: [`AGENT_CHALLENGE_CHECKLIST.md`](./AGENT_CHALLENGE_CHECKLIST.md).  
CI gate: `cargo run -p xtask -- agent-challenge-check`.

---

## 0. Document conventions

| Notation | Meaning |
|----------|---------|
| `u8`/`u16`/`u32`/`u64` | SCALE fixed-width little-endian unsigned integers |
| `[u8; N]` | Fixed-length byte array |
| `Bytes` / `Vec<T>` | SCALE compact length + payload |
| `scale(T)` | Canonical SCALE encoding |
| `sha256` / `sha512` | SHA-256 / SHA-512 digests |
| `‖` | Byte concatenation |
| HTTP JSON | Allowed **only** on the challenge↔miner CVM hop (§4). Forbidden in bundle leaves (BUNDLE_SPEC §1) |

Hotkeys inside SCALE structures are `[u8; 32]` raw public keys, not SS58 strings.

Cross-reference: leaf `ScoreOrAbsence`, `NoScoreReasonCode`, and `gbase-rawweight-v1` signing are defined in BUNDLE_SPEC §3. This document defines **when** each reason is chosen and **how** `Score { value: u64 }` is computed.

---

## 1. What runs where (topology)

Three distinct trust domains. Do not collapse them.

```text
Operator / master host (compose profile master)
  postgres · gateway · validator · updater · socket-proxy
  + agent-challenge service (registered challenge backend)
           | HTTPS (gateway TLS terminates, D20)
           | proxy path /challenge/agent-v1/*
           v
Challenge service process
  - challenge signing secret (mounted file only)
  - expected set from local trust root + metagraph (D24)
  - task issue, score, sign leaves, POST /v1/weights/raw
           | HTTPS to miner public URL
           v
Miner Phala TDX CVM (measured app-compose)
  - agent HTTP server
  - report_data builder (D10) inside measured image
  - quote + event-log for certify
  - NO challenge signing key
           | certify quote -> validator
           v
Validator
  - nonces, parse/replay/policy (D10/D11/D13)
  - Verified | Rejected | Parked
  - bundle verify per BUNDLE_SPEC; does NOT re-score agents (D19)
```

| Component | Host | TDX-measured? | Holds challenge sk? | Signs leaves? |
|-----------|------|---------------|---------------------|---------------|
| Miner agent HTTP | Miner Phala CVM | **Yes** | No | No |
| report_data + quote path | Miner Phala CVM | **Yes** | No | No |
| Challenge service | Operator backend (gateway-registered) | No | **Yes** (file) | **Yes** |
| Gateway | Master only (D3) | No | No (routing DB only, D18) | Bundle only |
| Validator | Validator host | No | No | Dissent / peer-root |

**Normative split:**

1. **Miner CVM** proves which measured code answered and liveness this epoch.  
2. **Challenge service** computes numeric scores and signs leaves.  
3. **Validator** enforces attestation policy and bundle cryptography. It does **not** re-derive scores from agent transcripts (challenge honesty is out of scope per D19).

---

## 2. Identifiers and versions

| Field | Value |
|-------|-------|
| `challenge_id` | UTF-8 `agent-v1` (SCALE `Bytes`) |
| `challenge_scoring_version` | `u16 = 1` |
| Bundle `protocol_version` | `1` ([`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md)) |
| Trust-root row | Owner-signed `config/challenges.toml` entry: id, public key, emission bps, `ParticipantPolicy` |

Bump `challenge_scoring_version` when task generation, scoring math, score-affecting HTTP schema, attestation precondition, or compose service/port/image contract changes.  
Leaf SCALE layout changes go through BUNDLE_SPEC `protocol_version`.

---

## 3. Attestation precondition (explicit)

### 3.1 Rule

**A same-epoch `Verified` attestation is a precondition for emitting `Score { value }` for that miner.**

| Attestation outcome this epoch | Leaf for an expected miner |
|--------------------------------|----------------------------|
| `Verified` | May be `Score` or `NoScore` (other reasons below) |
| `Rejected` | MUST `NoScore { reason: AttestationNotVerified }` (code **3**) |
| `Parked` | MUST `NoScore { reason: AttestationNotVerified }` (code **3**). Park grants **no** credit and never carries a prior `Verified` (D13) |
| Missing / undecided at seal deadline | MUST `NoScore { reason: AttestationNotVerified }` (code **3**) |

The challenge MUST NOT emit `Score` unless attestation status is `Verified` for `(netuid, epoch, miner_hotkey)` under D10 binding.

### 3.2 D10 `report_data` (miner CVM, measured code)

```text
report_data = SHA512(
  scale(b"gbase-attest-v1")
  ‖ scale(netuid: u16)
  ‖ scale(epoch: u64)
  ‖ scale(miner_pubkey: [u8; 32])
  ‖ scale(nonce: [u8; 32])
  ‖ scale(validator_hotkey: [u8; 32])
)
```

- Domain tag matches `crypto` / BUNDLE_SPEC appendix A: `gbase-attest-v1`.  
- Nonce: validator-issued 32 bytes, single-use, TTL strictly less than one epoch.  
- Construction code lives **inside** the measured compose so the allowlist pins it.  
- D11: only env **names** and `LAUNCH_TOKEN` **hash** are measured. Env **values** are not attested. Secrets are **mounted files**, never secret values in compose `environment:`.

### 3.3 Status channel

The challenge service reads attestation outcomes from its configured control-plane source (shared DB or authenticated internal API). It MUST NOT invent `Verified`. If the channel is unavailable at scoring time, treat as Missing → `AttestationNotVerified`.

### 3.4 Non-claims

Attestation does not prove env secret values (D11), challenge score honesty, or owner honesty (D19).

---

## 4. Challenge ↔ miner CVM protocol

Transport: HTTPS. `Content-Type: application/json; charset=utf-8`.

### 4.1 Miner endpoints (public base URL)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Liveness; body `ok` |
| `GET` | `/readyz` | Ready for tasks |
| `POST` | `/v1/task` | Execute one task (scoring path) |
| `GET` | `/v1/quote` | Certify path (task 38); not used for scoring |

### 4.2 `POST /v1/task` request

```json
{
  "protocol": "gbase-agent-task-v1",
  "challenge_id": "agent-v1",
  "challenge_scoring_version": 1,
  "netuid": 1,
  "epoch": 42,
  "miner_hotkey_hex": "<64 lowercase hex>",
  "task_id_hex": "<64 lowercase hex>",
  "task_blob_hex": "<64 lowercase hex>",
  "deadline_unix_ms": 0
}
```

| Field | Rule |
|-------|------|
| `protocol` | Exactly `gbase-agent-task-v1` |
| `challenge_id` | Exactly `agent-v1` |
| `challenge_scoring_version` | `1` |
| `miner_hotkey_hex` | Must match CVM configured hotkey |
| `task_id_hex` / `task_blob_hex` | From §5.2 |
| `deadline_unix_ms` | Absolute deadline; miner SHOULD stop after |

### 4.3 `POST /v1/task` success response (HTTP 200)

```json
{
  "protocol": "gbase-agent-task-v1",
  "challenge_id": "agent-v1",
  "epoch": 42,
  "miner_hotkey_hex": "<64 hex>",
  "task_id_hex": "<64 hex>",
  "answer_digest_hex": "<64 hex>",
  "worker_ms": 12,
  "agent_version": "1"
}
```

| Field | Rule |
|-------|------|
| `answer_digest_hex` | Hex of `sha256(answer_bytes)` per §5.3 |
| `worker_ms` | Miner-reported; **ignored for scoring**. Challenge uses its own wall clock (§5.4) |
| `agent_version` | Exactly `"1"` for scoring_version 1 |

### 4.4 Error map

| Outcome | `NoScoreReasonCode` |
|---------|---------------------|
| HTTP 400 / 403 / schema fail / hotkey mismatch | `InvalidResponse` (2) |
| HTTP 500 / `agent_internal` | `MinerError` (4) |
| 503 exhausted / transport fail / deadline | `Timeout` (1) |
| Rate limit (HTTP 429) | `RateLimited` (5) |
| Challenge-side fault after retries exhausted | `ChallengeInternal` (6) still MUST cover participant |

### 4.5 Timing constants

```text
SOFT_MS      = 2_000
HARD_MS      = 10_000
CONNECT_MS   = 3_000
MAX_ATTEMPTS = 2
```

`duration_ms` is challenge-side wall time of the **successful** attempt, or time to final failure. Do not sum retry attempts for latency credit.

### 4.6 Submission idempotency

At most one signed leaf per `(challenge_id, epoch, miner_hotkey)`.  
Gateway `POST /v1/weights/raw` is idempotent on that key. Challenge retries 5xx; never submits two conflicting `ScoreOrAbsence` values for the same key.

---

## 5. Score meaning and scoring rule (`challenge_scoring_version = 1`)

### 5.1 Meaning

`Score { value: u64 }` is this challenge's raw weight input to BUNDLE_SPEC §6 aggregation.

| Property | Rule |
|----------|------|
| Ordering | Higher is better |
| Range | `0 ..= SCORE_MAX` |
| `SCORE_MAX` | `1_000_000` |
| Zero score | Valid `Score(0)` means attempted and failed correctness (or zero latency credit at HARD boundary) |
| `NoScore` | Signed absence; aggregation later treats raw as 0 **after** D24 completeness |

### 5.2 Task generation (pure, offline)

```text
task_id = sha256(
  b"gbase-agent-task-id-v1" ‖
  scale(netuid: u16) ‖
  scale(epoch: u64) ‖
  miner_hotkey: [u8; 32]
)

task_blob = sha256(
  b"gbase-agent-task-blob-v1" ‖
  task_id ‖
  scale(challenge_scoring_version: u16)   // 1
)
```

No RNG. No I/O. Fixtures fix `(netuid, epoch, miner_hotkey)` only.

### 5.3 Expected answer (pure, offline)

Honest agent:

```text
answer_bytes  = b"gbase-agent-answer-v1" ‖ task_blob
answer_digest = sha256(answer_bytes)
```

### 5.4 Scoring function (pure, integer-only)

```text
function score_miner(...) -> ScoreOrAbsence:

  if attestation != Verified:
    return NoScore(AttestationNotVerified)        // 3

  if terminal timeout / transport exhausted:
    return NoScore(Timeout)                       // 1
  if miner HTTP 500 / agent_internal:
    return NoScore(MinerError)                     // 4
  if schema / 400 / 403 / hotkey mismatch:
    return NoScore(InvalidResponse)               // 2
  if HTTP 429 exhausted:
    return NoScore(RateLimited)                   // 5

  // HTTP 200 body present
  if resp fields disagree with request epoch/challenge_id/task_id:
    return NoScore(InvalidResponse)
  if resp.agent_version != "1":
    return NoScore(InvalidResponse)

  expected = sha256(b"gbase-agent-answer-v1" ‖ task_blob)
  if resp.answer_digest != expected:
    return Score(0)

  if duration_ms > HARD_MS:
    return NoScore(Timeout)
  if duration_ms <= SOFT_MS:
    return Score(SCORE_MAX)                       // 1_000_000

  // Linear decay on integer lattice:
  // value = floor( SCORE_MAX * (HARD_MS - duration_ms) / (HARD_MS - SOFT_MS) )
  span = HARD_MS - SOFT_MS                        // 8000
  value = (SCORE_MAX * (HARD_MS - duration_ms)) / span
  return Score(value as u64)
```

Bare floating point is forbidden in the scorer implementation (same spirit as D8 for this pure function).

### 5.5 Worked fixture F1 (task 36 MUST match)

**Inputs:**

```text
netuid       = 1
epoch        = 7
miner_hotkey = 32 bytes of 0x11
attestation  = Verified
duration_ms  = 2000
answer       = correct
```

**Derived digests (pinned):**

```text
task_id_hex =
  4a590b2abf87da6bccd97d8fbe5d2e774bdbda3ad421119688010537be2b31ec

task_blob_hex =
  8c5430ceb95b9e422026baf2eaddb4c9c723923c6353164fe9b0905a47f9a29f

answer_digest_hex =
  83180b08e05630496531a158d174ce69ba857d854d8692087947706c159a487c
```

**Result:** `Score { value: 1_000_000 }`

### 5.6 Additional fixture table (task 36)

Same `netuid=1`, `epoch=7`, `miner_hotkey=[0x11;32]`, correct answer, `Verified`, unless noted:

| Fixture id | Variation | Expected leaf |
|------------|-----------|---------------|
| F1 | `duration_ms=2000` | `Score(1_000_000)` |
| F2 | `duration_ms=0` | `Score(1_000_000)` |
| F3 | `duration_ms=6000` | `Score(500_000)` |
| F4 | `duration_ms=10000` | `Score(0)` |
| F5 | `duration_ms=10001` | `NoScore(Timeout)` |
| F6 | wrong `answer_digest` | `Score(0)` |
| F7 | attestation `Parked` | `NoScore(AttestationNotVerified)` |
| F8 | attestation `Missing` | `NoScore(AttestationNotVerified)` |
| F9 | attestation `Rejected` | `NoScore(AttestationNotVerified)` |
| F10 | HTTP schema fail | `NoScore(InvalidResponse)` |
| F11 | second miner `hotkey=[0x22;32]`, correct, `duration_ms=2000` | `Score(1_000_000)` with digests below |

**F11 digests:**

```text
task_id_hex =
  d954306fba3943a86bb69aedfd08f2bca850eb2adabaaf5efe2ad2728dbf3412
answer_digest_hex =
  05157d001bb1ec9ef5acc7140d0221141d2fbc14a830ce32893793f30470c0aa
```

### 5.7 Reference assertions (task 36 offline tests)

```text
assert score(F1) == Score(1_000_000)
assert score(F3) == Score(500_000)
assert score(F5) == NoScore(Timeout)
assert score(F7) == NoScore(AttestationNotVerified)
assert hex(task_id(F1)) == pinned task_id_hex
assert hex(answer_digest(F1)) == pinned answer_digest_hex
```

---

## 6. Key custody (challenge signing key)

| Rule | Requirement |
|------|-------------|
| Algorithm | sr25519 (same as BUNDLE_SPEC leaf sigs) |
| Public key | Committed in owner-signed `config/challenges.toml` for `agent-v1` |
| Secret key | **Never** in git. Stored `age`-encrypted offline; decrypted to a **mode 0600 file** on the challenge host |
| Runtime load | Challenge process reads secret from file path env `GBASE_CHALLENGE_SK_FILE` |
| Gateway DB | **Must not** store challenge secrets or be consulted for leaf provenance (D18) |
| Signing domain | `gbase-rawweight-v1` over `RawWeightBodyV1` (BUNDLE_SPEC §3.4) |
| Rotation | D21 dual-accept window via trust-root release; keygen via `trustroot keygen` (task 18) |
| Compromise | Rotate trust root; D19 still applies until rotation lands |

Validators verify leaf signatures **only** against their **local** trust-root copy.

---

## 7. Declared participant set and `NoScore` reasons (D24)

### 7.1 Who defines the set

The challenge service does **not** choose the expected set unilaterally.

1. Read `ParticipantPolicy` for `agent-v1` from **local** owner-signed `config/challenges.toml`.  
2. Read `metagraph_at(block_hash)` for the epoch's `block_B` / `block_hash` (BUNDLE_SPEC §4).  
3. Run BUNDLE_SPEC §7.2 `expected_participants(agent-v1, policy, block_hash, chain)`.  
4. That set `E` is the coverage obligation.

Validators derive the **same** `E` independently. A bundle whose leaves cover only a proper subset of `E` is rejected (BUNDLE_SPEC §7.3).

### 7.2 Coverage obligation

For every `h ∈ E`, the challenge MUST produce exactly one signed leaf `(challenge_id=agent-v1, miner_hotkey=h, epoch)` with either `Score` or `NoScore`.  
**Silence is a bug.** Missing leaves fail gateway seal and validator verify.

### 7.3 Reason selection (normative priority)

Evaluate in order; first match wins:

| Priority | Condition | Reason |
|----------|-----------|--------|
| 1 | Not in `E` | Do not emit a leaf (extra leaves rejected by validators) |
| 2 | Attestation not `Verified` | `AttestationNotVerified` (3) |
| 3 | Deadline / timeout / transport | `Timeout` (1) |
| 4 | Rate limited | `RateLimited` (5) |
| 5 | Schema / bad response | `InvalidResponse` (2) |
| 6 | Miner explicit 500 | `MinerError` (4) |
| 7 | Challenge internal fault | `ChallengeInternal` (6) |
| 8 | Else scored | `Score(value)` per §5 |

**Forbidden:**

| Code | Name | Rule |
|------|------|------|
| 0 | `NotAttempted` | MUST NOT skip work for an `h ∈ E` under normal operation. A leaf is still required if ever used |
| 7 | `PolicySkip` | MUST NOT shrink `E` (BUNDLE_SPEC §3.3.1). Coverage comes from owner policy + metagraph only |

### 7.4 Default trust-root policy

Initial testnet policy is whatever the signed `challenges.toml` carries (typically `AllMetagraphHotkeys` or `StakeAtLeast`). Emission bps are D23/trust-root, not defined here.

---

## 8. Leaf emission and gateway POST

After scoring each `h ∈ E`:

```text
body = RawWeightBodyV1 {
  challenge_id: b"agent-v1",
  miner_hotkey: h,
  epoch,
  score_or_absence: Score(v) | NoScore(reason),
}
sig = sr25519_sign(challenge_sk, tag "gbase-rawweight-v1" ‖ scale(body))
leaf = LeafV1 { ... body fields ..., challenge_sig: sig }
POST /v1/weights/raw  with the gateway-accepted leaf envelope
```

Gateway verifies against **its** local trust root (defence in depth) and appends. Unknown challenge id → reject. Wrong key → reject.

---

## 9. Compose services, ports, image contract (tasks 37/38)

### 9.1 Miner CVM `app-compose.json` services (normative names)

| Service name | Role | Container port | Published |
|--------------|------|----------------|-----------|
| `agent` | HTTP agent (§4) | `8080` | Phala ingress → 8080 |
| `attest-helper` | Builds `report_data`, exports quote + event log for certify | `8081` (loopback inside CVM) | **Not** public |

No other long-lived services required for v1. No docker.sock. No `:latest` tags.

### 9.2 Image contract

```text
images:
  agent:
    repository: ghcr.io/baseintelligence/gbase-agent
    // digest-pinned only, e.g. repo@sha256:<64 hex>
  attest-helper:
    repository: ghcr.io/baseintelligence/gbase-attest-helper
    // digest-pinned only
```

| Rule | Requirement |
|------|-------------|
| Tags | Digest pins only; rendered compose must contain zero `:latest` |
| `allowed_envs` names | May include `GBASE_NETUID`, `GBASE_MINER_HOTKEY_FILE`, `GBASE_LAUNCH_TOKEN_HASH` |
| Secret values | **Files** under `/run/gbase/` (or equivalent), never env values |
| `LAUNCH_TOKEN` | Hash appears in measured compose (D11); raw token only as file if needed |
| compose-hash | Canonical JSON → SHA-256 per `compose-hash`; must match RTMR3 event and `mr_config_id` prefix |

### 9.3 `miner deploy` (task 37) obligations

1. Render compose from the template that satisfies §9.1–§9.2.  
2. Compute compose-hash **offline**; print it.  
3. `phala deploy` (or `--no-deploy` for hash-only).  
4. Miner funds their own Phala account (R3).  
5. Register public base URL with gateway routing for `agent-v1`.

### 9.4 `miner certify` (task 38) obligations

1. Request fresh nonce from validator.  
2. Inside CVM, build `report_data` per §3.2.  
3. Submit quote + event log.  
4. Validator: parse → replay → compose-hash → policy → redeem nonce → `Verified`/`Rejected`/`Parked`.

### 9.5 Challenge service process (operator side, not miner CVM)

| Item | Contract |
|------|----------|
| Binary / image | `agent-challenge` digest-pinned |
| Listen port | `8090` (internal) |
| Secrets | `GBASE_CHALLENGE_SK_FILE` mount 0600 |
| Trust root | Local `config/challenges.toml` path |
| Health | `/healthz`, `/readyz` on 8090 |

Gateway remains the only public TLS terminator (D20).

---

## 10. Challenge trait shape (task 36 implementers)

Informative surface (behaviour must match this doc):

```text
trait Challenge {
  fn challenge_id(&self) -> &str;                    // "agent-v1"
  fn scoring_version(&self) -> u16;                  // 1
  fn expected_set(&self, ctx: &EpochCtx) -> BTreeSet<Hotkey>;
  fn score_one(&self, ctx: &EpochCtx, miner: Hotkey)
      -> ScoreOrAbsence;
  fn score_epoch(&self, ctx: &EpochCtx)
      -> BTreeMap<Hotkey, ScoreOrAbsence>;           // full cover of expected_set
  fn sign_leaf(&self, ...) -> LeafV1;
  fn submit_all(&self, gateway: &GatewayClient) -> Result<()>;
}
```

Prism (out of scope) must remain accommodable without breaking `agent-v1` leaf bytes.

---

## 11. Security claim boundary

D19 applies unchanged:

- Valid leaf signatures prove the **challenge key** attested those scores, not that the scores are fair.  
- TEE attestation proves measured miner code and D10 binding, not challenge honesty.  
- Owner signs trust roots and operates the gateway: owner compromise remains residual risk (R12).

---

## 12. Verification checklist (implementers)

1. Topology matches §1 (no challenge sk in CVM; no agent re-score in validator).  
2. Attestation precondition §3 enforced before any `Score`.  
3. Task/answer digests match §5.5 pinned fixtures.  
4. F1–F11 table §5.6 green in offline tests (task 36).  
5. Every `h ∈ E` gets a signed leaf; no silence (D24).  
6. `PolicySkip` never used to drop coverage.  
7. Compose render matches §9; no `:latest`; no secrets in `environment:`.  
8. Leaves verify under BUNDLE_SPEC `protocol_version = 1` and local trust root.  
9. E2E scores (task 47) are consistent with this rule, not hardcoded toys.

---

## Appendix A. Cross-links

| Topic | Document |
|-------|----------|
| Leaf SCALE, `NoScoreReasonCode`, rawweight domain | BUNDLE_SPEC §3 |
| Aggregation of scores | BUNDLE_SPEC §6 |
| Participant derivation | BUNDLE_SPEC §7 |
| D19 claim | BUNDLE_SPEC §11.1 |
| Bundle `protocol_version` | BUNDLE_SPEC §2 (**value 1**) |
| D10/D11/D13/D18/D24 decisions | plan `gbase-rust-subnet` |

---

## Appendix B. Related plan decisions

| Decision | Sections here |
|----------|---------------|
| D10 liveness binding | §3.2, §9.4 |
| D11 env names only | §3.2, §9.2 |
| D13 park no credit | §3.1 |
| D18 local challenge keys | §6, §8 |
| D19 honest claim | §1, §11 |
| D20 gateway TLS only | §1, §9.5 |
| D23/D24 participants + shares | §7 |
| D21 key rotation | §6 |

---

**End of frozen AGENT_CHALLENGE challenge_scoring_version=1 (bundle protocol_version=1).**
