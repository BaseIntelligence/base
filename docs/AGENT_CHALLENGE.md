# base Agent Challenge Specification

**Status:** FROZEN (task 16 re-freeze — scoring_version 2 / SWE Harbor packs)  
**Prior freeze:** task 9 wave gate (scoring_version 1 / echo) — **superseded** by this document  
**Normative for:** `agent-challenge` service, `miner` deploy/certify, miner runner (`base-agent`), validator attestation integration  
**challenge_id:** `agent-v1`  
**challenge_scoring_version:** `2`  
**Bundle leaf protocol:** [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) **`protocol_version = 1`**

This file is the single source of truth for the base agent challenge at **scoring_version 2**: topology, pack-based task dispatch, the `model.patch` contract, operator-side held-out grading, pure integer scoring, work receipts, challenge key custody, D24 participant coverage, and the Phala compose image/port contract (including the measured miner socket-proxy).

Where this document and any other source disagree on **agent-challenge behaviour**, **this document wins**.  
Where this document and [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md) disagree on **leaf bytes, signatures, aggregation, or bundle verify**, **BUNDLE_SPEC wins**.

Checklist map: [`AGENT_CHALLENGE_CHECKLIST.md`](./AGENT_CHALLENGE_CHECKLIST.md).  
CI gate: `cargo run -p xtask -- agent-challenge-check`.

### Re-freeze gate note (task 16)

Task 9 froze scoring_version **1** (SHA-256 echo answer + latency decay). Owner decision Q2=B bumps the same `challenge_id` (`agent-v1`) to scoring_version **2** (Harbor SWE packs + pure correctness). This rewrite is an intentional freeze invalidation + re-freeze: checklist pins and `xtask agent-challenge-check` update **in the same commit**. Bundle `protocol_version` remains **1** — do not confuse the two version axes.

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
| HTTP JSON | Allowed on challenge↔miner hops (§4). Forbidden in bundle leaves (BUNDLE_SPEC §1) |

Hotkeys inside SCALE structures are `[u8; 32]` raw public keys, not SS58 strings.

Cross-reference: leaf `ScoreOrAbsence`, `NoScoreReasonCode`, and `base-rawweight-v1` signing are defined in BUNDLE_SPEC §3. This document defines **when** each reason is chosen and **how** `Score { value: u64 }` is computed under scoring_version 2.

### 0.1 Invariants preserved across the version bump (I1–I8)

| id | invariant |
|----|-----------|
| I1 | Same-epoch `Verified` attestation gates `Score`; else `NoScore{AttestationNotVerified}` (3) |
| I2 | Exactly one leaf per expected `(challenge_id, hotkey)` per epoch; silence is a bug (D24) |
| I3 | No floating point in the scorer |
| I4 | Challenge signing key never inside the miner CVM (D18) |
| I5 | Validators verify and recompute; never re-score agents (D19) |
| I6 | Merkle root absent from the on-chain weight payload (D5) |
| I7 | Expected set `E` is sealed at `block_B`; late registrations wait for the next epoch |
| I8 | Bundle `protocol_version` stays **1** |

---

## 1. What runs where (topology)

Three distinct trust domains. Do not collapse them.

```text
Operator / master host (compose profile master)
  postgres · gateway · validator · updater · socket-proxy (operator Docker API)
  + agent-challenge service (registered challenge backend)
           | HTTPS (gateway TLS terminates, D20)
           | proxy path /challenge/agent-v1/*
           v
Challenge service process (orchestrator)
  - challenge signing secret (mounted file only)
  - expected set from local trust root + metagraph (D24)
  - pack catalog + select_pack(epoch, hotkey)
  - dispatch stripped descriptors; collect model.patch + work receipt
  - HarborVerifier (held-out tests) via operator socket-proxy
  - score, sign leaves, POST /v1/weights/raw
           | HTTPS to miner public URL
           v
Miner Phala TDX CVM (measured app-compose)
  - base-agent runner HTTP server (:8080)
  - measured socket-proxy (Docker Engine allowlist; see §9.1)
  - pack environment containers (agent workload only)
  - CVM-local work-receipt key (not the challenge sk)
  - report_data builder (D10) + attest-helper (:8081)
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
| Miner agent runner | Miner Phala CVM | **Yes** | No | No (signs **work receipt** only) |
| Miner socket-proxy | Miner Phala CVM | **Yes** (in compose-hash) | No | No |
| report_data + quote path | Miner Phala CVM | **Yes** | No | No |
| Challenge service | Operator backend | No | **Yes** (file) | **Yes** (leaves) |
| Operator HarborVerifier | Operator host via socket-proxy | No | No | No |
| Gateway | Master only (D3) | No | No (routing DB only, D18) | Bundle only |
| Validator | Validator host | No | No | Dissent / peer-root |

**Normative split:**

1. **Miner CVM** proves which measured code ran, executes packs behind a measured socket-proxy, and returns `model.patch` plus a work receipt.  
2. **Challenge service** selects packs, grades held-out tests operator-side, computes numeric scores, and signs leaves.  
3. **Validator** enforces attestation policy and bundle cryptography. It does **not** re-derive scores from agent transcripts (challenge honesty is out of scope per D19).

---

## 2. Identifiers and versions

| Field | Value |
|-------|-------|
| `challenge_id` | UTF-8 `agent-v1` (SCALE `Bytes`) |
| `challenge_scoring_version` | `u16 = 2` |
| Bundle `protocol_version` | `1` ([`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md)) |
| Trust-root row | Owner-signed `config/challenges.toml` entry: id, public key, emission bps, `ParticipantPolicy` |

Bump `challenge_scoring_version` when task generation, scoring math, score-affecting HTTP schema, attestation precondition, pack selection, or compose service/port/image contract changes.  
Leaf SCALE layout changes go through BUNDLE_SPEC `protocol_version` (still **1** for this freeze).

### 2.1 Historical scoring_version 1 (retired)

scoring_version **1** used a pure echo answer  
`answer_digest = sha256(b"base-agent-answer-v1" ‖ task_blob)`  
and integer latency decay with historical constants `SOFT_MS = 2_000` and `HARD_MS = 10_000`.  
Those constants are **not** live scoring inputs under version 2. Offline code may retain v1 digest helpers solely for golden regression of the retired formulas.

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
  scale(b"base-attest-v1")
  ‖ scale(netuid: u16)
  ‖ scale(epoch: u64)
  ‖ scale(miner_pubkey: [u8; 32])
  ‖ scale(nonce: [u8; 32])
  ‖ scale(validator_hotkey: [u8; 32])
)
```

- Domain tag matches `crypto` / BUNDLE_SPEC appendix A: `base-attest-v1`.  
- Nonce: validator-issued 32 bytes, single-use, TTL strictly less than one epoch.  
- Construction code lives **inside** the measured compose so the allowlist pins it.  
- D11: only env **names** and `LAUNCH_TOKEN` **hash** are measured. Env **values** are not attested. Secrets are **mounted files**, never secret values in compose `environment:`.

### 3.3 Status channel

The challenge service reads attestation outcomes from its configured control-plane source (shared DB or authenticated internal API). It MUST NOT invent `Verified`. If the channel is unavailable at scoring time, treat as Missing → `AttestationNotVerified`.

### 3.4 What v2 attestation does **not** prove

Attestation proves **which measured code** answered a **fresh, bound** challenge for this epoch. It does **not** prove:

- env secret values (D11);
- challenge score honesty or owner honesty (D19);
- that `model.patch` is free of external cheating (public PR leakage, open egress);
- that the operator-side Harbor grade is fair (operator is inside the D19 trust boundary);
- that a work receipt implies a correct solve — receipts bind identity of the patch bytes, not reward.

---

## 4. Challenge ↔ miner CVM protocol

Transport: HTTPS. `Content-Type: application/json; charset=utf-8`.

Two JSON surfaces coexist:

1. **Dispatch** (`base-agent-dispatch-v1`) — orchestrator ↔ runner task descriptor / result (normative for scoring_version 2).  
2. **Legacy task envelope** (`base-agent-task-v1`) — retained only where older probes still speak it; live scoring uses dispatch + operator grade.

### 4.1 Miner endpoints (public base URL)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz` | Liveness; body `ok` |
| `GET` | `/readyz` | Ready for tasks |
| `GET` | `/v1/capacity` | Effective `max_concurrency` and current load |
| `POST` | `/v1/task` | Accept one dispatch descriptor |
| `GET` | `/v1/task/{id}` | Poll status; when complete, `model.patch` + receipt |
| `GET` | `/v1/quote` | Certify path; not used for scoring |

### 4.2 Dispatch descriptor (`TaskDescriptorV1`)

```json
{
  "protocol": "base-agent-dispatch-v1",
  "challenge_id": "agent-v1",
  "scoring_version": 2,
  "epoch": 42,
  "miner_hotkey_hex": "<64 lowercase hex>",
  "pack_id": "<catalog pack id>",
  "deadline_unix_ms": 0
}
```

| Field | Rule |
|-------|------|
| `protocol` | Exactly `base-agent-dispatch-v1` |
| `challenge_id` | Exactly `agent-v1` |
| `scoring_version` | `2` |
| `pack_id` | From §5.2 `select_pack` |
| `deadline_unix_ms` | Absolute deadline; runner MUST stop after (R1) |

The descriptor carries a **stripped** pack projection only: instruction + environment image digest / base commit metadata. It MUST NOT include `solution/`, `tests/test.patch`, or `grader.py` (anti-cheat).

### 4.2.1 Agent egress posture (scoring_version 2 — LOCKED default)

**Default: OPEN** (`AgentEgressPosture::Open`). The miner pack-environment container is **not** network-locked down in v2 (`NetworkDisabled=false`).

| Posture | Default? | Meaning |
|---------|----------|---------|
| **OPEN** | **yes** | Agent container may use the network. Honest claim: stripping protects **grading-channel integrity** (held-out `solution/` / `tests/` / `grader.py` never reach the miner), **not** miner honesty. |
| Allowlisted egress proxy | **no** (OFF by default) | Optional stronger mode: only miner model endpoint. Not implied by v2; enable only via explicit miner config. |

**Metis B6 residual risk (explicit):** Harbor packs are public merged PRs (`repository_url` + `base_commit_hash`). With OPEN egress an agent can fetch the upstream fix. Stripping is therefore **not cheat-proof**. D19 already disclaims score honesty and owner honesty; OPEN is consistent with that boundary, not a silent hole.

Contrast: operator-side `HarborVerifier` grades with `network_disabled: true`. That offline grade path is independent of the miner agent egress default.

Pack `task.toml` `allow_internet = false` is the **image build** offline contract (deps baked at build). It is **not** the CVM agent egress posture.

### 4.2.2 Pack execution and model key (Q3=A)

The runner resolves `pack_id` to a stripped Harbor projection, pulls the digest-pinned environment image through the measured socket-proxy allowlist, runs a reference agent against `instruction`, and collects `/logs/artifacts/model.patch`.

| Item | Contract |
|------|----------|
| Artifact | `/logs/artifacts/model.patch` (unified diff) |
| Deadline | `min(deadline_unix_ms remaining, agent.timeout_sec)` → hard stop → `status=timed_out`, no patch, **still signed** work receipt |
| Containers | Owned name prefix `base-verify-agent-` (subset of `base-verify-`); unconditional teardown after each attempt |
| Model key (Q3=A) | Miner-supplied file mount (e.g. `/run/base/model_key`); env carries **path only** (`MODEL_KEY_FILE`); **never** key bytes in logs, compose `environment:` values, or measured env values |
| Receipt | `patch_sha256 = sha256(model.patch bytes)`; sr25519 under `base-agent-work-receipt-v1` |

### 4.3 Dispatch result (`TaskResultV1`)

```json
{
  "protocol": "base-agent-dispatch-v1",
  "challenge_id": "agent-v1",
  "scoring_version": 2,
  "epoch": 42,
  "miner_hotkey_hex": "<64 hex>",
  "pack_id": "<pack id>",
  "status": "completed",
  "model_patch": "diff --git ...",
  "patch_sha256_hex": "<64 hex>",
  "receipt_sig_hex": "<128 hex>"
}
```

| Field | Rule |
|-------|------|
| `status` | `completed` \| `timed_out` \| `failed` |
| `model_patch` | Unified diff text when produced; omitted/empty on timeout |
| `patch_sha256_hex` | Hex of `sha256(model.patch bytes)`; zero digest when no patch |
| `receipt_sig_hex` | sr25519 over work-receipt body (§5.6) |

### 4.4 Error map (challenge↔miner hop)

| Outcome | `NoScoreReasonCode` |
|---------|---------------------|
| HTTP 400 / 403 / schema fail / hotkey mismatch | `InvalidResponse` (2) |
| HTTP 500 / `agent_internal` | `MinerError` (4) |
| 503 exhausted / transport fail / epoch deadline | `Timeout` (1) |
| Rate limit (HTTP 429) | `RateLimited` (5) |
| Challenge-side fault after retries exhausted | `ChallengeInternal` (6) still MUST cover participant |

### 4.5 Timing constants (protocol, not latency scoring)

```text
CONNECT_MS   = 3_000
MAX_ATTEMPTS = 2
```

`duration_ms` (challenge-side wall time) is **informational only** under scoring_version 2. It does **not** decay `Score`. Epoch deadline is a hard `Timeout` boundary (R1), not a soft/hard latency lattice.

### 4.6 Submission idempotency

At most one signed leaf per `(challenge_id, epoch, miner_hotkey)`.  
Gateway `POST /v1/weights/raw` is idempotent on that key. Challenge retries 5xx; never submits two conflicting `ScoreOrAbsence` values for the same key.

---

## 5. Score meaning and scoring rule (`challenge_scoring_version = 2`)

### 5.1 Meaning

`Score { value: u64 }` is this challenge's raw weight input to BUNDLE_SPEC §6 aggregation.

| Property | Rule |
|----------|------|
| Ordering | Higher is better |
| Range | `0 ..= SCORE_MAX` |
| `SCORE_MAX` | `1_000_000` |
| Correct solve (reward 1) | `Score(SCORE_MAX)` |
| Incorrect / apply-fail / tests-fail (reward 0) | `Score(0)` |
| `NoScore` | Signed absence; aggregation later treats raw as 0 **after** D24 completeness |
| All-zero epoch | Production aggregation uses `PARTICIPATION_FLOOR = false` → empty final vector / no-submit (BUNDLE_SPEC §6.5) |

### 5.2 Pack selection (R1 cadence)

**Exactly one deterministically selected pack per miner per epoch.**

```text
// catalog: ordered PackId slice, identical across challenge replicas
// n = catalog.len(); n == 0 → EmptyCatalog (operator fault → ChallengeInternal leaves)

digest = sha256(b"base-agent-pack-select-v1" ‖ miner_hotkey)
seed   = u64::from_le_bytes(digest[0..8])   // little-endian
index  = seed.wrapping_add(epoch) % (n as u64)
pack_id = catalog[index]
```

Domain tag: `base-agent-pack-select-v1`. No RNG. No I/O. No wall clock.

**R1 deadline:** hard stop at approximately **60% of the epoch** (testnet tempo ≈ 72 min → ~43 min). Unfinished work → `NoScore(Timeout)`. No multi-pack cross-epoch carry-over in this freeze.

### 5.3 Task identity formulas (pure, offline, v2)

```text
task_id_v2 = sha256(
  b"base-agent-task-id-v2" ‖
  scale(netuid: u16) ‖
  scale(epoch: u64) ‖
  miner_hotkey: [u8; 32] ‖
  scale(pack_id: Vec<u8>) ‖   // UTF-8 pack id bytes
  scale(scoring_version: u16) // 2
)

task_blob_v2 = sha256(
  b"base-agent-task-blob-v2" ‖
  task_id_v2 ‖
  scale(scoring_version: u16) ‖
  scale(pack_id: Vec<u8>)
)

answer_digest_v2 = sha256(
  b"base-agent-answer-v2" ‖
  model_patch                 // raw returned model.patch bytes
)
```

Domain tags are distinct from v1, from `base-agent-work-receipt-v1`, and from `base-attest-v1`.  
`answer_digest_v2` is **not** equal to untagged `sha256(model.patch)` (receipt `patch_sha256`).

### 5.4 Operator-side grading (`HarborVerifier`)

The operator runs the pack's held-out `tests/` harness in a digest-pinned container through `docker_engine::AllowlistClient` + verifier allowlist (`HarborVerifier`).

| Rule | Requirement |
|------|-------------|
| Held-out bytes | Stay on the operator host; never projected into agent-facing types |
| Resolve (reward 1) | Every F2P **and** every P2P node passes |
| Dual-truth | `solution.patch` → reward 1; empty patch → reward 0 |
| Default timeout | pack `verifier.timeout_sec` or 1800 s |
| Docker path | Operator `socket-proxy` only — no raw `/var/run/docker.sock` on long-lived challenge process |

Binary reward maps to the score lattice via §5.5 / §7.3.

### 5.5 Scoring function (pure, integer-only, correctness-only)

```text
function score_miner(...) -> ScoreOrAbsence:

  if attestation != Verified:
    return NoScore(AttestationNotVerified)        // 3

  // duration_ms intentionally unused — no latency decay in v2

  if terminal timeout / transport exhausted / R1 deadline:
    return NoScore(Timeout)                       // 1
  if miner HTTP 500 / agent_internal:
    return NoScore(MinerError)                     // 4
  if schema / 400 / 403 / hotkey mismatch:
    return NoScore(InvalidResponse)               // 2
  if HTTP 429 exhausted:
    return NoScore(RateLimited)                   // 5
  if challenge / verifier operator fault exhausted:
    return NoScore(ChallengeInternal)             // 6

  // After HarborVerifier (or pure fixture path with expected_model_patch):
  if reward == 1 OR answer_digest_v2(model.patch) == expected:
    return Score(SCORE_MAX)                       // 1_000_000
  else:
    return Score(0)
```

Bare floating point is forbidden in the scorer implementation (I3).

### 5.6 Work receipt (R3)

Signed **inside** the miner CVM by a key that exists only in the measured compose (not the challenge sk; not required in the trust-root ceremony for validators).

Domain: `base-agent-work-receipt-v1` (`crypto::domain::WORK_RECEIPT`).

```text
WorkReceiptBodyV1 {
  challenge_id: Vec<u8>,       // b"agent-v1"
  scoring_version: u16,        // 2
  epoch: u64,
  miner_hotkey: [u8; 32],
  pack_id: Vec<u8>,
  patch_sha256: [u8; 32],      // untagged sha256(model.patch)
}
sig = sr25519_sign(cvm_receipt_sk, tag WORK_RECEIPT ‖ scale(body))
```

The **challenge service** verifies receipts against the receipt public key published in the measured miner compose. Validators do not need this key (D19 — no re-score).

A valid receipt proves the measured runner attested those patch bytes for that pack/epoch. It does **not** prove reward 1.

### 5.7 Worked fixture F1 v2 (offline MUST match)

**Inputs:**

```text
netuid       = 1
epoch        = 7
miner_hotkey = 32 bytes of 0x11
pack_id      = b"pack-fixture-001"
model.patch  = b"diff --git a/x b/x\n+hello\n"
attestation  = Verified
scoring_version = 2
```

**Derived digests (pinned):**

```text
task_id_v2_hex =
  b1c18e56abe993e20e8dadcb72c7a7cadee8975e5741d15d1acb37f5ea367644

task_blob_v2_hex =
  c563caca4fa3a7c5e834a88b0dae9eb1ef87f90fcddc9973e38d2730b347c441

answer_digest_v2_hex =
  703b806158d655e5d37a5b45e3cbdf1e04735517805377199d108ae2a45ead5d
```

**Result:** `Score { value: 1_000_000 }`

### 5.8 Additional fixture table (v2 successors of F1–F11)

Same `netuid=1`, `epoch=7`, `pack_id=pack-fixture-001`, fixture patch, unless noted:

| Fixture id | Variation | Expected leaf |
|------------|-----------|---------------|
| F1 | correct patch, `Verified` | `Score(1_000_000)` |
| F2 | `duration_ms=0`, correct | `Score(1_000_000)` (latency ignored) |
| F3 | `duration_ms=6000`, correct | `Score(1_000_000)` (no half-credit) |
| F4 | `duration_ms=10000`, correct | `Score(1_000_000)` |
| F5 | `duration_ms=10001` + `Http200` correct | `Score(1_000_000)`; Timeout only via `CallOutcome::Timeout` |
| F6 | wrong `answer_digest` / reward 0 | `Score(0)` |
| F7 | attestation `Parked` | `NoScore(AttestationNotVerified)` |
| F8 | attestation `Missing` | `NoScore(AttestationNotVerified)` |
| F9 | attestation `Rejected` | `NoScore(AttestationNotVerified)` |
| F10 | HTTP schema fail | `NoScore(InvalidResponse)` |
| F11 | second miner `hotkey=[0x22;32]`, correct | `Score(1_000_000)` with digests below |

**F11 digests:**

```text
task_id_v2_hex =
  b99762643336fbf7abeb2c07085ff3d64ee1fd8d1c98b149c57a36ec0396228f
answer_digest_v2_hex =
  703b806158d655e5d37a5b45e3cbdf1e04735517805377199d108ae2a45ead5d
  // same patch → same answer_digest_v2 (patch-only preimage)
```

### 5.9 Reference assertions (offline tests)

```text
assert score(F1) == Score(1_000_000)
assert score(F3) == Score(1_000_000)   // not 500_000
assert score(F5 Http200) == Score(1_000_000)
assert score(Timeout outcome) == NoScore(Timeout)
assert score(F7) == NoScore(AttestationNotVerified)
assert hex(task_id_v2(F1)) == pinned task_id_v2_hex
assert hex(answer_digest_v2(F1)) == pinned answer_digest_v2_hex
assert v1 echo answer does not validate as v2 correct
```

---

## 6. Key custody (challenge signing key)

| Rule | Requirement |
|------|-------------|
| Algorithm | sr25519 (same as BUNDLE_SPEC leaf sigs) |
| Public key | Committed in owner-signed `config/challenges.toml` for `agent-v1` |
| Secret key | **Never** in git. Stored `age`-encrypted offline; decrypted to a **mode 0600 file** on the challenge host |
| Runtime load | Challenge process reads secret from file path env `BASE_CHALLENGE_SK_FILE` |
| Gateway DB | **Must not** store challenge secrets or be consulted for leaf provenance (D18) |
| Signing domain | `base-rawweight-v1` over `RawWeightBodyV1` (BUNDLE_SPEC §3.4) |
| Rotation | D21 dual-accept window via trust-root release; keygen via `trustroot keygen` |
| Compromise | Rotate trust root; D19 still applies until rotation lands |
| Work-receipt key | Separate CVM-local key; **not** the challenge sk; published in measured compose |

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
| 7 | Challenge / verifier operator fault | `ChallengeInternal` (6) |
| 8 | Else scored | `Score(value)` per §5 |

### 7.4 Verifier-fault matrix (`map_verify_error`)

Operator-side Harbor faults must never silence a hotkey in `E` and must never be charged as miner solve failure. Park / `AttestationNotVerified` is D13 attestation-only — **not** reused for verifier outages.

| `VerifyError` | `ScoreOrAbsence` | Attribution |
|---------------|------------------|-------------|
| `Timeout` (verifier wall) | `NoScore(ChallengeInternal)` | operator |
| `Docker` (crash, pull, API) | `NoScore(ChallengeInternal)` | operator |
| `MalformedOutput` (junit/reward parse) | `NoScore(ChallengeInternal)` | operator |
| `Staging` | `NoScore(ChallengeInternal)` | operator |
| `MissingHeldOut` | `NoScore(ChallengeInternal)` | operator |
| `ApplyFailed` | `Score { value: 0 }` | miner |
| `RewardZero` | `Score { value: 0 }` | miner |
| `Reward(1)` | `Score { SCORE_MAX }` | miner solve |
| `Reward(0)` | `Score { 0 }` | miner solve |

Operator faults may retry at most `MAX_VERIFY_RETRIES = 2` (total attempts `1 + 2`) for retryable classes only; then emit `ChallengeInternal`. Never unbounded retries past seal.

**Forbidden:**

| Code | Name | Rule |
|------|------|------|
| 0 | `NotAttempted` | MUST NOT skip work for an `h ∈ E` under normal operation |
| 7 | `PolicySkip` | MUST NOT shrink `E` (BUNDLE_SPEC §3.3.1) |

### 7.5 Default trust-root policy

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
sig = sr25519_sign(challenge_sk, tag "base-rawweight-v1" ‖ scale(body))
leaf = LeafV1 { ... body fields ..., challenge_sig: sig }
POST /v1/weights/raw  with the gateway-accepted leaf envelope
```

Gateway verifies against **its** local trust root (defence in depth) and appends. Unknown challenge id → reject. Wrong key → reject.

---

## 9. Compose services, ports, image contract (tasks 37/38)

### 9.1 Miner CVM `app-compose.json` services (normative names)

| Service name | Role | Container port | Published |
|--------------|------|----------------|-----------|
| `agent` | HTTP runner (§4) | `8080` | Phala ingress → 8080 |
| `socket-proxy` | Allowlisted Docker Engine API for pack env containers | loopback inside CVM | **Not** public |
| `attest-helper` | Builds `report_data`, exports quote + event log for certify | `8081` | Phala ingress → 8081, `Authorization: Bearer <launch token>` required |

#### 9.1.1 Measured socket-proxy supersedes v1 "No docker.sock"

**scoring_version 1** stated: "No docker.sock. No `:latest` tags."

**scoring_version 2 supersedes the docker.sock ban for the miner CVM only**, with explicit reason:

- SWE Harbor packs require nested containers (environment image) on the miner host (Q1=A).  
- The runner reaches Docker **only** through a method-allowlisted `socket-proxy` declared in the CVM `app-compose`, so the proxy is covered by compose-hash and RTMR3 (measured path).  
- Raw `/var/run/docker.sock` MUST NOT be mounted into the long-lived `agent` container.  
- On the **operator** side, the host socket remains solely on the operator `socket-proxy` (read-only); no raw sock on `agent-challenge`.

**Still forbidden everywhere:** `:latest` tags.

### 9.2 Image contract

```text
images:
  agent:
    repository: ghcr.io/baseintelligence/base/base-agent
    // digest-pinned only, e.g. repo@sha256:<64 hex>
  attest-helper:
    repository: ghcr.io/baseintelligence/base/base-attest-helper
    // digest-pinned only
  socket-proxy:
    // digest-pinned allowlisted proxy image (miner CVM + operator)
```

| Rule | Requirement |
|------|-------------|
| Tags | Digest pins only; rendered compose must contain zero `:latest` |
| `allowed_envs` names | May include `BASE_NETUID`, `BASE_MINER_HOTKEY_FILE`, `BASE_LAUNCH_TOKEN_HASH` |
| Secret values | **Files** under `/run/base/` (or equivalent), never env values |
| `LAUNCH_TOKEN` | Hash appears in measured compose (D11); raw token only as file if needed |
| compose-hash | Canonical JSON → SHA-256 per `compose-hash`; must match RTMR3 event and `mr_config_id` prefix |
| Work-receipt pubkey | Published in measured compose for challenge verification |

### 9.3 `miner deploy` obligations

1. Render compose from the template that satisfies §9.1–§9.2 (including measured socket-proxy).  
2. Compute compose-hash **offline**; print it.  
3. `phala deploy` (or `--no-deploy` for hash-only).  
4. Miner funds their own Phala account (R3).  
5. Register public base URL with gateway routing for `agent-v1`.

### 9.4 `miner certify` obligations

1. Request fresh nonce from validator.  
2. Inside CVM, build `report_data` per §3.2.  
3. Submit quote + event log.  
4. Validator: parse → replay → compose-hash → policy → redeem nonce → `Verified`/`Rejected`/`Parked`.

### 9.5 Challenge service process (operator side, not miner CVM)

| Item | Contract |
|------|----------|
| Binary / image | `agent-challenge` digest-pinned |
| Listen port | `8090` (internal) |
| Secrets | `BASE_CHALLENGE_SK_FILE` mount 0600 |
| Trust root | Local `config/challenges.toml` path |
| Health | `/healthz`, `/readyz` on 8090 |
| Docker | Operator socket-proxy only (HarborVerifier) |

Gateway remains the only public TLS terminator (D20).

---

## 10. Challenge trait shape (implementers)

Informative surface (behaviour must match this doc):

```text
trait Challenge {
  fn challenge_id(&self) -> &str;                    // "agent-v1"
  fn scoring_version(&self) -> u16;                  // 2
  fn expected_set(&self, ctx: &EpochCtx) -> BTreeSet<Hotkey>;
  fn select_pack(&self, epoch, hotkey) -> PackId;
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

D19 applies unchanged (see [`THREAT_MODEL.md`](./THREAT_MODEL.md) §1 — byte-identical freeze):

- Valid leaf signatures prove the **challenge key** attested those scores, not that the scores are fair.  
- TEE attestation proves measured miner code and D10 binding, not challenge honesty.  
- Owner signs trust roots and operates the gateway: owner compromise remains residual risk (R12).  
- Work receipts and Harbor grades sit inside the same honesty boundary as the challenge operator.

---

## 12. Parent-plan clause F7 restatement (SWE challenge)

Parent plan `base-rust-subnet` final verification **F7** ("The original request, clause by clause") includes:

> Agent Challenge on Phala, miners self-deploying, validators verifying cryptographically → tasks 36, 37, 38, 47.

Under scoring_version **2**, that clause means:

| Original clause fragment | SWE / v2 reading |
|--------------------------|------------------|
| Agent Challenge on Phala | Miner CVM runs `base-agent` + measured socket-proxy + attest-helper; packs are Harbor SWE tasks from the pinned deepagent catalog |
| Miners self-deploying | `miner deploy` / install script renders digest-pinned compose; miner funds Phala |
| Validators verifying cryptographically | D10 quote path + BUNDLE_SPEC leaf/bundle verify; validators **do not** re-score `model.patch` (D19) |
| End-to-end proof | Parent task 47 criteria, with scoring_version 2 leaves and attestation precondition |

Any F7 sub-clause not yet demonstrated live remains **explicitly unmet** — no silent scope reduction. This document freezes the **contract**; live e2e is a later wave.

---

## 13. Verification checklist (implementers)

1. Topology matches §1 (no challenge sk in CVM; measured miner socket-proxy; no agent re-score in validator).  
2. Attestation precondition §3 enforced before any `Score`.  
3. v2 task/answer digests match §5.7 pinned fixtures.  
4. F1–F11 v2 table §5.8 green in offline tests.  
5. Every `h ∈ E` gets a signed leaf; no silence (D24).  
6. Verifier faults map per §7.4; never Park for Docker/junit outages.  
7. `PolicySkip` never used to drop coverage.  
8. Compose render matches §9; no `:latest`; measured socket-proxy; no secrets in `environment:`.  
9. Leaves verify under BUNDLE_SPEC `protocol_version = 1` and local trust root.  
10. Work receipt domain is `base-agent-work-receipt-v1`, distinct from attest.  
11. Doc states what v2 attestation does **not** prove (§3.4).  
12. Agent egress default is **OPEN** (§4.2.1); allowlisted proxy OFF by default; Metis B6 residual stated.  
13. Model key is a mounted file path only (Q3=A); never logged (§4.2.2).

---

## Appendix A. Cross-links

| Topic | Document |
|-------|----------|
| Leaf SCALE, `NoScoreReasonCode`, rawweight domain | BUNDLE_SPEC §3 |
| Aggregation of scores / `PARTICIPATION_FLOOR` | BUNDLE_SPEC §6 |
| Participant derivation | BUNDLE_SPEC §7 |
| D19 claim | BUNDLE_SPEC §11.1 / THREAT_MODEL §1 |
| Bundle `protocol_version` | BUNDLE_SPEC §2 (**value 1**) |
| D10/D11/D13/D18/D24 decisions | plan `base-rust-subnet` |
| SWE deepagent plan | plan `base-agent-challenge-deepagent` |

---

## Appendix B. Related plan decisions

| Decision | Sections here |
|----------|---------------|
| D10 liveness binding | §3.2, §9.4 |
| D11 env names only | §3.2, §9.2 |
| D13 park no credit | §3.1 |
| D18 local challenge keys | §6, §8 |
| D19 honest claim | §1, §3.4, §11 |
| D20 gateway TLS only | §1, §9.5 |
| D23/D24 participants + shares | §7 |
| D21 key rotation | §6 |
| R1 one pack / epoch deadline | §5.2 |
| R3 work receipt | §5.6 |
| Q1=A miner-side runner | §1, §9.1 |
| Q2=B scoring_version 2 | §2, freeze note |
| Q3=A miner model key file | §4.2.2 |
| Agent egress OPEN default | §4.2.1 |

---

## Appendix C. Domain tag registry (agent-v1)

| Tag | Purpose |
|-----|---------|
| `base-agent-task-id-v2` | v2 task id preimage |
| `base-agent-task-blob-v2` | v2 task blob preimage |
| `base-agent-answer-v2` | v2 answer over `model.patch` |
| `base-agent-pack-select-v1` | deterministic pack index |
| `base-agent-work-receipt-v1` | CVM work receipt signatures |
| `base-agent-dispatch-v1` | JSON dispatch protocol label |
| `base-attest-v1` | D10 report_data (unchanged) |
| `base-rawweight-v1` | leaf signatures (unchanged) |
| `base-agent-task-id-v1` / `…-blob-v1` / `…-answer-v1` | **retired** v1 echo (historical only) |

---

**End of frozen AGENT_CHALLENGE challenge_scoring_version=2 (bundle protocol_version=1). Re-freeze gate: task 16.**
