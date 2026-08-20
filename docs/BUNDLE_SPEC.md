# base Epoch Bundle Specification

**Status:** FROZEN (task 8 wave gate)  
**Normative for:** `bundle`, `aggregate`, gateway seal, validator verify/recompute  
**Encoding:** parity SCALE (`parity-scale-codec`), little-endian multi-byte integers  
**protocol_version of this document:** `1`

This file is the single source of truth for hashed and signed epoch-bundle bytes.
Where this document and any other source disagree, **this document wins**, with one
scoped exception: the aggregation formula in §6 is defined by behavioural parity with
the BASE Python master, so that `/v1/weights/latest` can replace
`https://chain.joinbase.ai/v1/weights/latest` for clients already parsing it. There,
the pinned upstream and its frozen vectors win. This supersedes D16, which had made the
Python float/JSON vectors characterization only.

Checklist map: [`BUNDLE_SPEC_CHECKLIST.md`](./BUNDLE_SPEC_CHECKLIST.md).  
CI gate: `cargo run -p xtask -- spec-check`.

---

## 0. Document conventions

| Notation | Meaning |
|----------|---------|
| `u8`/`u16`/`u32`/`u64`/`u128` | SCALE fixed-width little-endian unsigned integers |
| `[u8; N]` | Fixed-length byte array, encoded as N raw bytes |
| `Vec<T>` | SCALE compact length prefix, then elements |
| `scale(T)` | Canonical SCALE encoding of value `T` |
| `sha256(bytes)` | SHA-256 digest, 32 bytes |
| `‖` | Byte concatenation |
| `checked_*` | Overflow must return error; never wrap, never panic-as-success |
| bps | Basis points; full mass = `10_000` |

Hotkeys are Substrate account IDs: `[u8; 32]` (raw public key bytes), not SS58 strings, inside all SCALE structures.

Challenge identifiers are UTF-8 strings encoded as SCALE `Bytes` (`Vec<u8>`), max length `64` bytes. Implementations MUST reject longer ids before hashing or signing.

---

## 1. Encoding law (a)

**SCALE only for every byte sequence that is hashed, merkle-leafed, or signed.**

| Allowed | Forbidden |
|---------|-----------|
| `parity-scale-codec` Encode/Decode of the types in this spec | JSON, MessagePack, protobuf, bincode, custom ad-hoc layouts |
| Sorted `Vec<(K, V)>` for maps (key order = ascending `scale(K)` byte order) | `HashMap` / unsorted maps in consensus paths |
| Integer fields only in aggregation inputs/outputs | `f32` / `f64` in consensus paths |

JSON MAY appear only on human HTTP error bodies and operator logs. It MUST NOT appear in:

- leaf preimages
- merkle inputs
- bundle body bytes under signature
- dissent payloads under signature
- peer root statements under signature
- on-chain `WeightsTlockPayload` construction inputs beyond the four fields in §12

**Maps:** any logical map is encoded as `Vec<(K, V)>` sorted by ascending `scale(K)`. Duplicate keys are invalid.

---

## 2. Protocol version (b)

```text
protocol_version: u16
```

| Rule | Requirement |
|------|-------------|
| Current | `1` |
| Compatibility | A validator that does not implement the received `protocol_version` MUST reject the bundle (`DissentReasonCode::ProtocolVersionUnsupported`) and MUST NOT submit weights derived from it |
| Schema change | Any change to field set, field order, enum discriminants, hash domain tags, aggregation semantics, or merkle construction is a **major** bump of `protocol_version` |
| Patch docs | Editorial doc fixes that do not change bytes MAY keep the same version; the frozen byte contract does not move |

`algorithm_version` (aggregation) is independent of `protocol_version` but lives inside the bundle body. Changing aggregation math bumps `algorithm_version` and, if the bundle field layout changes, also `protocol_version`.

---

## 3. Merkle construction (RFC 6962) and leaves (c)

### 3.1 Tree rules

Implementations MUST match `merkle` and RFC 6962 Certificate Transparency:

| Node kind | Hash |
|-----------|------|
| Leaf | `SHA256(0x00 ‖ leaf_data)` |
| Internal | `SHA256(0x01 ‖ left_hash ‖ right_hash)` |
| Odd node at a level | **Promote** the single child hash unchanged. **Never** duplicate a node as its own sibling (blocks CVE-2012-2459). |

Canonical leaf **order** is the caller's responsibility (D7): sort leaf preimages by `scale(challenge_id, miner_hotkey)` before calling `root`.

### 3.2 Empty-tree root (pinned)

When the leaf set is empty, the merkle root is exactly this 32-byte value (RFC 6962 §2.1 `MTH({}) = SHA-256()` of the empty input; **not** `hash_leaf(&[])`):

```text
EMPTY_ROOT =
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

This constant is also `merkle::EMPTY_ROOT`. Divergence is a bug.

### 3.3 Leaf payload

Each merkle leaf preimage is:

```text
LeafV1 = scale(
  challenge_id:    Bytes,           // Vec<u8>, UTF-8 challenge id
  miner_hotkey:    [u8; 32],
  epoch:           u64,
  score_or_absence: ScoreOrAbsence,
  challenge_sig:   [u8; 64]         // sr25519 signature bytes
)
```

```text
ScoreOrAbsence (SCALE enum, u8 discriminant):
  0 = Score { value: u64 }
  1 = NoScore { reason: NoScoreReasonCode }   // reason: u8
```

#### 3.3.1 `NoScoreReasonCode` (u8)

| Code | Name | Meaning |
|------|------|---------|
| 0 | `NotAttempted` | Challenge did not invoke the miner this epoch |
| 1 | `Timeout` | Miner failed to respond in time |
| 2 | `InvalidResponse` | Response failed schema/scoring preconditions |
| 3 | `AttestationNotVerified` | Required attestation missing or not `Verified` this epoch |
| 4 | `MinerError` | Miner returned an explicit error |
| 5 | `RateLimited` | Challenge rate-limited the miner |
| 6 | `ChallengeInternal` | Challenge-side fault; still must cover the participant |
| 7 | `PolicySkip` | Reserved; MUST NOT be used to shrink the expected set (D24). Validators reject bundles that use this to omit coverage |
| 8–255 | _reserved_ | Reject unknown codes on verify |

Absence is **signed by the challenge key** over the same domain as scores (see §3.4). Silence is not absence.

### 3.4 Challenge signature

Domain-separated message (plan task 14 tags):

```text
msg = "base-rawweight-v1" as length-prefixed UTF-8 tag ‖ scale(RawWeightBodyV1)

RawWeightBodyV1 = scale(
  challenge_id:     Bytes,
  miner_hotkey:     [u8; 32],
  epoch:            u64,
  score_or_absence: ScoreOrAbsence
)
```

`challenge_sig` is sr25519 over `msg`, verifiable with the challenge public key from the **local** owner-signed trust root (`config/challenges.toml`), never from gateway HTTP (D18).

### 3.5 Sort key before tree build

```text
sort_key(leaf) = scale(challenge_id, miner_hotkey)
```

Leaves MUST be sorted by ascending `sort_key` byte order before `merkle_root = root(leaf_preimages)`.  
Stable sort; duplicate `(challenge_id, miner_hotkey)` pairs are invalid.

---

## 4. Bundle body and block pin (d)

### 4.1 `EpochBundleV1` body (unsigned structural type)

Field order is normative. SCALE encode in this order only:

```text
EpochBundleBodyV1 = scale(
  protocol_version:     u16,                 // must be 1 for this doc
  epoch:                u64,
  netuid:               u16,
  block_B:              u64,                 // epoch_end_block (inclusive end of epoch window)
  block_hash:           [u8; 32],            // hash of block_B
  metagraph_root:       [u8; 32],            // §4.3
  algorithm_version:    u16,                 // aggregation; 1 = §6
  emission_shares:      Vec<(Bytes /*challenge_id*/, u16 /*bps*/)>,  // sorted by challenge_id
  measurements_digest:  [u8; 32],            // sha256 of local measurements trust root body
  uid_map:              Vec<([u8; 32] /*hotkey*/, u16 /*uid*/)>,     // sorted by hotkey
  leaves:               Vec<LeafV1>,         // sorted by scale(challenge_id, miner_hotkey)
  merkle_root:          [u8; 32],            // recomputed over leaf preimages
  final_vector:         Vec<(u16 /*uid*/, u16 /*weight*/)>,          // sorted by uid
  gateway_hotkey:       [u8; 32]
)
```

Signed envelope:

```text
EpochBundleV1 = scale(
  body: EpochBundleBodyV1,
  gateway_sig: [u8; 64]   // sr25519 over tag "base-bundle-v1" ‖ scale(body)
)
```

### 4.2 `block_B` and `block_hash`

| Field | Definition |
|-------|------------|
| `block_B` | `epoch_end_block`: the last block number belonging to `epoch` under the subnet tempo/epoch schedule used by the chain client |
| `block_hash` | Block hash of `block_B` as returned by the chain (`block_hash(block_B)`). Pinning **hash**, not only height, removes intra-block ambiguity |
| Consistency | `chain.block_hash(block_B) == bundle.block_hash` or reject |

### 4.3 `metagraph_root` (D7)

```text
MetagraphRow = (hotkey: [u8; 32], uid: u16, stake: u64)
rows = metagraph_at(block_hash) projected to MetagraphRow
rows sorted by ascending hotkey bytes
metagraph_root = sha256( scale(rows as Vec<MetagraphRow>) )
```

`metagraph_at` MUST be queried at `block_hash`, never at a bare block number alone.

`uid_map` in the bundle MUST equal `{(hotkey, uid)}` from those rows, sorted by hotkey. Stake is in the metagraph root preimage but not repeated in `uid_map`.

---

## 5. Emission shares from owner-signed trust root (f)

| Rule | Requirement |
|------|-------------|
| Source of truth | Owner-signed `config/challenges.toml` (trust root), loaded from **local disk** on every validator and on the gateway (D23, D18) |
| Bundle role | Gateway **copies** shares into `emission_shares`; it does not invent them |
| Sum | `sum(bps) == 10_000` exactly, or reject |
| Sort | `emission_shares` sorted by ascending `scale(challenge_id)` |
| Validator check | Re-read local trust root; require byte-equal challenge_id set and equal bps per id. Mismatch → reject (`EmissionShareMismatch`) |
| Unknown challenge | Leaf `challenge_id` absent from local trust root → reject (D18) |

Share values are `u16` basis points. No floats.

---

## 6. Aggregation formula, algorithm_version = 1 (e)

Pure function. No I/O.

**Authority.** This section specifies the algorithm served by the BASE master at
`https://chain.joinbase.ai/v1/weights/latest`. The normative reference is
`base.master.aggregator.aggregate_challenge_weights` at
`8249563774ee2e71c41ae2cfac182ff32aa35dd1`. The Rust implementation
(`aggregate::aggregate_python`) is a bit-for-bit port and MUST stay one:
`crates/aggregate/tests/differential.rs` compares both against a live interpreter,
and the frozen vectors under
`crates/aggregate/tests/vectors/python/8249563774ee2e71c41ae2cfac182ff32aa35dd1/`
pin the result. Where implementation and this prose disagree, the frozen vectors win.

This algorithm is specified in IEEE-754 binary64. That is a deliberate exception to
D8 for this section only; every other consensus surface (merkle, encoding, payload)
stays integer. `xtask/consensus-crates.txt` records the exception as a per-token
waiver so `HashMap` and `wrapping_*` remain banned in `aggregate`.

### 6.1 Inputs

```text
VerifiedLeaf {
  challenge_id: Bytes,
  miner_hotkey: [u8; 32],
  score_or_absence: ScoreOrAbsence,  // Score(u64) or NoScore(_)
}

shares: Vec<(challenge_id, bps: u16)>   // sum bps = 10000
uid_map: Vec<(hotkey, uid: u16)>
algorithm_version: u16                  // must be 1
```

Only leaves that already passed signature and participant-set checks enter aggregation.

### 6.2 Constants

```text
CHAIN_U16_MAX:       u16  = 65_535
ZERO_MINER_BURN_UID: u16  = 0
EPS:                 f64  = 1e-12
BPS_DENOM:           u128 = 10_000
min_allowed_weights: u32  = 1
max_weight_limit:    u16  = 65_535
algorithm_version:   u16  = 1
```

`min_allowed_weights` and `max_weight_limit` are pinned constants, not chain reads.
The deployed master constructs `AggregationService(session_factory, freshness_seconds=…)`
without passing either, so it runs on the `aggregate_challenge_weights` defaults.
Pinning them keeps recompute a pure function of the signed body.

### 6.3 Float determinism (normative)

Seal and independent recompute MUST produce bit-identical `f64`. Three rules make
that hold, and each has a regression test:

1. **Summation.** CPython's builtin `sum()` over floats is Neumaier-compensated, not
   a naive fold. `sum([0.1] * 10)` is exactly `1.0`; a fold gives `0.9999999999999999`.
   Implementations MUST reproduce `builtin_sum_impl`, including the trailing
   `if c != 0.0 { result += c }` correction. This makes the algorithm sensitive to the
   interpreter version; the port targets CPython 3.12.
2. **Iteration order.** Python `dict` preserves insertion order and float summation is
   order-dependent. The Rust adapter re-derives a canonical order inside the aggregation
   (challenges ascending by `challenge_id`, hotkeys ascending by key bytes) so no caller
   can move a ulp by reordering its input.
3. **Rounding.** `round()` is half-to-even. `0.5 * 65535 = 32767.5 → 32768` and
   `0.3 * 65535 = 19660.5 → 19660`. Half-away-from-zero is wrong and gives `19661`.

`max(x, 0.0)` follows Python, not Rust: `max(NaN, 0.0)` is `NaN`.

### 6.4 Per-challenge normalization

`ok_results` = challenges whose share is present. For each, the absolute fraction is

```text
frac[slug] = max(emission_percent, 0.0) / 100.0      // bps / 100.0
```

`emission_percent` is an **absolute** share of 100, not a share renormalized across
challenges: a challenge with `emission_percent = e` owns `e / 100` of the whole vector.
If `alloc_total = sum(frac) > 1.0`, every `frac` is divided by `alloc_total` so the
total is exactly `1.0`. Under-allocation is **not** scaled up; the slack burns.

Within a challenge, miner weights are cleaned then normalized to sum 1.0:

```text
clean:     drop non-finite, drop weight <= 0
normalize: w / sum(clean)            // {} when sum <= 0
```

`Score { value: v }` contributes `v as f64`; `NoScore` contributes nothing.

Each miner accumulates `frac[slug] * normalized_weight`, in first-appearance order.
Challenges with `share <= 0.0` are skipped; a `NaN` share is **not** skipped, because
NaN comparisons are false.

### 6.5 Mapping to uids, and burn

Hotkey scores map through `uid_map`. A hotkey that is **absent** from `uid_map`, or that
maps to `ZERO_MINER_BURN_UID`, is dropped and its mass burns. This is a deliberate
divergence from the integer path, which raised `MissingUidMapEntry`.

```text
miner_total = sum(uid_scores)
burn        = 1.0 - miner_total
if burn > EPS: uid_scores[0] += burn        // appended last
normalized  = uid_scores / sum(uid_scores)
```

### 6.6 Zero-miner burn

If `miner_total <= EPS` (no real miner scored), the vector is built by
`build_zero_miner_weights`:

```text
max_fraction = min(max_weight_limit / 65535, 1.0)
required     = max(min_allowed_weights, 1, ceil(1.0 / max_fraction))
candidates   = [burn_uid] + sorted(set(uid_map values) - {burn_uid})
if candidates.len() < required -> ZeroMinerWeightError (never submit)
weight       = 1.0 / required, assigned to candidates[..required]
```

With the pinned constants this is `{0: 1.0}`: a full burn to the subnet owner.
`hotkey_weights` is empty on this branch.

There is no "empty vector, no-submit" case: an epoch with no scoring miner burns.

### 6.7 Output vector

```text
uids:          Vec<u16>              // ascending
weights:       Vec<f64>              // sums to 1.0
hotkey_weights: [(hotkey, f64)]      // kept miners only, first-appearance order
final_vector:  Vec<(uid, u16)>       // round_ties_even(weight * 65535)
```

`sum(final_vector.weight)` is **not** guaranteed to equal `65_535`. Python performs no
post-rounding renormalization, so roughly 30% of vectors sum to `65_536` or `65_534`.
Implementations MUST NOT add a correction step: it would break parity. No consensus
invariant, dissent check, or chain submission path depends on the sum.

`hotkey_weights` excludes the burn uid, per `burn-uid0.v1`.

### 6.8 Worked numeric example

One challenge at `emission_percent = 100`, three miners, `uid_map` mapping each to a
non-zero uid.

| Miner | UID | raw |
|-------|-----|-----|
| A | 0 | 50 |
| B | 1 | 30 |
| C | 2 | 20 |

A sits on the burn uid, so its mass burns. B and C normalize to `0.3` and `0.2` of the
whole, and the remaining `0.5` burns to uid 0:

```text
uids         = [0, 1, 2]
weights      = [0.5, 0.3, 0.2]
final_vector = [(0, 32768), (1, 19660), (2, 13107)]   // sum = 65535
```

Note `0.3 * 65535 = 19660.5` rounds **down** to `19660` under half-to-even, while
`0.5 * 65535 = 32767.5` rounds **up** to `32768`.

### 6.9 Quarantine re-aggregation (D6)

1. Drop quarantined `challenge_id`s from both leaves and shares.
2. Let `surv = sum(remaining bps)`. If `surv == 0` → error (escalate class B).
3. If `surv < min_share_mass_bps` → caller escalates class B (no submit).
4. Else re-apportion the remaining mass to sum exactly `10_000` by Hamilton
   largest-remainder on the remaining share weights, house `10_000`, tie-break by
   ascending `scale(challenge_id)`. This apportions **shares**, which are integers;
   it is unaffected by §6.3.

Then re-run §6.4–§6.7 on the surviving leaves. The quarantine submission vector uses
the same algorithm as a normal epoch, so the two are never built differently.

Default `min_share_mass_bps = 5000` (config; D6).

### 6.10 Failure policy

`ZeroMinerWeightError` (not enough candidate uids to satisfy `min_allowed_weights`)
→ class B, dissent `EmptyScoreVectorNoSubmit`. Malformed input (bad shares sum,
duplicate uid_map entry, wrong `algorithm_version`) → class B, dissent
`AggregationOverflow`. Never submit a vector the chain would reject.

---

## 7. Expected participant set derivation (g) (D24)

Validators **derive** the expected set. They MUST NOT trust a set announced only by the gateway or challenge HTTP API.

### 7.1 Trust-root policy per challenge

In owner-signed `challenges.toml`, each challenge carries:

```text
ParticipantPolicy (SCALE enum for signed body; TOML maps 1:1):
  0 = AllMetagraphHotkeys
      // every hotkey in metagraph_at(block_hash)
  1 = StakeAtLeast { min_stake: u64 }
      // stake >= min_stake at metagraph_at(block_hash)
  2 = ExplicitAllowlist { hotkeys: Vec<[u8;32]> }
      // sorted unique hotkeys; intersection with metagraph
  3 = AllExceptDenyList { hotkeys: Vec<[u8;32]> }
      // metagraph hotkeys minus deny list
```

### 7.2 Derivation algorithm (normative)

```text
function expected_participants(challenge_id, policy, block_hash, chain) -> BTreeSet<[u8;32]>:
  rows = chain.metagraph_at(block_hash)   // hotkey, uid, stake, ...
  meta_keys = set(rows.hotkey)

  match policy:
    AllMetagraphHotkeys:
      S = meta_keys
    StakeAtLeast(min_stake):
      S = { h | row.hotkey = h and row.stake >= min_stake }
    ExplicitAllowlist(list):
      A = set(list)
      S = A ∩ meta_keys
      // hotkeys in allowlist but not on metagraph are ignored (not expected)
    AllExceptDenyList(deny):
      S = meta_keys \ set(deny)

  return S sorted by hotkey bytes
```

### 7.3 Completeness rule

Let `E_c = expected_participants(c, ...)` for each challenge `c` in the local trust root with `bps > 0`.

For every `c` and every `h ∈ E_c`, the bundle `leaves` MUST contain **exactly one** leaf with `(challenge_id=c, miner_hotkey=h)` whose `ScoreOrAbsence` is either `Score` or `NoScore`.

Reject if:

| Failure | Notes |
|---------|-------|
| Missing leaf for any `(c,h)` in some `E_c` | Censorship / omission |
| Bundle's implied set is a **proper subset** of `E_c` | D24 explicit |
| Extra leaf for unknown challenge id | D18 |
| Extra leaf for hotkey not in `E_c` (bps > 0) | Reject (strict coverage) |
| Leaf for a challenge with `bps == 0` | Ignored — not in any `E_c`; seal strips before merkle |
| Duplicate `(c,h)` | Reject |

Gateway seal MUST fail closed rather than publish an incomplete bundle.

---

## 8. Final vector comparison (h)

```text
final_vector: Vec<(u16 /*uid*/, u16 /*weight|)>  // sorted by uid ascending
```

Two vectors `V_a`, `V_b` match if and only if **both**:

1. **Full equality:** same length and pairwise equal `(uid, weight)` at every index.  
2. **Digest equality:**  
   `sha256(scale(V_a)) == sha256(scale(V_b))`  
   where `scale` is the canonical SCALE encoding of `Vec<(u16,u16)>`.

Validators compare:

- gateway `body.final_vector` vs local `aggregate(...)` output using the dual check above;
- `expected_vector_hash = sha256(scale(local_vector))` in dissent messages.

---

## 9. Distribution and caching (i)

| Endpoint | Behavior |
|----------|----------|
| `GET /v1/bundle/{epoch}` | Returns the sealed `EpochBundleV1` SCALE bytes (content-type `application/octet-stream`) or 404 |
| `GET /v1/bundle/root/{root}` | Lookup by `merkle_root` hex (64 lowercase hex chars); returns same body or 404 |
| `GET /v1/weights/latest` | Sealed projection of the **newest revision** of the highest chain-scale sealed epoch (`sealed: true`, with `revision` / `vector_digest`). If no sealed bundle is available or the stored bytes cannot be decoded, MUST return **200** with the fail-closed **burn vector** under `burn-uid0.v1` (`uids: [0]`, `weights: [1.0]`, `sealed: false`) — never 404. Validators MUST NOT treat the burn vector itself as a Match / submit path. When latest is `sealed: false`, a validator MAY independently re-compare and submit a previously verified sealed bundle persisted on disk. When latest is `sealed: true`, fetch/verify failure MUST NOT use that file |
| Tip reseal | Within the live tip epoch, when challenge leaves tip-supersede (`POST /v1/weights/raw` with a changed `payload_digest` for the same `(challenge_id, epoch, miner_hotkey)`), the gateway MAY append a new `epoch_bundle.revision` whose merkle/vector reflect the updated leaves. Re-seal with unchanged merkle root and `final_vector` is a no-op (returns the existing seal). Older epochs are not rewritten by the tip sealer walk |
| Validator mirroring | Validators MAY re-serve a bundle they have verified and persisted; peers SHOULD prefer multi-source fetch |
| **No last-known-good** | MUST NOT fall back to a previous epoch's bundle, root, or vector when the current epoch fetch/verify fails. Failure → class B / degraded path, not stale success (aligns with D13 spirit for attestation). The unsealed burn response on `/v1/weights/latest` is an operator-safety default, not a last-known-good seal; a validator-local copy of a previously **verified** seal may still be used for set-weights while latest is burn (master restart with an empty seal store). Mid-epoch tip revision bumps are intentional tip-tracking, not last-known-good |

Gateway signature and leaf signatures are always verified against local trust roots after fetch.

---

## 10. Dissent (j)

```text
DissentBodyV1 = scale(
  protocol_version:      u16,
  epoch:                 u64,
  bundle_root:           [u8; 32],   // merkle_root of the disputed bundle, or 0x00.. if none
  expected_vector_hash:  [u8; 32],   // sha256(scale(local final_vector))
  actual_vector_hash:    [u8; 32],   // sha256(scale(gateway final_vector)) or 0x00.. if absent
  reason_code:           DissentReasonCode  // u8
)

DissentV1 = scale(
  body: DissentBodyV1,
  validator_hotkey: [u8; 32],
  signature: [u8; 64]   // sr25519 over tag "base-dissent-v1" ‖ scale(body)
)
```

### 10.1 `DissentReasonCode` (u8), fully enumerated

| Code | Name | Typical class |
|------|------|----------------|
| 0 | `VectorMismatch` | A — inputs ok, peer roots ok, gateway vector ≠ recomputation |
| 1 | `LeafSignatureInvalid` | B |
| 2 | `LeafChallengeKeyUnknown` | B (D18) |
| 3 | `IncompleteParticipantSet` | B (D24) |
| 4 | `MerkleRootMismatch` | B |
| 5 | `EmissionShareMismatch` | B (D23) |
| 6 | `MetagraphRootMismatch` | B |
| 7 | `BlockHashMismatch` | B |
| 8 | `ProtocolVersionUnsupported` | B |
| 9 | `PeerRootConflict` | B |
| 10 | `PeerSampleInsufficient` | Degraded / no submit (D26) |
| 11 | `ShareMassBelowThreshold` | B after quarantine |
| 12 | `BundleSignatureInvalid` | B |
| 13 | `AggregationOverflow` | B |
| 14 | `EmptyScoreVectorNoSubmit` | no-submit |
| 15 | `UidMapMismatch` | B |
| 16 | `MeasurementsDigestMismatch` | B |
| 17 | `DuplicateLeaf` | B |
| 18 | `QuarantineExhausted` | B |
| 19–255 | _reserved_ | treat as unknown; still persist raw dissent bytes |

---

## 11. Security claim, quarantine, peer sample (k)

### 11.1 D19 claim (verbatim)

The following paragraph is **normative wording** for docs and threat models. Do not weaken or inflate:

> base guarantees *no equivocation between validators* and *no undetected deviation by the gateway from the owner-signed challenge and measurement artifacts*. It does **not** guarantee (i) that a challenge's scores are honest, (ii) that the owner is honest — the owner signs the trust roots and runs the gateway, so a malicious owner can authorize a dishonest challenge or a backdoored measurement, (iii) completeness beyond what D24 provides, nor (iv) **chain-anchored, third-party-auditable non-equivocation** — per D5 the property is peer-consensus plus local evidence, verifiable by the participating validators and not by an outside observer after the fact.

### 11.2 Mismatch outcomes (D6)

| Class | Condition | Action |
|-------|-----------|--------|
| **A** | Inputs verify; peer roots agree; gateway `final_vector` ≠ local recompute | Submit **local** vector + `DissentV1{ reason: VectorMismatch }` |
| **Quarantine** | One or more challenges' leaves unverifiable/absent, and surviving emission mass ≥ `min_share_mass_bps` | Drop bad challenges, `renormalize_after_quarantine`, aggregate, submit; metric `base_challenge_quarantined_total` |
| **B** | Inputs unverifiable; peer roots conflict; surviving mass `< min_share_mass_bps`; or other hard failures | **No** weight submission; signed dissent; alarm |

Default `min_share_mass_bps = 5000` (half of `10_000`).

### 11.3 Peer sample (D26)

| Rule | Requirement |
|------|-------------|
| `min_peer_sample` | Default `1`. May be `0` only when the metagraph contains no other validator with `validator_permit` (single-validator testnet) |
| Below threshold | Do not submit; status `Degraded`; dissent `PeerSampleInsufficient` |
| Identity | Peer responses authenticated by **sr25519 over response body** bound to metagraph hotkey — never IP allowlists alone |
| Root exchange | `GET`-style peer API returns signed `(epoch, merkle_root)` under tag `base-root-v1` |

---

## 12. On-chain weight payload: no merkle root (l) (D5)

**The merkle root is NOT in the on-chain weight payload.**

`WeightsTlockPayload` is frozen by the runtime to exactly:

```text
WeightsTlockPayload = {
  hotkey:      AccountId,      // [u8; 32]
  uids:        Vec<u16>,
  values:      Vec<u16>,       // parallel to uids
  version_key: u64
}
```

There is **no** field for a 256-bit merkle root. `version_key` is 64 bits and MUST NOT be overloaded as a root prefix.

Non-equivocation rests on:

1. In-epoch signed peer root exchange (hotkey-authenticated HTTPS).  
2. Durable local persistence of the signed bundle and peer root statements.  
3. Optional commitments-pallet announcement **only if** metadata snapshot proves the pallet exists — not required by this spec.

Any code or doc that claims the merkle root is committed inside `WeightsTlockPayload` is wrong and MUST be corrected.

---

## 13. Gateway bundle signature

```text
msg = tag "base-bundle-v1" ‖ scale(EpochBundleBodyV1)
gateway_sig = sr25519_sign(gateway_hotkey_sk, msg)
```

`gateway_hotkey` MUST equal on-chain `SubnetOwnerHotkey` for master-only gateway operation (D3); validators still verify the signature cryptographically and MAY additionally check owner equality.

---

## 14. Verification checklist (implementers)

A `verify(bundle, chain, local_trust_root)` implementation MUST check, in order:

1. `protocol_version` supported  
2. `gateway_sig` over body  
3. `block_hash` matches `chain.block_hash(block_B)`  
4. `metagraph_root` and `uid_map` match `metagraph_at(block_hash)`  
5. `emission_shares` equal local trust root and sum to `10_000`  
6. `measurements_digest` equals local measurements digest  
7. Every leaf challenge key known locally; every `challenge_sig` valid  
8. Participant-set completeness (§7)  
9. Leaf sort order canonical; `merkle_root` recomputes  
10. `algorithm_version == 1` and `final_vector` equals `aggregate(...)` under §8 dual equality  
11. No duplicate leaves  

Failure modes map to §10.1 reason codes.

---

## 15. Byte-stability requirements

- Re-encoding a decoded bundle MUST yield identical bytes.  
- Golden vectors in `bundle` / `aggregate` tests pin this document's field order.  
- A doc test in `bundle` (task 19) MUST fail if SCALE field order drifts from §4.1.

---

## Appendix A. Domain tags (length-prefixed UTF-8)

| Tag string | Used for |
|------------|----------|
| `base-bundle-v1` | Epoch bundle body |
| `base-rawweight-v1` | Challenge leaf body |
| `base-dissent-v1` | Dissent body |
| `base-root-v1` | Peer `(epoch, merkle_root)` |
| `base-trustroot-v1` | Owner trust-root body |
| `base-attest-v1` | Attestation bindings (see AGENT_CHALLENGE / D10; out of scope for leaf math) |

Length-prefix rule: SCALE `Bytes` encoding of the tag string UTF-8 bytes (compact length + bytes), then payload. Task 14 owns the exact sign helper; this appendix names the tags only.

---

## Appendix B. Related decisions

| Decision | Spec sections |
|----------|---------------|
| D4 verifiable aggregation | §4, §6, §14 |
| D5 no on-chain merkle | §12 (l) |
| D6 quarantine / classes | §6.9, §11.2 |
| D7 SCALE + metagraph_root | §1, §4.3 |
| D8 integer / Hamilton / checked | §6 |
| D9 RFC6962 + EMPTY_ROOT | §3 |
| D18 local challenge keys | §3.4, §5, §14 |
| D19 honest claim | §11.1 |
| D23 emission shares in trust root | §5 |
| D24 absence + derived set | §3.3, §7 |
| D26 peer sample | §11.3 |

---

**End of frozen BUNDLE_SPEC protocol_version=1.**
