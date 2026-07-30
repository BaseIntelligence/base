# gbase Epoch Bundle Specification

**Status:** FROZEN (task 8 wave gate)  
**Normative for:** `bundle`, `aggregate`, gateway seal, validator verify/recompute  
**Encoding:** parity SCALE (`parity-scale-codec`), little-endian multi-byte integers  
**protocol_version of this document:** `1`

This file is the single source of truth for hashed and signed epoch-bundle bytes.
Where this document and any other source disagree, **this document wins**.
Python BASE float/JSON vectors are characterization only (plan D16).

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
msg = "gbase-rawweight-v1" as length-prefixed UTF-8 tag ‖ scale(RawWeightBodyV1)

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
  gateway_sig: [u8; 64]   // sr25519 over tag "gbase-bundle-v1" ‖ scale(body)
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

Pure function. No I/O. No floats. Accumulators are `u128` with **`checked_*` only**.  
Bare `+`/`*`/`-` on accumulators and all `wrapping_*` are forbidden in consensus crates (D8).

### 6.1 Inputs

```text
VerifiedLeaf {
  challenge_id: Bytes,
  miner_hotkey: [u8; 32],
  score_or_absence: ScoreOrAbsence,  // Score(u64) or NoScore(_)
}

shares: Vec<(challenge_id, bps: u16)>   // sum bps = 10000
uid_map: Vec<(hotkey, uid: u16)>        // bijective for all hotkeys appearing in leaves
algorithm_version: u16                  // must be 1
```

Only leaves that already passed signature and participant-set checks enter aggregation.

### 6.2 Constants

```text
BPS_DENOM:          u128 = 10_000
HOUSE:              u16  = 65_535        // u16::MAX
FIXED:              u128 = 1_000_000_000_000   // 10^12 fixed-point scale for per-challenge norms
algorithm_version:  u16  = 1
```

### 6.3 Per-challenge normalization

For each challenge `c` with share `s_c > 0`:

1. Let `L_c` be all verified leaves with `challenge_id == c`.
2. For each leaf, define raw score:
   - `Score { value: v }` → `raw = v` as `u128`
   - `NoScore { .. }` → `raw = 0`
3. `sum_c = checked_sum(raw over L_c)`. If overflow → hard error (do not submit).
4. If `sum_c == 0`: every miner in `L_c` contributes `0` from this challenge (skip weight add).
5. Else, for each miner `m` in `L_c` with raw `r_m`:

```text
// fixed-point fraction of challenge mass, then weight by share bps
// norm_m = floor(r_m * FIXED / sum_c)          // in 0..FIXED
// term_m = floor(norm_m * s_c / BPS_DENOM)     // share-weighted
//
// Combined without separate storage (checked):
term_m = r_m
  .checked_mul(FIXED)?
  .checked_mul(u128::from(s_c))?
  .checked_div(sum_c)?
  .checked_div(BPS_DENOM)?
acc[m] = acc[m].checked_add(term_m)?
```

Miners that do not appear under challenge `c` get no term from `c` (they must still have been covered by completeness before aggregation; see §7).

Challenges with `s_c == 0` are ignored (should not appear if shares are well-formed).

### 6.4 Cross-challenge combine

After all challenges:

```text
total = checked_sum(acc[m] for all m in acc)
```

### 6.5 All-zero → all-zero, no-submit

If `total == 0` OR every `acc[m] == 0`:

- Output weight vector is empty **or** all pairs `(uid, 0)` for uid_map (implementations MUST pick **empty `Vec`** as the canonical all-zero vector).
- Classification: **no-submit** condition (not a division-by-zero). Validators/gateway MUST NOT call `commit_timelocked_*` with a fabricated uniform vector.
- Dissent MAY use `DissentReasonCode::EmptyScoreVectorNoSubmit` when this blocks submission after an otherwise valid bundle path.

### 6.6 Hamilton largest-remainder apportionment (house = 65_535)

When `total > 0`:

For each miner `m` with `acc_m` and `uid_m` from `uid_map`:

```text
// exact quota in u128
prod_m     = acc_m.checked_mul(u128::from(HOUSE))?   // HOUSE = 65535
floor_m    = prod_m / total                           // integer division
remainder_m = prod_m % total
```

Let `sum_floors = checked_sum(floor_m)`.  
Overflow of any `floor_m` above `u16::MAX` as a single seat count is impossible because `sum_floors <= HOUSE` when math is correct; if `floor_m > u128::from(u16::MAX)` → hard error.

```text
seats_left = u128::from(HOUSE).checked_sub(sum_floors)?   // leftover seats to assign
```

Assign one extra seat to the `seats_left` miners with the **largest** `remainder_m`.  
**Tie-break:** lower `uid` wins (ascending UID).  
If remainders and UIDs still tie (identical uid is impossible), lower hotkey bytes win.

```text
weight[uid_m] = u16::try_from(floor_m + extra_m)?
```

### 6.7 Output vector

```text
final_vector: Vec<(uid: u16, weight: u16)>
```

- Include only UIDs with `weight > 0` **or** include all UIDs from `uid_map` with zeros?  
  **Canonical rule:** include **every** UID present in `uid_map` that appears as a miner hotkey in at least one leaf, with the computed weight (possibly zero only if that miner had zero acc but others had positive total — actually if total > 0 and acc_m = 0, weight is 0).  
  **Tighter canonical rule used by gbase:** emit **all** `(uid, weight)` for every uid in `uid_map` whose hotkey appears in the verified leaf set, sorted by ascending `uid`. Zero weights are kept so vector equality is stable.  
  Exception: the all-zero/no-submit case uses empty `Vec` (§6.5).

- Sorted by ascending `uid`.
- `sum(weight) == HOUSE` (65_535) whenever `total > 0`.

### 6.8 Worked numeric example (Hamilton)

Three miners, one challenge, shares = 10_000 bps on that challenge.

| Miner | UID | raw score |
|-------|-----|-----------|
| A | 1 | 50 |
| B | 2 | 30 |
| C | 3 | 20 |

`sum_c = 100`, `s_c = 10000`, `FIXED = 10^12`.

```text
term_A = 50 * 10^12 * 10000 / 100 / 10000 = 50 * 10^12 / 100 = 5 * 10^11
term_B = 30 * 10^12 / 100 = 3 * 10^11
term_C = 20 * 10^12 / 100 = 2 * 10^11
total  = 10^12
```

Quotas with `HOUSE = 65535`:

```text
prod_A = 5e11 * 65535 = 32767500000000000 ; floor_A = 32767 ; rem_A = 5000000000000
prod_B = 3e11 * 65535 = 19660500000000000 ; floor_B = 19660 ; rem_B = 5000000000000
prod_C = 2e11 * 65535 = 13107000000000000 ; floor_C = 13107 ; rem_C = 0
sum_floors = 32767 + 19660 + 13107 = 65534
seats_left = 1
```

Remainders: A and B tie at `5e12`; **ascending UID** → A (uid 1) gets the extra seat.

```text
weights = [(1, 32768), (2, 19660), (3, 13107)]
sum = 65535
```

### 6.9 Quarantine renormalization (D6 helper)

When challenges `D` are dropped:

```text
renormalize_after_quarantine(shares, dropped_ids) -> Result<Vec<(id, bps)>>
```

1. Remove dropped ids from shares.  
2. Let `surv = sum(remaining bps)`. If `surv == 0` → error (escalate class B).  
3. If `surv < min_share_mass_bps` → caller escalates class B (no submit).  
4. Else re-apportion remaining mass to sum exactly `10_000` by the same Hamilton rule on the remaining share weights as "scores", house `10_000`, tie-break by ascending `scale(challenge_id)`:

```text
// treat each remaining bps as acc; HOUSE_SHARES = 10000
// identical largest-remainder as §6.6 with uid replaced by challenge_id sort key
```

Then re-run §6.3–§6.7 on surviving leaves only.

Default `min_share_mass_bps = 5000` (config; D6).

### 6.10 Overflow policy

Any `checked_*` failure → aggregation returns `Err(Overflow)`.  
Callers MUST treat overflow as class B (no submit + dissent `AggregationOverflow`).  
Debug panics on overflow are not an acceptable substitute: release builds must not wrap.

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
| Extra leaf for hotkey not in `E_c` | Reject (strict coverage) |
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
| Validator mirroring | Validators MAY re-serve a bundle they have verified and persisted; peers SHOULD prefer multi-source fetch |
| **No last-known-good** | MUST NOT fall back to a previous epoch's bundle, root, or vector when the current epoch fetch/verify fails. Failure → class B / degraded path, not stale success (aligns with D13 spirit for attestation) |

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
  signature: [u8; 64]   // sr25519 over tag "gbase-dissent-v1" ‖ scale(body)
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

> gbase guarantees *no equivocation between validators* and *no undetected deviation by the gateway from the owner-signed challenge and measurement artifacts*. It does **not** guarantee (i) that a challenge's scores are honest, (ii) that the owner is honest — the owner signs the trust roots and runs the gateway, so a malicious owner can authorize a dishonest challenge or a backdoored measurement, (iii) completeness beyond what D24 provides, nor (iv) **chain-anchored, third-party-auditable non-equivocation** — per D5 the property is peer-consensus plus local evidence, verifiable by the participating validators and not by an outside observer after the fact.

### 11.2 Mismatch outcomes (D6)

| Class | Condition | Action |
|-------|-----------|--------|
| **A** | Inputs verify; peer roots agree; gateway `final_vector` ≠ local recompute | Submit **local** vector + `DissentV1{ reason: VectorMismatch }` |
| **Quarantine** | One or more challenges' leaves unverifiable/absent, and surviving emission mass ≥ `min_share_mass_bps` | Drop bad challenges, `renormalize_after_quarantine`, aggregate, submit; metric `gbase_challenge_quarantined_total` |
| **B** | Inputs unverifiable; peer roots conflict; surviving mass `< min_share_mass_bps`; or other hard failures | **No** weight submission; signed dissent; alarm |

Default `min_share_mass_bps = 5000` (half of `10_000`).

### 11.3 Peer sample (D26)

| Rule | Requirement |
|------|-------------|
| `min_peer_sample` | Default `1`. May be `0` only when the metagraph contains no other validator with `validator_permit` (single-validator testnet) |
| Below threshold | Do not submit; status `Degraded`; dissent `PeerSampleInsufficient` |
| Identity | Peer responses authenticated by **sr25519 over response body** bound to metagraph hotkey — never IP allowlists alone |
| Root exchange | `GET`-style peer API returns signed `(epoch, merkle_root)` under tag `gbase-root-v1` |

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
msg = tag "gbase-bundle-v1" ‖ scale(EpochBundleBodyV1)
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
| `gbase-bundle-v1` | Epoch bundle body |
| `gbase-rawweight-v1` | Challenge leaf body |
| `gbase-dissent-v1` | Dissent body |
| `gbase-root-v1` | Peer `(epoch, merkle_root)` |
| `gbase-trustroot-v1` | Owner trust-root body |
| `gbase-attest-v1` | Attestation bindings (see AGENT_CHALLENGE / D10; out of scope for leaf math) |

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
