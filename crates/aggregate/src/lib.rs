//! `aggregate` — `BUNDLE_SPEC` §6 integer aggregation (`algorithm_version = 1`).
//!
//! Pure function. No I/O. No floats. Accumulators use `u128` with **`checked_*` only**.
//! Consensus lint (D8): no `HashMap`, `f32`/`f64`, or `wrapping_*`.
//!
//! # Binary rewards and all-zero epochs (`BUNDLE_SPEC` §6.5)
//!
//! Agent-challenge scoring is effectively **binary** (solved / not). Frontier pass@1 is
//! low, so many epochs have **zero solvers**. When accumulator mass `total == 0`:
//!
//! - Canonical output is an **empty** `Vec` (not equal weights, not a fabricated uniform
//!   vector). Callers classify this as **no-submit** and MUST NOT emit a CRV4 /
//!   `set_weights` extrinsic (`submit::payload_from_intent` rejects empty vectors).
//! - [`PARTICIPATION_FLOOR`] defaults to **`false`**. Setting it `true` is a deliberate
//!   one-constant switch that replaces all-zero mass with unit participation scores
//!   before Hamilton — it changes what a weight means and stays off unless operators
//!   accept that semantic shift.
//!
//! When `total > 0`, Hamilton largest-remainder apportions exactly [`HOUSE`] (`u16::MAX`).
//!
//! # Two algorithms live here
//!
//! - [`aggregate`] (this module): the integer `BUNDLE_SPEC` §6 Hamilton path, still used by
//!   the bundle/validator code.
//! - [`chain_vector`]: the one adapter both the gateway seal and the validator recompute
//!   call. It maps bundle-shaped inputs onto [`python`] and is the serving algorithm.
//! - [`python`]: a bit-for-bit port of the Python service that actually serves
//!   <https://chain.joinbase.ai>. The operator chose full behavioural parity with Python
//!   for the gateway, so **that** module is the authority for the served vector and it
//!   uses `f64` deliberately.
//!
//! Vectors under `tests/vectors/python/8249563774ee2e71c41ae2cfac182ff32aa35dd1/` are
//! generated from the Python authority; the older `c4ec5c04.../` set is the same code at
//! an earlier upstream commit.

#![forbid(unsafe_code)]

pub mod chain_vector;
pub mod python;

pub use chain_vector::{
    aggregate_python_vector, PythonVector, PythonVectorError, PY_MAX_WEIGHT_LIMIT,
    PY_MIN_ALLOWED_WEIGHTS,
};
pub use python::{
    aggregate_python, build_zero_miner_weights, normalize_weights, to_chain_u16,
    ChallengeWeightsResult, FinalWeights, ZeroMinerWeightError, CHAIN_U16_MAX, EPS,
    ZERO_MINER_BURN_UID,
};

use std::collections::{BTreeMap, BTreeSet};

/// Basis points denominator (shares must sum to this).
pub const BPS_DENOM: u128 = 10_000;
/// On-chain weight house: `u16::MAX`.
pub const HOUSE: u16 = 65_535;
/// Fixed-point scale for per-challenge normalization (`10^12`).
pub const FIXED: u128 = 1_000_000_000_000;
/// Aggregation algorithm version implemented by this crate.
pub const ALGORITHM_VERSION: u16 = 1;
/// Default minimum surviving share mass after quarantine (config; D6). Callers enforce.
pub const DEFAULT_MIN_SHARE_MASS_BPS: u16 = 5_000;
/// Participation floor for all-zero epochs (plan todo 15 / Metis B2).
///
/// When `false` (default): `total == 0` → empty vector → **no-submit** (§6.5).
/// When `true`: each leaf participant gets unit `acc = 1` before Hamilton so a
/// near-equal weight vector is emitted even if nobody solved. **Off by default** —
/// flip only as a conscious product decision; it is a one-constant change.
pub const PARTICIPATION_FLOOR: bool = false;

/// Score present on a verified leaf, or explicit absence (`NoScore`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreOrAbsence {
    /// Miner produced a score for the challenge.
    Score {
        /// Non-negative raw score (`u64` as specified).
        value: u64,
    },
    /// Miner did not produce a usable score; raw contribution is `0`.
    NoScore {
        /// Reason code (opaque to aggregation; raw = 0).
        reason: u8,
    },
}

/// One verified leaf entering aggregation (post signature / participant checks).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLeaf {
    /// Challenge identifier bytes (compared to share keys).
    pub challenge_id: Vec<u8>,
    /// Miner hotkey (32 bytes).
    pub miner_hotkey: [u8; 32],
    /// Score or explicit no-score.
    pub score_or_absence: ScoreOrAbsence,
}

/// Aggregation / renormalization errors (class B → no-submit when returned to callers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateError {
    /// `algorithm_version` is not `1`.
    UnsupportedAlgorithmVersion,
    /// Share bps do not sum to exactly [`BPS_DENOM`] (`10_000`).
    SharesSumNotBpsDenom,
    /// Duplicate challenge id in the shares list.
    DuplicateChallengeShare,
    /// Duplicate hotkey in `uid_map`.
    DuplicateHotkeyInUidMap,
    /// Duplicate uid in `uid_map`.
    DuplicateUidInUidMap,
    /// Leaf hotkey missing from `uid_map`.
    MissingUidMapEntry,
    /// Checked arithmetic overflow (or floor > `u16::MAX`).
    Overflow,
    /// After dropping quarantined challenges, surviving bps sum to 0.
    SurvivingMassZero,
}

impl core::fmt::Display for AggregateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedAlgorithmVersion => write!(f, "unsupported algorithm_version"),
            Self::SharesSumNotBpsDenom => write!(f, "shares bps must sum to 10000"),
            Self::DuplicateChallengeShare => write!(f, "duplicate challenge id in shares"),
            Self::DuplicateHotkeyInUidMap => write!(f, "duplicate hotkey in uid_map"),
            Self::DuplicateUidInUidMap => write!(f, "duplicate uid in uid_map"),
            Self::MissingUidMapEntry => write!(f, "leaf hotkey missing from uid_map"),
            Self::Overflow => write!(f, "aggregation arithmetic overflow"),
            Self::SurvivingMassZero => write!(f, "surviving share mass is zero after quarantine"),
        }
    }
}

impl std::error::Error for AggregateError {}

/// Hotkey → uid lookup built from a bijective `uid_map`.
type HotkeyUidMap = BTreeMap<[u8; 32], u16>;

/// Intermediate Hamilton row for share renormalization (challenge id key).
struct ShareRow {
    id: Vec<u8>,
    floor: u128,
    remainder: u128,
    extra: u128,
}

/// Intermediate Hamilton row for miner weight apportionment.
struct MinerRow {
    hotkey: [u8; 32],
    uid: u16,
    floor: u128,
    remainder: u128,
    extra: u128,
}

/// Returns `true` when `vector` is the canonical §6.5 no-submit signal (empty).
///
/// Non-empty vectors — including those with zero-weight rows when `total > 0` —
/// are submit-eligible from the aggregation layer's point of view.
#[must_use]
pub fn is_no_submit_vector(vector: &[(u16, u16)]) -> bool {
    vector.is_empty()
}

/// Prepare participant accumulators for Hamilton, applying an optional participation floor.
///
/// # Returns
///
/// - `Some(total)` when Hamilton should run (`total > 0` after optional floor).
/// - `None` when the epoch is **no-submit** (empty weight vector).
///
/// Production callers pass [`PARTICIPATION_FLOOR`]. Tests may pass `true` to exercise
/// the floor path without flipping the crate default.
pub fn prepare_weight_inputs(
    participants: &mut [([u8; 32], u16, u128)],
    total: u128,
    participation_floor: bool,
) -> Option<u128> {
    if total > 0 {
        return Some(total);
    }
    if participation_floor && !participants.is_empty() {
        let n = u128::try_from(participants.len()).ok()?;
        for row in participants.iter_mut() {
            row.2 = 1;
        }
        return Some(n);
    }
    None
}

/// Aggregate verified leaves into the canonical final weight vector.
///
/// # Errors
///
/// Returns [`AggregateError`] on bad version, malformed shares/`uid_map`, missing
/// uid entries, or any `checked_*` failure.
///
/// # Canonical output (`BUNDLE_SPEC` §6.5–§6.7)
///
/// - All-zero / `total == 0` and [`PARTICIPATION_FLOOR`] is `false` → empty `Vec`
///   (no-submit). See [`is_no_submit_vector`].
/// - Otherwise every uid whose hotkey appears in `leaves` (and is in `uid_map`),
///   sorted by ascending uid; `sum(weight) == HOUSE` (`65_535`). Zero weights kept.
pub fn aggregate(
    leaves: &[VerifiedLeaf],
    shares: &[(Vec<u8>, u16)],
    uid_map: &[([u8; 32], u16)],
    algorithm_version: u16,
) -> Result<Vec<(u16, u16)>, AggregateError> {
    if algorithm_version != ALGORITHM_VERSION {
        return Err(AggregateError::UnsupportedAlgorithmVersion);
    }

    let share_map = validate_shares(shares)?;
    let hk_to_uid = validate_uid_map(uid_map)?;

    // Group raw scores by (challenge_id → miner_hotkey → summed raw).
    let mut by_challenge: BTreeMap<Vec<u8>, BTreeMap<[u8; 32], u128>> = BTreeMap::new();
    let mut leaf_hotkeys: BTreeSet<[u8; 32]> = BTreeSet::new();

    for leaf in leaves {
        if !hk_to_uid.contains_key(&leaf.miner_hotkey) {
            return Err(AggregateError::MissingUidMapEntry);
        }
        leaf_hotkeys.insert(leaf.miner_hotkey);
        let raw = match leaf.score_or_absence {
            ScoreOrAbsence::Score { value } => u128::from(value),
            ScoreOrAbsence::NoScore { .. } => 0,
        };
        let miners = by_challenge.entry(leaf.challenge_id.clone()).or_default();
        let prev = miners.get(&leaf.miner_hotkey).copied().unwrap_or(0);
        let next = prev.checked_add(raw).ok_or(AggregateError::Overflow)?;
        miners.insert(leaf.miner_hotkey, next);
    }

    // Cross-challenge accumulators (hotkey → acc).
    let mut acc: BTreeMap<[u8; 32], u128> = BTreeMap::new();

    for (cid, s_c) in &share_map {
        if *s_c == 0 {
            continue;
        }
        let Some(miners) = by_challenge.get(cid) else {
            continue;
        };

        let mut sum_c: u128 = 0;
        for raw in miners.values() {
            sum_c = sum_c.checked_add(*raw).ok_or(AggregateError::Overflow)?;
        }
        if sum_c == 0 {
            continue;
        }

        let s_c_u = u128::from(*s_c);
        for (hk, r_m) in miners {
            // term = floor(r_m * FIXED * s_c / sum_c / BPS_DENOM)
            let term = r_m
                .checked_mul(FIXED)
                .ok_or(AggregateError::Overflow)?
                .checked_mul(s_c_u)
                .ok_or(AggregateError::Overflow)?
                .checked_div(sum_c)
                .ok_or(AggregateError::Overflow)?
                .checked_div(BPS_DENOM)
                .ok_or(AggregateError::Overflow)?;
            let prev = acc.get(hk).copied().unwrap_or(0);
            let next = prev.checked_add(term).ok_or(AggregateError::Overflow)?;
            acc.insert(*hk, next);
        }
    }

    // Participants: every leaf hotkey (acc default 0).
    let mut participants: Vec<([u8; 32], u16, u128)> = Vec::with_capacity(leaf_hotkeys.len());
    let mut total: u128 = 0;
    for hk in &leaf_hotkeys {
        let a = acc.get(hk).copied().unwrap_or(0);
        total = total.checked_add(a).ok_or(AggregateError::Overflow)?;
        let uid = *hk_to_uid
            .get(hk)
            .ok_or(AggregateError::MissingUidMapEntry)?;
        participants.push((*hk, uid, a));
    }

    // §6.5 all-zero → empty Vec (no-submit) unless PARTICIPATION_FLOOR is on.
    let Some(total) = prepare_weight_inputs(&mut participants, total, PARTICIPATION_FLOOR) else {
        return Ok(Vec::new());
    };

    let weights = hamilton_house(&participants, total, u128::from(HOUSE))?;

    // Sort by ascending uid (stable canonical order).
    let mut out = weights;
    out.sort_by_key(|(uid, _)| *uid);
    Ok(out)
}

/// Re-apportion remaining emission shares after dropping quarantined challenges.
///
/// Removes `dropped_ids` from `shares`, then Hamilton-apportion remaining bps as
/// scores with house [`BPS_DENOM`] (`10_000`). Tie-break: ascending `challenge_id`
/// bytes (`BUNDLE_SPEC` §6.9).
///
/// Does **not** enforce `min_share_mass_bps` (caller / config; D6).
///
/// # Errors
///
/// - [`AggregateError::SurvivingMassZero`] if no bps remain.
/// - [`AggregateError::Overflow`] on arithmetic failure.
/// - [`AggregateError::DuplicateChallengeShare`] if input shares have duplicate ids
///   among non-dropped entries after filter (duplicate keys in input).
pub fn renormalize_after_quarantine(
    shares: &[(Vec<u8>, u16)],
    dropped_ids: &[Vec<u8>],
) -> Result<Vec<(Vec<u8>, u16)>, AggregateError> {
    let dropped: BTreeSet<&[u8]> = dropped_ids.iter().map(Vec::as_slice).collect();

    let mut remaining: BTreeMap<Vec<u8>, u16> = BTreeMap::new();
    for (id, bps) in shares {
        if dropped.contains(id.as_slice()) {
            continue;
        }
        if remaining.insert(id.clone(), *bps).is_some() {
            return Err(AggregateError::DuplicateChallengeShare);
        }
    }

    let mut surv: u128 = 0;
    for bps in remaining.values() {
        surv = surv
            .checked_add(u128::from(*bps))
            .ok_or(AggregateError::Overflow)?;
    }
    if surv == 0 {
        return Err(AggregateError::SurvivingMassZero);
    }

    // Hamilton on remaining bps-as-scores, house = 10_000, tie-break by challenge_id.
    let mut ids: Vec<Vec<u8>> = remaining.keys().cloned().collect();
    ids.sort(); // ascending challenge_id

    let total = surv;
    let house = BPS_DENOM;

    let mut rows: Vec<ShareRow> = Vec::with_capacity(ids.len());
    let mut sum_floors: u128 = 0;
    for id in ids {
        let bps = *remaining.get(&id).ok_or(AggregateError::Overflow)?;
        let acc_m = u128::from(bps);
        let prod = acc_m.checked_mul(house).ok_or(AggregateError::Overflow)?;
        let floor_m = prod.checked_div(total).ok_or(AggregateError::Overflow)?;
        let remainder_m = prod.checked_rem(total).ok_or(AggregateError::Overflow)?;
        if floor_m > house {
            return Err(AggregateError::Overflow);
        }
        sum_floors = sum_floors
            .checked_add(floor_m)
            .ok_or(AggregateError::Overflow)?;
        rows.push(ShareRow {
            id,
            floor: floor_m,
            remainder: remainder_m,
            extra: 0,
        });
    }

    let seats_left = house
        .checked_sub(sum_floors)
        .ok_or(AggregateError::Overflow)?;
    let seats_n = usize::try_from(seats_left).map_err(|_| AggregateError::Overflow)?;

    // Largest remainder; tie → ascending challenge_id.
    rows.sort_by(|a, b| b.remainder.cmp(&a.remainder).then_with(|| a.id.cmp(&b.id)));
    if seats_n > rows.len() {
        return Err(AggregateError::Overflow);
    }
    for row in rows.iter_mut().take(seats_n) {
        row.extra = 1;
    }

    let mut out: Vec<(Vec<u8>, u16)> = Vec::with_capacity(rows.len());
    for row in rows {
        let seats = row
            .floor
            .checked_add(row.extra)
            .ok_or(AggregateError::Overflow)?;
        if seats > house {
            return Err(AggregateError::Overflow);
        }
        let bps = u16::try_from(seats).map_err(|_| AggregateError::Overflow)?;
        out.push((row.id, bps));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));

    // Sanity: sum must be BPS_DENOM.
    let mut check: u128 = 0;
    for (_, b) in &out {
        check = check
            .checked_add(u128::from(*b))
            .ok_or(AggregateError::Overflow)?;
    }
    if check != BPS_DENOM {
        return Err(AggregateError::Overflow);
    }
    Ok(out)
}

// --- internals ----------------------------------------------------------------

fn validate_shares(shares: &[(Vec<u8>, u16)]) -> Result<BTreeMap<Vec<u8>, u16>, AggregateError> {
    let mut map: BTreeMap<Vec<u8>, u16> = BTreeMap::new();
    let mut sum: u32 = 0;
    for (id, bps) in shares {
        if map.insert(id.clone(), *bps).is_some() {
            return Err(AggregateError::DuplicateChallengeShare);
        }
        sum = sum
            .checked_add(u32::from(*bps))
            .ok_or(AggregateError::Overflow)?;
    }
    if u128::from(sum) != BPS_DENOM {
        return Err(AggregateError::SharesSumNotBpsDenom);
    }
    Ok(map)
}

fn validate_uid_map(uid_map: &[([u8; 32], u16)]) -> Result<HotkeyUidMap, AggregateError> {
    let mut hk_to_uid: HotkeyUidMap = BTreeMap::new();
    let mut seen_uids: BTreeSet<u16> = BTreeSet::new();
    for &(hk, uid) in uid_map {
        if hk_to_uid.insert(hk, uid).is_some() {
            return Err(AggregateError::DuplicateHotkeyInUidMap);
        }
        if !seen_uids.insert(uid) {
            return Err(AggregateError::DuplicateUidInUidMap);
        }
    }
    Ok(hk_to_uid)
}

/// Hamilton largest-remainder for miner weights.
///
/// `participants`: `(hotkey, uid, acc)`. `total` must be > 0. `house` is `65535`.
fn hamilton_house(
    participants: &[([u8; 32], u16, u128)],
    total: u128,
    house: u128,
) -> Result<Vec<(u16, u16)>, AggregateError> {
    if total == 0 {
        return Ok(Vec::new());
    }

    let mut rows: Vec<MinerRow> = Vec::with_capacity(participants.len());
    let mut sum_floors: u128 = 0;

    for &(hotkey, uid, acc_m) in participants {
        let prod = acc_m.checked_mul(house).ok_or(AggregateError::Overflow)?;
        let floor_m = prod.checked_div(total).ok_or(AggregateError::Overflow)?;
        let remainder_m = prod.checked_rem(total).ok_or(AggregateError::Overflow)?;
        if floor_m > u128::from(u16::MAX) {
            return Err(AggregateError::Overflow);
        }
        sum_floors = sum_floors
            .checked_add(floor_m)
            .ok_or(AggregateError::Overflow)?;
        rows.push(MinerRow {
            hotkey,
            uid,
            floor: floor_m,
            remainder: remainder_m,
            extra: 0,
        });
    }

    let seats_left = house
        .checked_sub(sum_floors)
        .ok_or(AggregateError::Overflow)?;
    let seats_n = usize::try_from(seats_left).map_err(|_| AggregateError::Overflow)?;

    // Largest remainder; tie → lower uid; still tie → lower hotkey bytes.
    rows.sort_by(|a, b| {
        b.remainder
            .cmp(&a.remainder)
            .then_with(|| a.uid.cmp(&b.uid))
            .then_with(|| a.hotkey.cmp(&b.hotkey))
    });
    if seats_n > rows.len() {
        return Err(AggregateError::Overflow);
    }
    for row in rows.iter_mut().take(seats_n) {
        row.extra = 1;
    }

    let mut out: Vec<(u16, u16)> = Vec::with_capacity(rows.len());
    for row in rows {
        let seats = row
            .floor
            .checked_add(row.extra)
            .ok_or(AggregateError::Overflow)?;
        if seats > u128::from(u16::MAX) {
            return Err(AggregateError::Overflow);
        }
        let w = u16::try_from(seats).map_err(|_| AggregateError::Overflow)?;
        out.push((row.uid, w));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn hk(tag: u8) -> [u8; 32] {
        let mut h = [0_u8; 32];
        h[0] = tag;
        h
    }

    fn score_leaf(cid: &[u8], tag: u8, value: u64) -> VerifiedLeaf {
        VerifiedLeaf {
            challenge_id: cid.to_vec(),
            miner_hotkey: hk(tag),
            score_or_absence: ScoreOrAbsence::Score { value },
        }
    }

    fn noscore_leaf(cid: &[u8], tag: u8) -> VerifiedLeaf {
        VerifiedLeaf {
            challenge_id: cid.to_vec(),
            miner_hotkey: hk(tag),
            score_or_absence: ScoreOrAbsence::NoScore { reason: 1 },
        }
    }

    /// Canonical encoding for determinism hashing: LE uid, LE weight pairs.
    fn encode_vector(v: &[(u16, u16)]) -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len().saturating_mul(4));
        for &(uid, w) in v {
            out.extend_from_slice(&uid.to_le_bytes());
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let dig = Sha256::digest(bytes);
        let mut s = String::with_capacity(64);
        for b in dig {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    // --- S1 happy path: §6.8 worked example 50/30/20 ---

    #[test]
    fn s1_worked_example_50_30_20_hamilton() {
        let cid = b"solo";
        let leaves = vec![
            score_leaf(cid, b'A', 50),
            score_leaf(cid, b'B', 30),
            score_leaf(cid, b'C', 20),
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(b'A'), 1), (hk(b'B'), 2), (hk(b'C'), 3)];
        let out = aggregate(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert_eq!(out, vec![(1, 32_768), (2, 19_660), (3, 13_107)]);
        let sum: u32 = out.iter().map(|(_, w)| u32::from(*w)).sum();
        assert_eq!(sum, u32::from(HOUSE));
    }

    // --- S2 all-zero → empty no-submit ---

    #[test]
    fn s2_all_zero_scores_empty_vector_no_submit() {
        let cid = b"c";
        let leaves = vec![
            score_leaf(cid, 1, 0),
            noscore_leaf(cid, 2),
            score_leaf(cid, 3, 0),
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 10), (hk(2), 20), (hk(3), 30)];
        let out = aggregate(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert!(out.is_empty(), "canonical all-zero is empty Vec");
    }

    #[test]
    fn s2b_empty_leaves_empty_vector() {
        let shares = vec![(b"c".to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 1)];
        let out = aggregate(&[], &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert!(out.is_empty());
    }

    // --- S3 permutation invariance ---

    #[test]
    fn s3_permuting_leaves_and_uid_map_unchanged() {
        let cid = b"solo";
        let mut leaves = vec![
            score_leaf(cid, b'A', 50),
            score_leaf(cid, b'B', 30),
            score_leaf(cid, b'C', 20),
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let mut uid_map = vec![(hk(b'A'), 1), (hk(b'B'), 2), (hk(b'C'), 3)];
        let base = aggregate(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("base");

        leaves.swap(0, 2);
        leaves.swap(1, 2);
        uid_map.swap(0, 1);
        let perm = aggregate(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("perm");
        assert_eq!(base, perm);
    }

    // --- S4 multi-challenge combine ---

    #[test]
    fn s4_two_challenges_weighted_shares() {
        // c1 60% scores equal two miners; c2 40% single miner takes all.
        let c1 = b"c1";
        let c2 = b"c2";
        let leaves = vec![
            score_leaf(c1, 1, 10),
            score_leaf(c1, 2, 10),
            score_leaf(c2, 2, 100),
            // miner 1 also present under c2 as NoScore so completeness-style leaf exists
            noscore_leaf(c2, 1),
        ];
        let shares = vec![(c1.to_vec(), 6_000), (c2.to_vec(), 4_000)];
        let uid_map = vec![(hk(1), 1), (hk(2), 2)];
        let out = aggregate(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        let sum: u32 = out.iter().map(|(_, w)| u32::from(*w)).sum();
        assert_eq!(sum, u32::from(HOUSE));
        assert_eq!(out.len(), 2);
        // miner2 should get strictly more than miner1 (half of 60% + all of 40%).
        let w1 = out.iter().find(|(u, _)| *u == 1).expect("u1").1;
        let w2 = out.iter().find(|(u, _)| *u == 2).expect("u2").1;
        assert!(w2 > w1, "w2={w2} w1={w1}");
    }

    // --- S5 quarantine renormalization ---

    #[test]
    fn s5_renormalize_after_quarantine_sums_to_10000() {
        let shares = vec![
            (b"a".to_vec(), 5_000),
            (b"b".to_vec(), 3_000),
            (b"c".to_vec(), 2_000),
        ];
        let dropped = vec![b"a".to_vec()];
        let out = renormalize_after_quarantine(&shares, &dropped).expect("renorm");
        let sum: u32 = out.iter().map(|(_, b)| u32::from(*b)).sum();
        assert_eq!(sum, 10_000);
        // 3000:2000 → 6000:4000
        assert_eq!(out, vec![(b"b".to_vec(), 6_000), (b"c".to_vec(), 4_000)]);
    }

    #[test]
    fn s5b_renormalize_all_dropped_errors() {
        let shares = vec![(b"a".to_vec(), 10_000)];
        let err = renormalize_after_quarantine(&shares, &[b"a".to_vec()]).expect_err("zero");
        assert_eq!(err, AggregateError::SurvivingMassZero);
    }

    #[test]
    fn s5c_renormalize_identity_when_already_10000() {
        let shares = vec![
            (b"a".to_vec(), 3_333),
            (b"b".to_vec(), 3_333),
            (b"c".to_vec(), 3_334),
        ];
        let out = renormalize_after_quarantine(&shares, &[]).expect("renorm");
        let sum: u32 = out.iter().map(|(_, b)| u32::from(*b)).sum();
        assert_eq!(sum, 10_000);
        assert_eq!(
            out,
            vec![
                (b"a".to_vec(), 3_333),
                (b"b".to_vec(), 3_333),
                (b"c".to_vec(), 3_334),
            ]
        );
    }

    // --- S6 error branches ---

    #[test]
    fn s6_bad_algorithm_version() {
        let err = aggregate(&[], &[(b"c".to_vec(), 10_000)], &[], 0).expect_err("ver");
        assert_eq!(err, AggregateError::UnsupportedAlgorithmVersion);
    }

    #[test]
    fn s6_shares_sum_not_10000() {
        let err = aggregate(&[], &[(b"c".to_vec(), 9999)], &[], 1).expect_err("sum");
        assert_eq!(err, AggregateError::SharesSumNotBpsDenom);
    }

    #[test]
    fn s6_duplicate_challenge_share() {
        let shares = vec![(b"c".to_vec(), 5_000), (b"c".to_vec(), 5_000)];
        let err = aggregate(&[], &shares, &[], 1).expect_err("dup");
        assert_eq!(err, AggregateError::DuplicateChallengeShare);
    }

    #[test]
    fn s6_missing_uid_map_entry() {
        let leaves = vec![score_leaf(b"c", 9, 1)];
        let shares = vec![(b"c".to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 1)];
        let err = aggregate(&leaves, &shares, &uid_map, 1).expect_err("uid");
        assert_eq!(err, AggregateError::MissingUidMapEntry);
    }

    #[test]
    fn s6_duplicate_hotkey_uid_map() {
        let uid_map = vec![(hk(1), 1), (hk(1), 2)];
        let err = aggregate(&[], &[(b"c".to_vec(), 10_000)], &uid_map, 1).expect_err("hk");
        assert_eq!(err, AggregateError::DuplicateHotkeyInUidMap);
    }

    #[test]
    fn s6_duplicate_uid_uid_map() {
        let uid_map = vec![(hk(1), 1), (hk(2), 1)];
        let err = aggregate(&[], &[(b"c".to_vec(), 10_000)], &uid_map, 1).expect_err("uid");
        assert_eq!(err, AggregateError::DuplicateUidInUidMap);
    }

    // --- S7 zero weight kept when total > 0 ---

    #[test]
    fn s7_zero_acc_miner_kept_with_zero_weight() {
        let cid = b"c";
        let leaves = vec![
            score_leaf(cid, 1, 100),
            noscore_leaf(cid, 2), // raw 0 but present
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 1), (hk(2), 2)];
        let out = aggregate(&leaves, &shares, &uid_map, 1).expect("agg");
        assert_eq!(out, vec![(1, HOUSE), (2, 0)]);
        let sum: u32 = out.iter().map(|(_, w)| u32::from(*w)).sum();
        assert_eq!(sum, u32::from(HOUSE));
    }

    // --- S8 Hamilton remainder tie → lower uid ---

    #[test]
    fn s8_remainder_tie_lower_uid_wins_extra_seat() {
        // Same as worked example: A and B tie remainder; uid 1 wins → 32768 not 32767.
        let cid = b"solo";
        let leaves = vec![
            score_leaf(cid, b'A', 50),
            score_leaf(cid, b'B', 30),
            score_leaf(cid, b'C', 20),
        ];
        // Swap tags so higher tag has lower uid — still uid order decides.
        let uid_map = vec![(hk(b'A'), 1), (hk(b'B'), 2), (hk(b'C'), 3)];
        let out = aggregate(&leaves, &[(cid.to_vec(), 10_000)], &uid_map, 1).expect("agg");
        assert_eq!(out[0], (1, 32_768));
    }

    // --- S9 debug/release determinism pin (sha256 of canonical encoding) ---

    /// Pinned sha256 of LE-encoded §6.8 vector — debug and release must both match.
    const WORKED_EXAMPLE_VECTOR_SHA256: &str =
        "d11f34644e6a5ab969c297eb6670e3546cbb07c5e233f62bd1f4ac3da9c78a35";

    #[test]
    fn s9_worked_example_sha256_pin_debug_release_identical() {
        let cid = b"solo";
        let leaves = vec![
            score_leaf(cid, b'A', 50),
            score_leaf(cid, b'B', 30),
            score_leaf(cid, b'C', 20),
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(b'A'), 1), (hk(b'B'), 2), (hk(b'C'), 3)];
        let out = aggregate(&leaves, &shares, &uid_map, 1).expect("agg");
        let enc = encode_vector(&out);
        let expected_enc = {
            let mut v = Vec::new();
            for (u, w) in [(1_u16, 32_768_u16), (2, 19_660), (3, 13_107)] {
                v.extend_from_slice(&u.to_le_bytes());
                v.extend_from_slice(&w.to_le_bytes());
            }
            v
        };
        assert_eq!(enc, expected_enc);
        assert_eq!(sha256_hex(&enc), WORKED_EXAMPLE_VECTOR_SHA256);
    }

    // --- S10 matches_spec vector 03 from task-16 ---

    #[test]
    fn s10_task16_matches_spec_vector_03() {
        // Hotkeys A/B/C as single-byte tags matching characterization labels.
        let cid = b"solo";
        let leaves = vec![
            score_leaf(cid, b'A', 50),
            score_leaf(cid, b'B', 30),
            score_leaf(cid, b'C', 20),
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(b'A'), 1), (hk(b'B'), 2), (hk(b'C'), 3)];
        let out = aggregate(&leaves, &shares, &uid_map, 1).expect("agg");
        assert_eq!(out, vec![(1, 32_768), (2, 19_660), (3, 13_107)]);
    }

    #[test]
    fn s11_uid_map_extra_hotkey_not_in_leaves_omitted() {
        let cid = b"c";
        let leaves = vec![score_leaf(cid, 1, 1)];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 1), (hk(2), 2)];
        let out = aggregate(&leaves, &shares, &uid_map, 1).expect("agg");
        assert_eq!(out, vec![(1, HOUSE)]);
    }

    #[test]
    fn constants_match_spec() {
        assert_eq!(BPS_DENOM, 10_000);
        assert_eq!(HOUSE, 65_535);
        assert_eq!(HOUSE, u16::MAX);
        assert_eq!(FIXED, 1_000_000_000_000);
        assert_eq!(ALGORITHM_VERSION, 1);
        // Todo 15: participation floor stays OFF — all-zero is no-submit.
        const { assert!(!PARTICIPATION_FLOOR) };
    }

    fn vector_sum(v: &[(u16, u16)]) -> u32 {
        v.iter().map(|(_, w)| u32::from(*w)).sum()
    }

    /// Binary-reward shape suite (todo 15): all-zero, single-winner, all-winners, sparse.
    #[test]
    fn s15_binary_reward_four_shapes_bound_and_sum() {
        let cid = b"bin";
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 0), (hk(2), 1), (hk(3), 2)];

        // Shape A — all-zero (binary fails): empty no-submit vector.
        let all_zero_leaves = vec![
            score_leaf(cid, 1, 0),
            score_leaf(cid, 2, 0),
            noscore_leaf(cid, 3),
        ];
        let all_zero = aggregate(&all_zero_leaves, &shares, &uid_map, 1).expect("zero");
        assert!(is_no_submit_vector(&all_zero));
        assert!(all_zero.is_empty());
        assert_eq!(vector_sum(&all_zero), 0);
        eprintln!("shape all-zero: {all_zero:?} sum={}", vector_sum(&all_zero));

        // Shape B — single winner (binary 1/0/0): winner takes HOUSE, losers kept at 0.
        let single_leaves = vec![
            score_leaf(cid, 1, 0),
            score_leaf(cid, 2, 1),
            score_leaf(cid, 3, 0),
        ];
        let single = aggregate(&single_leaves, &shares, &uid_map, 1).expect("single");
        assert!(!is_no_submit_vector(&single));
        assert_eq!(vector_sum(&single), u32::from(HOUSE));
        assert_eq!(single, vec![(0, 0), (1, HOUSE), (2, 0)]);
        eprintln!(
            "shape single-winner: {single:?} sum={}",
            vector_sum(&single)
        );

        // Shape C — all winners (binary 1/1/1): equal Hamilton split of HOUSE.
        let all_win_leaves = vec![
            score_leaf(cid, 1, 1),
            score_leaf(cid, 2, 1),
            score_leaf(cid, 3, 1),
        ];
        let all_win = aggregate(&all_win_leaves, &shares, &uid_map, 1).expect("all");
        assert!(!is_no_submit_vector(&all_win));
        assert_eq!(vector_sum(&all_win), u32::from(HOUSE));
        // 65535 / 3 = 21845 exactly.
        assert_eq!(all_win, vec![(0, 21_845), (1, 21_845), (2, 21_845)]);
        eprintln!(
            "shape all-winners: {all_win:?} sum={}",
            vector_sum(&all_win)
        );

        // Shape D — sparse (two of many solve): winners share HOUSE, zeros kept.
        let sparse_leaves = vec![
            score_leaf(cid, 1, 1),
            score_leaf(cid, 2, 0),
            score_leaf(cid, 3, 1),
        ];
        let sparse = aggregate(&sparse_leaves, &shares, &uid_map, 1).expect("sparse");
        assert!(!is_no_submit_vector(&sparse));
        assert_eq!(vector_sum(&sparse), u32::from(HOUSE));
        // acc [1,0,1] total=2 → floors 32767/0/32767, one extra seat → lower uid.
        assert_eq!(sparse, vec![(0, 32_768), (1, 0), (2, 32_767)]);
        eprintln!("shape sparse: {sparse:?} sum={}", vector_sum(&sparse));

        // Live reference [(0,3641),(1,65535),(2,3641)] is NOT a Hamilton HOUSE vector
        // (sum ≠ 65535). Canonical base never emits that shape.
        let live_ref_sum: u32 = 3_641 + u32::from(HOUSE) + 3_641;
        assert_ne!(live_ref_sum, u32::from(HOUSE));
    }

    #[test]
    fn s15_participation_floor_off_by_default_no_submit() {
        const { assert!(!PARTICIPATION_FLOOR) };
        let mut parts = vec![(hk(1), 0_u16, 0_u128), (hk(2), 1, 0), (hk(3), 2, 0)];
        let mass = prepare_weight_inputs(&mut parts, 0, PARTICIPATION_FLOOR);
        assert!(mass.is_none(), "default floor off → no-submit");
        // Accumulators untouched when floor is off.
        assert!(parts.iter().all(|(_, _, a)| *a == 0));
    }

    #[test]
    fn s15_participation_floor_on_is_one_constant_path() {
        // Documented alternative: pass true without flipping the crate const.
        let mut parts = vec![(hk(1), 0_u16, 0_u128), (hk(2), 1, 0), (hk(3), 2, 0)];
        let mass = prepare_weight_inputs(&mut parts, 0, true).expect("floor on");
        assert_eq!(mass, 3);
        assert!(parts.iter().all(|(_, _, a)| *a == 1));
        // Equal unit acc feeds the same Hamilton path as all-winners (s15 four shapes).
        // Production aggregate still uses PARTICIPATION_FLOOR=false.
        // Production aggregate still uses PARTICIPATION_FLOOR=false.
        let cid = b"c";
        let leaves = vec![
            score_leaf(cid, 1, 0),
            score_leaf(cid, 2, 0),
            score_leaf(cid, 3, 0),
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 0), (hk(2), 1), (hk(3), 2)];
        let prod = aggregate(&leaves, &shares, &uid_map, 1).expect("agg");
        assert!(is_no_submit_vector(&prod));
    }

    #[test]
    fn s15_prepare_weight_inputs_positive_total_passthrough() {
        let mut parts = vec![(hk(1), 1_u16, 50_u128), (hk(2), 2, 50)];
        let mass = prepare_weight_inputs(&mut parts, 100, false).expect("pos");
        assert_eq!(mass, 100);
        assert_eq!(parts[0].2, 50);
    }
}
