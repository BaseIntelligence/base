//! The one adapter from bundle-shaped inputs to the served Python vector.
//!
//! The gateway seal and the validator's independent recompute must produce identical
//! bytes, so both call [`aggregate_python_vector`] and nothing else. Duplicating the
//! conversion would let the two sides drift silently; keeping it here makes agreement a
//! property of the code rather than of two copies happening to match.
//!
//! Shapes differ between the two worlds:
//!
//! | bundle | Python |
//! |---|---|
//! | `emission_shares` in bps out of `10_000` | `emission_percent` out of `100` |
//! | leaves keyed by `(challenge_id, hotkey)` | per-challenge `weights` dict |
//! | `NoScore` / zero score | absent key (`_clean_weights` drops non-positive) |
//! | `uid_map` sorted by hotkey | `hotkey_to_uid` dict |
//!
//! Identifiers are hex strings because Python keys on `str`; hex is injective over
//! bytes, so it cannot merge two distinct challenges or hotkeys into one dict slot.

use std::collections::BTreeMap;

use crate::python::{
    aggregate_python, to_chain_u16, ChallengeWeightsResult, FinalWeights, ZeroMinerWeightError,
    CHAIN_U16_MAX,
};
use crate::{
    validate_shares, validate_uid_map, AggregateError, ScoreOrAbsence, VerifiedLeaf,
    ALGORITHM_VERSION,
};

/// `min_allowed_weights` the deployed Python master actually runs with.
///
/// Not a taste default: `base.master.service` constructs
/// `AggregationService(session_factory, freshness_seconds=...)` on `origin/main` without
/// passing `min_allowed_weights`, so `aggregate_challenge_weights` uses its own default
/// of `1`. Changing this changes the served vector on the zero-miner path.
pub const PY_MIN_ALLOWED_WEIGHTS: u32 = 1;

/// `max_weight_limit` the deployed Python master actually runs with.
///
/// Same provenance as [`PY_MIN_ALLOWED_WEIGHTS`]: the service never passes it, so the
/// `aggregate_challenge_weights` default of `65535` (no per-uid cap) applies.
pub const PY_MAX_WEIGHT_LIMIT: u16 = CHAIN_U16_MAX;

/// The served vector plus the floats it was rounded from.
///
/// `final_vector` is what goes into the signed bundle body and onto the chain;
/// `floats` is what `GET /v1/weights/latest` serves, because a u16 round-trip is not
/// the same number (`19660/65535 != 0.3`).
#[derive(Debug, Clone, PartialEq)]
pub struct PythonVector {
    /// `(uid, u16 weight)` ascending by uid.
    ///
    /// Does **not** always sum to `65535`: Python performs no post-rounding
    /// renormalisation, so banker's rounding leaves the total at `65534`/`65536` for a
    /// substantial fraction of inputs. Correcting it here would break parity.
    pub final_vector: Vec<(u16, u16)>,
    /// Exact `aggregate_challenge_weights` output (uids, weights, `kept_hotkeys`).
    ///
    /// `hotkey_weights` keys are lowercase hex of the 32-byte hotkey.
    pub floats: FinalWeights,
}

/// Why the served vector could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PythonVectorError {
    /// Malformed bundle-side inputs (shares, `uid_map`, arithmetic).
    Input(AggregateError),
    /// Python's `ZeroMinerWeightError`: no chain-valid burn vector exists.
    ZeroMiner(ZeroMinerWeightError),
}

impl core::fmt::Display for PythonVectorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Input(e) => write!(f, "{e}"),
            Self::ZeroMiner(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PythonVectorError {}

/// Lowercase hex, so byte identifiers become distinct Python `dict` keys.
fn hex_key(bytes: &[u8]) -> String {
    const DIGITS: [u8; 16] = *b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    out
}

/// Run the Python authority's aggregation over bundle-shaped inputs.
///
/// Deterministic regardless of input order: challenges are visited in ascending
/// `challenge_id`, hotkeys in ascending key bytes, so the insertion order Python's
/// float summation depends on is fixed by the data, not by the caller's `Vec` order.
///
/// Behaviour that differs from [`crate::aggregate`], on purpose, because Python does it:
///
/// - A leaf hotkey absent from `uid_map` is **not** an error; its share burns to uid 0.
/// - A hotkey mapped to uid 0 is dropped and its share burns (uid 0 is the sink).
/// - An all-zero epoch falls back to `build_zero_miner_weights` instead of emitting the
///   empty no-submit vector.
///
/// Every challenge in `shares` is passed with `ok = true`, including ones with no leaves.
/// An `ok` challenge with empty weights contributes nothing and its share burns, which is
/// numerically identical to `ok = false` — the only place `ok` changes the arithmetic is
/// the `alloc_total > 1.0` scale-back, which cannot trigger while shares sum to `10_000`
/// (i.e. `alloc_total == 1.0`). `ok_true_empty_challenge_matches_ok_false` proves it.
///
/// # Errors
///
/// [`PythonVectorError`] on bad `algorithm_version`, malformed shares/`uid_map`,
/// score overflow, or `ZeroMinerWeightError`.
pub fn aggregate_python_vector(
    leaves: &[VerifiedLeaf],
    shares: &[(Vec<u8>, u16)],
    uid_map: &[([u8; 32], u16)],
    algorithm_version: u16,
) -> Result<PythonVector, PythonVectorError> {
    if algorithm_version != ALGORITHM_VERSION {
        return Err(PythonVectorError::Input(
            AggregateError::UnsupportedAlgorithmVersion,
        ));
    }
    let share_map = validate_shares(shares).map_err(PythonVectorError::Input)?;
    let hk_to_uid = validate_uid_map(uid_map).map_err(PythonVectorError::Input)?;

    let mut by_challenge: BTreeMap<&[u8], BTreeMap<[u8; 32], u64>> = BTreeMap::new();
    for leaf in leaves {
        let raw = match leaf.score_or_absence {
            ScoreOrAbsence::Score { value } => value,
            ScoreOrAbsence::NoScore { .. } => 0,
        };
        let miners = by_challenge
            .entry(leaf.challenge_id.as_slice())
            .or_default();
        let prev = miners.get(&leaf.miner_hotkey).copied().unwrap_or(0);
        let next = prev
            .checked_add(raw)
            .ok_or(PythonVectorError::Input(AggregateError::Overflow))?;
        miners.insert(leaf.miner_hotkey, next);
    }

    let results: Vec<ChallengeWeightsResult> = share_map
        .iter()
        .map(|(challenge_id, bps)| ChallengeWeightsResult {
            slug: hex_key(challenge_id),
            emission_percent: f64::from(*bps) / 100.0,
            weights: challenge_weights(by_challenge.get(challenge_id.as_slice())),
            ok: true,
            error: None,
        })
        .collect();

    let hotkey_to_uid: Vec<(String, u16)> = hk_to_uid
        .iter()
        .map(|(hotkey, uid)| (hex_key(hotkey), *uid))
        .collect();

    let floats = aggregate_python(
        &results,
        &hotkey_to_uid,
        PY_MIN_ALLOWED_WEIGHTS,
        PY_MAX_WEIGHT_LIMIT,
    )
    .map_err(PythonVectorError::ZeroMiner)?;

    let final_vector: Vec<(u16, u16)> = floats
        .uids
        .iter()
        .copied()
        .zip(to_chain_u16(&floats.weights))
        .collect();

    Ok(PythonVector {
        final_vector,
        floats,
    })
}

/// Positive raw scores as Python weights; zero / `NoScore` entries are omitted.
///
/// Omitting is equivalent to passing `0.0` because `_clean_weights` drops non-positive
/// values before normalising; omitting just says so at the call site.
fn challenge_weights(miners: Option<&BTreeMap<[u8; 32], u64>>) -> Vec<(String, f64)> {
    let Some(miners) = miners else {
        return Vec::new();
    };
    miners
        .iter()
        .filter(|(_, raw)| **raw > 0)
        .map(|(hotkey, raw)| {
            // Scores above 2^53 lose precision here, exactly as they would when the
            // Python challenge API hands the master a JSON number.
            #[allow(clippy::cast_precision_loss)]
            (hex_key(hotkey), *raw as f64)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::python::aggregate_python;

    fn hk(tag: u8) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0] = tag;
        out
    }

    fn score(cid: &[u8], tag: u8, value: u64) -> VerifiedLeaf {
        VerifiedLeaf {
            challenge_id: cid.to_vec(),
            miner_hotkey: hk(tag),
            score_or_absence: ScoreOrAbsence::Score { value },
        }
    }

    fn noscore(cid: &[u8], tag: u8) -> VerifiedLeaf {
        VerifiedLeaf {
            challenge_id: cid.to_vec(),
            miner_hotkey: hk(tag),
            score_or_absence: ScoreOrAbsence::NoScore { reason: 1 },
        }
    }

    /// Python: `aggregate_challenge_weights([{c,100,{A:50,B:30,C:20}}], {A:1,B:2,C:3})`
    /// -> uids [1,2,3] weights [0.5, 0.3, 0.2] -> u16 [32768, 19660, 13107].
    #[test]
    fn spec_like_50_30_20_matches_python_floats() {
        let cid = b"solo";
        let leaves = vec![
            score(cid, b'A', 50),
            score(cid, b'B', 30),
            score(cid, b'C', 20),
        ];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(b'A'), 1), (hk(b'B'), 2), (hk(b'C'), 3)];
        let out =
            aggregate_python_vector(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert_eq!(out.floats.uids, vec![1, 2, 3]);
        assert_eq!(out.floats.weights, vec![0.5, 0.3, 0.2]);
        assert_eq!(
            out.final_vector,
            vec![(1, 32_768), (2, 19_660), (3, 13_107)]
        );
    }

    /// Input order must not move a single ulp: the adapter re-sorts before Python sums.
    #[test]
    fn permuting_leaves_and_uid_map_is_bit_identical() {
        let cid = b"c";
        let mut leaves = vec![score(cid, 1, 7), score(cid, 2, 11), score(cid, 3, 13)];
        let shares = vec![(cid.to_vec(), 10_000)];
        let mut uid_map = vec![(hk(1), 5), (hk(2), 6), (hk(3), 7)];
        let base =
            aggregate_python_vector(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("base");
        leaves.swap(0, 2);
        uid_map.swap(0, 1);
        let perm =
            aggregate_python_vector(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("perm");
        assert_eq!(base.final_vector, perm.final_vector);
        for (a, b) in base.floats.weights.iter().zip(&perm.floats.weights) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// `ok = true` with no weights is arithmetically the same as `ok = false`, because
    /// the share simply never reaches a miner and burns. Argued in the docs, proved here.
    #[test]
    fn ok_true_empty_challenge_matches_ok_false() {
        let full = ChallengeWeightsResult {
            slug: "full".to_owned(),
            emission_percent: 30.0,
            weights: vec![("a".to_owned(), 1.0)],
            ok: true,
            error: None,
        };
        let empty_ok = ChallengeWeightsResult {
            slug: "empty".to_owned(),
            emission_percent: 70.0,
            weights: Vec::new(),
            ok: true,
            error: None,
        };
        let mut empty_not_ok = empty_ok.clone();
        empty_not_ok.ok = false;
        let uids = vec![("a".to_owned(), 4u16)];

        let with_ok = aggregate_python(
            &[full.clone(), empty_ok],
            &uids,
            PY_MIN_ALLOWED_WEIGHTS,
            PY_MAX_WEIGHT_LIMIT,
        )
        .expect("ok=true");
        let without_ok = aggregate_python(
            &[full, empty_not_ok],
            &uids,
            PY_MIN_ALLOWED_WEIGHTS,
            PY_MAX_WEIGHT_LIMIT,
        )
        .expect("ok=false");

        assert_eq!(with_ok.uids, without_ok.uids);
        for (a, b) in with_ok.weights.iter().zip(&without_ok.weights) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        // And the burn really is there: 30% to the miner, 70% to uid 0.
        assert_eq!(with_ok.uids, vec![0, 4]);
        assert_eq!(with_ok.weights, vec![0.7, 0.3]);
    }

    /// Python drops hotkeys it cannot map and burns their share; so do we.
    #[test]
    fn unknown_hotkey_burns_instead_of_erroring() {
        let cid = b"c";
        let leaves = vec![score(cid, 1, 1), score(cid, 9, 1)];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 3)];
        let out =
            aggregate_python_vector(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert_eq!(out.floats.uids, vec![0, 3]);
        assert_eq!(out.floats.weights, vec![0.5, 0.5]);
    }

    /// uid 0 is the burn sink, never a miner slot.
    #[test]
    fn hotkey_on_uid_zero_is_dropped_and_burns() {
        let cid = b"c";
        let leaves = vec![score(cid, 1, 1), score(cid, 2, 1)];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 0), (hk(2), 4)];
        let out =
            aggregate_python_vector(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert_eq!(out.floats.uids, vec![0, 4]);
        assert_eq!(out.floats.weights, vec![0.5, 0.5]);
        assert_eq!(out.floats.hotkey_weights.len(), 1);
        assert_eq!(out.floats.hotkey_weights[0].0, hex_key(&hk(2)));
    }

    /// All-zero epoch: Python's zero-miner fallback, not the integer path's empty vector.
    #[test]
    fn all_zero_scores_fall_back_to_burn_vector() {
        let cid = b"c";
        let leaves = vec![score(cid, 1, 0), noscore(cid, 2)];
        let shares = vec![(cid.to_vec(), 10_000)];
        let uid_map = vec![(hk(1), 1), (hk(2), 2)];
        let out =
            aggregate_python_vector(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert_eq!(out.floats.uids, vec![0]);
        assert_eq!(out.floats.weights, vec![1.0]);
        assert_eq!(out.final_vector, vec![(0, 65_535)]);
        assert!(out.floats.hotkey_weights.is_empty());
    }

    /// Two challenges, absolute shares, remainder burns (Python vector 02 shape).
    #[test]
    fn two_challenges_absolute_shares_burn_remainder() {
        let c1 = b"c1";
        let c2 = b"c2";
        let leaves = vec![score(c1, 1, 1), score(c2, 2, 1)];
        let shares = vec![(c1.to_vec(), 3_000), (c2.to_vec(), 2_000)];
        // Shares must still sum to 10_000 on the bundle side, so pad with a third.
        let shares = {
            let mut s = shares;
            s.push((b"c3".to_vec(), 5_000));
            s
        };
        let uid_map = vec![(hk(1), 1), (hk(2), 2)];
        let out =
            aggregate_python_vector(&leaves, &shares, &uid_map, ALGORITHM_VERSION).expect("agg");
        assert_eq!(out.floats.uids, vec![0, 1, 2]);
        assert_eq!(out.floats.weights, vec![0.5, 0.3, 0.2]);
    }

    #[test]
    fn bad_algorithm_version_and_shares_are_input_errors() {
        let err =
            aggregate_python_vector(&[], &[(b"c".to_vec(), 10_000)], &[], 0).expect_err("version");
        assert_eq!(
            err,
            PythonVectorError::Input(AggregateError::UnsupportedAlgorithmVersion)
        );
        let err = aggregate_python_vector(&[], &[(b"c".to_vec(), 9_999)], &[], ALGORITHM_VERSION)
            .expect_err("shares");
        assert_eq!(
            err,
            PythonVectorError::Input(AggregateError::SharesSumNotBpsDenom)
        );
    }

    #[test]
    fn zero_miner_without_candidates_is_a_zero_miner_error() {
        // No uid_map at all: not even uid 0 padding can reach min_allowed_weights=1.
        let cid = b"c";
        let leaves = vec![score(cid, 1, 0)];
        let shares = vec![(cid.to_vec(), 10_000)];
        let out = aggregate_python_vector(&leaves, &shares, &[], ALGORITHM_VERSION).expect("agg");
        // uid 0 is always a candidate, so this still succeeds with a pure burn vector.
        assert_eq!(out.final_vector, vec![(0, 65_535)]);
    }

    #[test]
    fn hex_key_is_injective_and_lowercase() {
        assert_eq!(hex_key(&[0x00, 0xff, 0x0a]), "00ff0a");
        assert_ne!(hex_key(b"ab"), hex_key(b"ba"));
    }
}
