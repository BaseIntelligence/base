//! Bit-for-bit port of the authoritative Python weight aggregation.
//!
//! Upstream: `src/base/master/aggregator.py` at BASE commit
//! `8249563774ee2e71c41ae2cfac182ff32aa35dd1` (`origin/main`), the code serving
//! <https://chain.joinbase.ai>. That Python implementation **is** the authority for the
//! served vector, so this module reproduces it exactly — floats included. The integer
//! Hamilton path in [`crate::aggregate`] is a different algorithm and stays untouched.
//!
//! # Why order matters
//!
//! Python `dict` preserves insertion order and `sum()` over `float` is **not**
//! associative, so `{a: x, b: y}` and `{b: y, a: x}` can produce different totals in the
//! last ulp. Every map here therefore keeps caller insertion order ([`OrderedMap`]);
//! only the final vector is sorted by uid, exactly like `sorted(normalized.items())`.
//!
//! # Rounding
//!
//! Python's `round()` on a `float` is round-half-to-**even**. `f64::round` is
//! half-away-from-zero and would give `round(0.3 * 65535) == 19661` instead of `19660`.
//! [`to_chain_u16`] uses [`f64::round_ties_even`].

use std::collections::BTreeMap;

/// Bittensor weights are encoded as u16; `65535/65535 == 1.0` is the full share.
pub const CHAIN_U16_MAX: u16 = 65_535;
/// uid 0 == the validator/subnet owner; unallocated mass burns here.
pub const ZERO_MINER_BURN_UID: u16 = 0;
/// Float tolerance: residual miner mass / burn at or below this is treated as 0.
pub const EPS: f64 = 1e-12;

/// Python's two-argument `max(a, b)`, which returns `b` only when `b > a`.
///
/// `f64::max` disagrees on NaN (`f64::NAN.max(0.0) == 0.0`, `max(nan, 0.0) is nan`).
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// One challenge's contribution, mirroring `base.schemas.weights.ChallengeWeightsResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChallengeWeightsResult {
    /// Challenge slug. Duplicate slugs collapse exactly like a Python `dict` key.
    pub slug: String,
    /// Absolute share of 100 owned by this challenge (not renormalized across challenges).
    pub emission_percent: f64,
    /// Raw per-hotkey weights in caller insertion order.
    pub weights: Vec<(String, f64)>,
    /// Whether the challenge produced a usable result; `false` entries are dropped.
    pub ok: bool,
    /// Optional error string carried alongside `ok == false` (unused by aggregation).
    pub error: Option<String>,
}

/// Final aggregated vector, mirroring `base.schemas.weights.FinalWeights`.
#[derive(Debug, Clone, PartialEq)]
pub struct FinalWeights {
    /// uids ascending (`sorted(normalized.items())`).
    pub uids: Vec<u16>,
    /// Weights aligned with [`FinalWeights::uids`]; sums to ~1.0.
    pub weights: Vec<f64>,
    /// Python's `kept_hotkeys`, in first-appearance order. Empty on the zero-miner path.
    pub hotkey_weights: Vec<(String, f64)>,
}

/// No chain-valid weight vector can be built for the zero-miner case.
///
/// Message text matches `ZeroMinerWeightError` raised by
/// `base.master.aggregator.build_zero_miner_weights` verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZeroMinerWeightError {
    /// `max_weight_limit <= 0`: no positive weight is admissible.
    NonPositiveMaxWeightLimit {
        /// The offending limit (u16 space).
        max_weight_limit: u16,
    },
    /// Fewer usable uids than the chain minimum / per-weight cap requires.
    NotEnoughUids {
        /// uids needed to satisfy both constraints.
        required: usize,
        /// Chain `min_allowed_weights` as passed in.
        min_allowed_weights: u32,
        /// Chain `max_weight_limit` as passed in.
        max_weight_limit: u16,
        /// Usable candidate uids actually available.
        available: usize,
    },
}

impl core::fmt::Display for ZeroMinerWeightError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveMaxWeightLimit { max_weight_limit } => write!(
                f,
                "max_weight_limit={max_weight_limit} admits no positive weight"
            ),
            Self::NotEnoughUids {
                required,
                min_allowed_weights,
                max_weight_limit,
                available,
            } => write!(
                f,
                "cannot build a chain-valid zero-miner weight vector: need {required} uids \
                 (min_allowed_weights={min_allowed_weights}, \
                 max_weight_limit={max_weight_limit}) but only {available} usable uid(s) \
                 available"
            ),
        }
    }
}

impl std::error::Error for ZeroMinerWeightError {}

/// Insertion-ordered map with `O(log n)` lookup — the Rust stand-in for a Python `dict`.
///
/// Re-assigning an existing key keeps its original position, matching `dict.__setitem__`.
#[derive(Debug, Clone)]
struct OrderedMap<K: Ord + Clone, V: Copy> {
    entries: Vec<(K, V)>,
    index: BTreeMap<K, usize>,
}

impl<K: Ord + Clone, V: Copy> OrderedMap<K, V> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            index: BTreeMap::new(),
        }
    }

    fn get(&self, key: &K) -> Option<V> {
        self.index
            .get(key)
            .and_then(|&i| self.entries.get(i))
            .map(|e| e.1)
    }

    fn set(&mut self, key: &K, value: V) {
        if let Some(&i) = self.index.get(key) {
            if let Some(slot) = self.entries.get_mut(i) {
                slot.1 = value;
            }
            return;
        }
        self.index.insert(key.clone(), self.entries.len());
        self.entries.push((key.clone(), value));
    }

    fn iter(&self) -> impl Iterator<Item = &(K, V)> {
        self.entries.iter()
    }
}

impl<K: Ord + Clone> OrderedMap<K, f64> {
    /// `defaultdict(float)` semantics: `map[key] += value`, inserting `0.0` on first touch.
    fn add(&mut self, key: &K, value: f64) {
        let current = self.get(key).unwrap_or(0.0);
        self.set(key, current + value);
    }

    /// `sum(map.values())` in insertion order — see [`py_sum`].
    fn sum(&self) -> f64 {
        py_sum(self.entries.iter().map(|e| e.1))
    }
}

/// `CPython`'s builtin `sum()` over floats.
///
/// **Not** a naive fold: since `CPython` 3.12 `sum()` uses the improved Kahan-Babuska
/// algorithm by Neumaier, so `sum([0.1] * 10)` is exactly `1.0` while a naive left fold
/// gives `0.9999999999999999`. Matching the served vector bit-for-bit therefore requires
/// reproducing the compensation, including the final `if c != 0.0` guard that keeps
/// `CPython`'s negative-zero handling. Ground truth: `CPython` 3.12.3
/// `Python/bltinmodule.c`, `builtin_sum_impl`.
///
/// A `CPython` 3.11 or older service would sum naively; if the deployment ever downgrades,
/// this function is the single place that has to change.
fn py_sum<I: IntoIterator<Item = f64>>(values: I) -> f64 {
    let mut result = 0.0_f64;
    let mut c = 0.0_f64;
    for x in values {
        let t = result + x;
        if result.abs() >= x.abs() {
            c += (result - t) + x;
        } else {
            c += (x - t) + result;
        }
        result = t;
    }
    if c != 0.0 {
        result += c;
    }
    result
}

/// Port of `_clean_weights`: drop non-finite and non-positive weights, keep order.
fn clean_weights(raw: &[(String, f64)]) -> OrderedMap<String, f64> {
    let mut cleaned = OrderedMap::new();
    for (hotkey, value) in raw {
        let weight = *value;
        if !weight.is_finite() {
            continue;
        }
        if weight <= 0.0 {
            continue;
        }
        cleaned.set(hotkey, weight);
    }
    cleaned
}

/// Port of `normalize_weights`: clean, then divide by the insertion-ordered total.
///
/// Returns an empty vector when the surviving mass is `<= 0` (Python returns `{}`).
#[must_use]
pub fn normalize_weights(raw: &[(String, f64)]) -> Vec<(String, f64)> {
    let cleaned = clean_weights(raw);
    let total = cleaned.sum();
    if total <= 0.0 {
        return Vec::new();
    }
    cleaned
        .iter()
        .map(|(hotkey, value)| (hotkey.clone(), value / total))
        .collect()
}

/// Port of `build_zero_miner_weights` with the caller defaults used by the aggregator
/// (`burn_uid = 0`, `allow_burn_self_vote = True`).
///
/// `available_uids` is the raw `hotkey_to_uid.values()` sequence; it is deduplicated and
/// sorted here exactly as `sorted(set(...))` does.
///
/// # Errors
///
/// [`ZeroMinerWeightError`] when no chain-valid fallback vector exists.
pub fn build_zero_miner_weights(
    min_allowed_weights: u32,
    max_weight_limit: u16,
    available_uids: &[u16],
) -> Result<Vec<(u16, f64)>, ZeroMinerWeightError> {
    if max_weight_limit == 0 {
        return Err(ZeroMinerWeightError::NonPositiveMaxWeightLimit { max_weight_limit });
    }

    let max_fraction = (f64::from(max_weight_limit) / f64::from(CHAIN_U16_MAX)).min(1.0);

    let mut candidates: Vec<u16> = vec![ZERO_MINER_BURN_UID];
    let mut sorted: Vec<u16> = available_uids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    for uid in sorted {
        if uid == ZERO_MINER_BURN_UID {
            continue;
        }
        candidates.push(uid);
    }

    // `math.ceil(1.0 / max_fraction)`; max_fraction is in (0, 1] so the quotient is in
    // [1, 65535] and the cast cannot lose information.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let cap_required = (1.0_f64 / max_fraction).ceil() as usize;
    let required = usize::try_from(min_allowed_weights)
        .unwrap_or(usize::MAX)
        .max(1)
        .max(cap_required);

    if candidates.len() < required {
        return Err(ZeroMinerWeightError::NotEnoughUids {
            required,
            min_allowed_weights,
            max_weight_limit,
            available: candidates.len(),
        });
    }

    #[allow(clippy::cast_precision_loss)]
    let weight = 1.0_f64 / required as f64;
    Ok(candidates
        .into_iter()
        .take(required)
        .map(|uid| (uid, weight))
        .collect())
}

/// Port of `aggregate_challenge_weights` — the served aggregation.
///
/// `emission_percent` is an **absolute** share of 100: an OK challenge with
/// `emission_percent = e` owns `e/100` of the vector. Anything not landing on a real
/// miner burns to [`ZERO_MINER_BURN_UID`]. Over-allocation (`sum > 100`) is scaled back
/// to exactly 1.0, so there is no burn in that case.
///
/// `hotkey_to_uid` is a `dict` in Python: duplicate hotkeys collapse with **last value
/// wins**, and `.values()` (used for the zero-miner fallback) sees the deduplicated set.
///
/// # Errors
///
/// [`ZeroMinerWeightError`] when no miner scored and no chain-valid burn vector exists.
pub fn aggregate_python(
    results: &[ChallengeWeightsResult],
    hotkey_to_uid: &[(String, u16)],
    min_allowed_weights: u32,
    max_weight_limit: u16,
) -> Result<FinalWeights, ZeroMinerWeightError> {
    let mut hk_to_uid: OrderedMap<String, u16> = OrderedMap::new();
    for (hotkey, uid) in hotkey_to_uid {
        hk_to_uid.set(hotkey, *uid);
    }

    let ok_results: Vec<&ChallengeWeightsResult> = results.iter().filter(|r| r.ok).collect();

    // frac: dict keyed by slug — duplicate slugs collapse, last emission_percent wins.
    let mut frac: OrderedMap<String, f64> = OrderedMap::new();
    for result in &ok_results {
        frac.set(&result.slug, py_max(result.emission_percent, 0.0) / 100.0);
    }
    let alloc_total = frac.sum();
    if alloc_total > 1.0 {
        let scaled: Vec<(String, f64)> = frac
            .iter()
            .map(|(slug, value)| (slug.clone(), value / alloc_total))
            .collect();
        for (slug, value) in &scaled {
            frac.set(slug, *value);
        }
    }

    let mut hotkey_scores: OrderedMap<String, f64> = OrderedMap::new();
    for result in &ok_results {
        let share = frac.get(&result.slug).unwrap_or(0.0);
        if share <= 0.0 {
            continue;
        }
        for (hotkey, weight) in normalize_weights(&result.weights) {
            hotkey_scores.add(&hotkey, share * weight);
        }
    }

    let mut uid_scores: OrderedMap<u16, f64> = OrderedMap::new();
    let mut kept_hotkeys: Vec<(String, f64)> = Vec::new();
    for (hotkey, score) in hotkey_scores.iter() {
        let Some(uid) = hk_to_uid.get(hotkey) else {
            continue;
        };
        if uid == ZERO_MINER_BURN_UID {
            continue;
        }
        uid_scores.add(&uid, *score);
        kept_hotkeys.push((hotkey.clone(), *score));
    }

    let miner_total = uid_scores.sum();
    let normalized: Vec<(u16, f64)> = if miner_total <= EPS {
        kept_hotkeys.clear();
        let available: Vec<u16> = hk_to_uid.iter().map(|e| e.1).collect();
        build_zero_miner_weights(min_allowed_weights, max_weight_limit, &available)?
    } else {
        let mut final_scores = uid_scores;
        let burn = 1.0 - miner_total;
        if burn > EPS {
            final_scores.add(&ZERO_MINER_BURN_UID, burn);
        }
        let total = final_scores.sum();
        final_scores
            .iter()
            .map(|(uid, value)| (*uid, value / total))
            .collect()
    };

    let mut ordered = normalized;
    ordered.sort_by_key(|(uid, _)| *uid);

    Ok(FinalWeights {
        uids: ordered.iter().map(|(uid, _)| *uid).collect(),
        weights: ordered.iter().map(|(_, w)| *w).collect(),
        hotkey_weights: kept_hotkeys,
    })
}

/// Encode normalized float weights into the chain u16 space: `round(w * 65535)`.
///
/// Python's `round()` is round-half-to-even, so `0.5 -> 32768` but `0.3 -> 19660`
/// (`0.3 * 65535` is exactly `19660.5` in binary64). Out-of-range inputs saturate;
/// aggregated weights are always in `[0, 1]` so saturation is unreachable in practice.
#[must_use]
pub fn to_chain_u16(weights: &[f64]) -> Vec<u16> {
    weights
        .iter()
        .map(|w| {
            let scaled = (w * f64::from(CHAIN_U16_MAX)).round_ties_even();
            if scaled.is_nan() || scaled <= 0.0 {
                return 0;
            }
            if scaled >= f64::from(CHAIN_U16_MAX) {
                return CHAIN_U16_MAX;
            }
            // 0 < scaled < 65535 and integral: the cast is exact.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            {
                scaled as u16
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn cr(slug: &str, pct: f64, weights: &[(&str, f64)]) -> ChallengeWeightsResult {
        ChallengeWeightsResult {
            slug: slug.to_owned(),
            emission_percent: pct,
            weights: weights.iter().map(|(h, w)| ((*h).to_owned(), *w)).collect(),
            ok: true,
            error: None,
        }
    }

    fn uids(pairs: &[(&str, u16)]) -> Vec<(String, u16)> {
        pairs.iter().map(|(h, u)| ((*h).to_owned(), *u)).collect()
    }

    #[test]
    fn py_sum_matches_cpython_neumaier_compensation() {
        // CPython 3.12: sum([0.1] * 10) == 1.0; a naive left fold gives 0.9999999999999999.
        let tenths = [0.1_f64; 10];
        assert_eq!(py_sum(tenths.iter().copied()).to_bits(), 1.0_f64.to_bits());
        assert_ne!(
            tenths.iter().fold(0.0_f64, |a, b| a + b).to_bits(),
            1.0_f64.to_bits()
        );
        // Classic catastrophic-cancellation case: naive gives 0.0, Neumaier gives 2.0.
        assert_eq!(
            py_sum([1e100, 1.0, -1e100, 1.0]).to_bits(),
            2.0_f64.to_bits()
        );
        assert_eq!(py_sum(std::iter::empty()).to_bits(), 0.0_f64.to_bits());
        // Python's `sum()` starts from int 0, so a lone -0.0 comes back as +0.0.
        assert!(py_sum([-0.0_f64]).is_sign_positive());
    }

    #[test]
    fn bankers_rounding_matches_python_round() {
        // Python: round(0.5 * 65535) == 32768, round(0.3 * 65535) == 19660.
        assert_eq!(to_chain_u16(&[0.5]), vec![32_768]);
        assert_eq!(to_chain_u16(&[0.3]), vec![19_660]);
        // f64::round would give 19661 here.
        assert!((0.3_f64 * 65_535.0).round() - 19_661.0 == 0.0);
        assert!((0.3_f64 * 65_535.0).round_ties_even() - 19_660.0 == 0.0);
        assert_eq!(to_chain_u16(&[0.0, 1.0]), vec![0, 65_535]);
    }

    #[test]
    fn single_challenge_two_miners_equal() {
        let out = aggregate_python(
            &[cr("c", 100.0, &[("a", 1.0), ("b", 1.0)])],
            &uids(&[("a", 1), ("b", 2)]),
            1,
            CHAIN_U16_MAX,
        )
        .unwrap();
        assert_eq!(out.uids, vec![1, 2]);
        assert_eq!(out.weights, vec![0.5, 0.5]);
        assert_eq!(to_chain_u16(&out.weights), vec![32_768, 32_768]);
    }

    #[test]
    fn unallocated_share_burns_to_uid_zero() {
        let out = aggregate_python(
            &[cr("c", 30.0, &[("a", 1.0)])],
            &uids(&[("a", 30)]),
            1,
            CHAIN_U16_MAX,
        )
        .unwrap();
        assert_eq!(out.uids, vec![0, 30]);
        assert_eq!(out.weights, vec![0.7, 0.3]);
    }

    #[test]
    fn over_allocation_scales_back_without_burn() {
        let out = aggregate_python(
            &[cr("a", 80.0, &[("ha", 1.0)]), cr("b", 80.0, &[("hb", 1.0)])],
            &uids(&[("ha", 1), ("hb", 2)]),
            1,
            CHAIN_U16_MAX,
        )
        .unwrap();
        assert_eq!(out.uids, vec![1, 2]);
        assert_eq!(out.weights, vec![0.5, 0.5]);
    }

    #[test]
    fn uid_zero_hotkey_is_dropped_and_burns() {
        let out = aggregate_python(
            &[cr("c", 100.0, &[("val", 1.0), ("a", 1.0)])],
            &uids(&[("val", 0), ("a", 4)]),
            1,
            CHAIN_U16_MAX,
        )
        .unwrap();
        assert_eq!(out.uids, vec![0, 4]);
        assert_eq!(out.weights, vec![0.5, 0.5]);
        assert_eq!(out.hotkey_weights, vec![("a".to_owned(), 0.5)]);
    }

    #[test]
    fn clean_weights_filters_nonfinite_and_nonpositive() {
        let raw = vec![
            ("a".to_owned(), 1.0),
            ("neg".to_owned(), -1.0),
            ("zero".to_owned(), 0.0),
            ("nan".to_owned(), f64::NAN),
            ("inf".to_owned(), f64::INFINITY),
            ("b".to_owned(), 3.0),
        ];
        assert_eq!(
            normalize_weights(&raw),
            vec![("a".to_owned(), 0.25), ("b".to_owned(), 0.75)]
        );
    }

    #[test]
    fn zero_miner_fallback_pads_for_min_allowed_weights() {
        let out = aggregate_python(
            &[cr("empty", 100.0, &[])],
            &uids(&[("val", 0), ("a", 5), ("b", 7)]),
            3,
            CHAIN_U16_MAX,
        )
        .unwrap();
        assert_eq!(out.uids, vec![0, 5, 7]);
        assert!(out.hotkey_weights.is_empty());
        let third = 1.0_f64 / 3.0;
        assert_eq!(out.weights, vec![third, third, third]);
    }

    #[test]
    fn zero_miner_errors_when_candidates_insufficient() {
        let err = aggregate_python(
            &[cr("empty", 100.0, &[])],
            &uids(&[("val", 0)]),
            3,
            CHAIN_U16_MAX,
        )
        .expect_err("not enough uids");
        assert_eq!(
            err.to_string(),
            "cannot build a chain-valid zero-miner weight vector: need 3 uids \
             (min_allowed_weights=3, max_weight_limit=65535) but only 1 usable uid(s) available"
        );
    }

    #[test]
    fn max_weight_limit_forces_extra_padding_uids() {
        // 20000/65535 = 0.30518 -> ceil(1/0.30518) = 4 uids required.
        let out = aggregate_python(
            &[cr("empty", 100.0, &[])],
            &uids(&[("v", 0), ("a", 1), ("b", 2), ("c", 3), ("d", 4)]),
            1,
            20_000,
        )
        .unwrap();
        assert_eq!(out.uids, vec![0, 1, 2, 3]);
        assert_eq!(out.weights, vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn zero_max_weight_limit_errors() {
        let err =
            build_zero_miner_weights(1, 0, &[0, 1]).expect_err("no positive weight admissible");
        assert_eq!(
            err.to_string(),
            "max_weight_limit=0 admits no positive weight"
        );
    }

    #[test]
    fn ordered_map_reassignment_keeps_position() {
        let mut m: OrderedMap<String, f64> = OrderedMap::new();
        m.set(&"a".to_owned(), 1.0);
        m.set(&"b".to_owned(), 2.0);
        m.set(&"a".to_owned(), 9.0);
        let keys: Vec<&str> = m.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["a", "b"]);
        assert_eq!(m.get(&"a".to_owned()), Some(9.0));
    }
}
