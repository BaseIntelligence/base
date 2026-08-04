//! Integer Elo, deterministic pairing, elimination.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::cast_possible_truncation)]

use std::collections::{BTreeMap, BTreeSet};

use design_challenge_task::{
    ELIMINATION_BOTTOM_BPS, ELIMINATION_COOLDOWN_ROUNDS, PAIR_ID_DOMAIN, SCORE_MAX,
};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Starting Elo.
pub const ELO_INITIAL: i32 = 1000;
/// Fixed K-factor (integer).
pub const ELO_K: i32 = 32;

/// One pairwise outcome (winner beat loser).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotationOutcome {
    /// Winner run id.
    pub winner_run_id: String,
    /// Loser run id.
    pub loser_run_id: String,
    /// Created ordering key.
    pub created_at_ms: u64,
    /// Stable id for tie-break.
    pub id: String,
}

/// Per-miner rating aggregate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinerRating {
    /// Hotkey.
    pub miner_hotkey: String,
    /// Elo.
    pub rating: i32,
    /// Wins.
    pub wins: u32,
    /// Losses.
    pub losses: u32,
}

/// Pairing candidate run.
#[derive(Debug, Clone)]
pub struct PairCandidate {
    /// Run id.
    pub run_id: String,
    /// Miner hotkey.
    pub miner_hotkey: String,
    /// Prompt id.
    pub prompt_id: String,
}

/// Generated pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairSpec {
    /// Pair id.
    pub id: String,
    /// Prompt.
    pub prompt_id: String,
    /// Run A.
    pub run_a_id: String,
    /// Run B.
    pub run_b_id: String,
}

/// Rating errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum RatingError {
    /// Bad input.
    #[error("rating: {0}")]
    Invalid(String),
}

/// Integer expected score × 1000 for Elo (logistic approximation table-free).
///
/// `expected_a = 1000 / (1 + 10^((rb-ra)/400))` via integer pow approx.
#[must_use]
pub fn expected_score_millis(ra: i32, rb: i32) -> i32 {
    // Use: 1000 / (1 + 10^((rb-ra)/400)) ≈ via scaled integer.
    let diff = rb.saturating_sub(ra);
    // Approximate 10^(d/400) with 2^(d*10/400*log2(10)) ≈ 2^(d/120) roughly (log2(10)≈3.32 → 400/3.32≈120).
    // Clamp for stability.
    let clamped = diff.clamp(-1200, 1200);
    let exp_approx = if clamped >= 0 {
        // 10^(x/400) ≈ (1000 + 4*x)/1000 for modest x; better: use millidiv table.
        millipow10(clamped)
    } else {
        let inv = millipow10(-clamped);
        // 1 / inv_pow → millipow returns millis of 10^; so 10^(-x) = 1000^2 / millipow(x) / 1000
        if inv == 0 {
            1000
        } else {
            1_000_000 / inv
        }
    };
    // expected = 1000 / (1 + 10^diff) = 1000 * 1000 / (1000 + exp_approx)
    let denom = 1000i64.saturating_add(i64::from(exp_approx));
    if denom <= 0 {
        return 500;
    }
    i32::try_from(1_000_000i64 / denom).unwrap_or(500)
}

/// `round(1000 * 10^(x/400))` for x in 0..=1200.
fn millipow10(x: i32) -> i32 {
    // Piecewise linear on decades of 100 Elo points.
    // 10^(0)=1, 10^(0.25)≈1.778, 10^(0.5)≈3.162, 10^(0.75)≈5.623, 10^1=10, ...
    const TABLE: &[(i32, i32)] = &[
        (0, 1000),
        (100, 1778),
        (200, 3162),
        (300, 5623),
        (400, 10000),
        (500, 17783),
        (600, 31623),
        (700, 56234),
        (800, 100000),
        (900, 177828),
        (1000, 316228),
        (1100, 562341),
        (1200, 1000000),
    ];
    let x = x.clamp(0, 1200);
    for w in TABLE.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x <= x1 {
            let t = i64::from(x - x0);
            let span = i64::from(x1 - x0).max(1);
            let y = i64::from(y0) + (i64::from(y1 - y0) * t) / span;
            return i32::try_from(y).unwrap_or(y0);
        }
    }
    1_000_000
}

/// Apply one game: winner gains, loser loses (integer Elo).
#[must_use]
pub fn apply_game(winner: i32, loser: i32) -> (i32, i32) {
    let exp_w = expected_score_millis(winner, loser);
    // score_w = 1000, delta = K * (score - expected) / 1000
    let delta_w = (ELO_K * (1000 - exp_w)) / 1000;
    let delta_l = (ELO_K * (0 - (1000 - exp_w))) / 1000;
    (
        winner.saturating_add(delta_w),
        loser.saturating_add(delta_l),
    )
}

/// Aggregate Elo over a deterministic annotation sequence.
///
/// `run_to_miner` maps run_id → miner_hotkey. Annotations must be pre-sorted
/// by `(created_at_ms, id)`.
pub fn aggregate_elo(
    run_to_miner: &BTreeMap<String, String>,
    outcomes: &[AnnotationOutcome],
) -> Result<Vec<MinerRating>, RatingError> {
    let mut ratings: BTreeMap<String, MinerRating> = BTreeMap::new();
    let ensure = |map: &mut BTreeMap<String, MinerRating>, hk: &str| {
        map.entry(hk.to_owned()).or_insert_with(|| MinerRating {
            miner_hotkey: hk.to_owned(),
            rating: ELO_INITIAL,
            wins: 0,
            losses: 0,
        });
    };
    for o in outcomes {
        let w_hk = run_to_miner
            .get(&o.winner_run_id)
            .ok_or_else(|| RatingError::Invalid(format!("unknown winner {}", o.winner_run_id)))?;
        let l_hk = run_to_miner
            .get(&o.loser_run_id)
            .ok_or_else(|| RatingError::Invalid(format!("unknown loser {}", o.loser_run_id)))?;
        if w_hk == l_hk {
            continue;
        }
        ensure(&mut ratings, w_hk);
        ensure(&mut ratings, l_hk);
        let rw = ratings[w_hk].rating;
        let rl = ratings[l_hk].rating;
        let (nw, nl) = apply_game(rw, rl);
        if let Some(r) = ratings.get_mut(w_hk) {
            r.rating = nw;
            r.wins = r.wins.saturating_add(1);
        }
        if let Some(r) = ratings.get_mut(l_hk) {
            r.rating = nl;
            r.losses = r.losses.saturating_add(1);
        }
    }
    Ok(ratings.into_values().collect())
}

/// Normalize ratings onto `[0, SCORE_MAX]` lattice (integer).
#[must_use]
pub fn ratings_to_scores(ratings: &[MinerRating]) -> BTreeMap<String, u64> {
    if ratings.is_empty() {
        return BTreeMap::new();
    }
    let min_r = ratings.iter().map(|r| r.rating).min().unwrap_or(0);
    let max_r = ratings.iter().map(|r| r.rating).max().unwrap_or(0);
    let span = (max_r - min_r).max(1);
    ratings
        .iter()
        .map(|r| {
            let num = i64::from(r.rating - min_r) * i64::try_from(SCORE_MAX).unwrap_or(1);
            let v = u64::try_from(num / i64::from(span)).unwrap_or(0);
            (r.miner_hotkey.clone(), v.min(SCORE_MAX))
        })
        .collect()
}

/// Build round-robin pairs per prompt (order by run_id for determinism).
#[must_use]
pub fn build_pairs(round_id: u64, candidates: &[PairCandidate]) -> Vec<PairSpec> {
    let mut by_prompt: BTreeMap<String, Vec<&PairCandidate>> = BTreeMap::new();
    for c in candidates {
        by_prompt.entry(c.prompt_id.clone()).or_default().push(c);
    }
    let mut out = Vec::new();
    for (prompt_id, mut group) in by_prompt {
        group.sort_by_key(|c| &c.run_id);
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let a = group[i];
                let b = group[j];
                if a.miner_hotkey == b.miner_hotkey {
                    continue;
                }
                let id = pair_id(round_id, &prompt_id, &a.run_id, &b.run_id);
                out.push(PairSpec {
                    id,
                    prompt_id: prompt_id.clone(),
                    run_a_id: a.run_id.clone(),
                    run_b_id: b.run_id.clone(),
                });
            }
        }
    }
    out.sort_by(|x, y| x.id.cmp(&y.id));
    out
}

/// Pair id digest.
#[must_use]
pub fn pair_id(round_id: u64, prompt_id: &str, run_a: &str, run_b: &str) -> String {
    let (a, b) = if run_a <= run_b {
        (run_a, run_b)
    } else {
        (run_b, run_a)
    };
    let mut h = Sha256::new();
    h.update(PAIR_ID_DOMAIN);
    h.update(round_id.to_be_bytes());
    h.update(prompt_id.as_bytes());
    h.update(a.as_bytes());
    h.update(b.as_bytes());
    hex::encode(h.finalize())
}

/// Bottom 20% (at least 1) miners to eliminate; returns hotkeys + until_round.
#[must_use]
pub fn elimination_set(ratings: &[MinerRating], current_round: u64) -> BTreeSet<(String, u64)> {
    if ratings.is_empty() {
        return BTreeSet::new();
    }
    let mut sorted = ratings.to_vec();
    sorted.sort_by(|a, b| {
        a.rating
            .cmp(&b.rating)
            .then(a.miner_hotkey.cmp(&b.miner_hotkey))
    });
    let n = sorted.len();
    let mut k = (n as u64 * u64::from(ELIMINATION_BOTTOM_BPS) / 10_000) as usize;
    k = k.max(1).min(n);
    let until = current_round.saturating_add(ELIMINATION_COOLDOWN_ROUNDS);
    sorted
        .into_iter()
        .take(k)
        .map(|r| (r.miner_hotkey, until))
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn elo_deterministic() {
        let mut map = BTreeMap::new();
        map.insert("r1".into(), "m1".into());
        map.insert("r2".into(), "m2".into());
        let outcomes = vec![
            AnnotationOutcome {
                winner_run_id: "r1".into(),
                loser_run_id: "r2".into(),
                created_at_ms: 1,
                id: "a".into(),
            },
            AnnotationOutcome {
                winner_run_id: "r1".into(),
                loser_run_id: "r2".into(),
                created_at_ms: 2,
                id: "b".into(),
            },
        ];
        let a = aggregate_elo(&map, &outcomes).unwrap();
        let b = aggregate_elo(&map, &outcomes).unwrap();
        assert_eq!(a, b);
        let m1 = a.iter().find(|r| r.miner_hotkey == "m1").unwrap();
        let m2 = a.iter().find(|r| r.miner_hotkey == "m2").unwrap();
        assert!(m1.rating > m2.rating);
    }

    #[test]
    fn elimination_bottom_20() {
        let ratings: Vec<_> = (0..10)
            .map(|i| MinerRating {
                miner_hotkey: format!("m{i:02}"),
                rating: 1000 + i * 10,
                wins: 0,
                losses: 0,
            })
            .collect();
        let elim = elimination_set(&ratings, 5);
        assert_eq!(elim.len(), 2); // 20% of 10
        assert!(elim.contains(&("m00".into(), 9)));
        assert!(elim.contains(&("m01".into(), 9)));
    }

    #[test]
    fn pairing_skips_same_miner() {
        let c = vec![
            PairCandidate {
                run_id: "a".into(),
                miner_hotkey: "m1".into(),
                prompt_id: "p".into(),
            },
            PairCandidate {
                run_id: "b".into(),
                miner_hotkey: "m1".into(),
                prompt_id: "p".into(),
            },
            PairCandidate {
                run_id: "c".into(),
                miner_hotkey: "m2".into(),
                prompt_id: "p".into(),
            },
        ];
        let pairs = build_pairs(1, &c);
        assert_eq!(pairs.len(), 2); // a-c, b-c
    }
}
