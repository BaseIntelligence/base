//! Sealed weight vector + trust-root emission shares → site contract.
//!
//! The arena frames carry no invented emission figures: the split comes from
//! the owner-signed trust root (always available) and the per-hotkey weights
//! from the latest sealed bundle (only when a seal exists). A missing or
//! undecodable bundle degrades to `sealed: false` with an empty hotkey list —
//! the same fail-closed posture as `/v1/weights/latest`, without duplicating
//! its burn payload.

use bundle::{EpochBundleBodyV1, EpochBundleV1, ScoreOrAbsence};
use site_types::{EmissionShare, HotkeyWeight, SiteWeights};
use trustroot::{ChallengesBody, BPS_DENOM};
use weights_api::{SealRecord, BURN_UID};

/// Configured emission split from the trust root (`slug → 0..1`), sorted by slug.
#[must_use]
pub fn configured_shares(challenges: Option<&ChallengesBody>) -> Vec<(String, f64)> {
    let mut shares: Vec<(String, f64)> = challenges
        .map(|c| {
            c.emission_shares()
                .into_iter()
                .map(|(id, bps)| {
                    (
                        String::from_utf8_lossy(&id).into_owned(),
                        f64::from(bps) / f64::from(BPS_DENOM),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    shares.sort_by(|a, b| a.0.cmp(&b.0));
    shares
}

/// Sealed bundle projection used by the site contract.
#[derive(Debug, Clone)]
pub struct SealedView {
    /// Sealed epoch.
    pub epoch: u64,
    /// ISO-8601 seal time.
    pub computed_at: String,
    /// Per-hotkey weights (SS58), burn uid excluded.
    pub hotkeys: Vec<(String, f64)>,
    /// Fraction of the vector on the burn uid, 0..1.
    pub burn_share: f64,
}

/// Project the latest sealed bundle: per-hotkey weights + burn fraction.
///
/// `None` when no seal exists or the stored bytes cannot be decoded.
#[must_use]
pub fn sealed_view(latest: Option<(Vec<u8>, SealRecord)>) -> Option<SealedView> {
    let (bytes, seal) = latest?;
    let bundle = EpochBundleV1::decode_bytes(&bytes).ok()?;
    let latest = weights_api::build_latest(&bundle, seal);
    let total: f64 = latest.weights.iter().sum();
    let burn = latest
        .uids
        .iter()
        .position(|u| *u == weights_api::BURN_UID)
        .and_then(|i| latest.weights.get(i))
        .copied()
        .unwrap_or(0.0);
    let burn_share = if total > 0.0 { burn / total } else { 1.0 };
    let hotkeys: Vec<(String, f64)> = latest
        .hotkey_weights
        .entries()
        .iter()
        .map(|(hk, w)| (hk.clone(), *w))
        .collect();
    Some(SealedView {
        epoch: latest.epoch?,
        computed_at: latest.computed_at,
        hotkeys,
        burn_share,
    })
}

/// Full `/v1/site/weights` payload.
#[must_use]
pub fn site_weights(
    netuid: u16,
    challenges: Option<&ChallengesBody>,
    latest: Option<(Vec<u8>, SealRecord)>,
) -> SiteWeights {
    let emission_shares = configured_shares(challenges)
        .into_iter()
        .map(|(arena, share)| EmissionShare { arena, share })
        .collect();
    match sealed_view(latest) {
        Some(view) => {
            let mut hotkey_weights: Vec<HotkeyWeight> = view
                .hotkeys
                .into_iter()
                .map(|(hotkey, weight)| HotkeyWeight { hotkey, weight })
                .collect();
            hotkey_weights.sort_by(|a, b| {
                b.weight
                    .partial_cmp(&a.weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            SiteWeights {
                sealed: true,
                epoch: Some(view.epoch),
                netuid,
                emission_shares,
                hotkey_weights,
                burn_share: view.burn_share,
                computed_at: Some(view.computed_at),
            }
        }
        None => SiteWeights {
            sealed: false,
            epoch: None,
            netuid,
            emission_shares,
            hotkey_weights: Vec::new(),
            burn_share: 1.0,
            computed_at: None,
        },
    }
}

/// Effective on-chain weight of one arena for the latest seal.
///
/// This is the fraction of the sealed vector that challenge `slug` actually
/// placed on real miners (not uid-0 burn). It equals the configured trust-root
/// share when that challenge's positive scores all map to non-burn UIDs, and
/// is lower (possibly `0`) when its share burns — e.g. empty / unknown-hotkey
/// leaves. Scaling every arena by the *global* burn fraction is wrong: when
/// design burns 50% and prism lands 50%, prism must stay `0.5`, not `0.25`.
///
/// Returns `0` when nothing is sealed or the bundle cannot be decoded.
#[must_use]
pub fn arena_weight(
    challenges: Option<&ChallengesBody>,
    latest: Option<(Vec<u8>, SealRecord)>,
    slug: &str,
) -> f64 {
    // Unknown to the trust root → never invent a weight (coding / paused arenas).
    if configured_shares(challenges).iter().all(|(s, _)| s != slug) {
        return 0.0;
    }
    let Some((bytes, _)) = latest else {
        return 0.0;
    };
    let Ok(bundle) = EpochBundleV1::decode_bytes(&bytes) else {
        return 0.0;
    };
    delivered_arena_weight(&bundle.body, slug)
}

/// Mass of the sealed vector attributable to `slug`'s scored, mapped miners.
fn delivered_arena_weight(body: &EpochBundleBodyV1, slug: &str) -> f64 {
    let slug_bytes = slug.as_bytes();
    let Some((_, bps)) = body
        .emission_shares
        .iter()
        .find(|(id, _)| id.as_slice() == slug_bytes)
    else {
        return 0.0;
    };
    let share = f64::from(*bps) / f64::from(BPS_DENOM);
    if share <= 0.0 {
        return 0.0;
    }

    // Collapse duplicate hotkeys (last-writer in aggregate is sum-via-BTreeMap).
    let mut by_hotkey: Vec<([u8; 32], u64)> = Vec::new();
    for leaf in &body.leaves {
        if leaf.challenge_id.as_slice() != slug_bytes {
            continue;
        }
        let ScoreOrAbsence::Score { value } = leaf.score_or_absence else {
            continue;
        };
        if value == 0 {
            continue;
        }
        if let Some((_, acc)) = by_hotkey
            .iter_mut()
            .find(|(hk, _)| *hk == leaf.miner_hotkey)
        {
            *acc = acc.saturating_add(value);
        } else {
            by_hotkey.push((leaf.miner_hotkey, value));
        }
    }
    let raw_total: u128 = by_hotkey.iter().map(|(_, v)| u128::from(*v)).sum();
    if raw_total == 0 {
        return 0.0;
    }

    // Same within-challenge normalisation as aggregate_python: each positive
    // score owns raw/total of the challenge share; unmapped / burn-uid miners
    // drop their slice (it burns), so delivered ≤ configured share.
    #[allow(clippy::cast_precision_loss)]
    let total_f = raw_total as f64;
    by_hotkey.into_iter().fold(0.0, |acc, (hk, raw)| {
        let Some(uid) = body
            .uid_map
            .iter()
            .find(|(k, _)| *k == hk)
            .map(|(_, uid)| *uid)
        else {
            return acc;
        };
        if uid == BURN_UID {
            return acc;
        }
        #[allow(clippy::cast_precision_loss)]
        let w = (raw as f64) / total_f;
        acc + share * w
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use bundle::{EpochBundleBodyV1, LeafV1};
    use trustroot::{ChallengeEntry, ParticipantPolicy};

    fn entry(id: &str, bps: u16) -> ChallengeEntry {
        ChallengeEntry {
            id: id.as_bytes().to_vec(),
            public_key: [7u8; 32],
            emission_share_bps: bps,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }
    }

    fn leaf(challenge: &str, hk: u8) -> LeafV1 {
        leaf_score(challenge, hk, 42)
    }

    fn leaf_score(challenge: &str, hk: u8, value: u64) -> LeafV1 {
        LeafV1 {
            challenge_id: challenge.as_bytes().to_vec(),
            miner_hotkey: [hk; 32],
            epoch: 7,
            score_or_absence: bundle::ScoreOrAbsence::Score { value },
            challenge_sig: [0u8; 64],
        }
    }

    fn leaf_absent(challenge: &str, hk: u8) -> LeafV1 {
        LeafV1 {
            challenge_id: challenge.as_bytes().to_vec(),
            miner_hotkey: [hk; 32],
            epoch: 7,
            score_or_absence: bundle::ScoreOrAbsence::NoScore {
                reason: bundle::NoScoreReasonCode::NotAttempted,
            },
            challenge_sig: [0u8; 64],
        }
    }

    fn trust_root() -> ChallengesBody {
        ChallengesBody {
            challenges: vec![entry("design", 5_000), entry("prism", 5_000)],
        }
    }

    fn sealed_bundle() -> (Vec<u8>, SealRecord) {
        let body = EpochBundleBodyV1 {
            protocol_version: 1,
            epoch: 7,
            netuid: 100,
            block_b: 4242,
            block_hash: [1u8; 32],
            metagraph_root: [2u8; 32],
            algorithm_version: bundle::ALGORITHM_VERSION,
            emission_shares: vec![(b"design".to_vec(), 5_000), (b"prism".to_vec(), 5_000)],
            measurements_digest: [3u8; 32],
            uid_map: vec![([0xAAu8; 32], 0), ([0xBBu8; 32], 157)],
            leaves: vec![leaf("design", 0xBB), leaf("prism", 0xBB)],
            merkle_root: [4u8; 32],
            final_vector: vec![(0, 32_767), (157, 32_768)],
            gateway_hotkey: [5u8; 32],
        };
        let bytes = EpochBundleV1 {
            body,
            gateway_sig: [6u8; 64],
        }
        .encode_bytes();
        (
            bytes,
            SealRecord {
                sealed_at_micros: 1_785_572_565_992_448,
                revision: 1,
            },
        )
    }

    #[test]
    fn shares_come_from_trust_root() {
        let shares = configured_shares(Some(&trust_root()));
        assert_eq!(shares.len(), 2);
        assert!((shares[0].1 - 0.5).abs() < f64::EPSILON);
        assert_eq!(shares[0].0, "design");
        assert!((shares[1].1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn sealed_view_splits_burn_from_hotkeys() {
        let (bytes, seal) = sealed_bundle();
        let view = sealed_view(Some((bytes, seal))).unwrap();
        assert_eq!(view.epoch, 7);
        // Both challenges scored the same hotkey: it holds the full vector.
        assert_eq!(view.hotkeys.len(), 1);
        assert!((view.hotkeys[0].1 - 1.0).abs() < 0.001);
        assert!(view.burn_share.abs() < 0.001);
        // Both challenges delivered their full configured share onto miners.
        let root = trust_root();
        let latest = sealed_bundle();
        assert!((arena_weight(Some(&root), Some(latest.clone()), "design") - 0.5).abs() < 0.001);
        assert!((arena_weight(Some(&root), Some(latest), "prism") - 0.5).abs() < 0.001);
        let w = site_weights(100, Some(&root), Some(sealed_bundle()));
        assert!(w.sealed);
        assert_eq!(w.hotkey_weights.len(), 1);
        assert!(w.hotkey_weights[0].hotkey.starts_with('5'));
    }

    #[test]
    fn arena_weight_does_not_tax_prism_for_design_burn() {
        // Prod-shaped seal: design leaves are NoScore (share burns to uid 0),
        // prism WTA lands its full 0.5 on one miner. Global burn_share == 0.5,
        // but prism's arena weight must remain 0.5 — not share*(1-burn)=0.25.
        let body = EpochBundleBodyV1 {
            protocol_version: 1,
            epoch: 7,
            netuid: 100,
            block_b: 4242,
            block_hash: [1u8; 32],
            metagraph_root: [2u8; 32],
            algorithm_version: bundle::ALGORITHM_VERSION,
            emission_shares: vec![(b"design".to_vec(), 5_000), (b"prism".to_vec(), 5_000)],
            measurements_digest: [3u8; 32],
            uid_map: vec![([0xAAu8; 32], 0), ([0xBBu8; 32], 157)],
            leaves: vec![leaf_absent("design", 0xAA), leaf_score("prism", 0xBB, 99)],
            merkle_root: [4u8; 32],
            final_vector: vec![(0, 32_768), (157, 32_767)],
            gateway_hotkey: [5u8; 32],
        };
        let bytes = EpochBundleV1 {
            body,
            gateway_sig: [6u8; 64],
        }
        .encode_bytes();
        let latest = (
            bytes,
            SealRecord {
                sealed_at_micros: 1_785_572_565_992_448,
                revision: 1,
            },
        );
        let root = trust_root();
        let view = sealed_view(Some(latest.clone())).unwrap();
        assert!((view.burn_share - 0.5).abs() < 0.01);
        assert!((arena_weight(Some(&root), Some(latest.clone()), "prism") - 0.5).abs() < 0.001);
        assert!(arena_weight(Some(&root), Some(latest), "design").abs() < 0.001);
        assert!(arena_weight(Some(&root), Some(sealed_bundle()), "coding").abs() < f64::EPSILON);
    }

    #[test]
    fn unsealed_is_fail_closed() {
        let root = trust_root();
        let w = site_weights(100, Some(&root), None);
        assert!(!w.sealed);
        assert!(w.hotkey_weights.is_empty());
        assert!((w.burn_share - 1.0).abs() < f64::EPSILON);
        // Trust-root shares still surface; effective weight is 0.
        assert_eq!(w.emission_shares.len(), 2);
        assert!(arena_weight(Some(&root), None, "design").abs() < f64::EPSILON);
    }
}
