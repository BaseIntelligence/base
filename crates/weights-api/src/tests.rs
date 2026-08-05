//! Unit tests for the response projection.

use super::*;
use bundle::{EpochBundleBodyV1, EpochBundleV1, LeafV1, ScoreOrAbsence, ALGORITHM_VERSION};

pub(crate) fn leaf(challenge: &[u8], hotkey: u8, score: u64) -> LeafV1 {
    LeafV1 {
        challenge_id: challenge.to_vec(),
        miner_hotkey: [hotkey; 32],
        epoch: 7,
        score_or_absence: ScoreOrAbsence::Score { value: score },
        challenge_sig: [9u8; 64],
    }
}

pub(crate) fn bundle() -> EpochBundleV1 {
    let body = EpochBundleBodyV1 {
        protocol_version: 1,
        epoch: 7,
        netuid: 100,
        block_b: 4242,
        block_hash: [1u8; 32],
        metagraph_root: [2u8; 32],
        algorithm_version: ALGORITHM_VERSION,
        emission_shares: vec![
            (b"agent-challenge".to_vec(), 5_000),
            (b"prism".to_vec(), 5_000),
        ],
        measurements_digest: [3u8; 32],
        uid_map: vec![([0xAAu8; 32], 0), ([0xBBu8; 32], 157)],
        leaves: vec![leaf(b"agent-challenge", 0xBB, 10), leaf(b"prism", 0xBB, 20)],
        merkle_root: [4u8; 32],
        final_vector: vec![(0, 32_767), (157, 32_767)],
        gateway_hotkey: [5u8; 32],
    };
    EpochBundleV1 {
        body,
        gateway_sig: [6u8; 64],
    }
}

fn record() -> SealRecord {
    // 2026-08-01T08:22:45.992448Z
    SealRecord {
        sealed_at_micros: 1_785_572_565_992_448,
        revision: 1,
    }
}

#[test]
fn datetime_matches_pydantic_rendering() {
    let resp = build_latest(&bundle(), record());
    assert_eq!(resp.computed_at, "2026-08-01T08:22:45.992448Z");
    assert_eq!(resp.expires_at, "2026-08-01T08:34:45.992448Z");
    assert_eq!(resp.metagraph_updated_at, resp.computed_at);
}

#[test]
fn refresh_serve_freshness_slides_wall_clock_only() {
    let mut resp = build_latest(&bundle(), record());
    let digest = resp.vector_digest.clone();
    let vector_id = resp.vector_id.clone();
    let uids = resp.uids.clone();
    let weights = resp.weights.clone();
    refresh_serve_freshness(&mut resp);
    assert_ne!(resp.computed_at, "2026-08-01T08:22:45.992448Z");
    assert_eq!(resp.metagraph_updated_at, resp.computed_at);
    // expires_at is computed_at + 720s
    assert!(resp.expires_at > resp.computed_at);
    assert_eq!(resp.vector_digest, digest);
    assert_eq!(resp.vector_id, vector_id);
    assert_eq!(resp.uids, uids);
    assert_eq!(resp.weights, weights);
    assert!(resp.sealed);
}

/// Python authority for this body (`aggregate_challenge_weights`, both challenges at
/// 50%, hotkey `0xBB…` the sole scorer in each): uids `[157]`, weights `[1.0]`,
/// `hotkey_weights {bb…: 1.0}`. uid 0's hotkey never scored, so nothing burns.
#[test]
fn served_weights_are_the_raw_python_floats() {
    let resp = build_latest(&bundle(), record());
    assert_eq!(resp.uids, vec![157]);
    assert_eq!(resp.weights, vec![1.0]);
    assert_eq!(resp.hotkey_weights.entries().len(), 1);
    assert_eq!(resp.hotkey_weights.entries()[0].1, 1.0);
}

/// The response must carry the aggregator's double, not `final_vector[i] / sum`:
/// the u16 round-trip of `0.3` is `19660/65535`, which is a different number.
#[test]
fn floats_are_not_a_u16_round_trip() {
    let mut b = bundle();
    // 30 / 70 split across the two challenges -> uid 157 holds exactly 0.3.
    b.body.emission_shares = vec![
        (b"agent-challenge".to_vec(), 3_000),
        (b"prism".to_vec(), 7_000),
    ];
    b.body.leaves = vec![leaf(b"agent-challenge", 0xBB, 10)];
    let resp = build_latest(&b, record());
    assert_eq!(resp.uids, vec![0, 157]);
    assert_eq!(resp.weights[1].to_bits(), 0.3_f64.to_bits());
    assert_ne!(
        resp.weights[1].to_bits(),
        (19_660.0_f64 / 65_535.0).to_bits()
    );
}

#[test]
fn chain_domain_bytes_is_canonical_json() {
    let resp = build_latest(&bundle(), record());
    assert_eq!(
        resp.chain_domain_bytes.as_deref(),
        Some(r#"{"netuid":100,"uids":[157],"weights":[1.0]}"#)
    );
}

#[test]
fn sources_cover_every_expected_challenge() {
    let resp = build_latest(&bundle(), record());
    assert_eq!(resp.source_challenges.len(), 2);
    assert!(resp.source_challenges.iter().all(|c| c.ok));
    assert!(resp
        .source_challenges
        .iter()
        .all(|c| c.weights.entries().is_empty()));
    assert_eq!(resp.source_challenges[0].emission_percent, 50.0);
    assert_eq!(resp.source_snapshots.len(), 2);
    assert_eq!(resp.source_outcomes.len(), 2);
    assert_eq!(resp.source_outcomes[0].outcome, "accepted");
    assert_eq!(resp.source_outcomes[0].reason_code, "accepted");
    assert_eq!(
        resp.source_snapshots[0].payload_digest,
        resp.source_outcomes[0].payload_digest.clone().unwrap()
    );
}

#[test]
fn missing_source_is_reported_not_invented() {
    let mut b = bundle();
    b.body
        .leaves
        .retain(|l| l.challenge_id == b"prism".to_vec());
    let resp = build_latest(&b, record());
    let agent = &resp.source_challenges[0];
    assert!(!agent.ok);
    assert_eq!(agent.error.as_deref(), Some("missing"));
    assert_eq!(resp.source_snapshots.len(), 1);
    assert_eq!(resp.source_outcomes[0].snapshot_id, None);
}

#[test]
fn vector_identity_is_deterministic() {
    let first = build_latest(&bundle(), record());
    let second = build_latest(&bundle(), record());
    assert_eq!(first.vector_digest, second.vector_digest);
    assert_eq!(first.vector_id, second.vector_id);
    assert!(first.sealed);
    let mut altered = bundle();
    altered.body.epoch = 8;
    assert_ne!(
        build_latest(&altered, record()).vector_digest,
        first.vector_digest
    );
}

/// Nobody scored: Python's `build_zero_miner_weights` returns `{0: 1.0}` under the
/// deployed defaults (`min_allowed_weights=1`, `max_weight_limit=65535`).
#[test]
fn all_zero_epoch_serves_the_burn_vector() {
    let mut b = bundle();
    b.body.leaves = vec![leaf(b"agent-challenge", 0xBB, 0), leaf(b"prism", 0xBB, 0)];
    let resp = build_latest(&b, record());
    assert_eq!(resp.uids, vec![0]);
    assert_eq!(resp.weights, vec![1.0]);
    assert!(resp.hotkey_weights.entries().is_empty());
    assert!(resp.sealed);
}

#[test]
fn no_sealed_bundle_serves_fail_closed_burn_fallback() {
    let resp = crate::build_burn_fallback(100);
    assert!(!resp.sealed);
    assert_eq!(resp.epoch, None);
    assert_eq!(resp.uids, vec![0]);
    assert_eq!(resp.weights, vec![1.0]);
    assert_eq!(resp.final_vector, vec![(0, 65_535)]);
    assert_eq!(resp.burn_outcome, Some(true));
    assert_eq!(
        resp.burn_policy_version.as_deref(),
        Some(crate::BURN_POLICY_VERSION)
    );
    assert!(resp.merkle_root.is_empty());
    assert!(resp.hotkey_weights.entries().is_empty());
    assert_eq!(resp.netuid, 100);
}
