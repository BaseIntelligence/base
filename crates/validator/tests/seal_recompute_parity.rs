#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Gateway seal vs validator recompute: identical bytes **and** identical doubles.
//!
//! The validator recomputes to catch a cheating gateway, which only works if an honest
//! gateway is indistinguishable from the validator's own arithmetic. Comparing
//! `final_vector` alone is not enough now that `/v1/weights/latest` serves the
//! pre-rounding floats: two implementations can agree on `round(w * 65535)` while
//! disagreeing in the last ulp of `w`. Every assertion here is therefore on
//! `f64::to_bits()`.
//!
//! Expected values are derived from the Python authority
//! (`/root/prism-compute-plane/base/.venv/bin/python`, `base.master.aggregator`
//! `aggregate_challenge_weights` with the deployed defaults `min_allowed_weights=1`,
//! `max_weight_limit=65535`), never from what Rust happens to print.

use bundle::{
    build_sealed_bundle, make_signed_leaf, python_weights, python_weights_from_parts, sign_bundle,
    sort_leaves, EpochBundleV1, LeafV1, LocalTrustRoot, NoScoreReasonCode, ScoreOrAbsence,
    SealParams,
};
use chain::{FakeChain, FakeChainConfig};
use crypto::secret_from_bytes;
use sha2::{Digest, Sha256};
use trustroot::{
    measurements_digest, ChallengeEntry, ChallengesBody, MeasurementsBody, ParticipantPolicy,
};
use validator::{
    compare_bundle, independent_aggregate, independent_python_weights, ComparisonOutcome,
};

const EPOCH: u64 = 4_242;
const BLOCK_B: u64 = 100;

fn sk(tag: u8) -> [u8; 32] {
    let digest = Sha256::digest([0x5A, tag, 0xA5, tag]);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest);
    seed
}

fn pk_of(secret: &[u8; 32]) -> [u8; 32] {
    secret_from_bytes(secret)
        .expect("secret")
        .to_public()
        .to_bytes()
}

/// Miner hotkeys are opaque bytes to aggregation (only challenge keys sign), so we pin
/// them to ascending single-byte tags and read the uid assignment straight off the order.
fn hk(tag: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0] = tag;
    out
}

/// One score per (challenge, miner); `None` means an explicit `NoScore` leaf.
type Scores<'a> = &'a [(&'a [u8], u8, Option<u64>)];

/// Seal through the production path: `bundle::build_sealed_bundle`.
fn seal(
    challenges: &[(&[u8], u16)],
    miner_tags: &[u8],
    scores: Scores<'_>,
) -> (EpochBundleV1, LocalTrustRoot, FakeChain) {
    let csk = sk(1);
    let gsk = sk(2);
    let trust = LocalTrustRoot {
        challenges: ChallengesBody {
            challenges: challenges
                .iter()
                .map(|(id, bps)| ChallengeEntry {
                    id: (*id).to_vec(),
                    public_key: pk_of(&csk),
                    emission_share_bps: *bps,
                    policy: ParticipantPolicy::AllMetagraphHotkeys,
                })
                .collect(),
        },
        measurements_digest: measurements_digest(&MeasurementsBody::default()),
    };
    let chain = FakeChain::new(FakeChainConfig {
        current_block: BLOCK_B.max(10),
        hotkeys: miner_tags.iter().map(|t| hk(*t).to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    });

    let mut leaves: Vec<LeafV1> = Vec::new();
    for (id, _) in challenges {
        for tag in miner_tags {
            let value = scores
                .iter()
                .find(|(cid, t, _)| cid == id && t == tag)
                .and_then(|(_, _, v)| *v);
            let score = value.map_or(
                ScoreOrAbsence::NoScore {
                    reason: NoScoreReasonCode::Timeout,
                },
                |value| ScoreOrAbsence::Score { value },
            );
            leaves.push(make_signed_leaf(&csk, id, hk(*tag), EPOCH, score).expect("leaf"));
        }
    }
    sort_leaves(&mut leaves);

    let bundle = build_sealed_bundle(
        &chain,
        &trust,
        leaves,
        &SealParams {
            epoch: EPOCH,
            netuid: 1,
            block_b: BLOCK_B,
            gateway_secret: gsk,
        },
    )
    .expect("seal");
    (bundle, trust, chain)
}

/// The whole point: the validator's independent recompute must reproduce the sealed
/// vector and the served doubles exactly.
fn assert_seal_and_recompute_agree(
    label: &str,
    bundle: &EpochBundleV1,
    expected_uids: &[u16],
    expected_weight_bits: &[u64],
    expected_chain: &[(u16, u16)],
    expected_hotkey_tags: &[u8],
) {
    let local = independent_python_weights(&bundle.body).expect("recompute");

    assert_eq!(
        bundle.body.final_vector,
        independent_aggregate(&bundle.body).expect("recompute vector"),
        "{label}: sealed final_vector != independent recompute"
    );
    assert_eq!(
        bundle.body.final_vector, expected_chain,
        "{label}: sealed vector != Python authority"
    );

    assert_eq!(local.floats.uids, expected_uids, "{label}: uids");
    let got_bits: Vec<u64> = local.floats.weights.iter().map(|w| w.to_bits()).collect();
    assert_eq!(got_bits, expected_weight_bits, "{label}: weight bits");

    let expected_keys: Vec<String> = expected_hotkey_tags
        .iter()
        .map(|t| hex::encode(hk(*t)))
        .collect();
    let got_keys: Vec<String> = local
        .floats
        .hotkey_weights
        .iter()
        .map(|(k, _)| k.clone())
        .collect();
    assert_eq!(
        got_keys, expected_keys,
        "{label}: hotkey_weights keys/order"
    );

    // The gateway-side adapter (what seal ran) and the validator-side recompute are the
    // same function by construction; assert it anyway so a future split is caught here.
    let gateway = python_weights(&bundle.body).expect("gateway side");
    assert_eq!(gateway.final_vector, local.final_vector, "{label}: vector");
    for (a, b) in gateway
        .floats
        .weights
        .iter()
        .zip(&local.floats.weights)
        .chain(
            gateway
                .floats
                .hotkey_weights
                .iter()
                .map(|(_, v)| v)
                .zip(local.floats.hotkey_weights.iter().map(|(_, v)| v)),
        )
    {
        assert_eq!(a.to_bits(), b.to_bits(), "{label}: float bits");
    }
}

/// Two challenges, 60/40, with the uid-0 hotkey scoring so a remainder burns.
///
/// Python: `uids [0, 1, 2]`, `weights [0.1499999999999999, 0.45000000000000007, 0.4]`.
/// The first weight is *not* `0.15` — that residue is exactly what a float-inexact
/// reimplementation would round away, so it is the sharpest available parity probe.
#[test]
fn multi_challenge_with_burn_remainder_is_bit_identical() {
    let (bundle, trust, chain) = seal(
        &[(b"ch-a", 6_000), (b"ch-b", 4_000)],
        &[0x10, 0x20, 0x30],
        &[
            (b"ch-a", 0x10, Some(1)),
            (b"ch-a", 0x20, Some(1)),
            (b"ch-a", 0x30, Some(2)),
            (b"ch-b", 0x10, None),
            (b"ch-b", 0x20, Some(3)),
            (b"ch-b", 0x30, Some(1)),
        ],
    );
    assert_seal_and_recompute_agree(
        "multi-challenge burn",
        &bundle,
        &[0, 1, 2],
        &[
            0x3fc3_3333_3333_3330,
            0x3fdc_cccc_cccc_ccce,
            0x3fd9_9999_9999_999a,
        ],
        &[(0, 9_830), (1, 29_491), (2, 26_214)],
        &[0x20, 0x30],
    );
    assert!(
        matches!(
            compare_bundle(&bundle, &chain, &trust),
            ComparisonOutcome::Match { .. }
        ),
        "honest seal must compare as Match"
    );
}

/// Nobody scored: Python's `build_zero_miner_weights` fallback, `{0: 1.0}`.
#[test]
fn zero_miner_epoch_is_bit_identical() {
    let (bundle, trust, chain) = seal(&[(b"ch-a", 10_000)], &[0x10, 0x20, 0x30], &[]);
    assert_seal_and_recompute_agree(
        "zero miner",
        &bundle,
        &[0],
        &[0x3ff0_0000_0000_0000],
        &[(0, 65_535)],
        &[],
    );
    assert!(matches!(
        compare_bundle(&bundle, &chain, &trust),
        ComparisonOutcome::Match { .. }
    ));
}

/// The hotkey holding uid 0 is the burn sink, never a paid miner.
///
/// Python: scores `{uid0: 3, uid1: 1}` → `uids [0, 1]`, `weights [0.75, 0.25]`.
#[test]
fn hotkey_on_uid_zero_burns_bit_identical() {
    let (bundle, trust, chain) = seal(
        &[(b"ch-a", 10_000)],
        &[0x10, 0x20],
        &[(b"ch-a", 0x10, Some(3)), (b"ch-a", 0x20, Some(1))],
    );
    assert_seal_and_recompute_agree(
        "uid zero burns",
        &bundle,
        &[0, 1],
        &[0x3fe8_0000_0000_0000, 0x3fd0_0000_0000_0000],
        &[(0, 49_151), (1, 16_384)],
        &[0x20],
    );
    assert!(matches!(
        compare_bundle(&bundle, &chain, &trust),
        ComparisonOutcome::Match { .. }
    ));
}

/// A leaf hotkey absent from `uid_map` burns rather than aborting aggregation.
///
/// D24 completeness means the real seal path can never emit this body, so the bundle is
/// assembled by hand from the sealed one; `compare_bundle` would (correctly) reject it as
/// an incomplete participant set. What is under test is that both sides handle the
/// unmapped hotkey identically.
///
/// Python: `{uid0: 1, uid1: 1, unmapped: 2}` → `uids [0, 1]`, `weights [0.75, 0.25]`,
/// where uid 0's 0.25 and the unmapped 0.5 both burn.
#[test]
fn unknown_hotkey_burns_bit_identically_on_both_sides() {
    let (bundle, _trust, _chain) = seal(
        &[(b"ch-a", 10_000)],
        &[0x10, 0x20],
        &[(b"ch-a", 0x10, Some(1)), (b"ch-a", 0x20, Some(1))],
    );
    let mut body = bundle.body;
    body.leaves.push(
        make_signed_leaf(
            &sk(1),
            b"ch-a",
            hk(0x99),
            EPOCH,
            ScoreOrAbsence::Score { value: 2 },
        )
        .expect("leaf"),
    );
    sort_leaves(&mut body.leaves);
    body.merkle_root = bundle::compute_merkle_root(&body.leaves);
    body.final_vector =
        python_weights_from_parts(&body.leaves, &body.emission_shares, &body.uid_map)
            .expect("gateway side")
            .final_vector;
    let resealed = sign_bundle(&sk(2), body).expect("sign");

    assert_seal_and_recompute_agree(
        "unknown hotkey",
        &resealed,
        &[0, 1],
        &[0x3fe8_0000_0000_0000, 0x3fd0_0000_0000_0000],
        &[(0, 49_151), (1, 16_384)],
        &[0x20],
    );
}

/// `/v1/weights/latest` must serve the recompute's doubles, not a u16 round-trip.
#[test]
fn served_response_floats_match_the_recompute_bit_for_bit() {
    let (bundle, _trust, _chain) = seal(
        &[(b"ch-a", 6_000), (b"ch-b", 4_000)],
        &[0x10, 0x20, 0x30],
        &[
            (b"ch-a", 0x10, Some(1)),
            (b"ch-a", 0x20, Some(1)),
            (b"ch-a", 0x30, Some(2)),
            (b"ch-b", 0x10, None),
            (b"ch-b", 0x20, Some(3)),
            (b"ch-b", 0x30, Some(1)),
        ],
    );
    let local = independent_python_weights(&bundle.body).expect("recompute");
    let response = weights_api::build_latest(
        &bundle,
        weights_api::SealRecord {
            sealed_at_micros: 1_785_572_565_992_448,
            revision: 1,
        },
    );

    assert_eq!(response.uids, local.floats.uids);
    for (served, recomputed) in response.weights.iter().zip(&local.floats.weights) {
        assert_eq!(served.to_bits(), recomputed.to_bits());
    }
    // A u16 round-trip would give 29491/65535 here, which is a different double.
    assert_eq!(response.weights[1].to_bits(), 0x3fdc_cccc_cccc_ccce);
    assert_ne!(
        response.weights[1].to_bits(),
        (29_491.0_f64 / 65_535.0).to_bits()
    );

    let served_hotkeys: Vec<f64> = response
        .hotkey_weights
        .entries()
        .iter()
        .map(|(_, w)| *w)
        .collect();
    let recomputed_hotkeys: Vec<f64> = local
        .floats
        .hotkey_weights
        .iter()
        .map(|(_, w)| *w)
        .collect();
    assert_eq!(served_hotkeys.len(), recomputed_hotkeys.len());
    for (served, recomputed) in served_hotkeys.iter().zip(&recomputed_hotkeys) {
        assert_eq!(served.to_bits(), recomputed.to_bits());
    }
    // SS58 re-keying must preserve Python's `kept_hotkeys` order.
    let expected: Vec<String> = [0x20u8, 0x30]
        .iter()
        .map(|t| keystore::ss58_encode(&hk(*t), keystore::BITTENSOR_SS58_PREFIX))
        .collect();
    let got: Vec<String> = response
        .hotkey_weights
        .entries()
        .iter()
        .map(|(k, _)| k.clone())
        .collect();
    assert_eq!(got, expected);
}
