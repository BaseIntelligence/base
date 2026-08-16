//! `PRISM_EMISSION_MODE=sig` end-to-end leaf projection (v3, default-off).
//!
//! This lives in its own integration binary because [`EmissionMode::from_env`]
//! memoizes in a `OnceLock`: the knob must be set before the first read, which
//! is only reliable in a dedicated process. The default-path bit-identity
//! test lives in the crate's unit tests, where the env is unset.

use std::collections::BTreeMap;

use bundle::ScoreOrAbsence;
use challenge_common::{ExpectedParticipant, ExpectedSet, Hotkey};
use crypto::KEY_LEN;
use prism_emit::build_epoch_leaves_with;
use prism_registry::{
    sig::bps_to_lattice, AxisScore, EliteArchive, PairedOutcome, SigContext, BAND_BPS,
    CHAMPION_BPS, CHAMPION_FLOOR_BPS,
};
use prism_store::{EpochScoreRow, FinalScore};

fn sk() -> [u8; KEY_LEN] {
    let mut s = [7u8; KEY_LEN];
    s[0] = 0x42;
    s
}

/// uid 0 is the burn sink; the rest are ordinary miners.
fn expected(hks: &[[u8; KEY_LEN]]) -> ExpectedSet {
    ExpectedSet {
        block_hash: [0x77u8; 32],
        participants: hks
            .iter()
            .enumerate()
            .map(|(i, h)| ExpectedParticipant {
                hotkey: *h,
                uid: u16::try_from(i).unwrap_or(0),
            })
            .collect(),
    }
}

fn row(hk: [u8; KEY_LEN], score: u64) -> EpochScoreRow {
    EpochScoreRow {
        miner_hotkey: hex::encode(hk),
        arch_id: None,
        final_score: FinalScore::Score(score),
        weight_eligible: true,
    }
}

fn value_of(leaves: &BTreeMap<Hotkey, bundle::LeafV1>, hk: &Hotkey) -> u64 {
    match leaves.get(hk).map(|l| l.score_or_absence.clone()) {
        Some(ScoreOrAbsence::Score { value }) => value,
        other => panic!("expected a Score leaf, got {other:?}"),
    }
}

fn displacing_win(mean_gap: f64) -> PairedOutcome {
    PairedOutcome {
        n_paired: 200,
        n_decided: 180,
        n_wins: 120,
        win_rate_bps: 6_667,
        win_rate_lcb_bps: 6_000,
        mean_gap,
        displaces: true,
    }
}

#[test]
fn sig_mode_projects_shares_and_burns_the_remainder() {
    std::env::set_var("PRISM_EMISSION_MODE", "sig");

    // uid 0 = burn sink, then champion, challenger, and an axis holder.
    let (sink, champ, chal, looped) = ([0x01; 32], [0xAA; 32], [0xBB; 32], [0xCC; 32]);
    let exp = expected(&[sink, champ, chal, looped]);
    let batch = vec![
        row(champ, 800_000),
        row(chal, 900_000),
        row(looped, 300_000),
    ];

    let ctx = SigContext {
        incumbent: Some(hex::encode(champ)),
        tenure_days: 0,
        // The challenger cleared the paired test with a premium gap.
        challenger: Some((hex::encode(chal), displacing_win(0.05))),
        archive: EliteArchive::build(&[AxisScore {
            axis: "g3".into(),
            hotkey: hex::encode(looped),
            value: 0.95,
            gates_ok: true,
        }]),
        previous_bps: BTreeMap::new(),
    };

    let leaves = build_epoch_leaves_with(&sk(), 11, &exp, &batch, &BTreeMap::new(), Some(&ctx))
        .expect("emit sig leaves");
    assert_eq!(leaves.len(), 4, "D24 exact set");

    // Champion took the crown at the premium share.
    assert_eq!(value_of(&leaves, &chal), bps_to_lattice(CHAMPION_BPS));
    // Displaced incumbent falls to the leading band slot.
    assert_eq!(value_of(&leaves, &champ), bps_to_lattice(BAND_BPS[0]));
    // Axis holder: band rank 3 plus the whole exploration pool (sole slot).
    let looped_v = value_of(&leaves, &looped);
    assert!(
        looped_v > bps_to_lattice(BAND_BPS[1]),
        "frontier holder must earn band + explore, got {looped_v}"
    );

    // The remainder is a real burn leaf at uid 0, not silent dilution.
    let burn = value_of(&leaves, &sink);
    assert!(burn > 0, "unallocated share must burn to uid 0");

    // Conservation in lattice space: every leaf is share-proportional and
    // the total corresponds to the full 10 000 bps.
    let total: u64 = [sink, champ, chal, looped]
        .iter()
        .map(|hk| value_of(&leaves, hk))
        .sum();
    assert_eq!(total, bps_to_lattice(10_000), "shares + burn = 100 %");

    // Every leaf still verifies under the challenge key.
    let pk = challenge_common::public_key_from_secret(&sk()).expect("pk");
    for leaf in leaves.values() {
        challenge_common::verify_leaf_sig(leaf, &pk).expect("leaf sig");
    }
}

#[test]
fn sig_mode_holds_the_crown_against_a_clone() {
    std::env::set_var("PRISM_EMISSION_MODE", "sig");

    let (sink, champ, clone_hk) = ([0x01; 32], [0xAA; 32], [0xBB; 32]);
    let exp = expected(&[sink, champ, clone_hk]);
    // The clone's raw credit is one lattice tick HIGHER than the champion's
    // — under WTA that wins outright. Its paired outcome does not displace.
    let batch = vec![row(champ, 900_000), row(clone_hk, 900_001)];
    let ctx = SigContext {
        incumbent: Some(hex::encode(champ)),
        challenger: Some((hex::encode(clone_hk), PairedOutcome::hold())),
        ..SigContext::default()
    };

    let leaves = build_epoch_leaves_with(&sk(), 12, &exp, &batch, &BTreeMap::new(), Some(&ctx))
        .expect("emit sig leaves");

    assert_eq!(
        value_of(&leaves, &champ),
        bps_to_lattice(CHAMPION_FLOOR_BPS),
        "incumbent holds despite a higher point estimate"
    );
    assert_eq!(
        value_of(&leaves, &clone_hk),
        bps_to_lattice(BAND_BPS[0]),
        "the clone is capped at the band, never the crown"
    );
    // Champion floor 50 % + band 15 % = 65 %; 35 % burns.
    assert_eq!(value_of(&leaves, &sink), bps_to_lattice(3_500));
}

#[test]
fn sig_mode_is_deterministic_across_repeated_emissions() {
    std::env::set_var("PRISM_EMISSION_MODE", "sig");

    let (sink, a, b) = ([0x01; 32], [0xAA; 32], [0xBB; 32]);
    let exp = expected(&[sink, a, b]);
    let batch = vec![row(a, 700_000), row(b, 500_000)];
    let ctx = SigContext {
        incumbent: Some(hex::encode(a)),
        tenure_days: 5,
        ..SigContext::default()
    };

    let first = build_epoch_leaves_with(&sk(), 13, &exp, &batch, &BTreeMap::new(), Some(&ctx))
        .expect("emit")
        .iter()
        .map(|(hk, l)| (*hk, l.score_or_absence.clone()))
        .collect::<BTreeMap<_, _>>();
    for _ in 0..5 {
        let again = build_epoch_leaves_with(&sk(), 13, &exp, &batch, &BTreeMap::new(), Some(&ctx))
            .expect("emit")
            .iter()
            .map(|(hk, l)| (*hk, l.score_or_absence.clone()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(first, again, "identical inputs ⇒ identical leaf values");
    }
}
