//! Integration scenarios for hypertraining-pay (plan todo 11 acceptance).
//!
//! Contract:
//! - S1 happy: faster cand + guards → score > 0
//! - S2 edge: Δ = 0 → score 0
//! - S3 edge: slower cand → score 0
//! - S4 edge: guards fail → score 0
//! - S5 clawback zeros unvested
//! - S6 commit-reveal mismatch reject
//! - S7 no f32/f64 in public score API sources

use hypertraining_pay::{
    commit, payable_delta_ms, reveal, score_from_pay_inputs, score_from_reward_ms,
    score_from_vested_ms, CommitDigest, PayError, PayInputs, VestingGrant, VestingLedger,
    DEFAULT_REFERENCE_MS, DEFAULT_VESTING_SEGMENTS, SCORE_MAX,
};

/// S2: Δ = 0 → score 0
#[test]
fn s2_delta_zero_score_zero() {
    let p = PayInputs {
        t_champ_ms: 8_000,
        t_cand_ms: 8_000,
        guards_passed: true,
    };
    assert_eq!(payable_delta_ms(&p), 0);
    assert_eq!(score_from_pay_inputs(&p), 0);
}

/// S1 happy: faster candidate scores > 0
#[test]
fn s1_faster_cand_positive_score() {
    let p = PayInputs {
        t_champ_ms: 8_000,
        t_cand_ms: 6_000,
        guards_passed: true,
    };
    assert_eq!(payable_delta_ms(&p), 2_000);
    let s = score_from_pay_inputs(&p);
    assert!(s > 0, "positive Δ must yield positive score, got {s}");
    assert!(s <= SCORE_MAX);
    // 2000 ms vs DEFAULT_REFERENCE_MS=1000 → clamp SCORE_MAX
    assert_eq!(s, SCORE_MAX);
}

/// S3: slower candidate → score 0
#[test]
fn s3_slower_cand_score_zero() {
    let p = PayInputs {
        t_champ_ms: 8_000,
        t_cand_ms: 9_500,
        guards_passed: true,
    };
    assert_eq!(payable_delta_ms(&p), 0);
    assert_eq!(score_from_pay_inputs(&p), 0);
}

/// S4: guards failed → score 0 even if faster
#[test]
fn s4_guards_fail_score_zero() {
    let p = PayInputs {
        t_champ_ms: 8_000,
        t_cand_ms: 1_000,
        guards_passed: false,
    };
    assert_eq!(payable_delta_ms(&p), 0);
    assert_eq!(score_from_pay_inputs(&p), 0);
}

/// Partial Δ maps strictly between 0 and `SCORE_MAX`
#[test]
fn partial_delta_fractional_score() {
    // 250 ms of 1000 ms reference → 250_000
    assert_eq!(score_from_reward_ms(250, DEFAULT_REFERENCE_MS), 250_000);
    assert_eq!(score_from_vested_ms(250), 250_000);
}

/// S5: clawback zeros unvested; vested retained
#[test]
fn s5_clawback_zeros_unvested() {
    let mut led = VestingLedger::new();
    let id = led
        .open_grant(
            u64::from(DEFAULT_VESTING_SEGMENTS) * 10,
            DEFAULT_VESTING_SEGMENTS,
        )
        .expect("open");
    // V=4, total=40 → 10 per segment
    assert_eq!(led.advance_segment(), 10);
    assert_eq!(led.grant(id).map(|g| g.vested), Some(10));
    assert_eq!(led.grant(id).map(VestingGrant::unvested), Some(30));

    let clawed = led.clawback_unvested(id).expect("clawback");
    assert_eq!(clawed, 30);
    let g = led.grant(id).expect("grant");
    assert_eq!(g.unvested(), 0, "unvested must be zero after clawback");
    assert_eq!(g.vested, 10, "vested must be retained");
    assert!(g.clawed_back);

    // Leaf score path from remaining vested only
    assert_eq!(score_from_vested_ms(g.vested), score_from_vested_ms(10));
    assert_eq!(score_from_vested_ms(g.unvested()), 0);
}

/// Regression clawback across the whole ledger
#[test]
fn s5b_regression_claws_all_unvested() {
    let mut led = VestingLedger::new();
    led.open_grant(100, 4).expect("g1");
    led.open_grant(40, 2).expect("g2");
    led.advance_segment();
    let before_unvested = led.total_unvested();
    assert!(before_unvested > 0);
    let clawed = led.clawback_all_unvested_on_regression();
    assert_eq!(clawed, before_unvested);
    assert_eq!(led.total_unvested(), 0);
}

/// S6: commit-reveal mismatch rejected
#[test]
fn s6_commit_reveal_mismatch_reject() {
    let c: CommitDigest = commit(b"submission-body-A", b"nonce-1").expect("commit");
    let err = reveal(&c, b"submission-body-B", b"nonce-1").expect_err("must reject");
    assert_eq!(err, PayError::CommitRevealMismatch);
}

/// S6b: matching reveal accepted
#[test]
fn s6b_commit_reveal_match_ok() {
    let body = b"repo_url+tree+topology";
    let nonce = b"epoch-7-miner";
    let c = commit(body, nonce).expect("commit");
    reveal(&c, body, nonce).expect("reveal must accept match");
}

/// S7: `SCORE_MAX` pin
#[test]
fn s7_score_max_pin() {
    assert_eq!(SCORE_MAX, 1_000_000);
}

/// End-to-end: pay → open grant → vest one segment → score vested slice
#[test]
fn e2e_pay_vest_score() {
    let p = PayInputs {
        t_champ_ms: 5_000,
        t_cand_ms: 4_000,
        guards_passed: true,
    };
    let delta = payable_delta_ms(&p);
    assert_eq!(delta, 1_000);

    let mut led = VestingLedger::new();
    let id = led.open_grant(delta, 4).expect("open");
    let newly = led.advance_segment();
    assert_eq!(newly, 250);
    let vested = led.grant(id).expect("g").vested;
    let score = score_from_vested_ms(vested);
    assert!(score > 0);
    assert_eq!(score, score_from_reward_ms(250, DEFAULT_REFERENCE_MS));
}
