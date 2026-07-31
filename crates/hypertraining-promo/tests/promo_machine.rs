//! Integration scenarios for hypertraining-promo (plan todo 9).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use hypertraining_promo::{
    benjamini_hochberg, hash_hex, ChallengerId, CheckpointHash, DuelEvidence, HoldoutEvidence,
    PromoError, PromoState, PromotionMachine, RejectReason, ScreenEvidence, PROMOTION_K, SCREEN_K,
};

fn ck(b: u8) -> CheckpointHash {
    [b; 32]
}

fn screen_ok() -> ScreenEvidence {
    ScreenEvidence {
        kernel_passed: true,
        candidate_median_ms: 800,
        champion_median_ms: 1000,
        sign_coherent: true,
        k: SCREEN_K,
    }
}

fn duel_ok(p: f64) -> DuelEvidence {
    DuelEvidence {
        p_value: p,
        non_inferiority: true,
        physical_plausible: true,
        k: PROMOTION_K,
    }
}

fn full_promote(m: &mut PromotionMachine, cand: CheckpointHash) -> ChallengerId {
    let id = m.admit(cand);
    assert_eq!(
        m.advance_screen(id, screen_ok()).expect("ok"),
        PromoState::Screened
    );
    assert_eq!(
        m.advance_duel(id, duel_ok(0.01)).expect("ok"),
        PromoState::Duelled
    );
    assert_eq!(
        m.advance_holdout(id, HoldoutEvidence { sign_agrees: true })
            .expect("ok"),
        PromoState::Confirmed
    );
    assert_eq!(m.promote(id).expect("ok"), PromoState::Champion);
    id
}

/// S1 happy: full path to CHAMPION with lineage parent link.
#[test]
fn s1_happy_full_path_to_champion() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(0xAA)).expect("bootstrap");
    let id = full_promote(&mut m, ck(0xBB));
    assert_eq!(m.challenger(id).expect("ok").state, PromoState::Champion);
    assert_eq!(m.champion_hash(), Some(ck(0xBB)));
    let tip = m.lineage().tip().expect("tip");
    assert_eq!(tip.parent_hash, Some(ck(0xAA)));
    assert_eq!(tip.challenger_id, Some(id));
    // public entry hash is 32 bytes / printable hex
    assert_eq!(hash_hex(&tip.entry_hash).len(), 64);
}

/// S2 failure: screen fail → REJECTED.
#[test]
fn s2_screen_fail_rejected() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(1)).expect("bootstrap");
    let id = m.admit(ck(2));
    let st = m
        .advance_screen(
            id,
            ScreenEvidence {
                kernel_passed: true,
                candidate_median_ms: 1200,
                champion_median_ms: 1000,
                sign_coherent: true,
                k: SCREEN_K,
            },
        )
        .expect("ok");
    assert_eq!(st, PromoState::Rejected(RejectReason::ScreenFailed));
    // cannot promote rejected
    let err = m.promote(id).expect_err("err");
    assert!(matches!(
        err,
        PromoError::InvalidTransition {
            action: "promote",
            ..
        }
    ));
}

/// S3 duel BH failure → REJECTED.
#[test]
fn s3_duel_bh_fail_rejected() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(1)).expect("bootstrap");
    let id = m.admit(ck(2));
    m.advance_screen(id, screen_ok()).expect("ok");
    let st = m.advance_duel(id, duel_ok(0.40)).expect("ok");
    assert_eq!(st, PromoState::Rejected(RejectReason::DuelFailed));
}

/// S4 holdout disagree → REJECTED.
#[test]
fn s4_holdout_disagree_rejected() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(1)).expect("bootstrap");
    let id = m.admit(ck(2));
    m.advance_screen(id, screen_ok()).expect("ok");
    m.advance_duel(id, duel_ok(0.01)).expect("ok");
    let st = m
        .advance_holdout(id, HoldoutEvidence { sign_agrees: false })
        .expect("ok");
    assert_eq!(st, PromoState::Rejected(RejectReason::HoldoutDisagreed));
}

/// S5 rollback restores C(n-1).
#[test]
fn s5_rollback_restores_prior_hash() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(0x11)).expect("bootstrap");
    let _ = full_promote(&mut m, ck(0x22));
    assert_eq!(m.champion_hash(), Some(ck(0x22)));
    let restored = m.rollback().expect("ok");
    assert_eq!(restored, ck(0x11));
    assert_eq!(m.champion_hash(), Some(ck(0x11)));
    assert_eq!(m.lineage().len(), 1);
}

/// S6 BH across challengers: only small p survives.
#[test]
fn s6_bh_across_challengers() {
    let decisions = benjamini_hochberg(&[0.20, 0.01], 0.05).expect("ok");
    assert_eq!(decisions, vec![false, true]);

    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(1)).expect("bootstrap");
    let weak = m.admit(ck(2));
    let strong = m.admit(ck(3));
    m.advance_screen(weak, screen_ok()).expect("ok");
    m.advance_screen(strong, screen_ok()).expect("ok");
    let out = m
        .advance_duel_cohort(&[(weak, duel_ok(0.20)), (strong, duel_ok(0.01))])
        .expect("ok");
    assert_eq!(
        out,
        vec![
            (weak, PromoState::Rejected(RejectReason::DuelFailed)),
            (strong, PromoState::Duelled),
        ]
    );
}

/// S7 guards fail even with tiny p → REJECTED.
#[test]
fn s7_guards_fail_despite_significant_p() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(1)).expect("bootstrap");
    let id = m.admit(ck(2));
    m.advance_screen(id, screen_ok()).expect("ok");
    let st = m
        .advance_duel(
            id,
            DuelEvidence {
                p_value: 0.001,
                non_inferiority: false,
                physical_plausible: true,
                k: PROMOTION_K,
            },
        )
        .expect("ok");
    assert_eq!(st, PromoState::Rejected(RejectReason::DuelFailed));
}

/// Kernel gate fail at screen → REJECTED.
#[test]
fn kernel_fail_at_screen_rejects() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(1)).expect("bootstrap");
    let id = m.admit(ck(2));
    let st = m
        .advance_screen(
            id,
            ScreenEvidence {
                kernel_passed: false,
                candidate_median_ms: 100,
                champion_median_ms: 1000,
                sign_coherent: true,
                k: SCREEN_K,
            },
        )
        .expect("ok");
    assert_eq!(st, PromoState::Rejected(RejectReason::ScreenFailed));
}

/// Rollback without prior generation errors.
#[test]
fn rollback_without_prior_errors() {
    let mut m = PromotionMachine::new();
    m.bootstrap_champion(ck(1)).expect("bootstrap");
    let err = m.rollback().expect_err("err");
    assert_eq!(err, PromoError::NoPriorChampion);
}
