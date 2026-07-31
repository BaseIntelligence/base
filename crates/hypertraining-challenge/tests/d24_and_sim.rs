//! Integration: D24 leaf cover + sign verify + happy sim score (todo 12).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};

use hypertraining_challenge::{
    emit_signed_leaf_set, public_key_from_secret, run_sim_pipeline, search_faster_compiled,
    verify_leaf_sig, AttestationStatus, EpochCtx, Hotkey, HypertrainingChallenge,
    MapAttestationLookup, NoScoreReasonCode, PipelineOutcome, ScoreOrAbsence, CHALLENGE_ID,
    CHALLENGE_ID_BYTES, SCORE_MAX,
};
use hypertraining_cluster::{SegmentSeeds, SimBackend, Topology};
use hypertraining_eval::fixtures::fixture_equal_quality_pairs;
use chain::Metagraph;
use crypto::KEY_LEN;
use trustroot::{ChallengeEntry, ChallengesBody, ParticipantPolicy};

const CHAMP_SRC: &str = "def fused_gemm(a, b):\n    return a @ b\n";
const CHAMP_BIN: &[u8] =
    b".version 7.0\n.entry gemm {\nadd.u32 %r1, %r2, %r3;\nmul.f32 %f1, %f2, %f3;\n}\n";

fn mini() -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
    let m = schnorrkel::MiniSecretKey::generate_with(rand_core::OsRng);
    let sk = m.to_bytes();
    let pk = m
        .expand(schnorrkel::ExpansionMode::Ed25519)
        .to_public()
        .to_bytes();
    (sk, pk)
}

fn ctx(pk: [u8; KEY_LEN], hotkeys: Vec<Vec<u8>>) -> EpochCtx {
    EpochCtx {
        netuid: 1,
        epoch: 42,
        block_hash: [0xBEu8; 32],
        metagraph: Metagraph {
            netuid: 1,
            hotkeys,
            owner_hotkey: vec![0u8; 32],
        },
        challenges: ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: CHALLENGE_ID_BYTES.to_vec(),
                public_key: pk,
                emission_share_bps: 0,
                policy: ParticipantPolicy::AllMetagraphHotkeys,
            }],
        },
    }
}

/// S1: E=2 miners both get signed leaves under `challenge_id` hypertraining.
#[test]
fn s1_d24_two_miners_both_get_leaves() {
    let (sk, pk) = mini();
    let m1: Hotkey = [0x11; KEY_LEN];
    let m2: Hotkey = [0x22; KEY_LEN];
    let ch = HypertrainingChallenge::sim();
    let ctx = ctx(pk, vec![m1.to_vec(), m2.to_vec()]);
    let mut outcomes = BTreeMap::new();
    outcomes.insert(
        m1,
        PipelineOutcome::Measured {
            t_champ_ms: 10_000,
            t_cand_ms: 7_000,
            guards_passed: true,
        },
    );
    outcomes.insert(
        m2,
        PipelineOutcome::Measured {
            t_champ_ms: 10_000,
            t_cand_ms: 9_000,
            guards_passed: true,
        },
    );
    let attest = MapAttestationLookup {
        default: AttestationStatus::Verified,
        ..Default::default()
    };
    let scores = ch.score_epoch(&ctx, &outcomes, &attest).unwrap();
    assert_eq!(scores.len(), 2);
    let expected = ch.expected_set(&ctx).unwrap();
    let leaves = emit_signed_leaf_set(&sk, 42, &expected, &scores).unwrap();
    assert_eq!(leaves.len(), 2);
    for h in [m1, m2] {
        let leaf = leaves.get(&h).unwrap();
        assert_eq!(leaf.challenge_id, CHALLENGE_ID.as_bytes());
        assert_ne!(leaf.challenge_id, b"agent-v1");
        verify_leaf_sig(leaf, &pk).unwrap();
        match &leaf.score_or_absence {
            ScoreOrAbsence::Score { value } => {
                assert!(*value > 0 && *value <= SCORE_MAX);
            }
            ScoreOrAbsence::NoScore { reason } => {
                panic!("expected Score, got NoScore({reason:?})")
            }
        }
    }
}

/// S2: missing miner in outcomes → NoScore(ChallengeInternal), never silence.
#[test]
fn s2_missing_miner_noscore_not_silence() {
    let (sk, pk) = mini();
    let m1: Hotkey = [0x33; KEY_LEN];
    let m2: Hotkey = [0x44; KEY_LEN];
    let ch = HypertrainingChallenge::sim();
    let ctx = ctx(pk, vec![m1.to_vec(), m2.to_vec()]);
    let mut outcomes = BTreeMap::new();
    outcomes.insert(
        m1,
        PipelineOutcome::Measured {
            t_champ_ms: 8_000,
            t_cand_ms: 6_000,
            guards_passed: true,
        },
    );
    // m2 intentionally absent
    let attest = MapAttestationLookup {
        default: AttestationStatus::Verified,
        ..Default::default()
    };
    let scores = ch.score_epoch(&ctx, &outcomes, &attest).unwrap();
    assert_eq!(scores.len(), 2);
    assert!(matches!(
        scores[&m1],
        ScoreOrAbsence::Score { value } if value > 0
    ));
    assert_eq!(
        scores[&m2],
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    let expected: BTreeSet<_> = scores.keys().copied().collect();
    let leaves = emit_signed_leaf_set(&sk, 42, &expected, &scores).unwrap();
    assert_eq!(leaves.len(), 2);
    verify_leaf_sig(leaves.get(&m2).unwrap(), &pk).unwrap();
}

/// S3: `sign_leaf` verifies with challenge sk-derived pk.
#[test]
fn s3_sign_verify_with_challenge_sk() {
    let (sk, pk) = mini();
    assert_eq!(public_key_from_secret(&sk).unwrap(), pk);
    let ch = HypertrainingChallenge::sim();
    let miner = [0x55u8; KEY_LEN];
    let leaf = ch
        .sign_leaf(
            &sk,
            miner,
            9,
            ScoreOrAbsence::Score {
                value: SCORE_MAX / 2,
            },
        )
        .unwrap();
    assert_eq!(leaf.challenge_id, b"hypertraining");
    verify_leaf_sig(&leaf, &pk).unwrap();
    // Wrong pk fails
    let bad_pk = [0xFFu8; KEY_LEN];
    assert!(verify_leaf_sig(&leaf, &bad_pk).is_err());
}

/// S4: happy sim path — faster cand yields positive integer score.
#[test]
fn s4_happy_sim_faster_cand_positive_score() {
    let seeds = SegmentSeeds {
        run_seed: 11,
        aux_seed: 2,
    };
    let topo = Topology::new(2, 1, 1, 1);
    let budget = 5_000_000u64;
    let faster = search_faster_compiled(CHAMP_BIN, budget, &seeds, topo, 50_000).unwrap();
    let mut dedupe = hypertraining_challenge::default_dedupe().unwrap();
    let mut cluster = SimBackend::new();
    let (champ_loss, cand_loss) = fixture_equal_quality_pairs();
    let input = hypertraining_challenge::SimPipelineInput {
        cand_source: "def pipeline_overlap(x, y):\n    return compute(x)+y\n",
        cand_compiled: &faster,
        champ_source: CHAMP_SRC,
        champ_compiled: CHAMP_BIN,
        miner_id: "miner-fast",
        segment_index: 1,
        budget_tokens: budget,
        seeds,
        topology: topo,
        pkey_id: 7,
        noise_ms: 0,
        validator_lock: b"validator-lock-v1\n",
        admitted_files: vec![(
            "megatron/core/fusions/softmax.py".into(),
            b"x = 1\n".to_vec(),
        )],
        t_champ_ms_override: None,
        champ_loss,
        cand_loss,
    };
    let result = run_sim_pipeline(&input, &mut dedupe, &mut cluster).unwrap();
    assert!(result.score_u64 > 0 && result.score_u64 <= SCORE_MAX);
    assert!(result.cand_segment.wallclock_ms < result.t_champ_ms);
    assert!(result.eval.promote_allowed());

    // Feed into score_epoch → leaf
    let (sk, pk) = mini();
    let miner = [0x77u8; KEY_LEN];
    let ch = HypertrainingChallenge::sim();
    let ctx = ctx(pk, vec![miner.to_vec()]);
    let mut outcomes = BTreeMap::new();
    outcomes.insert(miner, result.outcome);
    let attest = MapAttestationLookup {
        default: AttestationStatus::Verified,
        ..Default::default()
    };
    let scores = ch.score_epoch(&ctx, &outcomes, &attest).unwrap();
    let leaf = ch
        .sign_leaf(&sk, miner, 42, scores[&miner].clone())
        .unwrap();
    verify_leaf_sig(&leaf, &pk).unwrap();
    assert!(matches!(
        leaf.score_or_absence,
        ScoreOrAbsence::Score { value } if value > 0
    ));
}
