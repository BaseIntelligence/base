//! Epoch-semantics regression suite for the prism leaf emission outbox.
//!
//! Covers the staging-proven bugs (evidence: staging prism-run6):
//! (a) independent scorers finalized in the same epoch locked each other out
//!     (append-only first-write-wins leaves + acceptance-epoch strict match);
//! (b) a submission accepted in epoch X but finalized in X+k never scored.
//!
//! Post-fix semantics: one D24-complete set per chain epoch, fresh batch =
//! every row finalized since the previous emitted epoch (exactly-once via
//! the `emitted_epoch` watermark + emit cursor), unioned with active
//! positive scores so winners keep weight until a better score supersedes.

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use challenge_common::{
    ExpectedParticipant, ExpectedSet, GatewayClient, GatewayClientConfig, DRY_RUN_BASE_URL,
};
use crypto::KEY_LEN;
use prism_emit::EpochEmitter;
use prism_store::{
    ArchitectureRecord, FinalScore, MemoryPrismStore, PrismStore, Stage, SubmissionState,
};

fn sk() -> [u8; KEY_LEN] {
    let mut s = [7u8; KEY_LEN];
    s[0] = 0x42;
    s
}

fn dry_emitter(store: &Arc<MemoryPrismStore>) -> EpochEmitter {
    let gateway = GatewayClient::new(GatewayClientConfig {
        base_url: DRY_RUN_BASE_URL.into(),
        max_attempts: 1,
        backoff: std::time::Duration::ZERO,
    })
    .expect("dry gateway");
    EpochEmitter::new(Arc::clone(store) as Arc<dyn PrismStore>, sk(), 541, gateway)
}

fn hk(byte: u8) -> String {
    hex::encode([byte; 32])
}

fn expected(bytes: &[u8]) -> ExpectedSet {
    ExpectedSet {
        block_hash: [0x77u8; 32],
        participants: bytes
            .iter()
            .enumerate()
            .map(|(i, b)| ExpectedParticipant {
                hotkey: [*b; 32],
                uid: u16::try_from(i).unwrap_or(0),
            })
            .collect(),
    }
}

/// A finalized (terminal + scored) submission row, accepted at `accept_epoch`.
fn scored_row(id: &str, hotkey: &str, accept_epoch: u64, score: FinalScore) -> SubmissionState {
    SubmissionState {
        id: id.into(),
        miner_hotkey: hotkey.into(),
        miner_coldkey: None,
        epoch: accept_epoch,
        netuid: 541,
        status: Stage::Terminated,
        architecture_py: "a".into(),
        training_py: "t".into(),
        tree_blob: None,
        label: None,
        pod_id: None,
        pod_provider: None,
        receipt: None,
        metrics_json: None,
        bpb: Some(2.0),
        arch_id: None,
        review: None,
        similarity: None,
        final_score: Some(score),
        retry_count: 0,
        error_detail: None,
        created_at_ms: 1,
        updated_at_ms: 1,
    }
}

fn leaf_soa(summary: &prism_emit::EmitSummary, byte: u8) -> ScoreOrAbsence {
    summary
        .signed
        .get(&[byte; 32])
        .unwrap_or_else(|| panic!("no leaf for {byte:#x}"))
        .score_or_absence
        .clone()
}

fn assert_score(soa: &ScoreOrAbsence, want: u64) {
    assert!(
        matches!(soa, ScoreOrAbsence::Score { value } if *value == want),
        "expected Score({want}), got {soa:?}"
    );
}

fn assert_not_attempted(soa: &ScoreOrAbsence) {
    assert!(
        matches!(
            soa,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::NotAttempted
            }
        ),
        "expected NoScore(NotAttempted), got {soa:?}"
    );
}

/// (a) Two independent scorers finalized in the same chain epoch must both
/// appear in that epoch's leaf set (first-write-wins lockout is gone: the
/// emission is a single batched set, not one emit per finalize).
#[tokio::test]
async fn independent_same_epoch_scorers_both_land() {
    let store = Arc::new(MemoryPrismStore::new());
    store
        .insert_queued(&scored_row(
            "sub-a",
            &hk(0xAA),
            7,
            FinalScore::Score(100_000),
        ))
        .await
        .unwrap();
    store
        .insert_queued(&scored_row(
            "sub-b",
            &hk(0xBB),
            7,
            FinalScore::Score(200_000),
        ))
        .await
        .unwrap();

    let em = dry_emitter(&store);
    let exp = expected(&[0xAA, 0xBB, 0xCC]);
    let s = em.tick(7, &exp).await.unwrap().expect("first tick emits");

    assert_eq!(s.epoch, 7);
    assert_eq!(s.leaves, 3, "D24-complete set");
    assert_eq!(s.batch, 2, "both scorers in one batch");
    // WTA: only the argmax (BB=200k) keeps a positive Score leaf.
    assert_score(&leaf_soa(&s, 0xAA), 0);
    assert_score(&leaf_soa(&s, 0xBB), 200_000);
    assert_not_attempted(&leaf_soa(&s, 0xCC));
    assert_eq!(store.emit_cursor(541).await.unwrap(), Some(7));
}

/// (b) A submission accepted in epoch X but finalized in X+3 lands in the
/// next boundary's set (leaf epoch ≠ acceptance epoch). Outbox assignment
/// is exactly-once; the positive score then carries into later epochs.
#[tokio::test]
async fn late_finalize_scores_exactly_once() {
    let store = Arc::new(MemoryPrismStore::new());
    // Accepted in epoch 3; eval crossed three epoch boundaries before finalizing.
    store
        .insert_queued(&scored_row(
            "sub-a",
            &hk(0xAA),
            3,
            FinalScore::Score(500_000),
        ))
        .await
        .unwrap();

    let em = dry_emitter(&store);
    let exp = expected(&[0xAA, 0xCC]);

    let s = em.tick(7, &exp).await.unwrap().expect("emit at boundary");
    assert_eq!(s.batch, 1);
    assert_score(&leaf_soa(&s, 0xAA), 500_000);
    assert!(
        s.signed.values().all(|l| l.epoch == 7),
        "stamped with the emit epoch"
    );

    // Next epoch: outbox is empty (already assigned), but the positive score carries.
    let s8 = em.tick(8, &exp).await.unwrap().expect("epoch 8 emits");
    assert_eq!(s8.batch, 0, "no new finals since epoch 7");
    assert_score(&leaf_soa(&s8, 0xAA), 500_000);
    assert_eq!(store.emit_batch(541, 7).await.unwrap().len(), 1);
    assert!(store.emit_batch(541, 8).await.unwrap().is_empty());
    assert!(store.pending_emit_epochs(541).await.unwrap().is_empty());
}

/// Outbox assignment is exactly-once per finalize; positive scores still
/// carry into later epochs alongside any new finals (lattice max).
#[tokio::test]
async fn no_double_emission_across_epochs() {
    let store = Arc::new(MemoryPrismStore::new());
    store
        .insert_queued(&scored_row(
            "sub-a",
            &hk(0xAA),
            6,
            FinalScore::Score(100_000),
        ))
        .await
        .unwrap();

    let em = dry_emitter(&store);
    let exp = expected(&[0xAA, 0xBB]);

    let s = em.tick(7, &exp).await.unwrap().expect("emit");
    assert_eq!(s.batch, 1);
    // Tip refresh re-submits WTA for the same epoch; cursor stays put.
    let tip = em.tick(7, &exp).await.unwrap().expect("tip refresh");
    assert_eq!(tip.epoch, 7);
    assert_eq!(store.emit_cursor(541).await.unwrap(), Some(7));

    // A second scorer finalizes during epoch 7; only that row is freshly assigned.
    store
        .insert_queued(&scored_row(
            "sub-b",
            &hk(0xBB),
            7,
            FinalScore::Score(900_000),
        ))
        .await
        .unwrap();
    let s8 = em.tick(8, &exp).await.unwrap().expect("epoch 8 emits");
    assert_eq!(s8.batch, 1, "only the new row is freshly assigned");
    // WTA: BB's 900k beats AA's carried 100k — only BB emits positive.
    assert_score(&leaf_soa(&s8, 0xAA), 0);
    assert_score(&leaf_soa(&s8, 0xBB), 900_000);
    assert_eq!(store.emit_batch(541, 7).await.unwrap().len(), 1);
    assert_eq!(store.emit_batch(541, 8).await.unwrap().len(), 1);
}

/// Prod incident regression: winner emitted at epoch N, epoch N+1 seals with
/// only a Score(0) reject — winner weights must still appear (no burn).
#[tokio::test]
async fn positive_score_carries_when_next_epoch_is_reject_only() {
    let store = Arc::new(MemoryPrismStore::new());
    store
        .insert_queued(&scored_row(
            "winner",
            &hk(0xAA),
            24353,
            FinalScore::Score(800_000),
        ))
        .await
        .unwrap();

    let em = dry_emitter(&store);
    let exp = expected(&[0xAA, 0xBB]);
    let s = em.tick(24353, &exp).await.unwrap().expect("epoch 24353");
    assert_eq!(s.batch, 1);
    assert_score(&leaf_soa(&s, 0xAA), 800_000);

    store
        .insert_queued(&scored_row(
            "reject",
            &hk(0xBB),
            24354,
            FinalScore::Score(0),
        ))
        .await
        .unwrap();
    let s2 = em.tick(24354, &exp).await.unwrap().expect("epoch 24354");
    assert_eq!(s2.batch, 1, "reject is freshly assigned once");
    assert_score(&leaf_soa(&s2, 0xAA), 800_000);
    assert_score(&leaf_soa(&s2, 0xBB), 0);
    assert_eq!(store.emit_batch(541, 24353).await.unwrap().len(), 1);
    assert_eq!(store.emit_batch(541, 24354).await.unwrap().len(), 1);
    assert_eq!(store.active_score_rows(541).await.unwrap().len(), 1);
}

/// Competition credit semantics (prism-registry) are preserved through the
/// outbox — both when owner + challenger finalize in the same epoch and when
/// the challenger lands in a later epoch (the run6 scenario, minus the
/// epoch-boundary choreography the staging script needed pre-fix).
#[tokio::test]
async fn competition_credits_survive_batching() {
    let store = Arc::new(MemoryPrismStore::new());
    store
        .publish_arch(&ArchitectureRecord {
            arch_id: "arch_0123456789abcdef".into(),
            owner_hotkey: hk(0xAA),
            arch_digest: "00".repeat(32),
            architecture_py: "a".into(),
            source_submission: "sub-owner".into(),
            best_bpb: Some(2.0),
            created_at_ms: 1_000,
        })
        .await
        .unwrap();
    let mut owner = scored_row("sub-owner", &hk(0xAA), 6, FinalScore::Score(400_000));
    owner.arch_id = Some("arch_0123456789abcdef".into());
    store.insert_queued(&owner).await.unwrap();

    let em = dry_emitter(&store);
    let exp = expected(&[0xAA, 0xBB]);

    // Owner's arch submission emits first (own credit only).
    let s7 = em.tick(7, &exp).await.unwrap().expect("epoch 7");
    assert_score(&leaf_soa(&s7, 0xAA), 400_000);
    assert_not_attempted(&leaf_soa(&s7, 0xBB));

    // Challenger trains the published arch and lands in a later epoch.
    // TEMP: owner-arch credit off → WTA goes to the BPB/score winner (BB).
    let mut chall = scored_row("sub-chall", &hk(0xBB), 7, FinalScore::Score(900_000));
    chall.arch_id = Some("arch_0123456789abcdef".into());
    store.insert_queued(&chall).await.unwrap();
    let s8 = em.tick(8, &exp).await.unwrap().expect("epoch 8");
    assert_eq!(s8.batch, 1);
    const {
        assert!(
            !prism_registry::OWNER_ARCH_CREDIT_ENABLED,
            "update expectations when restoring owner-arch credit"
        );
    }
    assert_score(&leaf_soa(&s8, 0xBB), 900_000);
    assert_score(&leaf_soa(&s8, 0xAA), 0);

    // Same-epoch variant: both rows in one batch → same WTA outcome.
    let store2 = Arc::new(MemoryPrismStore::new());
    store2
        .publish_arch(&ArchitectureRecord {
            arch_id: "arch_0123456789abcdef".into(),
            owner_hotkey: hk(0xAA),
            arch_digest: "00".repeat(32),
            architecture_py: "a".into(),
            source_submission: "sub-owner".into(),
            best_bpb: Some(2.0),
            created_at_ms: 1_000,
        })
        .await
        .unwrap();
    store2.insert_queued(&owner).await.unwrap();
    store2.insert_queued(&chall).await.unwrap();
    let em2 = dry_emitter(&store2);
    let s = em2.tick(7, &exp).await.unwrap().expect("emit");
    assert_eq!(s.batch, 2);
    assert_score(&leaf_soa(&s, 0xBB), 900_000);
    assert_score(&leaf_soa(&s, 0xAA), 0);
}

/// Crash recovery: a batch assigned but never cursor-completed (crashed
/// mid-submit) is replayed from the sticky assignment, not re-assigned —
/// and after recovery the epoch counts as done.
#[tokio::test]
async fn crash_replays_sticky_assignment() {
    let store = Arc::new(MemoryPrismStore::new());
    store
        .insert_queued(&scored_row(
            "sub-a",
            &hk(0xAA),
            5,
            FinalScore::Score(700_000),
        ))
        .await
        .unwrap();

    // Simulate: batch assigned to epoch 9, process dies before submit/cursor.
    let assigned = store.assign_emit_batch(541, 9).await.unwrap();
    assert_eq!(assigned.len(), 1);
    assert_eq!(store.emit_cursor(541).await.unwrap(), None);

    let em = dry_emitter(&store);
    let exp = expected(&[0xAA]);
    // Even though the chain has moved to epoch 11, recovery replays epoch 9's
    // assigned set first, then emits the live epoch with the (empty) backlog.
    let s = em
        .tick(11, &exp)
        .await
        .unwrap()
        .expect("live emit after recovery");
    assert_eq!(store.emit_cursor(541).await.unwrap(), Some(11));
    assert!(store.pending_emit_epochs(541).await.unwrap().is_empty());
    assert_eq!(s.epoch, 11);
    assert_eq!(s.batch, 0, "the backlog was already assigned to epoch 9");
    // Active positive score still carries into the live epoch after recovery.
    assert_score(&leaf_soa(&s, 0xAA), 700_000);
    // The epoch-9 replay kept the score (sticky batch content).
    assert_eq!(store.emit_batch(541, 9).await.unwrap().len(), 1);
}

/// A manually retried row re-enters the outbox: its old leaf stays immutable
/// history in the old epoch; the new scoring emits once in a later epoch.
#[tokio::test]
async fn retry_reenters_outbox() {
    let store = Arc::new(MemoryPrismStore::new());
    store
        .insert_queued(&scored_row(
            "sub-a",
            &hk(0xAA),
            6,
            FinalScore::NoScore(6), // ChallengeInternal (infra failure)
        ))
        .await
        .unwrap();

    let em = dry_emitter(&store);
    let exp = expected(&[0xAA]);
    let s7 = em.tick(7, &exp).await.unwrap().expect("epoch 7");
    assert!(matches!(
        leaf_soa(&s7, 0xAA),
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    ));

    store.reset_for_retry("sub-a", true).await.unwrap();
    let row = store.get("sub-a").await.unwrap().expect("row");
    assert!(row.final_score.is_none());
    // Re-score via apply (mirrors the orchestrator's finalize write).
    store
        .apply(
            "sub-a",
            &prism_store::StatePatch {
                status: Some(Stage::Terminated),
                final_score: Some(FinalScore::Score(321_000)),
                ..prism_store::StatePatch::default()
            },
            None,
        )
        .await
        .unwrap();
    let row = store.get("sub-a").await.unwrap().expect("row");
    assert!(matches!(row.final_score, Some(FinalScore::Score(321_000))));

    let s8 = em.tick(8, &exp).await.unwrap().expect("epoch 8");
    assert_eq!(s8.batch, 1);
    assert_score(&leaf_soa(&s8, 0xAA), 321_000);
    // No replay of the epoch-7 absence into epoch 8's batch composition.
    assert_eq!(store.emit_batch(541, 7).await.unwrap().len(), 0);
    assert_eq!(store.emit_batch(541, 8).await.unwrap().len(), 1);
}
