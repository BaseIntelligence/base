//! Epoch dispatch loop: fake runners, deadline, single active signer.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_challenge::{
    hex32, run_epoch_dispatch, ActiveSignerRegistry, EpochDispatchClient, EpochDispatchConfig,
    EpochDispatchResult, EpochLoopError, ExpectedParticipant, ExpectedSet, MinerEpochOutcome,
    PinnedBlockHash, RunnerCapacity, CHALLENGE_ID, SCORING_VERSION,
};
use agent_dispatch::{TaskDescriptorV1, TaskResultV1, TaskStatusV1, DISPATCH_PROTOCOL};
use agent_pack::PackId;

#[derive(Clone, Copy, Debug)]
enum Behavior {
    Fast,
    Slow,
    Dead,
    NoCapacity,
}

struct FakeClient {
    by_miner: BTreeMap<[u8; 32], Behavior>,
    started: AtomicUsize,
    fast_before_slow: AtomicUsize,
}

impl FakeClient {
    fn new(pairs: &[([u8; 32], Behavior)]) -> Arc<Self> {
        let mut by_miner = BTreeMap::new();
        for (k, b) in pairs {
            by_miner.insert(*k, *b);
        }
        Arc::new(Self {
            by_miner,
            started: AtomicUsize::new(0),
            fast_before_slow: AtomicUsize::new(0),
        })
    }
}

impl EpochDispatchClient for FakeClient {
    async fn capacity(&self, miner: [u8; 32]) -> RunnerCapacity {
        match self.by_miner.get(&miner).copied().unwrap_or(Behavior::Fast) {
            Behavior::NoCapacity => RunnerCapacity {
                max_concurrency: 1,
                current_load: 1,
            },
            _ => RunnerCapacity {
                max_concurrency: 2,
                current_load: 0,
            },
        }
    }

    async fn run_pack(
        &self,
        miner: [u8; 32],
        descriptor: TaskDescriptorV1,
    ) -> Result<TaskResultV1, String> {
        self.started.fetch_add(1, Ordering::SeqCst);
        match self.by_miner.get(&miner).copied().unwrap_or(Behavior::Fast) {
            Behavior::NoCapacity => unreachable!("capacity gate should skip run_pack"),
            Behavior::Fast => {
                self.fast_before_slow.store(1, Ordering::SeqCst);
                Ok(ok_result(&descriptor, "fast-patch"))
            }
            Behavior::Slow => {
                tokio::time::sleep(Duration::from_millis(80)).await;
                Ok(ok_result(&descriptor, "slow-patch"))
            }
            Behavior::Dead => {
                std::future::pending::<()>().await;
                unreachable!()
            }
        }
    }
}

fn ok_result(d: &TaskDescriptorV1, patch: &str) -> TaskResultV1 {
    TaskResultV1 {
        protocol: DISPATCH_PROTOCOL.into(),
        challenge_id: d.challenge_id.clone(),
        scoring_version: d.scoring_version,
        epoch: d.epoch,
        miner_hotkey_hex: d.miner_hotkey_hex.clone(),
        pack_id: d.pack_id.clone(),
        status: TaskStatusV1::Completed,
        model_patch: Some(patch.into()),
        patch_sha256_hex: "00".repeat(32),
        receipt_sig_hex: "00".repeat(64),
    }
}

fn hk(b: u8) -> [u8; 32] {
    [b; 32]
}

fn expected_three() -> ExpectedSet {
    ExpectedSet {
        block_hash: {
            let mut h = [0u8; 32];
            h[0] = 0xBB;
            h
        },
        participants: vec![
            ExpectedParticipant {
                hotkey: hk(0xA1),
                uid: 1,
            },
            ExpectedParticipant {
                hotkey: hk(0xB2),
                uid: 2,
            },
            ExpectedParticipant {
                hotkey: hk(0xC3),
                uid: 3,
            },
        ],
    }
}

fn catalog() -> Vec<PackId> {
    vec![PackId::new("pack-alpha"), PackId::new("pack-beta")]
}

fn cfg(
    epoch: u64,
    expected: ExpectedSet,
    deadline: Duration,
    deadline_unix_ms: u64,
) -> EpochDispatchConfig {
    EpochDispatchConfig {
        challenge_id: CHALLENGE_ID.into(),
        scoring_version: SCORING_VERSION,
        epoch,
        expected,
        catalog: catalog(),
        deadline,
        deadline_unix_ms,
    }
}

fn format_outcome_table(result: &EpochDispatchResult) -> String {
    let mut s = format!(
        "epoch={} block_hash={} |E|={}\nhotkey                                    outcome              pack_id\n",
        result.epoch,
        hex32(&result.block_hash),
        result.outcomes.len()
    );
    for (key, o) in &result.outcomes {
        let (k, p) = match o {
            MinerEpochOutcome::Completed { pack_id, .. } => ("completed", pack_id.as_str()),
            MinerEpochOutcome::TimedOut { pack_id } => ("timed_out", pack_id.as_str()),
            MinerEpochOutcome::Failed { pack_id, .. } => ("failed", pack_id.as_str()),
            MinerEpochOutcome::CapacityExhausted { pack_id } => {
                ("capacity_exhausted", pack_id.as_str())
            }
        };
        let _ = writeln!(s, "{:<40} {k:<20} {p}", hex32(key));
    }
    s
}

/// S1 — fast + slow + dead → exactly |E| outcomes; table printable.
#[tokio::test]
async fn s1_epoch_fast_slow_dead_outcomes() {
    let expected = expected_three();
    let n = expected.participants.len();
    let client = FakeClient::new(&[
        (hk(0xA1), Behavior::Fast),
        (hk(0xB2), Behavior::Slow),
        (hk(0xC3), Behavior::Dead),
    ]);
    let signers = ActiveSignerRegistry::new();
    let config = cfg(7, expected, Duration::from_millis(120), 1_700_000_000_120);
    let t0 = Instant::now();
    let result = run_epoch_dispatch(&config, Arc::clone(&client), &signers)
        .await
        .expect("epoch");
    let elapsed = t0.elapsed();

    assert_eq!(result.outcomes.len(), n, "exactly |E| outcomes");
    let table = format_outcome_table(&result);
    eprintln!("OUTCOME_TABLE\n{table}");
    assert!(table.contains("completed"), "{table}");
    assert!(table.contains("timed_out"), "{table}");

    let mut kinds = BTreeMap::new();
    for (key, o) in &result.outcomes {
        let label = match o {
            MinerEpochOutcome::Completed { .. } => "completed",
            MinerEpochOutcome::TimedOut { .. } => "timed_out",
            MinerEpochOutcome::Failed { .. } => "failed",
            MinerEpochOutcome::CapacityExhausted { .. } => "capacity_exhausted",
        };
        kinds.insert(*key, label);
    }
    assert_eq!(kinds.get(&hk(0xA1)), Some(&"completed"));
    assert_eq!(kinds.get(&hk(0xB2)), Some(&"completed"));
    assert_eq!(kinds.get(&hk(0xC3)), Some(&"timed_out"));
    assert!(
        elapsed < Duration::from_millis(400),
        "parallel dispatch expected, elapsed={elapsed:?}"
    );
    assert_eq!(client.started.load(Ordering::SeqCst), 3);
    assert_eq!(client.fast_before_slow.load(Ordering::SeqCst), 1);
}

/// S2 — hang past deadline → `TimedOut`; epoch still completes with |E|.
#[tokio::test]
async fn s2_deadline_timeout_epoch_completes() {
    let expected = ExpectedSet {
        block_hash: [0xDD; 32],
        participants: vec![
            ExpectedParticipant {
                hotkey: hk(0x11),
                uid: 0,
            },
            ExpectedParticipant {
                hotkey: hk(0x22),
                uid: 1,
            },
        ],
    };
    let client = FakeClient::new(&[(hk(0x11), Behavior::Dead), (hk(0x22), Behavior::Fast)]);
    let signers = ActiveSignerRegistry::new();
    let config = cfg(3, expected, Duration::from_millis(50), 1_700_000_000_050);
    let result = run_epoch_dispatch(&config, client, &signers)
        .await
        .expect("completes");
    assert_eq!(result.outcomes.len(), 2);
    assert!(matches!(
        result.outcomes.get(&hk(0x11)),
        Some(MinerEpochOutcome::TimedOut { .. })
    ));
    assert!(matches!(
        result.outcomes.get(&hk(0x22)),
        Some(MinerEpochOutcome::Completed { .. })
    ));
}

/// S3 — second concurrent signer for same epoch is refused.
#[tokio::test]
async fn s3_second_signer_refused() {
    let signers = ActiveSignerRegistry::new();
    let g1 = signers.try_acquire("agent-v1", 99).expect("first");
    let err = signers.try_acquire("agent-v1", 99).expect_err("second");
    assert!(
        matches!(err, EpochLoopError::SignerAlreadyActive { epoch: 99, .. }),
        "{err:?}"
    );
    drop(g1);
    let _g2 = signers.try_acquire("agent-v1", 99).expect("after drop");
}

/// S3b — `run_epoch_dispatch` refuses when lease already held.
#[tokio::test]
async fn s3b_run_epoch_refuses_second_signer() {
    let expected = ExpectedSet {
        block_hash: [0xEE; 32],
        participants: vec![ExpectedParticipant {
            hotkey: hk(0x55),
            uid: 0,
        }],
    };
    let client = FakeClient::new(&[(hk(0x55), Behavior::Fast)]);
    let signers = ActiveSignerRegistry::new();
    let _held = signers.try_acquire("agent-v1", 42).unwrap();
    let config = cfg(42, expected, Duration::from_millis(100), 0);
    let err = run_epoch_dispatch(&config, client, &signers)
        .await
        .expect_err("refused");
    assert!(matches!(err, EpochLoopError::SignerAlreadyActive { .. }));
}

/// S4 — abort in-flight `JoinSet` without leak.
#[tokio::test]
async fn s4_cancel_aborts_inflight() {
    let expected = ExpectedSet {
        block_hash: [0xFF; 32],
        participants: vec![ExpectedParticipant {
            hotkey: hk(0x77),
            uid: 0,
        }],
    };
    let client = FakeClient::new(&[(hk(0x77), Behavior::Dead)]);
    let signers = ActiveSignerRegistry::new();
    let config = cfg(1, expected, Duration::from_secs(30), 0);
    let handle = tokio::spawn(async move { run_epoch_dispatch(&config, client, &signers).await });
    tokio::time::sleep(Duration::from_millis(20)).await;
    handle.abort();
    let err = handle.await.expect_err("aborted");
    assert!(err.is_cancelled());
}

/// Capacity gate: `free_slots==0` → `CapacityExhausted`, no `run_pack`.
#[tokio::test]
async fn capacity_exhausted_skips_run() {
    let expected = ExpectedSet {
        block_hash: [0x01; 32],
        participants: vec![ExpectedParticipant {
            hotkey: hk(0x99),
            uid: 0,
        }],
    };
    let client = FakeClient::new(&[(hk(0x99), Behavior::NoCapacity)]);
    let signers = ActiveSignerRegistry::new();
    let config = cfg(5, expected, Duration::from_millis(50), 0);
    let result = run_epoch_dispatch(&config, Arc::clone(&client), &signers)
        .await
        .unwrap();
    assert_eq!(client.started.load(Ordering::SeqCst), 0);
    assert!(matches!(
        result.outcomes.get(&hk(0x99)),
        Some(MinerEpochOutcome::CapacityExhausted { .. })
    ));
}

#[test]
fn pinned_block_api_smoke() {
    let p = PinnedBlockHash::new([0xAB; 32]);
    assert_eq!(p.as_bytes()[0], 0xAB);
}
