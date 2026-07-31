//! `ClusterBackend` contract tests (Sim + Real).

use hypertraining_cluster::{
    ClusterBackend, ClusterError, RealBackend, SegmentConfig, SegmentSeeds, SimBackend, Topology,
};

const SIM_BASE_WALLCLOCK_MS: u64 = 1_000;

fn topo_32() -> Topology {
    Topology::new(4, 2, 4, 1)
}

fn cfg(fp_byte: u8, noise_ms: u32) -> SegmentConfig {
    SegmentConfig {
        code_fingerprint: [fp_byte; 32],
        budget_tokens: 5_000_000_000,
        seeds: SegmentSeeds {
            run_seed: 42,
            aux_seed: 7,
        },
        master_topology: topo_32(),
        slot_topology: topo_32(),
        pkey_id: 0x1234,
        noise_ms,
    }
}

#[test]
fn sim_segment_completes_with_checkpoint_and_wallclock() {
    let mut backend = SimBackend::new();
    let c = cfg(0xAB, 0);
    let result = backend.run_segment(&c).expect("sim segment");
    assert!(result.wallclock_ms >= SIM_BASE_WALLCLOCK_MS);
    assert_ne!(result.checkpoint_hash, [0u8; 32]);
    assert_eq!(result.telemetry.backend, "sim");
    assert_eq!(result.telemetry.tokens_processed, c.budget_tokens);
    assert_eq!(result.seeds, c.seeds);
    assert_eq!(result.telemetry.pkey_id, c.pkey_id);
}

#[test]
fn sim_wallclock_deterministic_for_same_fingerprint() {
    let mut a = SimBackend::new();
    let mut b = SimBackend::new();
    let c = cfg(0x11, 0);
    let r1 = a.run_segment(&c).expect("a");
    let r2 = b.run_segment(&c).expect("b");
    assert_eq!(r1.wallclock_ms, r2.wallclock_ms);
    assert_eq!(r1.checkpoint_hash, r2.checkpoint_hash);
}

#[test]
fn sim_wallclock_changes_with_fingerprint() {
    let mut backend = SimBackend::new();
    let r1 = backend.run_segment(&cfg(0x01, 0)).expect("fp1");
    backend.release_slot(0x1234);
    let r2 = backend.run_segment(&cfg(0x02, 0)).expect("fp2");
    assert_ne!(r1.wallclock_ms, r2.wallclock_ms);
    assert_ne!(r1.checkpoint_hash, r2.checkpoint_hash);
}

#[test]
fn sim_noise_param_affects_wallclock_surface() {
    let mut backend = SimBackend::new();
    let quiet = backend
        .run_segment(&cfg(0x55, 0))
        .expect("quiet")
        .wallclock_ms;
    backend.release_slot(0x1234);
    let n1 = backend.run_segment(&cfg(0x55, 1)).expect("n1").wallclock_ms;
    backend.release_slot(0x1234);
    let n2 = backend
        .run_segment(&cfg(0x55, 50_000))
        .expect("n2")
        .wallclock_ms;
    assert!(
        n1 != quiet || n2 != quiet || n1 != n2,
        "noise_ms must be able to affect wallclock (got quiet={quiet} n1={n1} n2={n2})"
    );
}

#[test]
fn topology_mismatch_rejected() {
    let backend = SimBackend::new();
    let master = Topology::new(4, 2, 4, 1);
    let slot = Topology::new(8, 1, 4, 1);
    let err = backend
        .check_topology_mirror(master, slot)
        .expect_err("mismatch");
    assert_eq!(err, ClusterError::TopologyMismatch { master, slot });
}

#[test]
fn run_segment_rejects_topology_mismatch() {
    let mut backend = SimBackend::new();
    let mut c = cfg(0xCD, 0);
    c.slot_topology = Topology::new(1, 1, 1, 1);
    let err = backend.run_segment(&c).expect_err("topo");
    assert!(matches!(err, ClusterError::TopologyMismatch { .. }));
}

#[test]
fn exclusive_slot_busy_on_double_allocate() {
    let mut backend = SimBackend::new();
    let s1 = backend.allocate_exclusive_slot(9).expect("first");
    assert_eq!(s1.pkey_id, 9);
    let err = backend.allocate_exclusive_slot(9).expect_err("second");
    assert_eq!(err, ClusterError::SlotBusy { pkey_id: 9 });
}

#[test]
fn zero_budget_invalid() {
    let mut backend = SimBackend::new();
    let mut c = cfg(0x00, 0);
    c.budget_tokens = 0;
    let err = backend.run_segment(&c).expect_err("budget");
    assert!(matches!(err, ClusterError::InvalidConfig(_)));
}

#[test]
fn real_run_segment_returns_not_configured() {
    let mut backend = RealBackend::new();
    let err = backend
        .run_segment(&cfg(0x01, 0))
        .expect_err("real deferred");
    assert_eq!(err, ClusterError::NotConfigured);
    let msg = err.to_string();
    assert!(
        msg.contains("B300") && msg.contains("deferred"),
        "message must mention B300 deferred, got: {msg}"
    );
}

#[test]
fn real_allocate_and_topology_not_configured() {
    let mut backend = RealBackend::new();
    assert_eq!(
        backend.allocate_exclusive_slot(1),
        Err(ClusterError::NotConfigured)
    );
    assert_eq!(
        backend.check_topology_mirror(Topology::new(1, 1, 1, 1), Topology::new(1, 1, 1, 1)),
        Err(ClusterError::NotConfigured)
    );
}

#[test]
fn real_does_not_fabricate_wallclock() {
    let mut backend = RealBackend::new();
    assert!(backend.run_segment(&cfg(0x01, 0)).is_err());
}
