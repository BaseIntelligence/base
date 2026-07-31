//! Live dual-truth / failure QA for the operator-side held-out verifier.
//!
//! Gated on `GBASE_VERIFY_LIVE=1` and a reachable socket-proxy + env image.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_challenge::{HarborVerifier, HarborVerifierConfig, Verifier, VerifyError, ZeroReason};
use agent_pack::load_pack;
use docker_engine::{Allowlist, AllowlistClient, DockerApi, RunSpec, OWNED_NAME_PREFIX};

fn live_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn live_enabled() -> bool {
    std::env::var("GBASE_VERIFY_LIVE").ok().as_deref() == Some("1")
}

fn pack_dir() -> PathBuf {
    std::env::var("GBASE_VERIFY_PACK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/da_m18c_hf_pull/tasks/realpr-more-itertools-1136"))
}

fn docker_base() -> String {
    std::env::var("GBASE_DOCKER_BASE").unwrap_or_else(|_| "http://127.0.0.1:2375".into())
}

fn env_image() -> String {
    std::env::var("GBASE_VERIFY_IMAGE").unwrap_or_else(|_| {
        "gbase-verify-env-more-itertools-1136@sha256:462caa0ae2f4ce87509323a33c383eb6b5c364fff4350ba33c2c2bddae62537f"
            .into()
    })
}

fn make_verifier_at(work_root: PathBuf, reward_zero_as_err: bool) -> HarborVerifier {
    HarborVerifier::new(HarborVerifierConfig {
        docker_base: docker_base(),
        environment_image: env_image(),
        work_root,
        timeout_sec_override: Some(600),
        reward_zero_as_err,
    })
    .expect("verifier")
}

#[test]
fn dual_truth_solution_one_empty_zero() {
    let _guard = live_lock();
    if !live_enabled() {
        eprintln!("skip: set GBASE_VERIFY_LIVE=1");
        return;
    }
    let pack = load_pack(pack_dir()).expect("load pack");
    let solution = pack
        .held_out
        .solution_patch
        .clone()
        .expect("solution.patch present");
    let work = PathBuf::from(format!(
        "/tmp/gbase-verify-it-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let v = make_verifier_at(work.clone(), false);
    let r1 = v.grade(&pack, &solution).expect("sol grade");
    assert_eq!(r1.value(), 1, "solution.patch must resolve");
    let r0 = v.grade(&pack, b"").expect("empty grade");
    assert_eq!(r0.value(), 0, "empty patch must not resolve");
    assert_eq!(v.owned_count().expect("owned"), 0);
    let _ = std::fs::remove_dir_all(work);
    println!("DUAL_TRUTH {} then {}", r1.value(), r0.value());
}

#[test]
fn invalid_and_p2p_break_distinct_zero_reasons() {
    let _guard = live_lock();
    if !live_enabled() {
        eprintln!("skip: set GBASE_VERIFY_LIVE=1");
        return;
    }
    let pack = load_pack(pack_dir()).expect("load pack");
    let work = PathBuf::from(format!(
        "/tmp/gbase-verify-fail-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let v = make_verifier_at(work.clone(), true);

    let bad = v.grade(&pack, b"not-a-unified-diff\n").expect_err("bad");
    assert!(
        matches!(bad, VerifyError::ApplyFailed { .. }),
        "invalid patch → ApplyFailed, got {bad:?}"
    );

    // Empty patch: tests run, F2P fail → TestsFailed (not ApplyFailed).
    let empty = v.grade(&pack, b"").expect_err("empty");
    match empty {
        VerifyError::RewardZero {
            reason: ZeroReason::TestsFailed { f2p_failed, .. },
        } => assert!(f2p_failed > 0, "expected f2p failures"),
        other => panic!("empty should be TestsFailed reward0, got {other:?}"),
    }

    assert_eq!(
        v.owned_count().expect("owned"),
        0,
        "no surviving containers"
    );
    let _ = std::fs::remove_dir_all(work);
}

#[test]
fn timeout_is_typed_not_reward_zero() {
    let _guard = live_lock();
    if !live_enabled() {
        eprintln!("skip: set GBASE_VERIFY_LIVE=1");
        return;
    }
    let client = AllowlistClient::with_allowlist(docker_base(), Allowlist::verifier()).expect("c");
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let name = format!("{OWNED_NAME_PREFIX}timeout-{ns}");
    let err = client
        .run_owned(&RunSpec {
            name: name.clone(),
            image:
                "busybox@sha256:73aaf090f3d85aa34ee199857f03fa3a95c8ede2ffd4cc2cdb5b94e566b11662"
                    .into(),
            cmd: vec!["sh".into(), "-c".into(), "sleep 30".into()],
            binds: vec![],
            env: vec![],
            network_disabled: true,
            working_dir: None,
            timeout_sec: Some(2),
        })
        .expect_err("must timeout");
    assert!(
        matches!(err, docker_engine::DockerError::Timeout { timeout_sec: 2 }),
        "got {err:?}"
    );
    let mapped = agent_challenge::map_docker_timeout(&err).expect("map");
    assert!(matches!(mapped, VerifyError::Timeout { timeout_sec: 2 }));
    assert!(!matches!(mapped, VerifyError::RewardZero { .. }));
    let left = client.cleanup_owned().expect("cleanup");
    assert_eq!(left, 0);
    let _ = client.remove_container(&name);
}
