#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Offline fixture build scenarios (no network).
//!
//! S1 stable digest ×2 · S2 missing validator lock · S3 miner lock forbidden · S4 order/content.

use hypertraining_build::{
    AdmittedSource, BuildError, BuildRequest, FixtureBuilder, HermeticBuilder, LockMaterial,
    ValidatorLock, Wheelhouse, BUILD_DIGEST_DOMAIN, DIGEST_PREFIX,
};

fn sample_source() -> AdmittedSource {
    AdmittedSource::new([
        (
            "miner_ext/kernel.py".into(),
            b"def fuse():\n    pass\n".to_vec(),
        ),
        ("megatron/core/fusions/x.py".into(), b"# ok\n".to_vec()),
    ])
    .expect("source")
}

fn sample_lock() -> ValidatorLock {
    ValidatorLock::new(b"torch==2.5.0\ntransformer-engine==2.18.0\n").expect("lock")
}

fn request(source: AdmittedSource, lock: LockMaterial) -> BuildRequest {
    BuildRequest {
        source,
        lock,
        wheelhouse: Wheelhouse::empty(),
    }
}

#[test]
fn fixture_digest_stable_across_two_runs() {
    let builder = FixtureBuilder::new();
    let req = request(sample_source(), sample_lock().into());
    let a = builder.build(&req).expect("build a");
    let b = builder.build(&req).expect("build b");
    assert_eq!(a.image_digest, b.image_digest);
    assert!(a.image_digest.starts_with(DIGEST_PREFIX));
    assert_eq!(a.image_digest.len(), DIGEST_PREFIX.len() + 64);
    assert_eq!(a.builder_id, "fixture");
    assert!(
        a.image_digest
            .chars()
            .skip(DIGEST_PREFIX.len())
            .all(|c| c.is_ascii_hexdigit()),
        "digest hex body"
    );
}

#[test]
fn fixture_digest_independent_of_source_insert_order() {
    let builder = FixtureBuilder::new();
    let s1 = AdmittedSource::new([("a.py".into(), b"1".to_vec()), ("b.py".into(), b"2".to_vec())])
        .expect("s1");
    let s2 = AdmittedSource::new([("b.py".into(), b"2".to_vec()), ("a.py".into(), b"1".to_vec())])
        .expect("s2");
    let lock = sample_lock();
    let d1 = builder
        .build(&request(s1, lock.clone().into()))
        .expect("d1")
        .image_digest;
    let d2 = builder
        .build(&request(s2, lock.into()))
        .expect("d2")
        .image_digest;
    assert_eq!(d1, d2);
}

#[test]
fn different_source_yields_different_digest() {
    let builder = FixtureBuilder::new();
    let lock = sample_lock();
    let s1 = sample_source();
    let s2 = AdmittedSource::new([("other.py".into(), b"x".to_vec())]).expect("s2");
    let d1 = builder
        .build(&request(s1, lock.clone().into()))
        .expect("d1")
        .image_digest;
    let d2 = builder
        .build(&request(s2, lock.into()))
        .expect("d2")
        .image_digest;
    assert_ne!(d1, d2);
}

#[test]
fn missing_validator_lock_fails() {
    let err = ValidatorLock::new(Vec::<u8>::new()).expect_err("empty");
    assert_eq!(err, BuildError::MissingValidatorLock);
}

#[test]
fn miner_lock_forbidden() {
    let builder = FixtureBuilder::new();
    let req = request(
        sample_source(),
        LockMaterial::Miner {
            contents: b"miner-pin==1.0\n".to_vec(),
        },
    );
    let err = builder.build(&req).expect_err("miner lock");
    assert_eq!(err, BuildError::MinerLockForbidden);
}

#[test]
fn empty_source_rejected() {
    let err = AdmittedSource::new(std::iter::empty::<(String, Vec<u8>)>()).expect_err("empty");
    assert_eq!(err, BuildError::EmptySource);
}

#[test]
fn wheelhouse_changes_digest() {
    let builder = FixtureBuilder::new();
    let lock = sample_lock();
    let src = sample_source();
    let bare = BuildRequest {
        source: src.clone(),
        lock: lock.clone().into(),
        wheelhouse: Wheelhouse::empty(),
    };
    let with_wh = BuildRequest {
        source: src,
        lock: lock.into(),
        wheelhouse: Wheelhouse::new([("torch.whl".into(), b"wheel-bytes".to_vec())]).expect("wh"),
    };
    let d1 = builder.build(&bare).expect("bare").image_digest;
    let d2 = builder.build(&with_wh).expect("wh").image_digest;
    assert_ne!(d1, d2);
}

#[test]
fn build_digest_domain_pin() {
    assert_eq!(BUILD_DIGEST_DOMAIN, b"base-hypertraining-build-v1");
}
