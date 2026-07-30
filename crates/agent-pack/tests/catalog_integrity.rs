//! Integration: pinned catalog materialize / load / integrity.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::path::{Path, PathBuf};

use agent_pack::{
    build_manifest, compute_catalog_digest, load_catalog, manifest_to_bytes, materialize_catalog,
    materialize_catalog_with_pin, select_pack, CatalogEntry, CatalogError, CATALOG_DIGEST_DOMAIN,
    DEEPAGENT_PIN, MANIFEST_FILE_NAME,
};

fn fixture_tasks() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn stage_source(tmp: &Path, fixture_names: &[&str]) -> PathBuf {
    let tasks = tmp.join("tasks");
    fs::create_dir_all(&tasks).unwrap();
    let fixtures = fixture_tasks();
    for name in fixture_names {
        let src = fixtures.join(name);
        let dst = tasks.join(name);
        copy_tree(&src, &dst);
    }
    tasks
}

fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_tree(&from, &to);
        } else {
            fs::copy(&from, &to).unwrap();
        }
    }
}

#[test]
fn deepagent_pin_is_fixed_commit() {
    assert_eq!(DEEPAGENT_PIN, "4a16f063c83032ad4db2bb5a3099608bfdcb5fe2");
    assert_ne!(DEEPAGENT_PIN, "latest");
    assert_eq!(DEEPAGENT_PIN.len(), 40);
}

#[test]
fn catalog_digest_domain_tag() {
    assert_eq!(CATALOG_DIGEST_DOMAIN, b"gbase-agent-pack-catalog-v1");
}

#[test]
fn catalog_digest_order_independent() {
    let a = vec![
        CatalogEntry {
            pack_id: "b".into(),
            pack_digest: "dd".into(),
            environment_image_digest: "sha256:bb".into(),
        },
        CatalogEntry {
            pack_id: "a".into(),
            pack_digest: "cc".into(),
            environment_image_digest: "sha256:aa".into(),
        },
    ];
    let b = vec![a[1].clone(), a[0].clone()];
    assert_eq!(
        compute_catalog_digest(DEEPAGENT_PIN, &a),
        compute_catalog_digest(DEEPAGENT_PIN, &b)
    );
}

#[test]
fn catalog_digest_pin_sensitive() {
    let entries = vec![CatalogEntry {
        pack_id: "p".into(),
        pack_digest: "aa".into(),
        environment_image_digest: "sha256:00".into(),
    }];
    let d1 = compute_catalog_digest("pin-a", &entries);
    let d2 = compute_catalog_digest("pin-b", &entries);
    assert_ne!(d1, d2);
}

#[test]
fn empty_materialize_is_typed_empty_error() {
    let tmp = tempfile::tempdir().unwrap();
    let empty_src = tmp.path().join("empty_tasks");
    fs::create_dir_all(&empty_src).unwrap();
    let cache = tmp.path().join("cache");
    let err = materialize_catalog(&empty_src, &cache).expect_err("empty");
    assert_eq!(err, CatalogError::Empty);
    assert!(err.to_string().contains("empty"));
}

#[test]
fn floating_pin_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let src = stage_source(tmp.path(), &["minimal-ok"]);
    let cache = tmp.path().join("cache");
    let err = materialize_catalog_with_pin(&src, &cache, "latest").expect_err("float");
    assert!(matches!(err, CatalogError::FloatingPin(_)));
}

#[test]
fn materialize_and_load_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    let src = stage_source(tmp.path(), &["minimal-ok"]);
    let cache = tmp.path().join("cache");
    let m = materialize_catalog(&src, &cache).expect("materialize");
    assert_eq!(m.pin, DEEPAGENT_PIN);
    assert_eq!(m.entries.len(), 1);
    assert_eq!(m.entries[0].pack_id, "minimal-ok");
    assert_eq!(m.catalog_digest, compute_catalog_digest(&m.pin, &m.entries));

    let cat = load_catalog(&cache).expect("load");
    assert_eq!(cat.catalog_digest(), m.catalog_digest);
    assert_eq!(cat.len(), 1);
    assert!(!cat.is_empty());
    let ids = cat.pack_ids();
    assert_eq!(ids[0].as_str(), "minimal-ok");
}

#[test]
fn two_materializations_byte_identical_manifests() {
    let tmp = tempfile::tempdir().unwrap();
    let src = stage_source(tmp.path(), &["minimal-ok"]);
    let c1 = tmp.path().join("cache1");
    let c2 = tmp.path().join("cache2");
    let m1 = materialize_catalog(&src, &c1).expect("m1");
    let m2 = materialize_catalog(&src, &c2).expect("m2");
    assert_eq!(m1, m2);
    let b1 = fs::read(c1.join(MANIFEST_FILE_NAME)).unwrap();
    let b2 = fs::read(c2.join(MANIFEST_FILE_NAME)).unwrap();
    assert_eq!(b1, b2, "manifest bytes must be identical");
    assert_eq!(b1, manifest_to_bytes(&m1).unwrap());
}

#[test]
fn tampered_instruction_fails_integrity_naming_pack() {
    let tmp = tempfile::tempdir().unwrap();
    let src = stage_source(tmp.path(), &["minimal-ok"]);
    let cache = tmp.path().join("cache");
    materialize_catalog(&src, &cache).expect("materialize");

    let instr = cache.join("packs/minimal-ok/instruction.md");
    let mut body = fs::read(&instr).unwrap();
    body[0] ^= 0x01;
    fs::write(&instr, body).unwrap();

    let err = load_catalog(&cache).expect_err("tamper");
    let display = err.to_string();
    assert!(
        display.contains("minimal-ok"),
        "error display must name pack: {display}"
    );
    match err {
        CatalogError::Integrity { pack_id, message } => {
            assert_eq!(pack_id, "minimal-ok");
            assert!(
                message.contains("pack_digest"),
                "message should mention pack_digest: {message}"
            );
        }
        other => panic!("expected Integrity, got {other:?}"),
    }
}

#[test]
fn select_pack_works_with_catalog_order() {
    let tmp = tempfile::tempdir().unwrap();
    let src = stage_source(tmp.path(), &["minimal-ok"]);
    let cache = tmp.path().join("cache");
    materialize_catalog(&src, &cache).unwrap();
    let cat = load_catalog(&cache).unwrap();
    let ids = cat.pack_ids();
    let selected = select_pack(0, &[0x11; 32], &ids).expect("select");
    assert_eq!(selected.as_str(), "minimal-ok");
}

#[test]
fn build_manifest_sorts_entries() {
    let m = build_manifest(
        "pin",
        vec![
            CatalogEntry {
                pack_id: "z".into(),
                pack_digest: "1".into(),
                environment_image_digest: "sha256:1".into(),
            },
            CatalogEntry {
                pack_id: "a".into(),
                pack_digest: "2".into(),
                environment_image_digest: "sha256:2".into(),
            },
        ],
    );
    assert_eq!(m.entries[0].pack_id, "a");
    assert_eq!(m.entries[1].pack_id, "z");
}
