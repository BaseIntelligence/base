#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Integration tests: good fork admits; denylist touch / sealed edit reject.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use hypertraining_sealed::{
    admit, sealed_symbol_ast_hash, sha256_hex, AdmitError, AdmitInput, DatasetPin, SealedSurfaceV1,
    SegmentPin, DEFAULT_SEALED_SYMBOL_KEYS,
};

fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    load_tree_rec(dir, dir, &mut map);
    map
}

fn load_tree_rec(root: &Path, dir: &Path, map: &mut BTreeMap<String, Vec<u8>>) {
    let entries = fs::read_dir(dir).expect("read_dir");
    for ent in entries {
        let ent = ent.expect("entry");
        let path = ent.path();
        if path.is_dir() {
            load_tree_rec(root, &path, map);
        } else {
            let rel = path
                .strip_prefix(root)
                .expect("strip")
                .to_string_lossy()
                .replace('\\', "/");
            let bytes = fs::read(&path).expect("read file");
            map.insert(rel, bytes);
        }
    }
}

fn baseline_manifest(files: &BTreeMap<String, Vec<u8>>) -> SealedSurfaceV1 {
    let mut m = SealedSurfaceV1::with_pins(
        "basedeadbeef",
        DatasetPin {
            corpus: "fineweb-edu".into(),
            revision: "rev1".into(),
            order_seed: 42,
        },
        SegmentPin {
            tokens: 1_000_000,
            gbs: 8,
            seq_len: 2048,
        },
    );

    for path in [
        "megatron/core/datasets/blended.py",
        "megatron/core/num_microbatches_calculator.py",
    ] {
        let bytes = files.get(path).unwrap_or_else(|| panic!("missing {path}"));
        m.denylist_hashes.insert(path.to_owned(), sha256_hex(bytes));
    }

    let training = std::str::from_utf8(
        files
            .get("megatron/training/training.py")
            .expect("training.py"),
    )
    .expect("utf8");
    for key in DEFAULT_SEALED_SYMBOL_KEYS {
        let h = sealed_symbol_ast_hash(key, training).expect("ast hash");
        m.sealed_symbols.insert((*key).to_owned(), h);
    }
    m
}

#[test]
fn s1_good_fork_allowlisted_only_admits() {
    let root = fixtures_root().join("good_fork");
    let files = load_tree(&root);
    let manifest = baseline_manifest(&files);
    let changed = vec!["megatron/core/fusions/softmax.py".to_owned()];
    let input = AdmitInput {
        changed_paths: &changed,
        file_contents: &files,
        manifest: &manifest,
    };
    admit(&input).expect("good fork must admit");
}

#[test]
fn s2_denylist_path_touched_rejects() {
    let root = fixtures_root().join("denylist_touch");
    let files = load_tree(&root);
    // Manifest pins match good baseline content; fork also "touches" datasets.
    let good = load_tree(&fixtures_root().join("good_fork"));
    let manifest = baseline_manifest(&good);
    let changed = vec![
        "megatron/core/fusions/softmax.py".to_owned(),
        "megatron/core/datasets/blended.py".to_owned(),
    ];
    // Use denylist_touch tree contents (dataset file may differ).
    let input = AdmitInput {
        changed_paths: &changed,
        file_contents: &files,
        manifest: &manifest,
    };
    let err = admit(&input).expect_err("denylist touch must reject");
    match err {
        AdmitError::DenylistPathTouched { path } => {
            assert!(path.contains("datasets"), "path={path}");
        }
        other => panic!("expected DenylistPathTouched, got {other:?}"),
    }
}

#[test]
fn s3_sealed_symbol_edit_rejects() {
    let good = load_tree(&fixtures_root().join("good_fork"));
    let manifest = baseline_manifest(&good);
    let sealed = load_tree(&fixtures_root().join("sealed_edit"));
    // Only allowlisted path changed in the diff list; training.py content is
    // still checked via sealed_symbols against the fork tree.
    let changed = vec!["megatron/core/fusions/softmax.py".to_owned()];
    let input = AdmitInput {
        changed_paths: &changed,
        file_contents: &sealed,
        manifest: &manifest,
    };
    let err = admit(&input).expect_err("sealed edit must reject");
    match err {
        AdmitError::SealedSymbolMismatch { key } => {
            assert!(key.contains("num_floating_point_operations"), "key={key}");
        }
        other => panic!("expected SealedSymbolMismatch, got {other:?}"),
    }
}

#[test]
fn s4_denylist_hash_mismatch_without_path_in_diff() {
    let good = load_tree(&fixtures_root().join("good_fork"));
    let mut manifest = baseline_manifest(&good);
    // Corrupt expected hash while path is not in changed_paths.
    manifest
        .denylist_hashes
        .insert("megatron/core/datasets/blended.py".into(), "00".repeat(32));
    let changed = vec!["megatron/core/fusions/softmax.py".to_owned()];
    let input = AdmitInput {
        changed_paths: &changed,
        file_contents: &good,
        manifest: &manifest,
    };
    let err = admit(&input).expect_err("hash mismatch must reject");
    assert!(matches!(err, AdmitError::DenylistHashMismatch { .. }));
}

#[test]
fn s5_manifest_round_trip_json() {
    let good = load_tree(&fixtures_root().join("good_fork"));
    let m = baseline_manifest(&good);
    let json = serde_json::to_string_pretty(&m).expect("ser");
    let back: SealedSurfaceV1 = serde_json::from_str(&json).expect("de");
    assert_eq!(m, back);
    assert_eq!(back.kind, "sealed_surface.v1");
    assert_eq!(back.mlm_commit, "cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54");
    assert_eq!(back.te_version, "2.18.0+e7c550c5");
}
