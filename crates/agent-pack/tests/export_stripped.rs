//! Integration: stripped pack tree + tar.gz never carry solution/tests.
//!
//! Prefers real Harbor pack under `/tmp/da_m18c_hf_pull/tasks/realpr-more-itertools-1136`
//! when present; otherwise uses the in-crate `minimal-ok` fixture.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use agent_pack::{export_stripped_tar_gz, load_pack, write_stripped_tree};
use flate2::read::GzDecoder;
use tar::Archive;
use tempfile::tempdir;

fn pack_source() -> PathBuf {
    let real = PathBuf::from("/tmp/da_m18c_hf_pull/tasks/realpr-more-itertools-1136");
    if real.join("task.toml").is_file() {
        return real;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/minimal-ok")
}

fn assert_no_held_out_paths(root: &Path) {
    assert!(
        !root.join("solution").exists(),
        "solution/ must not exist under stripped tree"
    );
    assert!(
        !root.join("tests").exists(),
        "tests/ must not exist under stripped tree"
    );
    assert!(
        !root.join("solution.patch").exists(),
        "root solution.patch must not exist"
    );
    assert!(
        !root.join("grader.py").exists(),
        "root grader.py must not exist"
    );
    assert!(
        !root.join("test.patch").exists(),
        "root test.patch must not exist"
    );
}

#[test]
fn write_stripped_tree_excludes_solution_and_tests_and_reloads() {
    let src = pack_source();
    let pack = load_pack(&src).unwrap_or_else(|e| panic!("load {}: {e}", src.display()));

    let dest = tempdir().expect("tempdir");
    write_stripped_tree(&pack, dest.path()).expect("write_stripped_tree");

    assert_no_held_out_paths(dest.path());
    assert!(dest.path().join("task.toml").is_file());
    assert!(dest.path().join("instruction.md").is_file());
    assert!(dest.path().join("environment/Dockerfile").is_file());

    // Walk every file under dest — no path segment solution or tests.
    for entry in walkdir_files(dest.path()) {
        let rel = entry
            .strip_prefix(dest.path())
            .expect("prefix")
            .to_string_lossy()
            .replace('\\', "/");
        assert!(
            !rel.starts_with("solution/") && rel != "solution",
            "leaked path {rel}"
        );
        assert!(
            !rel.starts_with("tests/") && rel != "tests",
            "leaked path {rel}"
        );
        let body = fs::read_to_string(&entry).unwrap_or_default();
        if let Some(sol) = &pack.held_out.solution_patch {
            let needle = String::from_utf8_lossy(sol);
            if let Some(line) = needle.lines().find(|l| l.len() > 40) {
                assert!(
                    !body.contains(line),
                    "stripped file {rel} embeds solution body"
                );
            }
        }
    }

    let reloaded = load_pack(dest.path()).expect("load stripped tree");
    assert_eq!(reloaded.task_id, pack.task_id);
    assert_eq!(reloaded.base_commit_hash, pack.base_commit_hash);
    assert!(reloaded.held_out.solution_patch.is_none());
    assert!(reloaded.held_out.test_patch.is_none());
    assert!(reloaded.held_out.grader_py.is_none());

    let stripped = reloaded.strip();
    stripped.assert_total_keys().expect("total keys");
    assert_eq!(stripped.task_id, pack.task_id);
}

#[test]
fn export_stripped_tar_gz_is_valid_and_has_no_solution() {
    let src = pack_source();
    let pack = load_pack(&src).unwrap_or_else(|e| panic!("load {}: {e}", src.display()));

    let bytes = export_stripped_tar_gz(&pack).expect("export tar.gz");
    assert!(bytes.len() > 32, "tar.gz should be non-trivial");
    // gzip magic
    assert_eq!(&bytes[0..2], &[0x1f, 0x8b]);

    let dec = GzDecoder::new(bytes.as_slice());
    let mut archive = Archive::new(dec);
    let mut paths = Vec::new();
    for entry in archive.entries().expect("entries") {
        let entry = entry.expect("entry");
        let path = entry
            .path()
            .expect("path")
            .to_string_lossy()
            .replace('\\', "/");
        paths.push(path);
    }

    assert!(
        paths
            .iter()
            .any(|p| p == "task.toml" || p.ends_with("/task.toml")),
        "missing task.toml in tar: {paths:?}"
    );
    assert!(
        paths
            .iter()
            .any(|p| p == "instruction.md" || p.ends_with("/instruction.md")),
        "missing instruction.md in tar: {paths:?}"
    );
    assert!(
        paths
            .iter()
            .any(|p| p.contains("environment/") && p.ends_with("Dockerfile")),
        "missing environment/Dockerfile in tar: {paths:?}"
    );

    for p in &paths {
        let norm = p.trim_start_matches("./");
        assert!(
            !norm.starts_with("solution/") && !norm.contains("/solution/"),
            "tar leaked solution path {p}"
        );
        assert!(
            !norm.starts_with("tests/") && !norm.contains("/tests/"),
            "tar leaked tests path {p}"
        );
        assert_ne!(norm, "solution.patch");
        assert_ne!(norm, "grader.py");
        assert_ne!(norm, "test.patch");
    }

    // Unpack and load_pack
    let dest = tempdir().expect("tempdir");
    {
        let dec = GzDecoder::new(bytes.as_slice());
        let mut archive = Archive::new(dec);
        archive.unpack(dest.path()).expect("unpack");
    }
    // tar may nest under a single root or write flat — find task.toml
    let root = find_pack_root(dest.path()).expect("pack root with task.toml");
    assert_no_held_out_paths(&root);
    let reloaded = load_pack(&root).expect("load from tar");
    reloaded.strip().assert_total_keys().expect("total");
}

fn walkdir_files_rec(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walkdir_files_rec(&p, out);
        } else if p.is_file() {
            out.push(p);
        }
    }
}

fn walkdir_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walkdir_files_rec(root, &mut out);
    out
}

fn find_pack_root(dir: &Path) -> Option<PathBuf> {
    if dir.join("task.toml").is_file() {
        return Some(dir.to_path_buf());
    }
    let rd = fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(found) = find_pack_root(&p) {
                return Some(found);
            }
        }
    }
    None
}

#[test]
fn export_roundtrip_bytes_readable() {
    let src = pack_source();
    let pack = load_pack(&src).expect("load");
    let bytes = export_stripped_tar_gz(&pack).expect("export");
    let mut dec = GzDecoder::new(bytes.as_slice());
    let mut raw = Vec::new();
    dec.read_to_end(&mut raw).expect("gunzip");
    assert!(raw.len() > 100, "uncompressed tar should have content");
}
