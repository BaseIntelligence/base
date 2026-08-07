//! PRISM miner source-tree delivery.
//!
//! A v3 submission is a *tree* of miner-authored files, not two loose scripts.
//! This crate owns the one representation of that tree that every stage shares:
//!
//! * intake ([`prism-recipe`]) validates a ZIP into a [`StagedTree`],
//! * persistence stores [`StagedTree::pack`] (a deterministic USTAR blob) keyed
//!   by [`StagedTree::tree_sha256`], next to [`StagedTree::manifest`],
//! * delivery ([`prism-lium`]) uploads [`StagedTree::pod_entries`] into the pod
//!   workdir over the existing tar-over-SSH-stdin channel,
//! * review materializes the tree on disk with [`StagedTree::materialize`].
//!
//! Because the same bytes flow through all four stages, the tree hash computed
//! at intake is verifiable at every later stage.
//!
//! # Pod layout
//!
//! The tree is staged under [`POD_SUBMISSION_DIR`] rather than at the workdir
//! root. That keeps miner paths from ever colliding with harness-owned files
//! (`main.py`, `prismlib/`, `eval/`, `checkpoint.pt`, `eval-assets/`), and lets
//! the harness put the tree *last* on `sys.path` so miner modules can never
//! shadow the standard library or the harness packages. [`reserved_pod_path`]
//! additionally rejects trees that merely *contain* a harness package name, so
//! a miner cannot hijack scoring by re-ordering `sys.path` from inside their
//! own code.
//!
//! [`prism-recipe`]: https://docs.rs/prism-recipe
//! [`prism-lium`]: https://docs.rs/prism-lium

mod scan;
mod tar;

pub use scan::{executable_magic, looks_binary};
pub use tar::{untar, ustar, MAX_TAR_PATH_BYTES};

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Maximum number of files in one submission tree.
pub const MAX_TREE_FILES: usize = 128;
/// Maximum bytes for any single file in the tree.
///
/// Sized to admit a real Hugging Face `tokenizer.json` (~1.4 MiB for a 32k BPE
/// vocab, larger for 128k vocabs) with headroom.
pub const MAX_TREE_FILE_BYTES: usize = 4 * 1024 * 1024;
/// Maximum total uncompressed bytes across the whole tree.
pub const MAX_TREE_TOTAL_BYTES: usize = 16 * 1024 * 1024;
/// Maximum path length for a tree entry, in bytes.
///
/// Bounded by the USTAR 100-byte name field minus the [`POD_SUBMISSION_DIR`]
/// prefix the pod staging layout adds, so every accepted tree is guaranteed to
/// serialize without needing a PAX extension header.
pub const MAX_TREE_PATH_BYTES: usize = MAX_TAR_PATH_BYTES - POD_SUBMISSION_DIR.len() - 1;
/// Maximum number of files under the miner's `tokenizer/` directory.
pub const MAX_TOKENIZER_FILES: usize = 12;
/// Maximum total bytes under the miner's `tokenizer/` directory.
pub const MAX_TOKENIZER_TOTAL_BYTES: usize = 8 * 1024 * 1024;

/// Directory, relative to the pod workdir, the validated tree is staged into.
pub const POD_SUBMISSION_DIR: &str = "submission";
/// Harness-owned pointer file naming the tree's declared entry module.
///
/// Written next to the tree by [`StagedTree::pod_entries`]; read by
/// `prismlib.submission` on the pod. Miner trees cannot contain a file with
/// this name (intake strips dotfiles), so the pointer is never spoofable.
pub const ENTRY_MARKER: &str = ".prism_entry";

/// Path components a submission tree may not use, at any depth.
///
/// Shadowing is already impossible by `sys.path` ordering; this closes the
/// residual hole where miner code re-orders `sys.path` itself and thereby
/// hijacks harness scoring or telemetry with its own module of the same name.
const RESERVED_STEMS: &[&str] = &[
    "prismlib",
    "eval",
    "prism_telemetry",
    "sitecustomize",
    "usercustomize",
    "torch",
    "transformers",
    "datasets",
    "numpy",
];

/// Failure modes for tree serialization and staging.
#[derive(Debug, thiserror::Error)]
pub enum TreeError {
    /// A path is unusable in the archive or on the pod filesystem.
    #[error("bad tree path {path:?}: {why}")]
    Path {
        /// Offending path.
        path: String,
        /// Why it was rejected.
        why: &'static str,
    },
    /// The archive is truncated, mis-framed, or has a bad checksum.
    #[error("malformed tree archive: {0}")]
    Archive(&'static str),
    /// A file exceeded a cap.
    #[error("tree too large: {0}")]
    TooLarge(String),
    /// Filesystem failure while materializing the tree.
    #[error("materialize {path:?}: {source}")]
    Io {
        /// Path being written.
        path: String,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
}

/// One file in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeFile {
    /// Tree-relative path, `/`-separated.
    pub path: String,
    /// Byte length.
    pub bytes: usize,
    /// Lowercase hex SHA-256 of the file contents.
    pub sha256: String,
}

/// Auditable description of a tree: what it contains and what it hashes to.
///
/// Stored alongside the tar blob so operators, the agentic reviewer, and
/// scoring can reason about a submission without rehydrating the bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeManifest {
    /// Canonical tree hash; see [`StagedTree::tree_sha256`].
    pub tree_sha256: String,
    /// Declared entry module path within the tree.
    pub entry: String,
    /// Number of files.
    pub file_count: usize,
    /// Total uncompressed bytes.
    pub total_bytes: usize,
    /// Every file, sorted by path.
    pub files: Vec<TreeFile>,
}

/// A validated miner source tree, ready to persist and to stage on a pod.
///
/// Construction is intentionally cheap and total: validation lives at intake
/// (`prism-recipe`), which is the only place that has the miner's ZIP and the
/// banned-pattern corpus. Everything downstream just moves these bytes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StagedTree {
    files: BTreeMap<String, Vec<u8>>,
    entry: String,
}

impl StagedTree {
    /// Build a tree from already-validated files and a declared entry path.
    #[must_use]
    pub fn new(files: BTreeMap<String, Vec<u8>>, entry: String) -> Self {
        Self { files, entry }
    }

    /// Tree-relative path of the module the harness imports first.
    #[must_use]
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// Every file, keyed by tree-relative path, in sorted order.
    #[must_use]
    pub fn files(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.files
    }

    /// Contents of one tree-relative path, if present.
    #[must_use]
    pub fn get(&self, path: &str) -> Option<&[u8]> {
        self.files.get(path).map(Vec::as_slice)
    }

    /// Number of files in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.files.len()
    }

    /// Whether the tree carries no files.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Total uncompressed bytes across the tree.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.files.values().map(Vec::len).sum()
    }

    /// Canonical content hash: for each file in sorted-path order,
    /// `path ‖ 0x00 ‖ contents ‖ 0xff` feeds one SHA-256.
    ///
    /// Deliberately the same framing as `prism_recipe::SourceTree::
    /// canonical_sha256`, so one tree has exactly one identity everywhere: the
    /// digest that derives a v3 `submission_id`, the storage content-address,
    /// and the manifest identity are the same string. `prism-recipe` has a test
    /// pinning that equality.
    ///
    /// The declared entry is not hashed because it is a pure function of the
    /// files (a `prism.toml` in the tree, else the default entry name).
    #[must_use]
    pub fn tree_sha256(&self) -> String {
        let mut h = Sha256::new();
        for (path, bytes) in &self.files {
            h.update(path.as_bytes());
            h.update([0x00]);
            h.update(bytes);
            h.update([0xff]);
        }
        hex::encode(h.finalize())
    }

    /// Auditable manifest for this tree.
    #[must_use]
    pub fn manifest(&self) -> TreeManifest {
        TreeManifest {
            tree_sha256: self.tree_sha256(),
            entry: self.entry.clone(),
            file_count: self.files.len(),
            total_bytes: self.total_bytes(),
            files: self
                .files
                .iter()
                .map(|(path, bytes)| TreeFile {
                    path: path.clone(),
                    bytes: bytes.len(),
                    sha256: hex::encode(Sha256::digest(bytes)),
                })
                .collect(),
        }
    }

    /// Serialize to a deterministic USTAR blob suitable for content-addressed
    /// storage; [`StagedTree::unpack`] is its exact inverse.
    ///
    /// The entry pointer travels inside the blob so a stored tree is
    /// self-describing and needs no sidecar column to be rehydrated.
    ///
    /// # Errors
    /// A path is unrepresentable in USTAR (intake enforces the caps that
    /// prevent this).
    pub fn pack(&self) -> Result<Vec<u8>, TreeError> {
        let mut entries: Vec<(&str, &[u8])> = vec![(ENTRY_MARKER, self.entry.as_bytes())];
        entries.extend(self.files.iter().map(|(p, b)| (p.as_str(), b.as_slice())));
        ustar(&entries)
    }

    /// Inverse of [`StagedTree::pack`].
    ///
    /// # Errors
    /// Malformed archive, or a blob with no entry pointer.
    pub fn unpack(blob: &[u8]) -> Result<Self, TreeError> {
        let mut files = untar(blob)?;
        let entry = files
            .remove(ENTRY_MARKER)
            .ok_or(TreeError::Archive("missing entry marker"))?;
        let entry = String::from_utf8(entry).map_err(|_| TreeError::Archive("entry not utf-8"))?;
        Ok(Self {
            files,
            entry: entry.trim().to_string(),
        })
    }

    /// Archive entries for pod staging: every tree file under
    /// [`POD_SUBMISSION_DIR`], plus the entry pointer.
    ///
    /// Returned owned so callers can merge these with harness files into a
    /// single upload tar without borrowing gymnastics.
    #[must_use]
    pub fn pod_entries(&self) -> Vec<(String, Vec<u8>)> {
        let mut out = vec![(
            format!("{POD_SUBMISSION_DIR}/{ENTRY_MARKER}"),
            self.entry.as_bytes().to_vec(),
        )];
        out.extend(
            self.files
                .iter()
                .map(|(p, b)| (format!("{POD_SUBMISSION_DIR}/{p}"), b.clone())),
        );
        out
    }

    /// Write the tree into `dir`, creating parent directories as needed.
    ///
    /// Returns the tree-relative paths written, sorted, so a caller can hand
    /// the agentic reviewer an exact inventory of what it can read.
    ///
    /// # Errors
    /// Filesystem failure creating a directory or writing a file.
    pub fn materialize(&self, dir: &Path) -> Result<Vec<String>, TreeError> {
        let mut written = Vec::with_capacity(self.files.len());
        for (path, bytes) in &self.files {
            let dest = dir.join(path);
            if let Some(parent) = dest.parent() {
                std::fs::create_dir_all(parent).map_err(|source| TreeError::Io {
                    path: parent.display().to_string(),
                    source,
                })?;
            }
            std::fs::write(&dest, bytes).map_err(|source| TreeError::Io {
                path: dest.display().to_string(),
                source,
            })?;
            written.push(path.clone());
        }
        Ok(written)
    }
}

/// Whether a tree-relative path collides with a harness-owned name.
///
/// See [`RESERVED_STEMS`] for the rationale: this blocks module-name hijacking
/// even though `sys.path` ordering already blocks accidental shadowing.
#[must_use]
pub fn reserved_pod_path(path: &str) -> bool {
    path.split('/').any(|part| {
        let stem = part.strip_suffix(".py").unwrap_or(part);
        RESERVED_STEMS.contains(&stem)
    })
}

/// Deterministic ustar of the harness package plus miner sources for pod
/// staging over SSH stdin (never argv — see BUG-5).
///
/// * Two-script submissions (`tree == None`): `architecture.py` /
///   `training.py` land at the workdir root (legacy layout).
/// * Source-tree submissions: the validated tree is namespaced under
///   [`POD_SUBMISSION_DIR`] via [`StagedTree::pod_entries`]; root seam files
///   are omitted so the harness loads the real tree (sibling imports +
///   `tokenizer/` resolve correctly).
///
/// # Errors
/// Path / size failures from [`ustar`].
pub fn pack_harness_upload(
    harness_files: &[(&str, &[u8])],
    architecture_py: &[u8],
    training_py: &[u8],
    tree: Option<&StagedTree>,
) -> Result<Vec<u8>, TreeError> {
    let mut owned: Vec<(String, Vec<u8>)> = harness_files
        .iter()
        .map(|(p, b)| ((*p).to_owned(), b.to_vec()))
        .collect();
    if let Some(t) = tree {
        owned.extend(t.pod_entries());
    } else {
        owned.push(("architecture.py".into(), architecture_py.to_vec()));
        owned.push(("training.py".into(), training_py.to_vec()));
    }
    let refs: Vec<(&str, &[u8])> = owned
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();
    ustar(&refs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> StagedTree {
        let mut files = BTreeMap::new();
        files.insert("architecture.py".to_string(), b"import kernels\n".to_vec());
        files.insert("kernels/flash.py".to_string(), b"def attend():\n".to_vec());
        files.insert("tokenizer/tokenizer.json".to_string(), vec![7u8; 4096]);
        StagedTree::new(files, "architecture.py".to_string())
    }

    #[test]
    fn pack_round_trips_and_is_deterministic() {
        let t = tree();
        let blob = t.pack().expect("pack");
        assert_eq!(blob, t.pack().expect("pack again"));
        let back = StagedTree::unpack(&blob).expect("unpack");
        assert_eq!(back, t);
        assert_eq!(back.entry(), "architecture.py");
        assert_eq!(back.tree_sha256(), t.tree_sha256());
    }

    #[test]
    fn tree_hash_tracks_content_and_paths() {
        let base = tree().tree_sha256();
        let mut edited = tree().files().clone();
        edited.insert(
            "kernels/flash.py".to_string(),
            b"def attend(): pass\n".to_vec(),
        );
        assert_ne!(
            StagedTree::new(edited, "architecture.py".into()).tree_sha256(),
            base
        );
        let mut renamed = tree().files().clone();
        let body = renamed.remove("kernels/flash.py").expect("kernel present");
        renamed.insert("kernels/other.py".to_string(), body);
        assert_ne!(
            StagedTree::new(renamed, "architecture.py".into()).tree_sha256(),
            base
        );
    }

    #[test]
    fn manifest_lists_every_file_with_digests() {
        let m = tree().manifest();
        assert_eq!(m.file_count, 3);
        assert_eq!(m.total_bytes, 15 + 14 + 4096);
        let paths: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "architecture.py",
                "kernels/flash.py",
                "tokenizer/tokenizer.json"
            ]
        );
        assert_eq!(
            m.files[0].sha256,
            hex::encode(Sha256::digest(b"import kernels\n"))
        );
    }

    #[test]
    fn pod_entries_are_namespaced_under_submission_dir() {
        let entries = tree().pod_entries();
        let names: Vec<&str> = entries.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(names[0], "submission/.prism_entry");
        assert!(names.contains(&"submission/kernels/flash.py"));
        assert!(
            names.iter().all(|n| n.starts_with("submission/")),
            "no entry may land at the workdir root: {names:?}"
        );
    }

    #[test]
    fn materialize_writes_nested_paths() {
        let dir = tempfile::tempdir().expect("tempdir");
        let written = tree().materialize(dir.path()).expect("materialize");
        assert_eq!(written.len(), 3);
        let helper = std::fs::read(dir.path().join("kernels/flash.py")).expect("read helper");
        assert_eq!(helper, b"def attend():\n");
    }

    #[test]
    fn reserved_paths_block_harness_module_names() {
        for bad in [
            "prismlib/scoring.py",
            "eval/common.py",
            "pkg/eval/rollup.py",
            "sitecustomize.py",
            "torch/__init__.py",
        ] {
            assert!(reserved_pod_path(bad), "{bad} must be reserved");
        }
        for ok in ["architecture.py", "kernels/eval_helper.py", "evaluate.py"] {
            assert!(!reserved_pod_path(ok), "{ok} must be allowed");
        }
    }

    #[test]
    fn tree_path_cap_fits_the_ustar_name_field() {
        assert!(MAX_TREE_PATH_BYTES + POD_SUBMISSION_DIR.len() < MAX_TAR_PATH_BYTES);
    }

    #[test]
    fn pack_harness_upload_namespaces_trees_and_keeps_two_script_root() {
        let harness = [("main.py", b"print(1)\n".as_slice())];
        let two = pack_harness_upload(&harness, b"arch", b"train", None).expect("two");
        let two_files = untar(&two).expect("untar two");
        assert!(two_files.contains_key("architecture.py"));
        assert!(two_files.contains_key("training.py"));
        assert!(!two_files.keys().any(|k| k.starts_with("submission/")));

        let staged = tree();
        let full = pack_harness_upload(&harness, b"arch", b"train", Some(&staged)).expect("tree");
        let files = untar(&full).expect("untar tree");
        assert!(files.contains_key("submission/kernels/flash.py"));
        assert!(files.contains_key("submission/tokenizer/tokenizer.json"));
        assert!(!files.contains_key("architecture.py"));
        assert_eq!(
            files.get("submission/.prism_entry").map(Vec::as_slice),
            Some(b"architecture.py".as_slice())
        );
    }
}
