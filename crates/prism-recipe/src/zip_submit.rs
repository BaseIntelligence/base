//! ZIP intake: the v1 two-script unpack (`architecture.py` + `training.py`)
//! plus the v3 **source-tree** contract.
//!
//! # Source-tree contract (v3)
//!
//! A source-tree submission is a ZIP project tree (flat or under one shared
//! top-level folder, which is stripped). Limits: ≤ [`MAX_TREE_FILES`] files,
//! ≤ [`MAX_TREE_FILE_BYTES`] per file, ≤ [`MAX_TREE_TOTAL_BYTES`] total
//! uncompressed, ≤ 8 MiB compressed. Entrypoint manifest: root `prism.toml`
//! with `entry = "<file.py>"`, else root `train.py` is the default entry.
//! The `architecture.py` / entry seams keep the v1 semantics:
//! `build_model(ctx)` / `train(model, ctx)` may be defined in the seam file
//! or re-exported from a module inside the tree (`from model.core import
//! build_model`); the defining module must live in the tree. Seam files and
//! their resolution targets stay under the v1 per-script cap
//! ([`MAX_SOURCE_BYTES`]) so the two-script downstream projection keeps
//! passing the legacy contract checks.
//!
//! # Tokenizer files (`tokenizer/`)
//!
//! The tokenizer is part of a submission (see the harness contract in
//! `harness/prismlib/tokenizer.py`): a tree may declare one as a
//! `build_tokenizer(ctx)` hook beside `build_model`, or as tokenizer files
//! under `tokenizer/`. Both are live — the whole validated tree is staged in
//! the pod workdir under `prism_tree::POD_SUBMISSION_DIR`, so
//! `tokenizer/tokenizer.json` resolves on the pod exactly as submitted.
//! Intake bounds that directory with the caps mirrored from the harness
//! ([`MAX_TOKENIZER_FILES`], [`MAX_TOKENIZER_TOTAL_BYTES`],
//! [`TOKENIZER_EXTENSIONS`]).
//!
//! Vendored pure-Python deps are allowed via a root `vendor.lock`
//! (sha256sum-style lines: `<64-hex sha256>  <path>`). Every file under
//! `vendor/` must be locked and must be `*.py`; locked paths must exist and
//! hash-match at intake.
//!
//! # What the scan covers
//!
//! Every file in the tree, not just the seams — a helper module, a `kernels/`
//! `.cu`, or a `vendor/` file is exactly as executable as `architecture.py`
//! once the tree is importable on the pod:
//!
//! * prebuilt artifacts are banned by extension *and* by content sniffing
//!   ([`prism_tree::executable_magic`]), so renaming an ELF does not help;
//! * files with the executable bit set in the ZIP are rejected;
//! * binary (non-UTF-8 or NUL-bearing) files are refused outright except as
//!   bounded `tokenizer/` assets, which are the only opaque bytes a submission
//!   may carry;
//! * every text file is scanned against the shared banned-pattern list
//!   ([`CHEATGUARD_PATTERNS_JSON`], also read by the harness-side
//!   `prismlib/cheatguard.py` AST audit): `ctypes`, `cuModuleLoadData`,
//!   `torch.utils.cpp_extension.load` (source-built `load_inline` remains
//!   allowed), and `base64.b64decode(` fed by a long base64-looking blob
//!   (embedded-binary heuristic);
//! * paths that collide with harness module names are refused
//!   ([`prism_tree::reserved_pod_path`]).
//!
//! Zip-slip/path-traversal: `enclosed_name` + explicit rejects (absolute,
//! `..`, `:` components, symlinks, duplicate normalized paths); macOS/Windows
//! junk (`__MACOSX`, dotfiles, `__pycache__`) is skipped.
//!
//! A two-script submission IS a trivial tree, but intake keeps routing it
//! through the byte-identical v1 path ([`sources_from_zip`] /
//! [`training_from_zip`]) so legacy `submission_id` hashes stay stable.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};

use sha2::{Digest, Sha256};
use thiserror::Error;
use zip::ZipArchive;

use crate::{check_contract, ContractError, MAX_SOURCE_BYTES};

/// ZIP submit errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ZipSubmitError {
    /// Archive failure.
    #[error("invalid zip submission: {0}")]
    Invalid(String),
    /// Contract shape.
    #[error(transparent)]
    Contract(#[from] ContractError),
}

/// Max files in a source-tree ZIP ([`prism_tree::MAX_TREE_FILES`]).
pub const MAX_TREE_FILES: usize = prism_tree::MAX_TREE_FILES;
/// Max bytes per tree file (uncompressed).
pub const MAX_TREE_FILE_BYTES: usize = prism_tree::MAX_TREE_FILE_BYTES;
/// Max total uncompressed bytes across the tree.
pub const MAX_TREE_TOTAL_BYTES: usize = prism_tree::MAX_TREE_TOTAL_BYTES;
/// Max compressed ZIP bytes for tree intake (defensive ceiling).
const MAX_TREE_ZIP_BYTES: usize = 20 * 1024 * 1024;
/// Entrypoint manifest filename at the tree root.
pub const TREE_MANIFEST_FILE: &str = "prism.toml";
/// Vendor hash-lock manifest filename at the tree root.
pub const VENDOR_LOCK_FILE: &str = "vendor.lock";
/// Tree directory carrying submitted tokenizer files (harness:
/// `prismlib.tokenizer.TOKENIZER_DIRNAME`).
pub const TOKENIZER_DIR: &str = "tokenizer/";
/// Max files under `tokenizer/` (harness: `MAX_TOKENIZER_FILES`).
pub const MAX_TOKENIZER_FILES: usize = prism_tree::MAX_TOKENIZER_FILES;
/// Max total bytes under `tokenizer/` (harness: `MAX_TOKENIZER_BYTES`).
pub const MAX_TOKENIZER_TOTAL_BYTES: usize = prism_tree::MAX_TOKENIZER_TOTAL_BYTES;
/// Allowed tokenizer file extensions (harness: `ALLOWED_EXTENSIONS`).
pub const TOKENIZER_EXTENSIONS: &[&str] = &["json", "txt", "model", "vocab", "merges", "bpe"];
/// Miner tokenizer hook (harness: `prismlib.tokenizer.HOOK_NAME`); must sit
/// beside `build_model` because the eval phase imports that module only.
pub const TOKENIZER_HOOK: &str = "build_tokenizer";
/// Default entry when no `prism.toml` manifest is present.
pub const DEFAULT_TREE_ENTRY: &str = "train.py";

/// Shared banned-pattern list (single source of truth). The Rust intake
/// scanner below and the harness-side `prismlib/cheatguard.py` AST audit both
/// read this document; keep them in sync when editing.
pub(crate) const CHEATGUARD_PATTERNS_JSON: &str =
    include_str!("../harness/cheatguard_patterns.json");

/// ZIP shape classification used by intake routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZipKind {
    /// Recipe ≥ 2.0 `AutoModel` patch layout (`automodel.base` + `automodel.patch`).
    Automodel,
    /// v1 two-script layout (flat, or one shared wrapper folder).
    TwoScript,
    /// v3 source-tree layout (subdirectories, `prism.toml`, `vendor/`, or
    /// `kernels/` present after wrapper-folder normalization).
    Tree,
}

/// A validated source-tree submission (v3). Files are kept raw (bytes) so
/// vendored sources hash-verify byte-exactly; paths are normalized,
/// wrapper-folder-stripped, zip-slip-safe relative POSIX paths.
#[derive(Debug, Clone)]
pub struct SourceTree {
    files: BTreeMap<String, Vec<u8>>,
    entry: String,
}

impl SourceTree {
    /// Build a tree from an in-memory file map (tests / attribution cell
    /// assembly). Does **not** validate — call [`SourceTree::validate`].
    #[must_use]
    pub fn from_files(files: BTreeMap<String, Vec<u8>>, entry: String) -> Self {
        Self { files, entry }
    }

    /// Normalized path → raw contents (sorted iteration order).
    #[must_use]
    pub fn files(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.files
    }

    /// Resolved entry file path (e.g. `train.py` or the `prism.toml` entry).
    #[must_use]
    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// UTF-8 view of one tree file.
    ///
    /// # Errors
    /// Missing file or non-UTF-8 contents.
    pub fn file_text(&self, path: &str) -> Result<&str, ZipSubmitError> {
        let bytes = self
            .files
            .get(path)
            .ok_or_else(|| ZipSubmitError::Invalid(format!("missing file: {path}")))?;
        std::str::from_utf8(bytes).map_err(|_| ZipSubmitError::Invalid(format!("non-utf8: {path}")))
    }

    /// `kernels/` subtree (path → contents), the kernel-swap unit for the
    /// 2×2 attribution matrix.
    #[must_use]
    pub fn kernels(&self) -> BTreeMap<String, Vec<u8>> {
        self.files
            .iter()
            .filter(|(p, _)| p.starts_with("kernels/"))
            .map(|(p, b)| (p.clone(), b.clone()))
            .collect()
    }

    /// Canonical content hash of the tree: for each file in sorted-path
    /// order, `path bytes ‖ 0x00 ‖ contents ‖ 0xff` feed one SHA-256 (same
    /// framing as [`prism_tree::StagedTree::tree_sha256`]). Zip
    /// ordering/metadata differences do not change the digest.
    #[must_use]
    pub fn canonical_sha256(&self) -> String {
        self.to_staged().tree_sha256()
    }

    /// Convert into the shared staged-tree representation used for
    /// persistence and pod delivery.
    #[must_use]
    pub fn to_staged(&self) -> prism_tree::StagedTree {
        prism_tree::StagedTree::new(self.files.clone(), self.entry.clone())
    }

    /// Pack a content-addressed delivery blob ([`prism_tree::StagedTree::pack`]).
    ///
    /// # Errors
    /// USTAR path / framing failures (intake caps prevent these in practice).
    pub fn pack_blob(&self) -> Result<Vec<u8>, ZipSubmitError> {
        self.to_staged()
            .pack()
            .map_err(|e| ZipSubmitError::Invalid(e.to_string()))
    }

    /// Two-script projection of the architecture seam: the root
    /// `architecture.py` source when it defines `build_model`, else the
    /// in-tree module it re-exports `build_model` from (real code, so the
    /// pre-LLM copy gate and agentic review see the implementation).
    ///
    /// # Errors
    /// Seam missing / unresolvable / non-UTF-8.
    pub fn architecture_projection(&self) -> Result<String, ZipSubmitError> {
        let seam = self
            .file_text("architecture.py")
            .map_err(|_| ZipSubmitError::Invalid("architecture.py missing at tree root".into()))?;
        if seam.contains("def build_model(") {
            return Ok(seam.to_owned());
        }
        if let Some(path) = resolve_reexport(seam, "build_model", &self.files) {
            return Ok(self.file_text(&path)?.to_owned());
        }
        Err(ZipSubmitError::Invalid(
            "architecture.py must define or re-export build_model(ctx)".into(),
        ))
    }

    /// Two-script projection of the training entry (same rule as
    /// [`SourceTree::architecture_projection`], for `train`).
    ///
    /// # Errors
    /// Entry missing / unresolvable / non-UTF-8.
    pub fn training_projection(&self) -> Result<String, ZipSubmitError> {
        let text = self.file_text(&self.entry).map_err(|_| {
            ZipSubmitError::Invalid(format!("entry {} missing from tree", self.entry))
        })?;
        if text.contains("def train(") {
            return Ok(text.to_owned());
        }
        if let Some(path) = resolve_reexport(text, "train", &self.files) {
            return Ok(self.file_text(&path)?.to_owned());
        }
        Err(ZipSubmitError::Invalid(format!(
            "entry {} must define or re-export train(model, ctx)",
            self.entry
        )))
    }

    /// Full tree validation (caps, seams, vendor lock, banned patterns).
    ///
    /// # Errors
    /// First contract violation found.
    pub fn validate(&self) -> Result<(), ZipSubmitError> {
        if self.files.len() > MAX_TREE_FILES {
            return Err(ZipSubmitError::Invalid(format!(
                "tree exceeds {MAX_TREE_FILES} files"
            )));
        }
        if !self.files.contains_key(&self.entry) {
            return Err(ZipSubmitError::Invalid(format!(
                "entry {} missing from tree",
                self.entry
            )));
        }
        let patterns = load_patterns();
        let mut total = 0usize;
        for (path, bytes) in &self.files {
            if bytes.len() > MAX_TREE_FILE_BYTES {
                return Err(ZipSubmitError::Invalid(format!("file too large: {path}")));
            }
            total = total.saturating_add(bytes.len());
            if prism_tree::reserved_pod_path(path) {
                return Err(ZipSubmitError::Invalid(format!(
                    "path collides with harness module name: {path}"
                )));
            }
            if has_banned_extension(path, &patterns) {
                return Err(ZipSubmitError::Invalid(format!(
                    "banned binary extension: {path}"
                )));
            }
            if let Some(magic) = prism_tree::executable_magic(bytes) {
                return Err(ZipSubmitError::Invalid(format!(
                    "prebuilt artifact ({magic}): {path}"
                )));
            }
            let is_tok = path.starts_with(TOKENIZER_DIR);
            if prism_tree::looks_binary(bytes) {
                if !is_tok {
                    return Err(ZipSubmitError::Invalid(format!(
                        "binary file outside {TOKENIZER_DIR}: {path}"
                    )));
                }
                continue;
            }
            // Every text file — helpers, kernels, vendor, seams — is scanned.
            let text = std::str::from_utf8(bytes)
                .map_err(|_| ZipSubmitError::Invalid(format!("non-utf8: {path}")))?;
            if let Some(v) = scan_source(text, path, &patterns).into_iter().next() {
                return Err(ZipSubmitError::Invalid(v));
            }
        }
        if total > MAX_TREE_TOTAL_BYTES {
            return Err(ZipSubmitError::Invalid(format!(
                "tree exceeds {MAX_TREE_TOTAL_BYTES} byte total budget"
            )));
        }
        // Seam presence + re-export resolution + v1 per-script caps (the
        // projections flow through legacy downstream contract checks).
        let arch = self.architecture_projection()?;
        let train = self.training_projection()?;
        if arch.len() > MAX_SOURCE_BYTES {
            return Err(ContractError::ArchitectureTooLarge(arch.len()).into());
        }
        if train.len() > MAX_SOURCE_BYTES {
            return Err(ContractError::TrainingTooLarge(train.len()).into());
        }
        self.validate_tokenizer(&arch, &train)?;
        self.validate_vendor_lock()
    }

    /// Tokenizer declaration checks (see the module docs § Tokenizer files).
    ///
    /// A `build_tokenizer` hook must sit in the architecture seam, never in
    /// the training entry: the eval phase imports the architecture module
    /// only, so a hook hiding in the entry would give train and eval two
    /// different tokenizers. `tokenizer/` files are bounded and delivered
    /// with the rest of the tree under `submission/tokenizer/` on the pod.
    fn validate_tokenizer(&self, arch: &str, train: &str) -> Result<(), ZipSubmitError> {
        let hook_marker = format!("def {TOKENIZER_HOOK}(");
        if train.contains(&hook_marker) && !arch.contains(&hook_marker) {
            return Err(ZipSubmitError::Invalid(format!(
                "{TOKENIZER_HOOK}(ctx) must be defined beside build_model (architecture seam), \
                 not in the training entry: the eval phase imports the architecture module only"
            )));
        }
        let tok_files: Vec<&String> = self
            .files
            .keys()
            .filter(|p| p.starts_with(TOKENIZER_DIR))
            .collect();
        if tok_files.is_empty() {
            return Ok(());
        }
        if tok_files.len() > MAX_TOKENIZER_FILES {
            return Err(ZipSubmitError::Invalid(format!(
                "{TOKENIZER_DIR} has {} files (max {MAX_TOKENIZER_FILES})",
                tok_files.len()
            )));
        }
        let mut total = 0usize;
        for path in &tok_files {
            let ext = std::path::Path::new(path.as_str())
                .extension()
                .map(|e| e.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default();
            if !TOKENIZER_EXTENSIONS.contains(&ext.as_str()) {
                return Err(ZipSubmitError::Invalid(format!(
                    "{path}: tokenizer files must be one of {TOKENIZER_EXTENSIONS:?}"
                )));
            }
            total = total.saturating_add(self.files.get(*path).map_or(0, Vec::len));
        }
        if total > MAX_TOKENIZER_TOTAL_BYTES {
            return Err(ZipSubmitError::Invalid(format!(
                "{TOKENIZER_DIR} is {total} bytes (max {MAX_TOKENIZER_TOTAL_BYTES})"
            )));
        }
        Ok(())
    }

    /// `vendor.lock` coverage: every `vendor/` file locked + hash-matched +
    /// `*.py`; every locked path exists under `vendor/`.
    fn validate_vendor_lock(&self) -> Result<(), ZipSubmitError> {
        let vendored: Vec<&String> = self
            .files
            .keys()
            .filter(|p| p.starts_with("vendor/"))
            .collect();
        let lock = self.files.get(VENDOR_LOCK_FILE);
        if vendored.is_empty() && lock.is_none() {
            return Ok(());
        }
        let Some(lock_bytes) = lock else {
            return Err(ZipSubmitError::Invalid(
                "vendor/ files require a root vendor.lock".into(),
            ));
        };
        let text = std::str::from_utf8(lock_bytes)
            .map_err(|_| ZipSubmitError::Invalid("non-utf8: vendor.lock".into()))?;
        let mut locked = std::collections::BTreeSet::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let (Some(hex_digest), Some(path), None) = (parts.next(), parts.next(), parts.next())
            else {
                return Err(ZipSubmitError::Invalid(format!(
                    "vendor.lock line malformed (want '<sha256>  <path>'): {line}"
                )));
            };
            if hex_digest.len() != 64 || !hex_digest.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(ZipSubmitError::Invalid(format!(
                    "vendor.lock bad sha256 for {path}"
                )));
            }
            if !path.starts_with("vendor/") {
                return Err(ZipSubmitError::Invalid(format!(
                    "vendor.lock path must live under vendor/: {path}"
                )));
            }
            let Some(bytes) = self.files.get(path) else {
                return Err(ZipSubmitError::Invalid(format!(
                    "vendor.lock references missing file: {path}"
                )));
            };
            let got = hex::encode(Sha256::digest(bytes));
            if !got.eq_ignore_ascii_case(hex_digest) {
                return Err(ZipSubmitError::Invalid(format!(
                    "vendor.lock sha256 mismatch: {path}"
                )));
            }
            locked.insert(path.to_owned());
        }
        for path in vendored {
            if !is_python_file(path) {
                return Err(ZipSubmitError::Invalid(format!(
                    "vendored deps must be pure Python (*.py): {path}"
                )));
            }
            if !locked.contains(path) {
                return Err(ZipSubmitError::Invalid(format!(
                    "vendored file missing from vendor.lock: {path}"
                )));
            }
        }
        Ok(())
    }
}

/// Classify ZIP bytes as `AutoModel` patch, v1 two-script, or v3 source-tree
/// (names only; wrapper-folder normalization and junk filtering match tree
/// extraction).
///
/// # Errors
/// Archive open / unsafe-path failures.
pub fn probe_zip_kind(zip_bytes: &[u8]) -> Result<ZipKind, ZipSubmitError> {
    if zip_bytes.len() > MAX_TREE_ZIP_BYTES {
        return Err(ZipSubmitError::Invalid("zip exceeds size budget".into()));
    }
    if prism_automodel::zip_is_automodel_layout(zip_bytes)
        .map_err(|e| ZipSubmitError::Invalid(e.to_string()))?
    {
        return Ok(ZipKind::Automodel);
    }
    let mut archive = ZipArchive::new(Cursor::new(zip_bytes))
        .map_err(|e| ZipSubmitError::Invalid(format!("zip open: {e}")))?;
    let mut names: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for i in 0..archive.len() {
        let file = archive
            .by_index(i)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip entry: {e}")))?;
        if file.is_dir() {
            continue;
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o170_000 == 0o120_000) {
            return Err(ZipSubmitError::Invalid("symlinks forbidden".into()));
        }
        let name = file
            .enclosed_name()
            .ok_or_else(|| ZipSubmitError::Invalid("unsafe zip path".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        let name = name.trim_start_matches("./");
        if name.is_empty() || name.contains("..") || name.starts_with('/') {
            return Err(ZipSubmitError::Invalid(format!("bad path: {name}")));
        }
        if is_junk_path(name) {
            continue;
        }
        names.insert(name.to_owned(), Vec::new());
    }
    let stripped = strip_common_prefix(names);
    let is_tree = stripped
        .keys()
        .any(|n| n.contains('/') || n == TREE_MANIFEST_FILE);
    Ok(if is_tree {
        ZipKind::Tree
    } else {
        ZipKind::TwoScript
    })
}

/// Parse + validate a source-tree ZIP (v3 contract; see module docs).
///
/// Accepts flat trees and single-wrapper-folder trees; a two-script ZIP is a
/// trivial tree and validates fine (intake still routes it through the v1
/// path for id stability).
///
/// # Errors
/// Archive, budget, seam, vendor-lock, or banned-pattern violations.
pub fn tree_from_zip(zip_bytes: &[u8]) -> Result<SourceTree, ZipSubmitError> {
    let files = extract_tree_entries(zip_bytes)?;
    let entry = match files.get(TREE_MANIFEST_FILE) {
        Some(bytes) => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| ZipSubmitError::Invalid("non-utf8: prism.toml".into()))?;
            parse_entry_from_toml(text).ok_or_else(|| {
                ZipSubmitError::Invalid("prism.toml must set entry = \"<file.py>\"".into())
            })?
        }
        None => {
            // Default entry: root `train.py`; the legacy two-script layout
            // (`training.py`) is accepted as the default too so a trivial
            // two-script tree validates without a manifest.
            if files.contains_key(DEFAULT_TREE_ENTRY) {
                DEFAULT_TREE_ENTRY.to_owned()
            } else {
                "training.py".to_owned()
            }
        }
    };
    if !is_python_file(&entry) {
        return Err(ZipSubmitError::Invalid(format!(
            "entry must be a python file: {entry}"
        )));
    }
    let tree = SourceTree { files, entry };
    tree.validate()?;
    Ok(tree)
}

/// Unpack ZIP bytes into `(architecture_py, training_py)`.
///
/// Accepts flat or one top-level folder. Rejects symlinks / traversal.
/// Source-tree ZIPs (subdirectories / `prism.toml` / `vendor/` present) are
/// rejected with a pointer to the JSON `zip_base64` intake, which validates
/// and retains the full tree; the raw `application/zip` path carries only
/// two scripts and cannot represent a tree honestly.
///
/// # Errors
/// Archive or contract failures.
pub fn sources_from_zip(zip_bytes: &[u8]) -> Result<(String, String), ZipSubmitError> {
    if matches!(probe_zip_kind(zip_bytes)?, ZipKind::Tree) {
        return Err(ZipSubmitError::Invalid(
            "source-tree ZIP detected: submit via the JSON zip_base64 field so the full tree is validated and retained".into(),
        ));
    }
    if zip_bytes.len() > MAX_SOURCE_BYTES.saturating_mul(4) {
        return Err(ZipSubmitError::Invalid("zip exceeds size budget".into()));
    }
    let mut archive = ZipArchive::new(Cursor::new(zip_bytes))
        .map_err(|e| ZipSubmitError::Invalid(format!("zip open: {e}")))?;

    let mut architecture_py = None;
    let mut training_py = None;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip entry: {e}")))?;
        if file.is_dir() {
            continue;
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o170_000 == 0o120_000) {
            return Err(ZipSubmitError::Invalid("symlinks forbidden".into()));
        }
        let name = file
            .enclosed_name()
            .ok_or_else(|| ZipSubmitError::Invalid("unsafe zip path".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        let name = name.trim_start_matches("./");
        if name.is_empty() || name.contains("..") || name.starts_with('/') {
            return Err(ZipSubmitError::Invalid(format!("bad path: {name}")));
        }
        let rel = normalize_rel(name);
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip read: {e}")))?;
        if data.len() > MAX_SOURCE_BYTES {
            return Err(ZipSubmitError::Invalid(format!("file too large: {rel}")));
        }
        let text = String::from_utf8(data)
            .map_err(|_| ZipSubmitError::Invalid(format!("non-utf8: {rel}")))?;
        match rel.as_str() {
            "architecture.py" => architecture_py = Some(text),
            "training.py" => training_py = Some(text),
            _ => {}
        }
    }

    let architecture_py =
        architecture_py.ok_or_else(|| ZipSubmitError::Invalid("architecture.py missing".into()))?;
    let training_py =
        training_py.ok_or_else(|| ZipSubmitError::Invalid("training.py missing".into()))?;
    check_contract(&architecture_py, &training_py)?;
    Ok((architecture_py, training_py))
}

/// Unpack ZIP bytes into `training.py` only — training-only submissions on a
/// published registry architecture. Archives carrying `architecture.py` are
/// rejected: the architecture source comes from the registry, never from the
/// miner (copy-gate integrity). Source-tree ZIPs are rejected: training-only
/// intake keeps the two-script layout in v3 (trees carry their own
/// architecture by construction).
///
/// # Errors
/// Archive or contract failures.
pub fn training_from_zip(zip_bytes: &[u8]) -> Result<String, ZipSubmitError> {
    if matches!(probe_zip_kind(zip_bytes)?, ZipKind::Tree) {
        return Err(ZipSubmitError::Invalid(
            "source-tree ZIP detected: training-only intake (arch_id) accepts the two-script layout only".into(),
        ));
    }
    if zip_bytes.len() > MAX_SOURCE_BYTES.saturating_mul(4) {
        return Err(ZipSubmitError::Invalid("zip exceeds size budget".into()));
    }
    let mut archive = ZipArchive::new(Cursor::new(zip_bytes))
        .map_err(|e| ZipSubmitError::Invalid(format!("zip open: {e}")))?;
    let mut training_py = None;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip entry: {e}")))?;
        if file.is_dir() {
            continue;
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o170_000 == 0o120_000) {
            return Err(ZipSubmitError::Invalid("symlinks forbidden".into()));
        }
        let name = file
            .enclosed_name()
            .ok_or_else(|| ZipSubmitError::Invalid("unsafe zip path".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        let name = name.trim_start_matches("./");
        if name.is_empty() || name.contains("..") || name.starts_with('/') {
            return Err(ZipSubmitError::Invalid(format!("bad path: {name}")));
        }
        let rel = normalize_rel(name);
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip read: {e}")))?;
        if data.len() > MAX_SOURCE_BYTES {
            return Err(ZipSubmitError::Invalid(format!("file too large: {rel}")));
        }
        let text = String::from_utf8(data)
            .map_err(|_| ZipSubmitError::Invalid(format!("non-utf8: {rel}")))?;
        match rel.as_str() {
            "architecture.py" => {
                return Err(ZipSubmitError::Invalid(
                    "architecture.py forbidden in training-only submission".into(),
                ));
            }
            "training.py" => training_py = Some(text),
            _ => {}
        }
    }
    let training_py =
        training_py.ok_or_else(|| ZipSubmitError::Invalid("training.py missing".into()))?;
    if !training_py.contains("def train(") {
        return Err(ZipSubmitError::Contract(ContractError::MissingTrain));
    }
    Ok(training_py)
}

fn normalize_rel(name: &str) -> String {
    let parts: Vec<&str> = name.split('/').filter(|p| !p.is_empty()).collect();
    match parts.as_slice() {
        [] => String::new(),
        [one] => (*one).to_owned(),
        [root, rest @ ..]
            if root
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') =>
        {
            rest.join("/")
        }
        _ => name.to_owned(),
    }
}

/// Extract tree entries: junk-filtered, dup-checked, cap-enforced,
/// wrapper-folder-stripped.
fn extract_tree_entries(zip_bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, ZipSubmitError> {
    if zip_bytes.len() > MAX_TREE_ZIP_BYTES {
        return Err(ZipSubmitError::Invalid("zip exceeds size budget".into()));
    }
    let patterns = load_patterns();
    let mut archive = ZipArchive::new(Cursor::new(zip_bytes))
        .map_err(|e| ZipSubmitError::Invalid(format!("zip open: {e}")))?;
    let mut files = BTreeMap::new();
    let mut total = 0usize;
    for i in 0..archive.len() {
        let file = archive
            .by_index(i)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip entry: {e}")))?;
        if file.is_dir() {
            continue;
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o170_000 == 0o120_000) {
            return Err(ZipSubmitError::Invalid("symlinks forbidden".into()));
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o111 != 0) {
            return Err(ZipSubmitError::Invalid(format!(
                "executable bit forbidden: {}",
                file.name()
            )));
        }
        let name = file
            .enclosed_name()
            .ok_or_else(|| ZipSubmitError::Invalid("unsafe zip path".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        let name = name.trim_start_matches("./");
        if name.is_empty() || name.contains("..") || name.starts_with('/') {
            return Err(ZipSubmitError::Invalid(format!("bad path: {name}")));
        }
        if name.split('/').any(|c| c.contains(':')) {
            return Err(ZipSubmitError::Invalid(format!(
                "bad path component: {name}"
            )));
        }
        if is_junk_path(name) {
            continue;
        }
        if files.contains_key(name) {
            return Err(ZipSubmitError::Invalid(format!("duplicate path: {name}")));
        }
        if files.len() >= MAX_TREE_FILES {
            return Err(ZipSubmitError::Invalid(format!(
                "tree exceeds {MAX_TREE_FILES} files"
            )));
        }
        if has_banned_extension(name, &patterns) {
            return Err(ZipSubmitError::Invalid(format!(
                "banned binary extension: {name}"
            )));
        }
        let mut data = Vec::new();
        file.take(MAX_TREE_FILE_BYTES as u64 + 1)
            .read_to_end(&mut data)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip read: {e}")))?;
        if data.len() > MAX_TREE_FILE_BYTES {
            return Err(ZipSubmitError::Invalid(format!("file too large: {name}")));
        }
        total = total.saturating_add(data.len());
        if total > MAX_TREE_TOTAL_BYTES {
            return Err(ZipSubmitError::Invalid(format!(
                "tree exceeds {MAX_TREE_TOTAL_BYTES} byte total budget"
            )));
        }
        files.insert(name.to_owned(), data);
    }
    Ok(strip_common_prefix(files))
}

/// Skip macOS/Windows zip junk and bytecode caches entirely (never part of
/// the tree, never scanned).
fn is_junk_path(name: &str) -> bool {
    name.split('/')
        .any(|c| c.is_empty() || c.starts_with('.') || c == "__MACOSX" || c == "__pycache__")
}

/// Case-insensitive `.py` extension check.
fn is_python_file(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("py"))
}

/// Strip a single shared top-level folder (one wrapper directory), matching
/// the v1 "flat or one top-level folder" rule for trees.
fn strip_common_prefix<V>(files: BTreeMap<String, V>) -> BTreeMap<String, V> {
    let mut prefix: Option<String> = None;
    for k in files.keys() {
        let Some((first, _)) = k.split_once('/') else {
            return files;
        };
        match &prefix {
            None => prefix = Some(first.to_owned()),
            Some(p) if p != first => return files,
            _ => {}
        }
    }
    let Some(p) = prefix else { return files };
    let skip = p.len() + 1;
    files
        .into_iter()
        .map(|(k, v)| (k[skip..].to_owned(), v))
        .collect()
}

/// Parse `entry = "<file.py>"` from a root `prism.toml` (line-level minimal
/// TOML: comments and other keys ignored).
fn parse_entry_from_toml(text: &str) -> Option<String> {
    for raw in text.lines() {
        let line = raw.split('#').next().unwrap_or("").trim();
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key.trim() == "entry" {
            let v = value.trim().trim_matches('"').trim_matches('\'').trim();
            if !v.is_empty() {
                return Some(v.to_owned());
            }
        }
    }
    None
}

/// Resolve `from a.b import <symbol>` in a seam file to a tree module that
/// defines it (`a/b.py` or `a/b/__init__.py` containing `def <symbol>(`).
fn resolve_reexport(seam: &str, symbol: &str, files: &BTreeMap<String, Vec<u8>>) -> Option<String> {
    let def_marker = format!("def {symbol}(");
    for line in seam.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("from ") else {
            continue;
        };
        let Some((module, names)) = rest.split_once(" import ") else {
            continue;
        };
        let imports_symbol = names.split(',').any(|n| {
            let n = n.trim().trim_start_matches('(').trim_end_matches(')');
            n.split(" as ").next().map(str::trim) == Some(symbol)
        });
        if !imports_symbol {
            continue;
        }
        let base = module.trim().replace('.', "/");
        for candidate in [format!("{base}.py"), format!("{base}/__init__.py")] {
            if let Some(bytes) = files.get(&candidate) {
                if std::str::from_utf8(bytes).is_ok_and(|t| t.contains(&def_marker)) {
                    return Some(candidate);
                }
            }
        }
    }
    None
}

/// Shared banned-pattern list (parsed from [`CHEATGUARD_PATTERNS_JSON`];
/// falls back to the built-in mirror so a corrupt embed can never panic).
#[derive(Debug, serde::Deserialize)]
#[serde(default)]
struct Patterns {
    banned_extensions: Vec<String>,
    banned_substrings: Vec<String>,
    banned_call_strings: Vec<String>,
    #[serde(rename = "banned_call_allowed_suffixes")]
    allowed_suffixes: Vec<String>,
    base64_decode_marker: String,
    base64_blob_min_len: usize,
}

impl Default for Patterns {
    fn default() -> Self {
        Self {
            banned_extensions: [".so", ".pyd", ".cubin", ".dll", ".pyc", ".pyo"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            banned_substrings: vec!["ctypes".into(), "cuModuleLoadData".into()],
            banned_call_strings: vec!["torch.utils.cpp_extension.load".into()],
            allowed_suffixes: vec!["_inline".into()],
            base64_decode_marker: "base64.b64decode(".into(),
            base64_blob_min_len: 256,
        }
    }
}

fn load_patterns() -> Patterns {
    serde_json::from_str(CHEATGUARD_PATTERNS_JSON).unwrap_or_default()
}

fn has_banned_extension(path: &str, patterns: &Patterns) -> bool {
    let lower = path.to_ascii_lowercase();
    patterns
        .banned_extensions
        .iter()
        .any(|ext| lower.ends_with(ext.as_str()))
}

/// Scan one `.py` source against the shared banned-pattern list (intake
/// hard-reject side; `cheatguard.py` mirrors these as AST findings).
fn scan_source(text: &str, path: &str, patterns: &Patterns) -> Vec<String> {
    let mut out = Vec::new();
    for s in &patterns.banned_substrings {
        if text.contains(s.as_str()) {
            out.push(format!("{path}: banned substring '{s}'"));
        }
    }
    for call in &patterns.banned_call_strings {
        for (idx, _) in text.match_indices(call.as_str()) {
            let after = &text[idx + call.len()..];
            if patterns
                .allowed_suffixes
                .iter()
                .any(|suf| after.starts_with(suf.as_str()))
            {
                continue;
            }
            out.push(format!(
                "{path}: banned call '{call}' (native code must build from source, e.g. load_inline)"
            ));
            break;
        }
    }
    if !patterns.base64_decode_marker.is_empty()
        && text.contains(patterns.base64_decode_marker.as_str())
        && has_b64_blob(text, patterns.base64_blob_min_len)
    {
        out.push(format!(
            "{path}: base64 blob feeding b64decode (embedded-binary heuristic)"
        ));
    }
    out
}

/// Heuristic: any run of ≥ `min_len` base64-alphabet characters in the
/// source (paired by the caller with a `base64.b64decode(` marker).
fn has_b64_blob(text: &str, min_len: usize) -> bool {
    let mut run = 0usize;
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b == b'+' || b == b'/' || b == b'=' {
            run += 1;
            if run >= min_len {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    fn zip_of(files: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ZipWriter::new(&mut buf);
            let opts = SimpleFileOptions::default();
            for (n, b) in files {
                w.start_file(*n, opts).unwrap();
                w.write_all(b.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    const ARCH: &str = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n";
    const TRAIN: &str = "def train(model, ctx):\n    return {}\n";

    #[test]
    fn unpacks_sources() {
        let z = zip_of(&[("architecture.py", ARCH), ("training.py", TRAIN)]);
        let (a, t) = sources_from_zip(&z).unwrap();
        assert!(a.contains("build_model"));
        assert!(t.contains("def train"));
    }

    #[test]
    fn wrapped_two_script_stays_two_script() {
        let z = zip_of(&[("proj/architecture.py", ARCH), ("proj/training.py", TRAIN)]);
        assert_eq!(probe_zip_kind(&z).unwrap(), ZipKind::TwoScript);
        let (a, t) = sources_from_zip(&z).unwrap();
        assert!(a.contains("build_model"));
        assert!(t.contains("def train"));
    }

    #[test]
    fn flat_tree_with_manifest() {
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("main_loop.py", TRAIN),
            ("prism.toml", "[prism]\nentry = \"main_loop.py\"\n"),
        ]);
        assert_eq!(probe_zip_kind(&z).unwrap(), ZipKind::Tree);
        let tree = tree_from_zip(&z).unwrap();
        assert_eq!(tree.entry(), "main_loop.py");
        assert_eq!(tree.training_projection().unwrap(), TRAIN);
        assert_eq!(tree.architecture_projection().unwrap(), ARCH);
        // Raw application/zip intake refuses trees with a clear pointer.
        let err = sources_from_zip(&z).unwrap_err().to_string();
        assert!(err.contains("zip_base64"), "{err}");
    }

    #[test]
    fn tree_with_submodule_and_reexport_seams() {
        let z = zip_of(&[
            (
                "wrap/architecture.py",
                "from model.core import build_model\n",
            ),
            ("wrap/train.py", "from model.loop import train\n"),
            ("wrap/model/core.py", ARCH),
            ("wrap/model/loop.py", TRAIN),
        ]);
        let tree = tree_from_zip(&z).unwrap();
        // Wrapper folder stripped; projections resolve to the defining module.
        assert!(tree.files().contains_key("model/core.py"));
        assert_eq!(tree.architecture_projection().unwrap(), ARCH);
        assert_eq!(tree.training_projection().unwrap(), TRAIN);
    }

    #[test]
    fn tree_rejects_missing_seam_and_entry() {
        let z = zip_of(&[("model/core.py", ARCH), ("train.py", TRAIN)]);
        let err = tree_from_zip(&z).unwrap_err().to_string();
        assert!(err.contains("architecture.py"), "{err}");
        let z = zip_of(&[("architecture.py", ARCH), ("model/loop.py", TRAIN)]);
        let err = tree_from_zip(&z).unwrap_err().to_string();
        assert!(err.contains("entry") && err.contains("missing"), "{err}");
    }

    #[test]
    fn tree_rejects_banned_extensions_and_patterns() {
        for (path, _) in [
            ("ops/kit.so", ""),
            ("ops/kit.cubin", ""),
            ("ops/kit.dll", ""),
            ("ops/kit.pyd", ""),
            ("ops/kit.pyc", ""),
        ] {
            let z = zip_of(&[("architecture.py", ARCH), ("train.py", TRAIN), (path, "x")]);
            let err = tree_from_zip(&z).unwrap_err().to_string();
            assert!(err.contains("binary extension"), "{path}: {err}");
        }
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("ops/ffi.py", "import ctypes\n"),
        ]);
        assert!(tree_from_zip(&z)
            .unwrap_err()
            .to_string()
            .contains("ctypes"));
        // load_inline (source-built) allowed; load( with prebuilt banned.
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            (
                "ops/build.py",
                "from torch.utils.cpp_extension import load_inline\n",
            ),
        ]);
        tree_from_zip(&z).unwrap();
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            (
                "ops/build.py",
                "torch.utils.cpp_extension.load(name='x', sources=[])\n",
            ),
        ]);
        assert!(tree_from_zip(&z)
            .unwrap_err()
            .to_string()
            .contains("banned call"));
    }

    #[test]
    fn tree_b64_blob_heuristic() {
        let blob = "A".repeat(300);
        let src = format!("import base64\ndef k():\n    raw = base64.b64decode(\"{blob}\")\n");
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("ops/k.py", &src),
        ]);
        assert!(tree_from_zip(&z)
            .unwrap_err()
            .to_string()
            .contains("base64"));
        // Short decode payload is fine (no binary-looking blob).
        let ok = "import base64\ndef k():\n    return base64.b64decode(\"aGVsbG8=\")\n";
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("ops/k.py", ok),
        ]);
        tree_from_zip(&z).unwrap();
    }

    #[test]
    fn tree_vendor_lock_verified() {
        let vendored = "def helper():\n    return 1\n";
        let digest = hex::encode(Sha256::digest(vendored.as_bytes()));
        let lock = format!("{digest}  vendor/helpers.py\n");
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("vendor/helpers.py", vendored),
            ("vendor.lock", &lock),
        ]);
        tree_from_zip(&z).unwrap();
        // Wrong hash fails.
        let bad = format!("{}  vendor/helpers.py\n", "0".repeat(64));
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("vendor/helpers.py", vendored),
            ("vendor.lock", &bad),
        ]);
        assert!(tree_from_zip(&z)
            .unwrap_err()
            .to_string()
            .contains("mismatch"));
        // Missing lock fails; non-py vendored file fails.
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("vendor/h.py", vendored),
        ]);
        assert!(tree_from_zip(&z)
            .unwrap_err()
            .to_string()
            .contains("vendor.lock"));
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("vendor/data.txt", "x"),
            (
                "vendor.lock",
                &format!("{}  vendor/data.txt\n", hex::encode(Sha256::digest(b"x"))),
            ),
        ]);
        assert!(tree_from_zip(&z)
            .unwrap_err()
            .to_string()
            .contains("pure Python"));
    }

    #[test]
    fn tree_tokenizer_hook_must_live_in_the_architecture_seam() {
        let hook = "\ndef build_tokenizer(ctx):\n    return ctx\n";
        // Beside build_model: accepted (this is the live declaration path).
        let z = zip_of(&[
            ("architecture.py", &format!("{ARCH}{hook}")),
            ("train.py", TRAIN),
            ("util.py", "x = 1\n"),
        ]);
        tree_from_zip(&z).unwrap();
        // In the training entry only: rejected (eval imports arch only).
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", &format!("{TRAIN}{hook}")),
            ("util.py", "x = 1\n"),
        ]);
        let err = tree_from_zip(&z).unwrap_err().to_string();
        assert!(err.contains("architecture seam"), "{err}");
    }

    #[test]
    fn tree_tokenizer_dir_is_accepted_and_bounded() {
        let vocab = "{\"a\": 0}\n";
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("tokenizer/tokenizer.json", vocab),
        ]);
        let tree = tree_from_zip(&z).unwrap();
        assert!(tree.files().contains_key("tokenizer/tokenizer.json"));
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("tokenizer/weights.bin", vocab),
        ]);
        let err = tree_from_zip(&z).unwrap_err().to_string();
        assert!(err.contains("tokenizer files must be one of"), "{err}");
        let many: Vec<(String, String)> = (0..=MAX_TOKENIZER_FILES)
            .map(|i| (format!("tokenizer/part{i}.json"), vocab.to_owned()))
            .collect();
        let mut files: Vec<(&str, &str)> = vec![("architecture.py", ARCH), ("train.py", TRAIN)];
        for (n, b) in &many {
            files.push((n, b));
        }
        let err = tree_from_zip(&zip_of(&files)).unwrap_err().to_string();
        assert!(err.contains(&format!("max {MAX_TOKENIZER_FILES}")), "{err}");
    }

    #[test]
    fn tree_accepts_realistic_tokenizer_json() {
        // ~1.4 MiB HF tokenizer.json must clear the per-file and tokenizer
        // total caps (the previous 1 MiB tokenizer budget rejected it).
        let big = format!("{{\"vocab\":{}}}", "x".repeat(1_400_000));
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("kernels/flash.py", "def attend():\n    return 1\n"),
            ("tokenizer/tokenizer.json", &big),
        ]);
        let tree = tree_from_zip(&z).unwrap();
        assert!(tree.files()["tokenizer/tokenizer.json"].len() > 1_400_000);
        let blob = tree.pack_blob().unwrap();
        let staged = prism_tree::StagedTree::unpack(&blob).unwrap();
        assert_eq!(staged.tree_sha256(), tree.canonical_sha256());
        assert!(staged.get("kernels/flash.py").is_some());
    }

    #[test]
    fn tree_rejects_banned_pattern_in_non_seam_helper() {
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("kernels/evil.py", "import ctypes\n"),
        ]);
        let err = tree_from_zip(&z).unwrap_err().to_string();
        assert!(err.contains("ctypes"), "{err}");
    }

    #[test]
    fn tree_rejects_oversize_file() {
        let huge = "x".repeat(MAX_TREE_FILE_BYTES + 1);
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("kernels/big.py", &huge),
        ]);
        let err = tree_from_zip(&z).unwrap_err().to_string();
        assert!(err.contains("too large") || err.contains("budget"), "{err}");
    }

    #[test]
    fn tree_rejects_traversal_and_counts() {
        let z = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("../evil.py", "x=1"),
        ]);
        assert!(tree_from_zip(&z).is_err());
        let many: Vec<(String, String)> = (0..=MAX_TREE_FILES)
            .map(|i| (format!("m/f{i}.py"), "x = 1\n".to_owned()))
            .collect();
        let mut files: Vec<(&str, &str)> = vec![("architecture.py", ARCH), ("train.py", TRAIN)];
        for (n, b) in &many {
            files.push((n, b));
        }
        let z = zip_of(&files);
        assert!(tree_from_zip(&z)
            .unwrap_err()
            .to_string()
            .contains(&MAX_TREE_FILES.to_string()));
    }

    #[test]
    fn tree_canonical_hash_is_order_independent() {
        let a = zip_of(&[
            ("architecture.py", ARCH),
            ("train.py", TRAIN),
            ("u.py", "x=1\n"),
        ]);
        let b = zip_of(&[
            ("u.py", "x=1\n"),
            ("train.py", TRAIN),
            ("architecture.py", ARCH),
        ]);
        assert_eq!(
            tree_from_zip(&a).unwrap().canonical_sha256(),
            tree_from_zip(&b).unwrap().canonical_sha256()
        );
    }

    #[test]
    fn two_script_zip_is_a_valid_trivial_tree() {
        // The trivial-tree equivalence holds for validation, but intake keeps
        // routing two-script ZIPs through the v1 path (id stability).
        let z = zip_of(&[("architecture.py", ARCH), ("training.py", TRAIN)]);
        assert_eq!(probe_zip_kind(&z).unwrap(), ZipKind::TwoScript);
        let tree = tree_from_zip(&z).unwrap();
        assert_eq!(tree.entry(), "training.py");
    }

    #[test]
    fn baseline_sources_pass_pattern_scan() {
        let pats = load_patterns();
        for src in [crate::BASELINE_ARCHITECTURE_PY, crate::BASELINE_TRAINING_PY] {
            assert!(scan_source(src, "baseline", &pats).is_empty());
        }
    }

    #[test]
    fn embedded_patterns_json_parses_to_default() {
        let parsed: Patterns = serde_json::from_str(CHEATGUARD_PATTERNS_JSON).unwrap();
        let d = Patterns::default();
        assert_eq!(parsed.banned_extensions, d.banned_extensions);
        assert_eq!(parsed.banned_substrings, d.banned_substrings);
        assert_eq!(parsed.banned_call_strings, d.banned_call_strings);
        assert_eq!(parsed.allowed_suffixes, d.allowed_suffixes);
        assert_eq!(parsed.base64_decode_marker, d.base64_decode_marker);
        assert_eq!(parsed.base64_blob_min_len, d.base64_blob_min_len);
    }
}
