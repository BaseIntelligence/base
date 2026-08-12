//! Recipe 2.0 ZIP / JSON intake: parse `automodel.base` + `automodel.patch`,
//! fail-closed apply onto the accepted pin, and materialise a file tree for
//! `tree_blob` persistence (including `.prism/` diff artifacts).

use std::collections::BTreeMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;
use zip::ZipArchive;

use crate::apply::{apply_patch, ApplyError};
use crate::classify::DiffStat;
use crate::env::{DEFAULT_TRAIN_ENTRY, ENV_FIXTURE, ENV_PIN_DIR};
use crate::pin::{
    fixture_pin_dir, AutomodelPin, AUTOMODEL_PIN, AUTOMODEL_PIN_ID, FIXTURE_PIN, FIXTURE_PIN_ID,
};

/// ZIP / JSON member carrying the pin id.
pub const MEMBER_BASE: &str = "automodel.base";
/// ZIP / JSON member carrying the unified diff.
pub const MEMBER_PATCH: &str = "automodel.patch";
/// Optional entry / recipe knobs.
pub const MEMBER_TOML: &str = "prism.toml";

/// Persisted under the packed tree for `GET …/diff`.
pub const META_BASE: &str = ".prism/automodel.base";
/// Persisted unified diff text.
pub const META_PATCH: &str = ".prism/automodel.patch";
/// Persisted classified diffstat JSON.
pub const META_DIFFSTAT: &str = ".prism/diffstat.json";

/// When `1`/`true`, accept recipe 1.x JSON two-script bodies (tests / transitional).
pub const ALLOW_LEGACY_ENV: &str = "PRISM_ALLOW_LEGACY_INTAKE";

/// Fail-closed intake errors (HTTP code suggested via [`IntakeError::code`]).
#[derive(Debug, Error)]
pub enum IntakeError {
    #[error("unsupported_layout: {0}")]
    UnsupportedLayout(String),
    #[error("recipe_version: {0}")]
    RecipeVersion(String),
    #[error("invalid automodel zip: {0}")]
    Invalid(String),
    #[error("pin mismatch: automodel.base={got:?} (accepted: {accepted})")]
    PinMismatch { got: String, accepted: String },
    #[error("pin checkout unavailable for {pin_id}: {detail}")]
    PinUnavailable { pin_id: String, detail: String },
    #[error(transparent)]
    Apply(#[from] ApplyError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl IntakeError {
    /// Stable API `code` field (`unsupported_layout` / `recipe_version` / `zip`).
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedLayout(_) => "unsupported_layout",
            Self::RecipeVersion(_) => "recipe_version",
            Self::PinMismatch { .. } | Self::PinUnavailable { .. } => "pin",
            Self::Apply(_) => "patch",
            Self::Invalid(_) | Self::Io(_) => "zip",
        }
    }
}

/// Members extracted from a recipe 2.0 submission ZIP (or JSON fields).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutomodelMembers {
    /// Trimmed pin id (`automodel.base`).
    pub pin_id: String,
    /// Raw unified-diff bytes (`automodel.patch`).
    pub patch: Vec<u8>,
    /// Optional `prism.toml` text.
    pub prism_toml: Option<String>,
}

/// Materialised apply result ready for `tree_blob` packing + diff API.
#[derive(Debug, Clone)]
pub struct MaterializedAutomodel {
    /// Accepted pin id.
    pub pin_id: String,
    /// Raw patch bytes (same as submitted).
    pub patch: Vec<u8>,
    /// UTF-8 patch text (patches are text-only by apply validate).
    pub patch_text: String,
    /// Classified diffstat.
    pub diffstat: DiffStat,
    /// Applied tree files including `.prism/` artifacts (path → bytes).
    pub files: BTreeMap<String, Vec<u8>>,
    /// Harness entry path under the applied tree.
    pub entry: String,
}

/// Whether recipe 2.0 still permits legacy JSON two-script bodies.
#[must_use]
pub fn allow_legacy_intake() -> bool {
    env_truthy(ALLOW_LEGACY_ENV)
}

/// Whether CI/Sim may accept [`FIXTURE_PIN_ID`] (`PRISM_AUTOMODEL_FIXTURE=1`).
///
/// Live rejects the fixture pin unless this flag is set — same spirit as
/// `DESIGN_FORCE_SIM` / host-sim stubs (not real-train proof).
#[must_use]
pub fn allow_fixture_intake() -> bool {
    env_truthy(ENV_FIXTURE)
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| {
        let v = v.trim();
        v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
    })
}

/// Deterministic `submission_id`: `sha256(pin_id ‖ 0x00 ‖ patch_bytes)`.
#[must_use]
pub fn submission_id_for_patch(pin_id: &str, patch: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(pin_id.as_bytes());
    h.update([0]);
    h.update(patch);
    hex::encode(h.finalize())
}

/// True when the ZIP name set includes both `AutoModel` members (after junk /
/// wrapper-folder normalization).
pub fn zip_is_automodel_layout(zip_bytes: &[u8]) -> Result<bool, IntakeError> {
    let names = zip_member_names(zip_bytes)?;
    Ok(names.contains(MEMBER_BASE) && names.contains(MEMBER_PATCH))
}

/// Extract `AutoModel` ZIP members (fail-closed on missing / non-UTF-8 base).
pub fn extract_automodel_zip(zip_bytes: &[u8]) -> Result<AutomodelMembers, IntakeError> {
    let files = extract_flat_members(zip_bytes)?;
    let base_raw = files
        .get(MEMBER_BASE)
        .ok_or_else(|| IntakeError::UnsupportedLayout(format!("missing {MEMBER_BASE}")))?;
    let patch = files
        .get(MEMBER_PATCH)
        .ok_or_else(|| IntakeError::UnsupportedLayout(format!("missing {MEMBER_PATCH}")))?
        .clone();
    let pin_id = std::str::from_utf8(base_raw)
        .map_err(|_| IntakeError::Invalid(format!("non-utf8: {MEMBER_BASE}")))?
        .trim()
        .to_owned();
    if pin_id.is_empty() {
        return Err(IntakeError::Invalid(format!("empty {MEMBER_BASE}")));
    }
    if pin_id.chars().any(char::is_whitespace) {
        return Err(IntakeError::Invalid(format!(
            "{MEMBER_BASE} must be a single-line pin id without whitespace"
        )));
    }
    let prism_toml = match files.get(MEMBER_TOML) {
        Some(b) => Some(
            std::str::from_utf8(b)
                .map_err(|_| IntakeError::Invalid(format!("non-utf8: {MEMBER_TOML}")))?
                .to_owned(),
        ),
        None => None,
    };
    Ok(AutomodelMembers {
        pin_id,
        patch,
        prism_toml,
    })
}

/// Resolve `automodel.base` to a known pin descriptor.
pub fn resolve_pin(pin_id: &str) -> Result<&'static AutomodelPin, IntakeError> {
    if pin_id == FIXTURE_PIN_ID {
        if !allow_fixture_intake() {
            return Err(IntakeError::PinMismatch {
                got: pin_id.to_owned(),
                accepted: format!(
                    "{AUTOMODEL_PIN_ID} (set {ENV_FIXTURE}=1 only for CI/Sim fixture)"
                ),
            });
        }
        return Ok(&FIXTURE_PIN);
    }
    if pin_id == AUTOMODEL_PIN_ID {
        return Ok(&AUTOMODEL_PIN);
    }
    let accepted = if allow_fixture_intake() {
        format!("{AUTOMODEL_PIN_ID}|{FIXTURE_PIN_ID}")
    } else {
        AUTOMODEL_PIN_ID.to_owned()
    };
    Err(IntakeError::PinMismatch {
        got: pin_id.to_owned(),
        accepted,
    })
}

/// Absolute directory of a clean pin checkout for `pin`.
pub fn pin_checkout_dir(pin: &AutomodelPin) -> Result<PathBuf, IntakeError> {
    if let Ok(dir) = std::env::var(ENV_PIN_DIR) {
        let dir = dir.trim();
        if !dir.is_empty() {
            let p = PathBuf::from(dir);
            if p.is_dir() {
                return Ok(p);
            }
            return Err(IntakeError::PinUnavailable {
                pin_id: pin.id.to_owned(),
                detail: format!("{ENV_PIN_DIR} is not a directory: {}", p.display()),
            });
        }
    }
    if pin.id == FIXTURE_PIN_ID {
        return Ok(fixture_pin_dir());
    }
    Err(IntakeError::PinUnavailable {
        pin_id: pin.id.to_owned(),
        detail: format!(
            "set {ENV_PIN_DIR} to the staged pin tree (or use {FIXTURE_PIN_ID} in Sim/CI)"
        ),
    })
}

/// Apply `members` onto a fresh pin copy and attach `.prism/` artifacts.
pub fn materialize(members: &AutomodelMembers) -> Result<MaterializedAutomodel, IntakeError> {
    let pin = resolve_pin(&members.pin_id)?;
    if pin.id != members.pin_id {
        return Err(IntakeError::PinMismatch {
            got: members.pin_id.clone(),
            accepted: pin.id.to_owned(),
        });
    }
    let src = pin_checkout_dir(pin)?;
    if pin.has_content_sha() {
        crate::pin::verify_pin_tree(pin, &src).map_err(|e| IntakeError::PinUnavailable {
            pin_id: pin.id.to_owned(),
            detail: e,
        })?;
    }
    let tmp = tempfile::tempdir()?;
    copy_pin_tree(&src, tmp.path())?;
    let apply = apply_patch(tmp.path(), &members.patch)?;
    let mut files = collect_tree_files(tmp.path())?;
    let patch_text = String::from_utf8(members.patch.clone())
        .map_err(|_| IntakeError::Invalid("automodel.patch must be UTF-8 text".into()))?;
    let diffstat_json = serde_json::to_vec(&apply.diffstat)
        .map_err(|e| IntakeError::Invalid(format!("diffstat json: {e}")))?;
    files.insert(
        META_BASE.into(),
        format!("{}\n", members.pin_id).into_bytes(),
    );
    files.insert(META_PATCH.into(), members.patch.clone());
    files.insert(META_DIFFSTAT.into(), diffstat_json);
    // Persist miner knobs into the applied tree + slim delivery blob so the
    // pod harness can resolve a non-default entry (e.g. Prism-shaped smoke).
    // Without this, expand_tree_blob_for_pod rematerializes with
    // DEFAULT_TRAIN_ENTRY and live eval imports NeMo train_ft (mlflow, …).
    if let Some(toml) = &members.prism_toml {
        files.insert(MEMBER_TOML.into(), toml.as_bytes().to_vec());
    }
    let entry = members
        .prism_toml
        .as_deref()
        .and_then(parse_entry_from_toml)
        .unwrap_or_else(|| DEFAULT_TRAIN_ENTRY.to_owned());
    Ok(MaterializedAutomodel {
        pin_id: members.pin_id.clone(),
        patch: members.patch.clone(),
        patch_text,
        diffstat: apply.diffstat,
        files,
        entry,
    })
}

/// Full ZIP → materialised tree (layout detect + apply).
pub fn intake_automodel_zip(zip_bytes: &[u8]) -> Result<MaterializedAutomodel, IntakeError> {
    if !zip_is_automodel_layout(zip_bytes)? {
        return Err(IntakeError::UnsupportedLayout(
            "expected automodel.base + automodel.patch (recipe ≥ 2.0)".into(),
        ));
    }
    let members = extract_automodel_zip(zip_bytes)?;
    materialize(&members)
}

/// Pack a **slim** delivery blob for DB persistence (`.prism/` + touched files).
///
/// The full `AutoModel` pin tree exceeds `prism_submission_tree_blob_len`
/// (~17 MiB). Pods rematerialize via [`expand_tree_blob_for_pod`] against
/// `PRISM_AUTOMODEL_PIN_DIR` at upload time.
pub fn pack_tree_blob(mat: &MaterializedAutomodel) -> Result<Vec<u8>, IntakeError> {
    let slim = slim_delivery_files(mat);
    prism_tree::StagedTree::new(slim, mat.entry.clone())
        .pack()
        .map_err(|e| IntakeError::Invalid(e.to_string()))
}

/// Files persisted in `tree_blob`: meta + patch-touched paths + entry + optional toml.
fn slim_delivery_files(mat: &MaterializedAutomodel) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for key in [
        META_BASE,
        META_PATCH,
        META_DIFFSTAT,
        mat.entry.as_str(),
        MEMBER_TOML,
    ]
    .into_iter()
    .chain(mat.diffstat.files.iter().map(|entry| entry.path.as_str()))
    {
        if let Some(v) = mat.files.get(key) {
            out.insert(key.to_owned(), v.clone());
        }
    }
    out
}

/// Expand a stored `AutoModel` (or legacy) tree blob into the full pod tree.
///
/// When `.prism/automodel.patch` is present, re-applies onto the live pin
/// checkout so the pod receives a complete applied `AutoModel` tree even though
/// the DB only kept the slim delta.
pub fn expand_tree_blob_for_pod(blob: &[u8]) -> Result<prism_tree::StagedTree, IntakeError> {
    let staged =
        prism_tree::StagedTree::unpack(blob).map_err(|e| IntakeError::Invalid(e.to_string()))?;
    let Some(patch) = staged.get(META_PATCH) else {
        return Ok(staged);
    };
    let pin_id = staged
        .get(META_BASE)
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| IntakeError::Invalid(format!("missing {META_BASE} in tree blob")))?;
    let prism_toml = staged
        .get(MEMBER_TOML)
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(str::to_owned)
        .or_else(|| {
            // Legacy slim blobs omitted prism.toml; recover entry from the
            // StagedTree entry pointer when it is not the stock train_ft path.
            let e = staged.entry().trim();
            if e.is_empty() || e == DEFAULT_TRAIN_ENTRY {
                None
            } else {
                Some(format!("entry = \"{e}\"\n"))
            }
        });
    let mat = materialize(&AutomodelMembers {
        pin_id: pin_id.to_owned(),
        patch: patch.to_vec(),
        prism_toml,
    })?;
    Ok(prism_tree::StagedTree::new(mat.files, mat.entry))
}

/// Wire-facing expand result for `SubmissionRequest` fields.
#[derive(Debug, Clone)]
pub struct ExpandedAutomodel {
    pub pin_id: String,
    pub patch_text: String,
    pub packed_tree: Vec<u8>,
    pub architecture_py: String,
    pub training_py: String,
}

impl ExpandedAutomodel {
    fn from_materialized(mat: MaterializedAutomodel) -> Result<Self, IntakeError> {
        let surface = crate::review::delta_surface_from_materialized(
            &mat.patch_text,
            &mat.diffstat,
            &mat.files,
        );
        Ok(Self {
            packed_tree: pack_tree_blob(&mat)?,
            pin_id: mat.pin_id,
            patch_text: mat.patch_text,
            // Copy / similarity / AST: patch fingerprint + touched-file Python.
            architecture_py: surface.architecture_py,
            // Trainer/data touches only (empty when arch-only); harness owns telemetry.
            training_py: surface.training_py,
        })
    }
}

/// Expand JSON `automodel_base`/`automodel_patch` or ZIP bytes into store fields.
///
/// Returns `Ok(None)` when neither JSON `AutoModel` fields nor an `AutoModel` ZIP
/// are present (caller may continue with legacy paths).
pub fn expand_automodel(
    base: Option<&str>,
    patch: Option<&str>,
    prism_toml: Option<&str>,
    zip_bytes: Option<&[u8]>,
) -> Result<Option<ExpandedAutomodel>, IntakeError> {
    let base = base.map(str::trim).filter(|s| !s.is_empty());
    let patch = patch.filter(|s| !s.is_empty());
    match (base, patch) {
        (Some(pin_id), Some(patch_text)) => {
            let members = AutomodelMembers {
                pin_id: pin_id.to_owned(),
                patch: patch_text.as_bytes().to_vec(),
                prism_toml: prism_toml.map(str::to_owned),
            };
            return Ok(Some(ExpandedAutomodel::from_materialized(materialize(
                &members,
            )?)?));
        }
        (None, None) => {}
        _ => {
            return Err(IntakeError::UnsupportedLayout(
                "automodel_base and automodel_patch must be provided together".into(),
            ));
        }
    }
    let Some(zip) = zip_bytes.filter(|z| !z.is_empty()) else {
        return Ok(None);
    };
    if !zip_is_automodel_layout(zip)? {
        return Err(IntakeError::UnsupportedLayout(
            "recipe ≥ 2.0 requires automodel.base + automodel.patch".into(),
        ));
    }
    Ok(Some(ExpandedAutomodel::from_materialized(
        intake_automodel_zip(zip)?,
    )?))
}

/// CI fixture ZIP: `FIXTURE_PIN_ID` + `fixtures/patches/happy.patch`.
pub fn fixture_automodel_zip() -> Result<Vec<u8>, IntakeError> {
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;
    let patch = fs::read(crate::pin::fixture_happy_patch_path())?;
    let mut buf = Cursor::new(Vec::new());
    {
        let mut w = ZipWriter::new(&mut buf);
        let opts = SimpleFileOptions::default();
        w.start_file(MEMBER_BASE, opts)
            .map_err(|e| IntakeError::Invalid(e.to_string()))?;
        w.write_all(format!("{FIXTURE_PIN_ID}\n").as_bytes())?;
        w.start_file(MEMBER_PATCH, opts)
            .map_err(|e| IntakeError::Invalid(e.to_string()))?;
        w.write_all(&patch)?;
        w.finish()
            .map_err(|e| IntakeError::Invalid(e.to_string()))?;
    }
    Ok(buf.into_inner())
}

/// Read diff artifacts from a packed / unpacked file map.
#[must_use]
pub fn diff_from_files(files: &BTreeMap<String, Vec<u8>>) -> Option<(String, DiffStat)> {
    let patch = files.get(META_PATCH)?;
    let patch_text = std::str::from_utf8(patch).ok()?.to_owned();
    let stat_bytes = files.get(META_DIFFSTAT)?;
    let diffstat: DiffStat = serde_json::from_slice(stat_bytes).ok()?;
    Some((patch_text, diffstat))
}

fn zip_member_names(zip_bytes: &[u8]) -> Result<std::collections::BTreeSet<String>, IntakeError> {
    Ok(extract_flat_members(zip_bytes)?.into_keys().collect())
}

fn extract_flat_members(zip_bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, IntakeError> {
    if zip_bytes.len() > crate::apply::MAX_PATCH_BYTES.saturating_mul(4) {
        return Err(IntakeError::Invalid("zip exceeds size budget".into()));
    }
    let mut archive = ZipArchive::new(Cursor::new(zip_bytes))
        .map_err(|e| IntakeError::Invalid(format!("zip open: {e}")))?;
    let mut files = BTreeMap::new();
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| IntakeError::Invalid(format!("zip entry: {e}")))?;
        if file.is_dir() {
            continue;
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o170_000 == 0o120_000) {
            return Err(IntakeError::Invalid("symlinks forbidden".into()));
        }
        let name = file
            .enclosed_name()
            .ok_or_else(|| IntakeError::Invalid("unsafe zip path".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        let name = name.trim_start_matches("./");
        if name.is_empty() || name.contains("..") || name.starts_with('/') {
            return Err(IntakeError::Invalid(format!("bad path: {name}")));
        }
        if is_junk_path(name) {
            continue;
        }
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| IntakeError::Invalid(format!("zip read: {e}")))?;
        files.insert(name.to_owned(), data);
    }
    Ok(strip_common_prefix(files))
}

fn is_junk_path(name: &str) -> bool {
    name.split('/')
        .any(|c| c.is_empty() || c.starts_with('.') || c == "__MACOSX" || c == "__pycache__")
}

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
    let Some(p) = prefix else {
        return files;
    };
    // Do not strip when AutoModel members already sit at the root.
    if files.contains_key(MEMBER_BASE) || files.contains_key(MEMBER_PATCH) {
        return files;
    }
    let skip = p.len() + 1;
    files
        .into_iter()
        .map(|(k, v)| (k[skip..].to_owned(), v))
        .collect()
}

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

fn copy_pin_tree(src: &Path, dst: &Path) -> Result<(), IntakeError> {
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s == ".git" || name_s.starts_with('.') {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        if from.is_dir() {
            fs::create_dir_all(&to)?;
            copy_pin_tree(&from, &to)?;
        } else if from.is_file() {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn collect_tree_files(root: &Path) -> Result<BTreeMap<String, Vec<u8>>, IntakeError> {
    let mut out = BTreeMap::new();
    collect_walk(root, root, &mut out)?;
    Ok(out)
}

fn collect_walk(
    root: &Path,
    dir: &Path,
    out: &mut BTreeMap<String, Vec<u8>>,
) -> Result<(), IntakeError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if name_s == ".git" || name_s.starts_with('.') {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            collect_walk(root, &path, out)?;
            continue;
        }
        if !path.is_file() {
            continue;
        }
        let rel = path
            .strip_prefix(root)
            .map_err(|_| IntakeError::Invalid("path outside pin root".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        out.insert(rel, fs::read(&path)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::env::with_fixture_env;
    use crate::pin::{fixture_happy_patch_path, FIXTURE_PIN_ID};
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    fn zip_of(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ZipWriter::new(&mut buf);
            let opts = SimpleFileOptions::default();
            for (n, b) in files {
                w.start_file(*n, opts).unwrap();
                w.write_all(b).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn happy_fixture_patch_intake() {
        with_fixture_env(true, || {
            let patch = fs::read(fixture_happy_patch_path()).unwrap();
            let z = zip_of(&[
                (MEMBER_BASE, format!("{FIXTURE_PIN_ID}\n").as_bytes()),
                (MEMBER_PATCH, &patch),
            ]);
            let mat = intake_automodel_zip(&z).unwrap();
            assert_eq!(mat.pin_id, FIXTURE_PIN_ID);
            assert!(mat
                .files
                .contains_key("nemo_automodel/components/models/toy/layers.py"));
            assert!(mat.files.contains_key(META_PATCH));
            let (text, stat) = diff_from_files(&mat.files).unwrap();
            assert!(text.contains("ToyModel"));
            assert!(!stat.files.is_empty());
            let id = submission_id_for_patch(&mat.pin_id, &mat.patch);
            assert_eq!(id.len(), 64);
        });
    }

    #[test]
    fn slim_blob_keeps_prism_toml_entry_through_expand() {
        with_fixture_env(true, || {
            let patch = fs::read(fixture_happy_patch_path()).unwrap();
            let custom_entry = "nemo_automodel/components/models/toy/model.py";
            let z = zip_of(&[
                (MEMBER_BASE, format!("{FIXTURE_PIN_ID}\n").as_bytes()),
                (MEMBER_PATCH, &patch),
                (
                    MEMBER_TOML,
                    format!("entry = \"{custom_entry}\"\n").as_bytes(),
                ),
            ]);
            let mat = intake_automodel_zip(&z).unwrap();
            assert_eq!(mat.entry, custom_entry);
            assert_eq!(
                std::str::from_utf8(mat.files.get(MEMBER_TOML).unwrap())
                    .unwrap()
                    .trim(),
                format!("entry = \"{custom_entry}\"")
            );
            let slim = pack_tree_blob(&mat).unwrap();
            let slim_staged = prism_tree::StagedTree::unpack(&slim).unwrap();
            assert!(
                slim_staged.get(MEMBER_TOML).is_some(),
                "slim delivery must carry prism.toml"
            );
            assert_eq!(slim_staged.entry(), custom_entry);
            let expanded = expand_tree_blob_for_pod(&slim).unwrap();
            assert_eq!(expanded.entry(), custom_entry);
            assert!(expanded.get(MEMBER_TOML).is_some());
        });
    }

    #[test]
    fn rejects_legacy_two_script_layout() {
        let z = zip_of(&[
            ("architecture.py", b"def build_model(ctx):\n    pass\n"),
            ("training.py", b"def train(model, ctx):\n    pass\n"),
        ]);
        let err = intake_automodel_zip(&z).unwrap_err();
        assert_eq!(err.code(), "unsupported_layout");
    }

    #[test]
    fn rejects_wrong_pin_id() {
        with_fixture_env(false, || {
            let patch = fs::read(fixture_happy_patch_path()).unwrap();
            let z = zip_of(&[
                (MEMBER_BASE, b"automodel@not-a-pin\n"),
                (MEMBER_PATCH, &patch),
            ]);
            let err = intake_automodel_zip(&z).unwrap_err();
            assert_eq!(err.code(), "pin");
        });
    }

    #[test]
    fn live_rejects_fixture_pin_without_allow() {
        with_fixture_env(false, || {
            let patch = fs::read(fixture_happy_patch_path()).unwrap();
            let z = zip_of(&[
                (MEMBER_BASE, format!("{FIXTURE_PIN_ID}\n").as_bytes()),
                (MEMBER_PATCH, &patch),
            ]);
            let err = intake_automodel_zip(&z).unwrap_err();
            assert_eq!(err.code(), "pin");
            assert!(err.to_string().contains(FIXTURE_PIN_ID));
        });
    }
}
