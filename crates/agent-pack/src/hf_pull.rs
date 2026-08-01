//! Real `HuggingFace` Hub downloader for the pinned deepagent task packs.
//!
//! Fetches `{subdir}/**` from a **pinned commit revision** of a dataset repo
//! over the public Hub HTTP API and writes one directory per pack into a
//! destination directory (`<dest>/<pack_id>/...`).
//!
//! Design constraints:
//! - The revision is a commit SHA by default, never `main`, so two replicas
//!   materialize byte-identical trees.
//! - Remote paths are untrusted input: every entry is re-validated to stay
//!   under `dest` before a single byte is written.
//! - Auth is optional; a token is only needed for private repos and is never
//!   logged or embedded in error text.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

use crate::sha256_bytes;

/// Dataset repo id holding the deepagent Harbor task packs.
pub const DEEPAGENT_HF_REPO: &str = "BaseIntelligence/deepagent";
/// Pinned dataset commit SHA (never a floating branch name).
pub const DEEPAGENT_HF_REVISION: &str = "5fe4e7834ba88a8ae34da8eea7a5f22e05503575";
/// Repo-relative directory that contains one subdirectory per pack.
pub const DEEPAGENT_HF_SUBDIR: &str = "tasks";
/// Env var overriding [`DEEPAGENT_HF_REPO`].
pub const ENV_HF_REPO: &str = "BASE_HF_REPO";
/// Env var overriding [`DEEPAGENT_HF_REVISION`].
pub const ENV_HF_REVISION: &str = "BASE_HF_REVISION";
/// Env var overriding [`DEEPAGENT_HF_SUBDIR`].
pub const ENV_HF_SUBDIR: &str = "BASE_HF_SUBDIR";
/// Env var capping how many packs a pull materializes.
pub const ENV_HF_MAX_PACKS: &str = "BASE_HF_MAX_PACKS";

const HF_ENDPOINT: &str = "https://huggingface.co";
const HTTP_TIMEOUT: Duration = Duration::from_mins(1);
const MAX_ATTEMPTS: u32 = 4;
const MAX_PAGES: usize = 64;
const USER_AGENT: &str = concat!("base-agent-pack/", env!("CARGO_PKG_VERSION"), " (hf-pull)");

/// Failures while listing or downloading a pinned `HuggingFace` dataset subtree.
///
/// Error text carries the URL and pin but never the bearer token.
#[derive(Debug, Error)]
pub enum HfPullError {
    /// Transport failure (DNS, TLS, timeout) after retries: `(url, message)`.
    #[error("hf request failed for {0}: {1}")]
    Http(String, String),
    /// Non-success HTTP status after retries: `(status, url)`.
    #[error("hf status {0} for {1}")]
    Status(u16, String),
    /// Tree listing was not the expected JSON array shape.
    #[error("hf tree json decode failed: {0}")]
    Json(String),
    /// A remote entry path escaped the destination directory.
    #[error("unsafe remote path rejected: {0}")]
    UnsafePath(String),
    /// Filesystem I/O while writing the pulled tree: `(path, message)`.
    #[error("hf pull I/O at {0}: {1}")]
    Io(PathBuf, String),
    /// The pinned revision has no files under the subdir: `(repo@rev, subdir)`.
    #[error("no files under `{1}` in {0}")]
    EmptySubtree(String, String),
    /// Downloaded bytes did not match the advertised LFS sha256.
    #[error("sha256 mismatch for {0}: expected {1}, got {2}")]
    Digest(String, String, String),
    /// The worker thread performing the blocking pull panicked.
    #[error("hf pull worker thread panicked")]
    WorkerPanic,
}

fn io_err(path: &Path, e: &std::io::Error) -> HfPullError {
    HfPullError::Io(path.to_path_buf(), e.to_string())
}

/// What to pull and from where.
#[derive(Debug, Clone)]
pub struct HfPullConfig {
    /// Dataset repo id, e.g. `BaseIntelligence/deepagent`.
    pub repo: String,
    /// Pinned revision (commit SHA strongly preferred over a branch name).
    pub revision: String,
    /// Repo-relative directory holding one subdirectory per pack.
    pub subdir: String,
    /// Optional cap on how many packs to materialize (deterministic prefix).
    pub max_packs: Option<usize>,
    /// Optional bearer token for private repos; never logged.
    pub token: Option<String>,
}

impl Default for HfPullConfig {
    fn default() -> Self {
        Self {
            repo: DEEPAGENT_HF_REPO.to_owned(),
            revision: DEEPAGENT_HF_REVISION.to_owned(),
            subdir: DEEPAGENT_HF_SUBDIR.to_owned(),
            max_packs: None,
            token: None,
        }
    }
}

impl HfPullConfig {
    /// Build a config from the pinned defaults, overridden by environment.
    ///
    /// Reads [`ENV_HF_REPO`], [`ENV_HF_REVISION`], [`ENV_HF_SUBDIR`],
    /// [`ENV_HF_MAX_PACKS`], and `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            repo: env_or(ENV_HF_REPO, DEEPAGENT_HF_REPO),
            revision: env_or(ENV_HF_REVISION, DEEPAGENT_HF_REVISION),
            subdir: env_or(ENV_HF_SUBDIR, DEEPAGENT_HF_SUBDIR),
            max_packs: non_empty_env(ENV_HF_MAX_PACKS).and_then(|v| v.parse().ok()),
            token: non_empty_env("HF_TOKEN").or_else(|| non_empty_env("HUGGING_FACE_HUB_TOKEN")),
        }
    }

    /// Human-readable `repo@revision` label safe to log.
    #[must_use]
    pub fn pin_label(&self) -> String {
        format!("{}@{}", self.repo, self.revision)
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn env_or(key: &str, fallback: &str) -> String {
    non_empty_env(key).unwrap_or_else(|| fallback.to_owned())
}

/// Download `subdir/**` at the pinned revision into `dest`, one dir per pack.
///
/// Existing files whose size already matches the Hub listing are left alone,
/// which makes repeated calls a cheap resume. Returns the sorted pack ids
/// materialized under `dest`.
///
/// # Errors
/// Returns [`HfPullError`] on transport/status failures after retries, on
/// malformed tree JSON, on a remote path that escapes `dest`, on digest
/// mismatch, or on local I/O failure.
pub fn pull_packs(cfg: &HfPullConfig, dest: &Path) -> Result<Vec<String>, HfPullError> {
    // `reqwest::blocking` panics when driven from inside a Tokio runtime, and
    // the challenge binary bootstraps the catalog from an async `serve` path.
    std::thread::scope(|s| {
        s.spawn(|| pull_packs_blocking(cfg, dest))
            .join()
            .unwrap_or(Err(HfPullError::WorkerPanic))
    })
}

fn pull_packs_blocking(cfg: &HfPullConfig, dest: &Path) -> Result<Vec<String>, HfPullError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| HfPullError::Http(HF_ENDPOINT.to_owned(), e.to_string()))?;

    let planned = plan_files(&fetch_tree(&client, cfg)?, &cfg.subdir, cfg.max_packs);
    if planned.is_empty() {
        return Err(HfPullError::EmptySubtree(
            cfg.pin_label(),
            cfg.subdir.clone(),
        ));
    }
    fs::create_dir_all(dest).map_err(|e| io_err(dest, &e))?;

    let mut pack_ids: Vec<String> = Vec::new();
    for file in &planned {
        download_file(&client, cfg, file, dest)?;
        if pack_ids.last() != Some(&file.pack_id) {
            pack_ids.push(file.pack_id.clone());
        }
    }
    Ok(pack_ids)
}

/// One entry from `GET /api/datasets/{repo}/tree/{rev}`.
#[derive(Debug, Clone, Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<LfsInfo>,
}

#[derive(Debug, Clone, Deserialize)]
struct LfsInfo {
    /// sha256 of the content, unlike the git-blob sha1 in the sibling `oid`.
    oid: String,
}

/// A file selected for download.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedFile {
    pack_id: String,
    /// Repo-relative path, e.g. `tasks/<pack>/task.toml`.
    repo_path: String,
    /// Path relative to `dest`, e.g. `<pack>/task.toml`.
    rel_path: String,
    size: u64,
    /// sha256 of content when the Hub exposes one (LFS entries only).
    sha256: Option<String>,
}

fn fetch_tree(
    client: &reqwest::blocking::Client,
    cfg: &HfPullConfig,
) -> Result<Vec<TreeEntry>, HfPullError> {
    let sub = cfg.subdir.trim_matches('/');
    let mut next = Some(format!(
        "{HF_ENDPOINT}/api/datasets/{}/tree/{}/{sub}?recursive=1",
        cfg.repo, cfg.revision
    ));
    let mut out: Vec<TreeEntry> = Vec::new();

    for _ in 0..MAX_PAGES {
        let Some(url) = next.take() else { break };
        let res = get_with_retry(client, &url, cfg.token.as_deref())?;
        let link = res
            .headers()
            .get(reqwest::header::LINK)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let body = res
            .text()
            .map_err(|e| HfPullError::Http(url.clone(), e.to_string()))?;
        let page: Vec<TreeEntry> =
            serde_json::from_str(&body).map_err(|e| HfPullError::Json(e.to_string()))?;
        if !page.is_empty() {
            next = link.as_deref().and_then(parse_next_link);
        }
        out.extend(page);
    }
    Ok(out)
}

/// Extract the `rel="next"` target from an RFC 5988 `Link` header.
fn parse_next_link(header: &str) -> Option<String> {
    let part = header.split(',').find(|p| p.contains("rel=\"next\""))?;
    let start = part.find('<')?.saturating_add(1);
    let end = part.find('>')?;
    part.get(start..end)
        .filter(|u| !u.is_empty())
        .map(str::to_owned)
}

/// Select the files to download: files only, under `subdir`, deterministic
/// pack ordering, truncated to `max_packs`.
fn plan_files(entries: &[TreeEntry], subdir: &str, max_packs: Option<usize>) -> Vec<PlannedFile> {
    let mut by_pack: BTreeMap<String, Vec<PlannedFile>> = BTreeMap::new();
    for e in entries.iter().filter(|e| e.kind == "file") {
        let Some((pack_id, rel_path)) = split_pack_path(subdir, &e.path) else {
            continue;
        };
        by_pack
            .entry(pack_id.clone())
            .or_default()
            .push(PlannedFile {
                pack_id,
                repo_path: e.path.clone(),
                rel_path,
                size: e.size,
                sha256: e.lfs.as_ref().map(|l| l.oid.clone()),
            });
    }

    let mut out = Vec::new();
    for (_, mut files) in by_pack.into_iter().take(max_packs.unwrap_or(usize::MAX)) {
        files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
        out.extend(files);
    }
    out
}

/// Split `subdir/<pack_id>/<tail>` into `(pack_id, "<pack_id>/<tail>")`.
///
/// Returns `None` for paths outside `subdir` and for files sitting directly in
/// `subdir` (those belong to no pack).
fn split_pack_path(subdir: &str, path: &str) -> Option<(String, String)> {
    let sub = subdir.trim_matches('/');
    let rest = if sub.is_empty() {
        path
    } else {
        path.strip_prefix(sub)?.strip_prefix('/')?
    };
    let (pack_id, tail) = rest.split_once('/')?;
    if pack_id.is_empty() || tail.is_empty() {
        return None;
    }
    Some((pack_id.to_owned(), rest.to_owned()))
}

/// Join an untrusted relative path onto `dest`, refusing any escape.
fn safe_join(dest: &Path, rel: &str) -> Result<PathBuf, HfPullError> {
    // Both separators are checked: on Unix `a\..\b` is a single path
    // component, so `Components` alone would miss a Windows-style traversal.
    let sane = !rel.is_empty()
        && !rel.starts_with('/')
        && !rel.starts_with('\\')
        && !rel.contains('\0')
        && rel
            .split(['/', '\\'])
            .all(|s| !s.is_empty() && s != "." && s != "..")
        && Path::new(rel)
            .components()
            .all(|c| matches!(c, Component::Normal(_)));
    let joined = dest.join(rel);
    if !sane || !joined.starts_with(dest) {
        return Err(HfPullError::UnsafePath(rel.to_owned()));
    }
    Ok(joined)
}

/// Unix mode for a pulled file: shell scripts stay executable.
fn file_mode(rel_path: &str) -> u32 {
    if Path::new(rel_path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("sh"))
    {
        0o755
    } else {
        0o644
    }
}

fn download_file(
    client: &reqwest::blocking::Client,
    cfg: &HfPullConfig,
    file: &PlannedFile,
    dest: &Path,
) -> Result<(), HfPullError> {
    let target = safe_join(dest, &file.rel_path)?;
    if fs::metadata(&target).is_ok_and(|m| m.is_file() && m.len() == file.size) {
        return Ok(());
    }

    let url = format!(
        "{HF_ENDPOINT}/datasets/{}/resolve/{}/{}",
        cfg.repo, cfg.revision, file.repo_path
    );
    let bytes = get_with_retry(client, &url, cfg.token.as_deref())?
        .bytes()
        .map_err(|e| HfPullError::Http(url.clone(), e.to_string()))?;

    if let Some(expected) = &file.sha256 {
        let found = crate::digest_hex(&sha256_bytes(&bytes));
        if !found.eq_ignore_ascii_case(expected) {
            return Err(HfPullError::Digest(
                file.repo_path.clone(),
                expected.clone(),
                found,
            ));
        }
    }

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| io_err(parent, &e))?;
    }
    let part = target.with_extension("part");
    fs::write(&part, &bytes).map_err(|e| io_err(&part, &e))?;
    set_mode(&part, file_mode(&file.rel_path))?;
    fs::rename(&part, &target).map_err(|e| io_err(&target, &e))
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<(), HfPullError> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|e| io_err(path, &e))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> Result<(), HfPullError> {
    Ok(())
}

fn get_with_retry(
    client: &reqwest::blocking::Client,
    url: &str,
    token: Option<&str>,
) -> Result<reqwest::blocking::Response, HfPullError> {
    let mut last = HfPullError::Http(url.to_owned(), "no attempt completed".to_owned());
    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(250u64 << attempt.min(5)));
        }
        let mut req = client.get(url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        match req.send() {
            Ok(res) if res.status().is_success() => return Ok(res),
            Ok(res) => {
                let s = res.status().as_u16();
                last = HfPullError::Status(s, url.to_owned());
                if !(s == 408 || s == 429 || (500..600).contains(&s)) {
                    return Err(last);
                }
            }
            Err(e) => last = HfPullError::Http(url.to_owned(), e.to_string()),
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::{
        file_mode, parse_next_link, plan_files, safe_join, split_pack_path, HfPullConfig,
        HfPullError, LfsInfo, TreeEntry, DEEPAGENT_HF_REVISION,
    };
    use std::path::Path;

    fn entry(kind: &str, path: &str, size: u64) -> TreeEntry {
        TreeEntry {
            kind: kind.to_owned(),
            path: path.to_owned(),
            size,
            lfs: None,
        }
    }

    fn sample_tree() -> Vec<TreeEntry> {
        vec![
            entry("directory", "tasks/zeta", 0),
            entry("file", "tasks/zeta/task.toml", 10),
            entry("file", "tasks/zeta/tests/test.sh", 20),
            entry("directory", "tasks/alpha", 0),
            entry("file", "tasks/alpha/task.toml", 11),
            entry("file", "tasks/mid/task.toml", 12),
            // Outside the subdir: must never be selected.
            entry("file", "assets/banner.png", 99),
            entry("file", "README.md", 5),
            // Directly inside the subdir: belongs to no pack.
            entry("file", "tasks/README.md", 5),
        ]
    }

    #[test]
    fn plan_selects_only_files_under_subdir() {
        let planned = plan_files(&sample_tree(), "tasks", None);
        let paths: Vec<&str> = planned.iter().map(|p| p.repo_path.as_str()).collect();
        assert_eq!(
            paths,
            vec![
                "tasks/alpha/task.toml",
                "tasks/mid/task.toml",
                "tasks/zeta/task.toml",
                "tasks/zeta/tests/test.sh",
            ]
        );
        assert!(planned.iter().all(|p| !p.repo_path.starts_with("assets/")));
    }

    #[test]
    fn plan_rel_paths_drop_the_subdir_prefix() {
        let planned = plan_files(&sample_tree(), "tasks", None);
        assert!(planned.iter().all(|p| !p.rel_path.starts_with("tasks/")));
        assert_eq!(planned[0].rel_path, "alpha/task.toml");
    }

    #[test]
    fn max_packs_selection_is_deterministic_and_sorted() {
        let two = plan_files(&sample_tree(), "tasks", Some(2));
        let ids: Vec<&str> = two.iter().map(|p| p.pack_id.as_str()).collect();
        assert_eq!(ids, vec!["alpha", "mid"]);

        // Input order must not change the selection.
        let mut shuffled = sample_tree();
        shuffled.reverse();
        let again = plan_files(&shuffled, "tasks", Some(2));
        assert_eq!(two, again);

        assert_eq!(plan_files(&sample_tree(), "tasks", Some(0)).len(), 0);
        assert_eq!(
            plan_files(&sample_tree(), "tasks", Some(99)).len(),
            plan_files(&sample_tree(), "tasks", None).len()
        );
    }

    #[test]
    fn lfs_sha256_is_carried_but_git_oid_is_not() {
        let mut e = entry("file", "tasks/alpha/blob.bin", 3);
        e.lfs = Some(LfsInfo {
            oid: "aa".repeat(32),
        });
        let planned = plan_files(&[e], "tasks", None);
        assert_eq!(planned[0].sha256.as_deref(), Some("aa".repeat(32).as_str()));
        assert!(plan_files(&sample_tree(), "tasks", None)
            .iter()
            .all(|p| p.sha256.is_none()));
    }

    #[test]
    fn split_pack_path_rejects_non_pack_entries() {
        assert_eq!(
            split_pack_path("tasks", "tasks/a/b.txt"),
            Some(("a".to_owned(), "a/b.txt".to_owned()))
        );
        assert_eq!(split_pack_path("tasks", "tasksy/a/b.txt"), None);
        assert_eq!(split_pack_path("tasks", "other/a/b.txt"), None);
        assert_eq!(split_pack_path("tasks", "tasks/only.txt"), None);
        assert_eq!(split_pack_path("tasks", "tasks/"), None);
    }

    #[test]
    fn safe_join_rejects_traversal_and_absolute_paths() {
        let dest = Path::new("/tmp/packs");
        for bad in [
            "../evil",
            "a/../../evil",
            "/etc/passwd",
            "a/./b",
            "..",
            "",
            "a//b",
            "a\\..\\evil",
            "\\windows",
        ] {
            let err = safe_join(dest, bad);
            assert!(
                matches!(err, Err(HfPullError::UnsafePath(_))),
                "expected rejection for {bad:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn safe_join_accepts_normal_nested_paths() {
        let dest = Path::new("/tmp/packs");
        let ok = safe_join(dest, "alpha/tests/test.sh").expect("normal path");
        assert_eq!(ok, dest.join("alpha").join("tests").join("test.sh"));
        assert!(ok.starts_with(dest));
    }

    #[test]
    fn shell_scripts_are_executable_others_are_not() {
        assert_eq!(file_mode("alpha/pre_artifacts.sh"), 0o755);
        assert_eq!(file_mode("alpha/solution/solve.sh"), 0o755);
        assert_eq!(file_mode("alpha/tests/test.sh"), 0o755);
        assert_eq!(file_mode("alpha/task.toml"), 0o644);
        assert_eq!(file_mode("alpha/instruction.md"), 0o644);
        assert_eq!(file_mode("alpha/sh"), 0o644);
    }

    #[test]
    fn next_link_parsing() {
        assert_eq!(
            parse_next_link("<https://hf.co/next?cursor=x>; rel=\"next\"").as_deref(),
            Some("https://hf.co/next?cursor=x")
        );
        assert_eq!(
            parse_next_link("<https://a/prev>; rel=\"prev\", <https://a/n>; rel=\"next\"")
                .as_deref(),
            Some("https://a/n")
        );
        assert_eq!(parse_next_link("<https://a/prev>; rel=\"prev\""), None);
        assert_eq!(parse_next_link(""), None);
    }

    #[test]
    fn default_config_pins_a_commit_sha_not_a_branch() {
        let cfg = HfPullConfig::default();
        assert_eq!(cfg.revision, DEEPAGENT_HF_REVISION);
        assert_eq!(cfg.revision.len(), 40);
        assert!(cfg.revision.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(cfg.revision, "main");
        assert_eq!(cfg.subdir, "tasks");
        assert!(cfg.token.is_none());
        assert!(cfg.pin_label().contains("BaseIntelligence/deepagent@"));
    }

    /// Real network pull. Ignored so CI stays offline-clean.
    #[test]
    #[ignore = "network: hits huggingface.co"]
    fn network_pull_two_packs() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg = HfPullConfig {
            max_packs: Some(2),
            ..HfPullConfig::from_env()
        };
        let ids = super::pull_packs(&cfg, tmp.path()).expect("pull");
        assert_eq!(ids.len(), 2, "expected 2 packs, got {ids:?}");
        for id in &ids {
            let dir = tmp.path().join(id);
            assert!(dir.join("task.toml").is_file(), "{id}: missing task.toml");
        }
        // Second call is a no-op resume and must report the same ids.
        let again = super::pull_packs(&cfg, tmp.path()).expect("resume");
        assert_eq!(ids, again);
    }
}
