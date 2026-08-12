//! Fail-closed unified-diff apply onto an `AutoModel` pin directory.
//!
//! Flow: validate patch bytes → prefer `git apply --check` then `git apply`;
//! if `git` is unavailable, fall back to a pure-Rust text apply. Rejects
//! path escape (`..`), absolute paths, binary blobs, and oversized patches.

use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

use thiserror::Error;

use crate::classify::{diffstat_from_parsed, DiffStat};

/// Maximum accepted unified-diff size (bytes).
pub const MAX_PATCH_BYTES: usize = 2 * 1024 * 1024;
/// Maximum number of files a single patch may touch.
pub const MAX_PATCH_FILES: usize = 512;

/// Errors from validate / apply (fail-closed).
#[derive(Debug, Error)]
pub enum ApplyError {
    #[error("patch oversized: {got} bytes (max {max})")]
    Oversized { got: usize, max: usize },
    #[error("patch touches too many files: {got} (max {max})")]
    TooManyFiles { got: usize, max: usize },
    #[error("invalid or empty unified diff")]
    InvalidPatch,
    #[error("path escape rejected: {path}")]
    PathEscape { path: String },
    #[error("absolute path rejected: {path}")]
    AbsolutePath { path: String },
    #[error("binary patch blob rejected: {path}")]
    BinaryBlob { path: String },
    #[error("patch conflict: {detail}")]
    Conflict { detail: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("git apply failed: {0}")]
    Git(String),
}

/// Result of a successful apply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApplyResult {
    /// Pin-relative paths touched by the patch (sorted).
    pub touched: Vec<String>,
    /// Classified diffstat derived from the patch text.
    pub diffstat: DiffStat,
}

/// Validate `patch` without touching the filesystem tree.
pub fn validate_patch(patch: &[u8]) -> Result<(), ApplyError> {
    let _ = parse_file_hunks(patch)?;
    Ok(())
}

/// Apply a unified diff onto `pin_dir` (fail-closed).
///
/// Prefers `git apply --check` + `git apply` when `git` is on `PATH`;
/// otherwise uses the pure-Rust text applicator (sufficient for CI fixtures).
pub fn apply_patch(pin_dir: &Path, patch: &[u8]) -> Result<ApplyResult, ApplyError> {
    let files = parse_file_hunks(patch)?;
    let result = result_from_parsed(&files);
    if git_available() {
        git_apply(pin_dir, patch)?;
    } else {
        rust_apply(pin_dir, &files)?;
    }
    Ok(result)
}

/// Force the pure-Rust applicator (tests / no-git environments).
pub fn apply_patch_rust(pin_dir: &Path, patch: &[u8]) -> Result<ApplyResult, ApplyError> {
    let files = parse_file_hunks(patch)?;
    let result = result_from_parsed(&files);
    rust_apply(pin_dir, &files)?;
    Ok(result)
}

fn result_from_parsed(files: &[ParsedFile]) -> ApplyResult {
    let diffstat = diffstat_from_parsed(files);
    let mut touched: Vec<String> = files.iter().map(|f| f.path.clone()).collect();
    touched.sort();
    touched.dedup();
    ApplyResult { touched, diffstat }
}

fn git_available() -> bool {
    Command::new("git")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn git_apply(pin_dir: &Path, patch: &[u8]) -> Result<(), ApplyError> {
    let tmp = pin_dir.join(".prism-automodel-pending.patch");
    fs::write(&tmp, patch)?;
    let run = |check: bool| -> Result<(), ApplyError> {
        let mut cmd = Command::new("git");
        cmd.arg("-C")
            .arg(pin_dir)
            .arg("apply")
            .arg("--whitespace=nowarn");
        if check {
            cmd.arg("--check");
        }
        cmd.arg(&tmp);
        let out = cmd.output().map_err(|e| ApplyError::Git(e.to_string()))?;
        if out.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_owned();
        let detail = if stderr.is_empty() {
            String::from_utf8_lossy(&out.stdout).trim().to_owned()
        } else {
            stderr
        };
        let detail = if detail.is_empty() {
            "git apply failed".into()
        } else {
            detail
        };
        if check || detail.to_ascii_lowercase().contains("conflict") {
            return Err(ApplyError::Conflict { detail });
        }
        Err(ApplyError::Git(detail))
    };
    let result = run(true).and_then(|()| run(false));
    let _ = fs::remove_file(&tmp);
    result
}

fn rust_apply(pin_dir: &Path, files: &[ParsedFile]) -> Result<(), ApplyError> {
    for file in files {
        let dest = safe_join(pin_dir, &file.path)?;
        if file.is_new {
            apply_new_file(&dest, file)?;
            continue;
        }
        if file.is_deleted {
            if !dest.is_file() {
                return Err(ApplyError::Conflict {
                    detail: format!("delete target missing: {}", file.path),
                });
            }
            fs::remove_file(&dest)?;
            continue;
        }
        let original = fs::read_to_string(&dest).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ApplyError::Conflict {
                    detail: format!("modify target missing: {}", file.path),
                }
            } else {
                ApplyError::Io(e)
            }
        })?;
        let updated = apply_hunks_to_text(&original, &file.hunks, &file.path)?;
        fs::write(&dest, updated)?;
    }
    Ok(())
}

fn apply_new_file(dest: &Path, file: &ParsedFile) -> Result<(), ApplyError> {
    if dest.exists() {
        return Err(ApplyError::Conflict {
            detail: format!("new file already exists: {}", file.path),
        });
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(dest, materialize_new_file(&file.hunks)?)?;
    Ok(())
}

fn materialize_new_file(hunks: &[Hunk]) -> Result<String, ApplyError> {
    let mut out = String::new();
    for hunk in hunks {
        for line in &hunk.lines {
            match line.kind {
                HunkLineKind::Add | HunkLineKind::Context => {
                    out.push_str(&line.text);
                    out.push('\n');
                }
                HunkLineKind::Delete => return Err(ApplyError::InvalidPatch),
            }
        }
    }
    Ok(out)
}

fn apply_hunks_to_text(original: &str, hunks: &[Hunk], path: &str) -> Result<String, ApplyError> {
    let had_trailing_nl = original.ends_with('\n');
    let mut lines: Vec<String> = original.lines().map(str::to_owned).collect();
    // Apply last-hunk-first so earlier hunk line numbers stay valid.
    for hunk in hunks.iter().rev() {
        apply_one_hunk(&mut lines, hunk, path)?;
    }
    let mut out = lines.join("\n");
    if (had_trailing_nl || original.is_empty()) && !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

fn apply_one_hunk(lines: &mut Vec<String>, hunk: &Hunk, path: &str) -> Result<(), ApplyError> {
    let start = hunk.old_start.saturating_sub(1);
    let mut idx = start;
    let mut old_consumed = 0usize;
    let mut new_lines: Vec<String> = Vec::new();
    for line in &hunk.lines {
        match line.kind {
            HunkLineKind::Context => {
                let existing = lines.get(idx).ok_or_else(|| ApplyError::Conflict {
                    detail: format!("{path}: context past EOF"),
                })?;
                if existing != &line.text {
                    return Err(ApplyError::Conflict {
                        detail: format!("{path}: context mismatch at line {}", idx + 1),
                    });
                }
                new_lines.push(existing.clone());
                idx = idx.saturating_add(1);
                old_consumed = old_consumed.saturating_add(1);
            }
            HunkLineKind::Delete => {
                let existing = lines.get(idx).ok_or_else(|| ApplyError::Conflict {
                    detail: format!("{path}: delete past EOF"),
                })?;
                if existing != &line.text {
                    return Err(ApplyError::Conflict {
                        detail: format!("{path}: delete mismatch at line {}", idx + 1),
                    });
                }
                idx = idx.saturating_add(1);
                old_consumed = old_consumed.saturating_add(1);
            }
            HunkLineKind::Add => new_lines.push(line.text.clone()),
        }
    }
    let end = start.saturating_add(old_consumed);
    if end > lines.len() {
        return Err(ApplyError::Conflict {
            detail: format!("{path}: hunk range past EOF"),
        });
    }
    lines.splice(start..end, new_lines);
    Ok(())
}

fn safe_join(pin_dir: &Path, rel: &str) -> Result<PathBuf, ApplyError> {
    check_rel_path(rel)?;
    let mut out = pin_dir.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(s) => out.push(s),
            Component::CurDir => {}
            _ => {
                return Err(ApplyError::PathEscape {
                    path: rel.to_owned(),
                });
            }
        }
    }
    if !out.starts_with(pin_dir) {
        return Err(ApplyError::PathEscape {
            path: rel.to_owned(),
        });
    }
    Ok(out)
}

fn check_rel_path(path: &str) -> Result<(), ApplyError> {
    if path.is_empty() || path == "/dev/null" {
        return Err(ApplyError::InvalidPatch);
    }
    if path.starts_with('/') || path.starts_with('\\') {
        return Err(ApplyError::AbsolutePath {
            path: path.to_owned(),
        });
    }
    if path.len() >= 2 && path.as_bytes()[1] == b':' {
        return Err(ApplyError::AbsolutePath {
            path: path.to_owned(),
        });
    }
    if path.split(['/', '\\']).any(|p| p == "..") {
        return Err(ApplyError::PathEscape {
            path: path.to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedFile {
    pub path: String,
    pub is_new: bool,
    pub is_deleted: bool,
    pub hunks: Vec<Hunk>,
}

#[derive(Debug, Clone)]
pub(crate) struct Hunk {
    pub old_start: usize,
    pub lines: Vec<HunkLine>,
}

#[derive(Debug, Clone)]
pub(crate) struct HunkLine {
    pub kind: HunkLineKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HunkLineKind {
    Context,
    Add,
    Delete,
}

/// Parse and validate a unified diff into per-file hunks.
pub(crate) fn parse_file_hunks(patch: &[u8]) -> Result<Vec<ParsedFile>, ApplyError> {
    let text = preprocess_patch(patch)?;
    let mut files: Vec<ParsedFile> = Vec::new();
    let mut current: Option<ParsedFile> = None;
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        if line.starts_with("diff --git ") {
            if let Some(f) = current.take() {
                files.push(f);
            }
            let file_path = path_from_diff_git(line)?;
            check_rel_path(&file_path)?;
            current = Some(ParsedFile {
                path: file_path,
                is_new: false,
                is_deleted: false,
                hunks: Vec::new(),
            });
        } else if line.starts_with("new file mode ") {
            if let Some(f) = current.as_mut() {
                f.is_new = true;
            }
        } else if line.starts_with("deleted file mode ") {
            if let Some(f) = current.as_mut() {
                f.is_deleted = true;
            }
        } else if let Some(rest) = line.strip_prefix("--- ") {
            handle_minus_header(rest, &mut current)?;
        } else if let Some(rest) = line.strip_prefix("+++ ") {
            handle_plus_header(rest, &mut current)?;
        } else if line.starts_with("@@ ") {
            let hunk = parse_hunk(line, &mut lines)?;
            let Some(f) = current.as_mut() else {
                return Err(ApplyError::InvalidPatch);
            };
            f.hunks.push(hunk);
        }
    }
    if let Some(f) = current.take() {
        files.push(f);
    }
    finalize_files(files)
}

fn preprocess_patch(patch: &[u8]) -> Result<&str, ApplyError> {
    if patch.len() > MAX_PATCH_BYTES {
        return Err(ApplyError::Oversized {
            got: patch.len(),
            max: MAX_PATCH_BYTES,
        });
    }
    if patch.is_empty() {
        return Err(ApplyError::InvalidPatch);
    }
    if patch.contains(&0) {
        return Err(ApplyError::BinaryBlob {
            path: "<patch>".into(),
        });
    }
    let text = std::str::from_utf8(patch).map_err(|_| ApplyError::BinaryBlob {
        path: "<patch>".into(),
    })?;
    let lower = text.to_ascii_lowercase();
    if lower.contains("git binary patch") || lower.contains("binary files ") {
        return Err(ApplyError::BinaryBlob {
            path: sniff_binary_path(text),
        });
    }
    Ok(text)
}

fn handle_minus_header(rest: &str, current: &mut Option<ParsedFile>) -> Result<(), ApplyError> {
    let p = strip_ab_prefix(rest.trim());
    if p == "/dev/null" {
        if let Some(f) = current.as_mut() {
            f.is_new = true;
        }
        return Ok(());
    }
    check_rel_path(p)?;
    if let Some(f) = current.as_mut() {
        if f.path.is_empty() {
            p.clone_into(&mut f.path);
        }
    }
    Ok(())
}

fn handle_plus_header(rest: &str, current: &mut Option<ParsedFile>) -> Result<(), ApplyError> {
    let p = strip_ab_prefix(rest.trim());
    if p == "/dev/null" {
        if let Some(f) = current.as_mut() {
            f.is_deleted = true;
        }
        return Ok(());
    }
    check_rel_path(p)?;
    if let Some(f) = current.as_mut() {
        p.clone_into(&mut f.path);
    }
    Ok(())
}

fn finalize_files(files: Vec<ParsedFile>) -> Result<Vec<ParsedFile>, ApplyError> {
    if files.is_empty() {
        return Err(ApplyError::InvalidPatch);
    }
    if files.len() > MAX_PATCH_FILES {
        return Err(ApplyError::TooManyFiles {
            got: files.len(),
            max: MAX_PATCH_FILES,
        });
    }
    for f in &files {
        check_rel_path(&f.path)?;
    }
    Ok(files)
}

fn sniff_binary_path(text: &str) -> String {
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            if let Ok(p) = path_from_diff_git(&format!("diff --git {rest}")) {
                return p;
            }
        }
    }
    "<binary>".into()
}

fn path_from_diff_git(line: &str) -> Result<String, ApplyError> {
    let rest = line
        .strip_prefix("diff --git ")
        .ok_or(ApplyError::InvalidPatch)?;
    let mut parts = rest.split_whitespace();
    let a = parts.next().ok_or(ApplyError::InvalidPatch)?;
    let b = parts.next().ok_or(ApplyError::InvalidPatch)?;
    let path = if b == "/dev/null" {
        strip_ab_prefix(a)
    } else {
        strip_ab_prefix(b)
    };
    if path.is_empty() {
        return Err(ApplyError::InvalidPatch);
    }
    Ok(path.to_owned())
}

fn strip_ab_prefix(path: &str) -> &str {
    let p = path.trim().trim_matches('"');
    p.strip_prefix("a/")
        .or_else(|| p.strip_prefix("b/"))
        .unwrap_or(p)
}

fn parse_hunk<'a, I>(header: &str, lines: &mut std::iter::Peekable<I>) -> Result<Hunk, ApplyError>
where
    I: Iterator<Item = &'a str>,
{
    let mut parts = header.split_whitespace();
    let _at = parts.next();
    let old = parts.next().ok_or(ApplyError::InvalidPatch)?;
    let new = parts.next().ok_or(ApplyError::InvalidPatch)?;
    let (old_start, old_count) = parse_hunk_range(old, true)?;
    let (_new_start, new_count) = parse_hunk_range(new, false)?;

    let mut hunk_lines = Vec::new();
    let mut old_seen = 0usize;
    let mut new_seen = 0usize;
    while old_seen < old_count || new_seen < new_count {
        let Some(line) = lines.next() else {
            break;
        };
        if line.starts_with('\\') {
            continue;
        }
        let parsed = classify_hunk_line(line)?;
        match parsed.0 {
            HunkLineKind::Add => new_seen = new_seen.saturating_add(1),
            HunkLineKind::Delete => old_seen = old_seen.saturating_add(1),
            HunkLineKind::Context => {
                old_seen = old_seen.saturating_add(1);
                new_seen = new_seen.saturating_add(1);
            }
        }
        hunk_lines.push(HunkLine {
            kind: parsed.0,
            text: parsed.1,
        });
    }
    Ok(Hunk {
        old_start,
        lines: hunk_lines,
    })
}

fn classify_hunk_line(line: &str) -> Result<(HunkLineKind, String), ApplyError> {
    if let Some(t) = line.strip_prefix('+') {
        return Ok((HunkLineKind::Add, t.to_owned()));
    }
    if let Some(t) = line.strip_prefix('-') {
        return Ok((HunkLineKind::Delete, t.to_owned()));
    }
    if let Some(t) = line.strip_prefix(' ') {
        return Ok((HunkLineKind::Context, t.to_owned()));
    }
    if line.is_empty() {
        return Ok((HunkLineKind::Context, String::new()));
    }
    Err(ApplyError::InvalidPatch)
}

fn parse_hunk_range(spec: &str, old: bool) -> Result<(usize, usize), ApplyError> {
    let spec = if old {
        spec.strip_prefix('-').unwrap_or(spec)
    } else {
        spec.strip_prefix('+').unwrap_or(spec)
    };
    let (start, count) = match spec.split_once(',') {
        Some((s, c)) => (s, c),
        None => (spec, "1"),
    };
    let start: usize = start.parse().map_err(|_| ApplyError::InvalidPatch)?;
    let count: usize = count.parse().map_err(|_| ApplyError::InvalidPatch)?;
    Ok((start, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pin::{fixture_happy_patch_path, fixture_pin_dir};
    use tempfile::tempdir;

    fn copy_fixture(dst: &Path) {
        copy_dir(&fixture_pin_dir(), dst);
    }

    fn copy_dir(src: &Path, dst: &Path) {
        fs::create_dir_all(dst).expect("mkdir");
        for entry in fs::read_dir(src).expect("read") {
            let entry = entry.expect("entry");
            let to = dst.join(entry.file_name());
            if entry.path().is_dir() {
                copy_dir(&entry.path(), &to);
            } else {
                fs::copy(entry.path(), &to).expect("copy");
            }
        }
    }

    #[test]
    fn happy_apply_git_or_rust() {
        let dir = tempdir().expect("tmp");
        let pin = dir.path().join("pin");
        copy_fixture(&pin);
        let patch = fs::read(fixture_happy_patch_path()).expect("patch");
        let result = apply_patch(&pin, &patch).expect("apply");
        assert!(result.touched.iter().any(|p| p.ends_with("model.py")));
        assert!(result.touched.iter().any(|p| p.ends_with("layers.py")));
        let model = fs::read_to_string(pin.join("nemo_automodel/components/models/toy/model.py"))
            .expect("model");
        assert!(model.contains("width: int = 16"));
        assert!(model.contains("return x * self.width"));
        assert!(pin
            .join("nemo_automodel/components/models/toy/layers.py")
            .is_file());
    }

    #[test]
    fn happy_apply_pure_rust() {
        let dir = tempdir().expect("tmp");
        let pin = dir.path().join("pin");
        copy_fixture(&pin);
        let patch = fs::read(fixture_happy_patch_path()).expect("patch");
        apply_patch_rust(&pin, &patch).expect("rust apply");
        let layers = fs::read_to_string(pin.join("nemo_automodel/components/models/toy/layers.py"))
            .expect("layers");
        assert!(layers.contains("def scale"));
    }

    #[test]
    fn reject_path_escape() {
        let patch = b"\
diff --git a/../evil.py b/../evil.py
--- a/../evil.py
+++ b/../evil.py
@@ -0,0 +1,1 @@
+pwned
";
        let err = validate_patch(patch).expect_err("escape");
        assert!(matches!(err, ApplyError::PathEscape { .. }));
    }

    #[test]
    fn reject_absolute_path() {
        let abs = b"\
diff --git a/foo b/foo
--- a/foo
+++ /tmp/absolute.py
@@ -0,0 +1,1 @@
+x
";
        let err = validate_patch(abs).expect_err("abs");
        assert!(matches!(err, ApplyError::AbsolutePath { .. }));
    }

    #[test]
    fn reject_bad_patch() {
        let err = validate_patch(b"not a patch").expect_err("bad");
        assert!(matches!(err, ApplyError::InvalidPatch));
    }

    #[test]
    fn reject_binary_marker() {
        let patch = b"\
diff --git a/nemo_automodel/x.bin b/nemo_automodel/x.bin
Binary files a/nemo_automodel/x.bin and b/nemo_automodel/x.bin differ
";
        let err = validate_patch(patch).expect_err("bin");
        assert!(matches!(err, ApplyError::BinaryBlob { .. }));
    }

    #[test]
    fn reject_conflict_on_mismatched_context() {
        let dir = tempdir().expect("tmp");
        let pin = dir.path().join("pin");
        copy_fixture(&pin);
        let patch = b"\
diff --git a/nemo_automodel/components/models/toy/model.py b/nemo_automodel/components/models/toy/model.py
--- a/nemo_automodel/components/models/toy/model.py
+++ b/nemo_automodel/components/models/toy/model.py
@@ -4,6 +4,6 @@
 class ToyModel:
-    def __init__(self, width: int = 999) -> None:
+    def __init__(self, width: int = 1) -> None:
         self.width = width
 
     def forward(self, x):
         return x
";
        let err = apply_patch_rust(&pin, patch).expect_err("conflict");
        assert!(matches!(err, ApplyError::Conflict { .. }));
    }

    #[test]
    fn reject_oversized() {
        let mut huge =
            Vec::from(&b"diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -0,0 +1,1 @@\n+y\n"[..]);
        huge.resize(MAX_PATCH_BYTES + 8, b' ');
        let err = validate_patch(&huge).expect_err("size");
        assert!(matches!(err, ApplyError::Oversized { .. }));
    }
}
