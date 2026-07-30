//! Per-crate non-test LOC cap (D1): max 1500 lines.

use std::fs;
use std::path::Path;

const MAX_NON_TEST_LOC: usize = 1500;

/// Count non-blank, non-line-comment source lines, skipping test-only paths and `#[cfg(test)]` modules.
pub fn run(workspace_root: &Path) -> Result<(), String> {
    let mut failures = Vec::new();
    let mut reported_any = false;

    for area in ["crates", "bins"] {
        let area_path = workspace_root.join(area);
        if !area_path.is_dir() {
            continue;
        }
        for entry in fs::read_dir(&area_path).map_err(|e| format!("read_dir {area}: {e}"))? {
            let entry = entry.map_err(|e| format!("read_dir entry: {e}"))?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let manifest = path.join("Cargo.toml");
            if !manifest.is_file() {
                continue;
            }
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_owned();
            let loc = count_crate_loc(&path)?;
            reported_any = true;
            println!("{area}/{name}: {loc} non-test LOC (cap {MAX_NON_TEST_LOC})");
            if loc > MAX_NON_TEST_LOC {
                failures.push(format!(
                    "{area}/{name}: {loc} non-test LOC exceeds cap {MAX_NON_TEST_LOC}"
                ));
            }
        }
    }

    if !reported_any {
        println!("loc-cap: no packages under crates/ or bins/ (ok)");
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n"))
    }
}

fn count_crate_loc(crate_dir: &Path) -> Result<usize, String> {
    let mut total = 0usize;
    let mut stack = vec![crate_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if name == "target" || name == ".git" {
                    continue;
                }
                if name == "tests" || name == "benches" || name == "examples" {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if name == "tests.rs" || name.ends_with("_tests.rs") {
                continue;
            }
            total = total.saturating_add(count_file_loc(&path)?);
        }
    }
    Ok(total)
}

fn count_file_loc(path: &Path) -> Result<usize, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(count_non_test_loc(&text))
}

/// Count LOC outside `#[cfg(test)]` module blocks (brace-balanced).
fn count_non_test_loc(source: &str) -> usize {
    let mut loc = 0usize;
    let mut depth_test: Option<usize> = None;
    let mut brace_depth = 0usize;
    let mut pending_cfg_test = false;

    for line in source.lines() {
        let trimmed = line.trim();

        if pending_cfg_test {
            if let Some(rest) = trimmed.strip_prefix("mod ") {
                if rest.contains('{') {
                    depth_test = Some(brace_depth);
                    pending_cfg_test = false;
                } else if rest.ends_with(';') {
                    pending_cfg_test = false;
                    continue;
                }
            } else if !trimmed.is_empty() && !trimmed.starts_with("//") && !trimmed.starts_with('#')
            {
                pending_cfg_test = false;
            }
        }

        if trimmed == "#[cfg(test)]" || trimmed.starts_with("#[cfg(test)]") {
            pending_cfg_test = true;
            continue;
        }

        if trimmed.contains("#[cfg(test)]") && trimmed.contains("mod ") {
            if trimmed.contains('{') {
                depth_test = Some(brace_depth);
            }
            continue;
        }

        let opens = line.chars().filter(|&c| c == '{').count();
        let closes = line.chars().filter(|&c| c == '}').count();

        if depth_test.is_none() && is_code_line(trimmed) {
            loc = loc.saturating_add(1);
        }

        brace_depth = brace_depth.saturating_add(opens);
        for _ in 0..closes {
            brace_depth = brace_depth.saturating_sub(1);
            if depth_test == Some(brace_depth) {
                depth_test = None;
            }
        }
        if let Some(start) = depth_test {
            if brace_depth < start {
                depth_test = None;
            }
        }
    }
    loc
}

fn is_code_line(trimmed: &str) -> bool {
    !trimmed.is_empty() && !trimmed.starts_with("//")
}

#[cfg(test)]
mod tests {
    use super::{count_non_test_loc, MAX_NON_TEST_LOC};
    use std::path::{Path, PathBuf};

    #[test]
    fn skips_cfg_test_module_body() {
        let src = r"
pub fn a() {}

#[cfg(test)]
mod tests {
    #[test]
    fn t() {
        let _x = 1;
        let _y = 2;
    }
}
";
        let n = count_non_test_loc(src);
        assert!(n <= 3, "expected only public fn lines, got {n}");
    }

    #[test]
    fn counts_normal_code() {
        let src = "pub fn a() {}\npub fn b() {}\n";
        assert_eq!(count_non_test_loc(src), 2);
    }

    #[test]
    fn cap_is_1500() {
        assert_eq!(MAX_NON_TEST_LOC, 1500);
    }

    #[test]
    fn run_on_workspace_ok() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = root.parent().map_or_else(PathBuf::new, Path::to_path_buf);
        assert!(!root.as_os_str().is_empty(), "workspace root");
        super::run(&root).expect("loc-cap should pass on this workspace");
    }
}
