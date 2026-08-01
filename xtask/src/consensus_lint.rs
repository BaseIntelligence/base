//! Consensus-crate lint (D7/D8).
//!
//! Reads `xtask/consensus-crates.txt` (one crate directory name per line).
//! Missing crate directories are skipped (future crates may be listed early).
//! Existing crates' non-test sources are scanned for forbidden tokens.
//!
//! A line may waive individual tokens with `name allow=f32,f64`. Waivers are
//! per token so a crate that needs one relaxation keeps every other rule.

use regex::Regex;
use std::fs;
use std::path::Path;

/// Tokens banned in consensus crates, as (waiver label, match pattern).
const BANNED_TOKENS: [(&str, &str); 4] = [
    ("HashMap", r"\bHashMap\b"),
    ("f32", r"\bf32\b"),
    ("f64", r"\bf64\b"),
    ("wrapping_", r"\bwrapping_"),
];

#[derive(Debug)]
struct CrateRule {
    name: String,
    allow: Vec<String>,
}

/// Behavior: crates listed in `consensus-crates.txt` that do not yet exist under
/// `crates/<name>/` are **skipped** with a note. Only existing trees are linted.
pub fn run(workspace_root: &Path) -> Result<(), String> {
    let list_path = workspace_root.join("xtask/consensus-crates.txt");
    let list =
        fs::read_to_string(&list_path).map_err(|e| format!("read {}: {e}", list_path.display()))?;

    let rules = parse_rules(&list)?;

    if rules.is_empty() {
        return Err("consensus-crates.txt lists no crates".into());
    }

    let mut violations = Vec::new();
    let mut linted = 0usize;

    for rule in &rules {
        let crate_dir = workspace_root.join("crates").join(&rule.name);
        if !crate_dir.is_dir() {
            println!("consensus-lint: skip missing crates/{}", rule.name);
            continue;
        }
        linted = linted.saturating_add(1);
        if !rule.allow.is_empty() {
            println!(
                "consensus-lint: crates/{} waives {}",
                rule.name,
                rule.allow.join(",")
            );
        }
        let token_re = token_regex(&rule.allow)?;
        scan_crate(&crate_dir, &rule.name, token_re.as_ref(), &mut violations)?;
    }

    println!(
        "consensus-lint: scanned {linted} existing crate(s), {} listed",
        rules.len()
    );

    if violations.is_empty() {
        Ok(())
    } else {
        for v in &violations {
            eprintln!("FORBIDDEN: {v}");
        }
        Err(format!(
            "{} consensus lint violation(s) (D7/D8: checked_* only; no HashMap/f32/f64/wrapping_*)",
            violations.len()
        ))
    }
}

/// Parse `name` / `name allow=f32,f64` lines, rejecting unknown waiver labels so
/// a typo fails the gate instead of silently waiving nothing.
fn parse_rules(list: &str) -> Result<Vec<CrateRule>, String> {
    let mut rules = Vec::new();
    for line in list.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else { continue };
        let mut allow = Vec::new();
        for field in fields {
            let Some(tokens) = field.strip_prefix("allow=") else {
                return Err(format!(
                    "consensus-crates.txt: unexpected field {field:?} on line {line:?} \
                     (only `allow=<token>[,<token>]` is supported)"
                ));
            };
            for token in tokens.split(',').map(str::trim).filter(|t| !t.is_empty()) {
                if !BANNED_TOKENS.iter().any(|(label, _)| *label == token) {
                    return Err(format!(
                        "consensus-crates.txt: unknown waiver token {token:?} for crate {name:?}"
                    ));
                }
                allow.push(token.to_owned());
            }
        }
        rules.push(CrateRule {
            name: name.to_owned(),
            allow,
        });
    }
    Ok(rules)
}

/// `None` when every token is waived, which means there is nothing left to match.
fn token_regex(allow: &[String]) -> Result<Option<Regex>, String> {
    let patterns: Vec<&str> = BANNED_TOKENS
        .iter()
        .filter(|(label, _)| !allow.iter().any(|a| a == label))
        .map(|(_, pattern)| *pattern)
        .collect();
    if patterns.is_empty() {
        return Ok(None);
    }
    Regex::new(&patterns.join("|"))
        .map(Some)
        .map_err(|e| format!("compile token regex: {e}"))
}

fn scan_crate(
    crate_dir: &Path,
    crate_name: &str,
    token_re: Option<&Regex>,
    violations: &mut Vec<String>,
) -> Result<(), String> {
    let mut stack = vec![crate_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
            let path = entry.path();
            let fname = entry.file_name();
            let fname = fname.to_string_lossy();
            if path.is_dir() {
                if fname == "target"
                    || fname == "tests"
                    || fname == "benches"
                    || fname == "examples"
                {
                    continue;
                }
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if fname == "tests.rs" || fname.ends_with("_tests.rs") {
                continue;
            }
            let rel = path
                .strip_prefix(crate_dir)
                .unwrap_or(path.as_path())
                .display()
                .to_string();
            let text =
                fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let prod = strip_cfg_test_modules(&text);
            for (idx, line) in prod.lines().enumerate() {
                let line_no = idx + 1;
                let trimmed = line.trim();
                if trimmed.starts_with("//") {
                    continue;
                }
                if token_re.is_some_and(|re| re.is_match(line)) {
                    violations.push(format!(
                        "crates/{crate_name}/{rel}:{line_no}: forbidden token: {trimmed}"
                    ));
                }
                if has_bare_u128_arithmetic(trimmed) {
                    violations.push(format!(
                        "crates/{crate_name}/{rel}:{line_no}: bare arithmetic near u128 (use checked_*): {trimmed}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Heuristic: line mentions `u128` and has bare `+`/`*`/`-` (not `checked_*` / `saturating_*`).
fn has_bare_u128_arithmetic(line: &str) -> bool {
    if !line.contains("u128") {
        return false;
    }
    if line.contains("checked_") || line.contains("saturating_") || line.contains("overflowing_") {
        return false;
    }
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '+' || c == '*' || c == '-' {
            // Look past whitespace for binary operands: `a + b`, `a+=1`.
            let prev = prev_non_ws(bytes, i);
            let next = next_non_ws(bytes, i + 1);
            let prev_ok = prev.is_ascii_alphanumeric() || prev == '_' || prev == ')' || prev == ']';
            let next_ok = next.is_ascii_alphanumeric() || next == '_' || next == '(' || next == '[';
            if next == '=' && prev_ok {
                return true;
            }
            if prev_ok && next_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn prev_non_ws(bytes: &[u8], idx: usize) -> char {
    let mut j = idx;
    while j > 0 {
        j -= 1;
        let c = bytes[j] as char;
        if !c.is_ascii_whitespace() {
            return c;
        }
    }
    ' '
}

fn next_non_ws(bytes: &[u8], idx: usize) -> char {
    let mut j = idx;
    while j < bytes.len() {
        let c = bytes[j] as char;
        if !c.is_ascii_whitespace() {
            return c;
        }
        j += 1;
    }
    ' '
}

/// Drop `#[cfg(test)]` module bodies so test-only code is not linted.
fn strip_cfg_test_modules(source: &str) -> String {
    let mut out = String::new();
    let mut depth_test: Option<usize> = None;
    let mut brace_depth = 0usize;
    let mut pending_cfg_test = false;

    for line in source.lines() {
        let trimmed = line.trim();

        if pending_cfg_test {
            if trimmed.starts_with("mod ") {
                if trimmed.contains('{') {
                    depth_test = Some(brace_depth);
                }
                pending_cfg_test = false;
                apply_braces(line, &mut brace_depth, &mut depth_test);
                continue;
            } else if !trimmed.is_empty() && !trimmed.starts_with('#') && !trimmed.starts_with("//")
            {
                pending_cfg_test = false;
            }
        }

        if trimmed == "#[cfg(test)]" || trimmed.starts_with("#[cfg(test)]") {
            if trimmed.contains("mod ") && trimmed.contains('{') {
                depth_test = Some(brace_depth);
                apply_braces(line, &mut brace_depth, &mut depth_test);
                continue;
            }
            pending_cfg_test = true;
            continue;
        }

        if depth_test.is_none() {
            out.push_str(line);
            out.push('\n');
        }

        apply_braces(line, &mut brace_depth, &mut depth_test);
    }
    out
}

fn apply_braces(line: &str, brace_depth: &mut usize, depth_test: &mut Option<usize>) {
    let opens = line.chars().filter(|&c| c == '{').count();
    let closes = line.chars().filter(|&c| c == '}').count();
    *brace_depth = brace_depth.saturating_add(opens);
    for _ in 0..closes {
        *brace_depth = brace_depth.saturating_sub(1);
        if *depth_test == Some(*brace_depth) {
            *depth_test = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{has_bare_u128_arithmetic, parse_rules, run, strip_cfg_test_modules, token_regex};
    use regex::Regex;
    use std::path::{Path, PathBuf};

    #[test]
    fn waiver_relaxes_only_the_named_token() {
        let rules = parse_rules("aggregate allow=f32,f64\nmerkle\n").expect("parse");
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].allow, vec!["f32".to_owned(), "f64".to_owned()]);
        assert!(rules[1].allow.is_empty());

        let re = token_regex(&rules[0].allow)
            .expect("regex")
            .expect("some tokens remain banned");
        assert!(!re.is_match("let x: f64 = 1.0;"));
        assert!(re.is_match("a.wrapping_add(1)"));
        assert!(re.is_match("use std::collections::HashMap;"));
    }

    #[test]
    fn unknown_waiver_token_is_rejected() {
        let err = parse_rules("aggregate allow=f65\n").expect_err("typo must fail the gate");
        assert!(err.contains("unknown waiver token"), "{err}");
        let err = parse_rules("aggregate deny=f64\n").expect_err("unknown field must fail");
        assert!(err.contains("unexpected field"), "{err}");
    }

    #[test]
    fn strips_test_mod() {
        let src = "pub fn ok() {}\n#[cfg(test)]\nmod t {\n use std::collections::HashMap;\n}\n";
        let out = strip_cfg_test_modules(src);
        assert!(out.contains("pub fn ok"));
        assert!(!out.contains("HashMap"));
    }

    #[test]
    fn token_regex_hits_f64_and_wrapping() {
        let re = Regex::new(r"\bHashMap\b|\bf32\b|\bf64\b|\bwrapping_").expect("re");
        assert!(re.is_match("let x: f64 = 1.0;"));
        assert!(re.is_match("a.wrapping_add(1)"));
        assert!(re.is_match("use std::collections::HashMap;"));
        assert!(!re.is_match("let x: u64 = 1;"));
    }

    #[test]
    fn bare_u128_detects_add() {
        assert!(has_bare_u128_arithmetic("let x: u128 = a + b;"));
        assert!(!has_bare_u128_arithmetic(
            "let x: u128 = a.checked_add(b).unwrap_or(0);"
        ));
        assert!(!has_bare_u128_arithmetic("let x: u64 = a + b;"));
    }

    #[test]
    fn run_skips_missing_listed_crates() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = root.parent().map_or_else(PathBuf::new, Path::to_path_buf);
        run(&root).expect("should pass when consensus crates are not yet created");
    }
}
